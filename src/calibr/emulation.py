"""Functions and types for constructing and fitting Gaussian process emulators."""

from collections.abc import Callable
from typing import NamedTuple, TypeAlias

import jax
import numpy as np
from emul.models import gaussian_process_with_isotropic_gaussian_observations
from emul.types import (
    CovarianceFunction,
    DataDict,
    MeanFunction,
    ParametersDict,
    PosteriorPredictiveFunctionFactory,
)
from jax import Array
from jax.typing import ArrayLike
from numpy.random import Generator

from .optimization import GlobalMinimizer, GlobalMinimizerKwarg, minimize_with_restarts

try:
    import mici

    MICI_IMPORTED = True
except ImportError:
    MICI_IMPORTED = False

try:
    import arviz

    ARVIZ_IMPORTED = True
except ImportError:
    ARVIZ_IMPORTED = False


#: Type alias for function which maps flat unconstrained vector to parameter dict.
ParameterTransformer: TypeAlias = Callable[[ArrayLike], ParametersDict]

#: Type alias for function generating random values for unconstrained parameters.
UnconstrainedParametersSampler: TypeAlias = Callable[[Generator, int | None], Array]

#: Type alias for function evaluating negative logarithm of prior density.
NegativeLogPriorDensity: TypeAlias = Callable[[ArrayLike], float]


class GaussianProcessModel(NamedTuple):
    """Wrapper for functions associated with a Gaussian process model."""

    neg_log_marginal_posterior: Callable[[ArrayLike], float]
    get_posterior_functions: PosteriorPredictiveFunctionFactory
    transform_parameters: ParameterTransformer
    sample_unconstrained_parameters: UnconstrainedParametersSampler


#: Type alias for function used to generate Gaussian process model given data dict.
GaussianProcessFactory: TypeAlias = Callable[[DataDict], GaussianProcessModel]


#: Type alias for function used to fit Gaussian process model parameters.
GaussianProcessParameterFitter: TypeAlias = Callable[
    [Generator, GaussianProcessModel], ParametersDict
]


def get_gaussian_process_factory(
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
    neg_log_prior_density: NegativeLogPriorDensity,
    transform_parameters: ParameterTransformer,
    sample_unconstrained_parameters: UnconstrainedParametersSampler,
) -> GaussianProcessFactory:
    """Construct a factory function generating Gaussian process models given data.

    Args:
        mean_function: Mean function for Gaussian process.
        covariance_function: Covariance function for Gaussian process.
        neg_log_prior_density: Negative logarithm of density of prior distribution on
            vector of unconstrained parameters for Gaussian process model.
        transform_parameters: Function which maps flat unconstrained parameter vector to
            a dictionary of (potential constrained) parameters, keyed by parameter name.
        sample_unconstrained_parameters: Function generating random values for
            unconstrained vector of Gaussian process parameters.

    Returns:
        Gaussian process factory function.
    """

    def gaussian_process_factory(data: DataDict) -> GaussianProcessModel:
        (
            neg_log_marginal_likelihood,
            get_posterior_functions,
        ) = gaussian_process_with_isotropic_gaussian_observations(
            data, mean_function, covariance_function
        )

        @jax.jit
        def neg_log_marginal_posterior(unconstrained_parameters: ArrayLike) -> float:
            parameters = transform_parameters(unconstrained_parameters)
            return neg_log_prior_density(
                unconstrained_parameters
            ) + neg_log_marginal_likelihood(parameters)

        return GaussianProcessModel(
            neg_log_marginal_posterior,
            get_posterior_functions,
            transform_parameters,
            sample_unconstrained_parameters,
        )

    return gaussian_process_factory


def fit_gaussian_process_parameters_map(
    rng: Generator,
    gaussian_process: GaussianProcessModel,
    *,
    minimize_function: GlobalMinimizer = minimize_with_restarts,
    **minimize_function_kwargs: GlobalMinimizerKwarg,
) -> ParametersDict:
    """Fit parameters of Gaussian process model by maximimizing posterior density.

    Finds maximum-a-posterior (MAP) estimate of Gaussian process parameters by
    minimizing negative logarithm of posterior density on parameters given data.

    Args:
        rng: Seeded NumPy random number generator.
        gaussian_process: Tuple of functions for Gaussian process model to fit.
        minimize_function: Function used to attempt to find global minimum of negative
            log posterior density function.
        **minimize_function_kwargs: Any keyword arguments to pass to
            `minimize_function` function used to optimize negative posterior log density
            function.

    Returns:
        Dictionary of parameters corresponding to maximum-a-posteriori estimate.
    """
    if minimize_function is minimize_with_restarts:
        minimize_function_kwargs.setdefault("number_minima_to_find", 1)
        minimize_function_kwargs.setdefault("maximum_minimize_calls", 100)
        minimize_function_kwargs.setdefault("minimize_method", "Newton-CG")
    unconstrained_parameters, _ = minimize_function(
        objective_function=gaussian_process.neg_log_marginal_posterior,
        sample_initial_state=lambda r: gaussian_process.sample_unconstrained_parameters(
            r,
            None,
        ),
        rng=rng,
        **minimize_function_kwargs,
    )
    return gaussian_process.transform_parameters(unconstrained_parameters)


if MICI_IMPORTED:

    def fit_gaussian_process_parameters_hmc(
        rng: Generator,
        gaussian_process: GaussianProcessModel,
        *,
        n_chain: int = 1,
        n_warm_up_iter: int = 500,
        n_main_iter: int = 1000,
        r_hat_threshold: float | None = None,
    ) -> ParametersDict:
        """
        Fit parameters of Gaussian process model by sampling posterior using HMC.

        Uses Hamiltonian Monte Carlo (HMC) to generate chain(s) of samples approximating
        posterior distribution on Gaussian process parameters given data.

        Args:
            rng: Seeded NumPy random number generator.
            gaussian_process: Tuple of functions for Gaussian process model to fit.
            n_chain: Number of Markov chains to simulate.
            n_warm_up_iter: Number of adaptive warm-up iterations to run for each chain.
            n_main_iter: Number of main sampling stage iterations to run for each chain.
            r_hat_threshold: If not `None`, specifies a maximum value for the
                (rank-normalized, split) R-hat convergence diagnostic computed from the
                chains, with R-hat values exceeding this threshold leading to an
                exception being raised. Requires `n_chain > 1` and for ArviZ package to
                be installed.

        Returns:
            Dictionary of parameters corresponding to approximate posterior sample.
        """
        if r_hat_threshold is not None and not ARVIZ_IMPORTED:
            msg = "R-hat convergence checks require ArviZ to be installed"
            raise RuntimeError(msg)
        value_and_grad_neg_log_marginal_posterior = jax.jit(
            jax.value_and_grad(gaussian_process.neg_log_marginal_posterior)
        )

        def grad_neg_log_marginal_posterior(
            unconstrained_variables: ArrayLike,
        ) -> tuple[Array, float]:
            value, grad = value_and_grad_neg_log_marginal_posterior(
                unconstrained_variables
            )
            return np.asarray(grad), float(value)

        init_states = gaussian_process.sample_unconstrained_parameters(rng, n_chain)

        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=gaussian_process.neg_log_marginal_posterior,
            grad_neg_log_dens=grad_neg_log_marginal_posterior,
        )
        integrator = mici.integrators.LeapfrogIntegrator(system)
        sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)

        final_states, traces, _ = sampler.sample_chains(
            n_warm_up_iter,
            n_main_iter,
            init_states,
            monitor_stats=["accept_stat", "step_size", "n_step", "diverging"],
            adapters=[
                mici.adapters.DualAveragingStepSizeAdapter(0.8),
                mici.adapters.OnlineCovarianceMetricAdapter(),
            ],
            n_process=1,
        )

        if n_chain > 1 and r_hat_threshold is not None:
            max_rhat = float(arviz.rhat(traces).to_array().max())
            if max_rhat > r_hat_threshold:
                msg = f"Chain convergence issue: max rank-normalized R-hat {max_rhat}"
                raise RuntimeError(msg)
        return gaussian_process.transform_parameters(final_states[0].pos)
