"""Acquisition functions for selecting new inputs points to evaluate model at."""

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from emul.types import (
    PosteriorPredictiveLookaheadVarianceReduction,
    PosteriorPredictiveMeanAndVariance,
)
from jax import Array
from jax.typing import ArrayLike

#: Type alias for acquisition functions which score a batch of inputs.
AcquisitionFunction: TypeAlias = Callable[[ArrayLike], float | Array]

#: Type alias for functions constructing acquisition function given Gaussian process
AcquisitionFunctionFactory: TypeAlias = Callable[
    [PosteriorPredictiveMeanAndVariance, PosteriorPredictiveLookaheadVarianceReduction],
    AcquisitionFunction,
]


def get_maximum_variance_greedy_batch_acquisition_functions(
    gp_mean_and_variance: PosteriorPredictiveMeanAndVariance,
    gp_lookahead_variance_reduction: PosteriorPredictiveLookaheadVarianceReduction,
) -> Callable:
    """
    Construct acquisition functions for greedy maximisation of variance.

    Selects next input points to evaluate target log-density at which maximise variance
    of an unnormalized target density approximation based on a Gaussian process emulator
    for the log-density function. After an initial point is chosen, subsequent points in
    the batch are selected in a greedy fashion by maximising the variance given the
    already selected point(s), assuming the log-density values at the already selected
    points are distributed according the Gaussian process predictive distribution.

    Args:
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for target log-density on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-density at a test-point given one or
            'pending' input points, when outputs associated with pending inputs are
            assumed to follow the posterior predictive distribution under the Gaussian
            process.

    Returns:
        The acquisition function to minimize to select new input point(s). The function
        takes one or two arguments. If a single argument is passed it corresponds to
        the acquisition function for the initial point in a batch. If two arguments are
        passed, the first argument corresponds to the new input point being chosen and
        the second argument to the already selected input point(s) in the batch.
    """

    def acquisition_function(
        new_input: ArrayLike, pending_inputs: ArrayLike | None = None
    ) -> Array:
        mean, variance = gp_mean_and_variance(new_input)
        if pending_inputs is None:
            lookahead_variance_reduction = 0
        else:
            lookahead_variance_reduction = gp_lookahead_variance_reduction(
                new_input, pending_inputs
            )
        lookahead_variance = variance - lookahead_variance_reduction
        return -2 * (mean + variance) - jnp.log1p(-jnp.exp(-lookahead_variance))

    return acquisition_function


def get_maximum_interquantile_range_greedy_batch_acquisition_functions(
    gp_mean_and_variance: PosteriorPredictiveMeanAndVariance,
    gp_lookahead_variance_reduction: PosteriorPredictiveLookaheadVarianceReduction,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> Callable:
    """
    Construct acquisition function for greedy maximisation of interquantile range.

    Selects next input points to evaluate target log-density at which maximise
    interquantile range of an unnormalized target density approximation based on a
    Gaussian process emulator for the log-density function. After an initial point is
    chosen, subsequent points in the batch are selected in a greedy fashion by
    maximising the interquantile range given the already selected point(s), assuming the
    log-density values at the already selected points are distributed according the
    Gaussian process predictive distribution.

    Args:
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for target log-density on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-density at a test-point given one or
            'pending' input points, when outputs associated with pending inputs are
            assumed to follow the posterior predictive distribution under the Gaussian
            process.
        quantile_interval: Lower and upper quantiles specifying inter-quantile range
            to optimize.

    Returns:
        The acquisition function to minimize to select new input point(s). The function
        takes one or two arguments. If a single argument is passed it corresponds to
        the acquisition function for the initial point in a batch. If two arguments are
        passed, the first argument corresponds to the new input point being chosen and
        the second argument to the already selected input point(s) in the batch.
    """
    lower = jsp.special.ndtri(quantile_interval[0])
    upper = jsp.special.ndtri(quantile_interval[1])

    def acquisition_function(
        new_input: ArrayLike, pending_inputs: ArrayLike | None = None
    ) -> Array:
        mean, variance = gp_mean_and_variance(new_input)
        if pending_inputs is None:
            lookahead_variance_reduction = 0
        else:
            lookahead_variance_reduction = gp_lookahead_variance_reduction(
                new_input, pending_inputs
            )
        lookahead_standard_deviation = (
            abs(variance - lookahead_variance_reduction) ** 0.5
        )
        return (
            -mean
            - upper * lookahead_standard_deviation
            - jnp.log1p(-jnp.exp(lookahead_standard_deviation * (lower - upper)))
        )

    return acquisition_function


def get_expected_integrated_variance_acquisition_function(
    gp_mean_and_variance: PosteriorPredictiveMeanAndVariance,
    gp_lookahead_variance_reduction: PosteriorPredictiveLookaheadVarianceReduction,
    integration_inputs: ArrayLike,
    integration_log_weights: ArrayLike,
) -> AcquisitionFunction:
    """
    Construct acquisition function for minimising expected integrated variance.

    Selects next input points to evaluate log-likelihood at which minimizes the
    expectation of the integral over the input space of the variance of an unnormalized
    posterior density approximation based on a Gaussian process emulator for the
    log-likelihood function, with the expectation being over the posterior predictive
    distribution on the unnormalized target density under the Gaussian process.

    Args:
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for target log-density on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-density at a test-point given one or
            'pending' input points, when outputs associated with pending inputs are
            assumed to follow the posterior predictive distribution under the Gaussian
            process.
        integration_inputs: Points to use when approximating integrals over input space.
        integration_log_weights: Logarithm of weights associated with each of points
            in `integration_inputs`.

    Returns:
        The acquisition function to minimize to select new input point(s).
    """
    mean_integration_inputs, variance_integration_inputs = jax.vmap(
        gp_mean_and_variance
    )(integration_inputs)

    def acquisition_function(new_inputs: ArrayLike) -> Array:
        lookahead_variance_reduction_integration_inputs = jax.vmap(
            gp_lookahead_variance_reduction, (0, None)
        )(integration_inputs, new_inputs)
        # We neglect the initial constant wrt θ* term in
        # Lᵛₜ(θ*) = ∫ exp(2mₜ(θ) + s²ₜ(θ)) (exp(s²ₜ(θ)) - exp(τ²ₜ(θ; θ*))) dθ
        # and use
        # -log ∫ exp(2mₜ(θ) + s²ₜ(θ) + τ²ₜ(θ; θ*)) dθ
        # corresponding to the negative logarithm of the negation of the second term
        # in the expected integrated variance design criterion.
        # This appears to give a more numerically stable objective function.
        return -jsp.special.logsumexp(
            integration_log_weights
            + 2 * mean_integration_inputs
            + variance_integration_inputs
            + lookahead_variance_reduction_integration_inputs
        )

    return acquisition_function


def get_integrated_median_interquantile_range_acquisition_function(
    gp_mean_and_variance: PosteriorPredictiveMeanAndVariance,
    gp_lookahead_variance_reduction: PosteriorPredictiveLookaheadVarianceReduction,
    integration_inputs: ArrayLike,
    integration_log_weights: ArrayLike,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> AcquisitionFunction:
    """
    Construct acquisition function for minimising integrated median interquantile range.

    Selects next input points to evaluate target log-density at which minimizes the
    integral over the input space of the median interquantile range of an unnormalized
    target density approximation based on a Gaussian process emulator for the
    log-density function, with the median being over the posterior predictive
    distribution on the unnormalized target density under the Gaussian process.

    Args:
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for target log-density on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-density at a test-point given one or
            'pending' input points, when outputs associated with pending inputs are
            assumed to follow the posterior predictive distribution under the Gaussian
            process.
        integration_inputs: Points to use when approximate integrals over input space.
        integration_log_weights: Logarithm of weights associated with each of points
            in `integration_inputs`.
        quantile_interval: Lower and upper quantiles specifying inter-quantile range
            to optimize.

    Returns:
        The acquisition function to minimize to select new input point(s).
    """
    lower = jsp.special.ndtri(quantile_interval[0])
    upper = jsp.special.ndtri(quantile_interval[1])
    mean_integration_inputs, variance_integration_inputs = jax.vmap(
        gp_mean_and_variance
    )(integration_inputs)

    def acquisition_function(new_inputs: ArrayLike) -> Array:
        lookahead_variance_reduction_integration_inputs = jax.vmap(
            gp_lookahead_variance_reduction, (0, None)
        )(integration_inputs, new_inputs)
        lookahead_standard_deviation_integration_inputs = (
            abs(
                variance_integration_inputs
                - lookahead_variance_reduction_integration_inputs
            )
            ** 0.5
        )
        return jsp.special.logsumexp(
            integration_log_weights
            + mean_integration_inputs
            + upper * lookahead_standard_deviation_integration_inputs
            + jnp.log1p(
                -jnp.exp(
                    (lower - upper) * lookahead_standard_deviation_integration_inputs
                )
            )
        )

    return acquisition_function
