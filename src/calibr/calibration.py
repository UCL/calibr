"""Functions for iteratively calibrating the parameters of a probabilistic model."""

from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np
from emul.types import DataDict, ParametersDict, PosteriorPredictiveMeanAndVariance
from jax import Array
from jax.typing import ArrayLike
from numpy.random import Generator

from .acquisition_functions import (
    AcquisitionFunction,
    AcquisitionFunctionFactory,
    get_integrated_median_interquantile_range_acquisition_function,
)
from .emulation import (
    GaussianProcessFactory,
    GaussianProcessModel,
    GaussianProcessParameterFitter,
    fit_gaussian_process_parameters_map,
)
from .optimization import GlobalMinimizer, minimize_with_restarts

#: Type alias for function generating random initial values for inputs.
InitialInputSampler: TypeAlias = Callable[[Generator, int], Array]

#: Type alias for callback function called at end of calibration iteration.
EndOfIterationCallback = Callable[
    [int, PosteriorPredictiveMeanAndVariance, DataDict, ArrayLike, float], None
]


def get_next_inputs_batch_by_joint_optimization(
    rng: Generator,
    acquisition_function: AcquisitionFunction,
    sample_initial_inputs: InitialInputSampler,
    batch_size: int,
    *,
    minimize_function: GlobalMinimizer = minimize_with_restarts,
    **minimize_function_kwargs,
) -> tuple[Array, float]:
    """
    Get next batch of inputs to evaluate by jointly optimizing acquisition function.

    Minimizes acquisition function over product of `batch_size` input spaces.

    Args:
        rng: NumPy random number generator for initializing optimization runs.
        acquisition_function: Scalar-valued function of a batch of inputs to optimize to
            find new batch of inputs to evaluate model for.
        sample_initial_inputs: Function outputting reasonable random initial values for
            batch of inputs when passed a random number generator and batch size. Used
            to initialize state for optimization runs.
        batch_size: Number of inputs in batch.
        minimize_function: Function used to attempt to find minimum of acquisition
            function.
        **minimize_function_kwargs: Any keyword arguments to pass to
            `minimize_function` function used to optimize acquisition function.

    Returns:
        Tuple of optimized inputs batch and corresponding value of acquisition function.
    """

    def acquisition_function_flat_input(flat_inputs: ArrayLike) -> float:
        return acquisition_function(flat_inputs.reshape((batch_size, -1)))

    if minimize_function is minimize_with_restarts:
        minimize_function_kwargs.setdefault("number_minima_to_find", 5)
        minimize_function_kwargs.setdefault("maximum_minimize_calls", 100)
        minimize_function_kwargs.setdefault("minimize_method", "Newton-CG")

    flat_inputs, min_acquisition_function = minimize_function(
        objective_function=acquisition_function_flat_input,
        sample_initial_state=lambda r: sample_initial_inputs(r, batch_size).flatten(),
        rng=rng,
        **minimize_function_kwargs,
    )

    return flat_inputs.reshape((batch_size, -1)), min_acquisition_function


def get_next_inputs_batch_by_greedy_optimization(
    rng: Generator,
    acquisition_function: AcquisitionFunction,
    sample_initial_inputs: InitialInputSampler,
    batch_size: int,
    *,
    minimize_function: GlobalMinimizer = minimize_with_restarts,
    **minimize_function_kwargs,
) -> tuple[Array, float]:
    """
    Get next batch of inputs to evaluate by greedily optimizing acquisition function.

    Sequentially minimizes acquisition function for `b` in 1 to `batch_size` by fixing
    `b - 1` inputs already optimized and minimizing over a single new input in each
    iteration.

    Args:
        rng: NumPy random number generator for initializing optimization runs.
        acquisition_function: Scalar-valued function of a batch of inputs to optimize to
            find new batch of inputs to evaluate model for.
        sample_initial_inputs: Function outputting reasonable random initial values for
            batch of inputs when passed a random number generator and batch size. Used
            to initialize state for optimization runs.
        batch_size: Number of inputs in batch.
        minimize_function: Function used to attempt to find minimum of (sequence of)
            acquisition functions.
        **minimize_function_kwargs: Any keyword arguments to pass to
            `minimize_function` function used to optimize acquisition function.

    Returns:
        Tuple of optimized inputs batch and corresponding value of acquisition function.
    """

    def acquisition_function_greedy(
        current_input: ArrayLike, fixed_inputs: list[ArrayLike]
    ) -> float:
        return acquisition_function(jnp.stack([current_input, *fixed_inputs]))

    fixed_inputs: list[ArrayLike] = []
    for _ in range(batch_size):
        current_input, min_acquisition_function = minimize_function(
            objective_function=partial(
                acquisition_function_greedy, fixed_inputs=fixed_inputs
            ),
            sample_initial_state=lambda r: sample_initial_inputs(r, 1).flatten(),
            rng=rng,
            **minimize_function_kwargs,
        )
        fixed_inputs.append(current_input)

    return np.stack(fixed_inputs), min_acquisition_function


def calibrate(  # noqa: PLR0913
    num_initial_inputs: int,
    batch_size: int,
    num_iterations: int,
    rng: Generator,
    sample_initial_inputs: InitialInputSampler,
    posterior_log_density_batch: Callable[[ArrayLike], Array],
    gaussian_process_factory: GaussianProcessFactory,
    get_integration_points_and_log_weights: Callable[
        [Generator, InitialInputSampler, PosteriorPredictiveMeanAndVariance],
        tuple[Array, Array],
    ],
    *,
    fit_gaussian_process_parameters: GaussianProcessParameterFitter = (
        fit_gaussian_process_parameters_map
    ),
    get_acquisition_function: AcquisitionFunctionFactory = (
        get_integrated_median_interquantile_range_acquisition_function
    ),
    get_next_inputs_batch: Callable[
        [Generator, AcquisitionFunction, InitialInputSampler, int], tuple[Array, float]
    ] = get_next_inputs_batch_by_joint_optimization,
    end_of_iteration_callback: EndOfIterationCallback | None = None,
) -> tuple[GaussianProcessModel, DataDict, ParametersDict]:
    """
    Estimate the posterior on the unknown inputs of an (expensive to evaluate) model.

    Iterates evaluating log density for posterior at a batch of inputs (unknown
    variables to infer posterior on), fitting a Gaussian process to the log density
    evaluations so far and optimizing an acquisition function using the Gaussian process
    emulator to choose a new batch of input points at which to evaluate the log density
    (by minimizing a measure of the expected uncertainty in the emulator about the log
    posterior density function).

    Args:
        num_initial_inputs: Number of initial inputs to evaluate posterior log density
            at to initialize calibration.
        batch_size: Size of batch of inputs to optimize for and evaluate log density at
            in each calibration iteration.
        num_iterations: Number of calibration iterations to perform. Total number of
            model posterior log density evaluations is
            `num_initial_inputs + batch_size * num_iterations`.
        rng: Seeded NumPy random number generator.
        sample_initial_inputs: Function outputting reasonable random initial values for
            batch of inputs when passed a random number generator and batch size.
        posterior_log_density_batch: Function computing logarithm of (unnormalized)
            posterior density on model inputs, for a batch of inputs (with passed
            argument being a two dimensional array with first dimension the batch
            index).
        gaussian_process_factory: Factory function generating Gaussian process models
            given a data dictionary.
        get_integration_points_and_log_weights: Function which outputs points in input
            space and corresponding (log) weights by which to estimate integrals over
            the input space in the acquisition function as a weighted sum. The function
            is passed a seeded random number generator, a function to sample random
            points in the input space and the current Gaussian process posterior
            predictive mean and variance function. The input points and weights may for
            example be generated according to a (deterministic) numerical quadrature
            rule for low dimensionalities, a stochastic (quasi-) Monte Carlo scheme for
            moderate dimensionalities or a Markov chain Monte Carlo or sequential Monte
            Carlo scheme for higher dimensionalities.
        fit_gaussian_process_parameters: Function which fits the parameters of a
            Gaussian process model given the current data (input-output pairs). Passed
            a seeded random number generator and tuple of Gaussian process model
            functions.
        get_acquisition_function: Factory function generating acquisition functions
            given Gaussian process posterior predictive functions.
        get_next_inputs_batch: Function which computes next batch of inputs to evaluate
            model at by optimizing the current acquisition function. Passed a seeded
            random number generator, acquisition function, input sampler and batch size.
        end_of_iteration_callback: Optional callback function evaluate at end of each
            calibration iteration, for example for logging metrics or plotting / saving
            intermediate outputs. Passed current iteration index, Gaussian process
            posterior mean and variance, data dictionary with all inputs and
            corresponding log density evaluations so far, batch of inputs selected in
            current iteration and corresponding optimized acquisition function value.

    Returns:
        Tuple of Gaussian process model, data dictionary containing all model inputs
        and log density evaluations and fitted Gaussian process model parameters at
        final iteration.
    """
    inputs = sample_initial_inputs(rng, num_initial_inputs)
    data = {"inputs": inputs, "outputs": posterior_log_density_batch(inputs)}
    for iteration_index in range(num_iterations + 1):
        gaussian_process = gaussian_process_factory(data)
        parameters = fit_gaussian_process_parameters(rng, gaussian_process)
        (
            posterior_mean_and_variance,
            lookahead_variance_reduction,
        ) = gaussian_process.get_posterior_functions(parameters)
        if iteration_index == num_iterations:
            # In final iteration, only fit Gaussian process to all model evaluations
            # computed so far without acquiring a new set of inputs to evaluate
            break
        (
            integration_inputs,
            integration_log_weights,
        ) = get_integration_points_and_log_weights(
            rng, sample_initial_inputs, posterior_mean_and_variance
        )
        acquisition_function = get_acquisition_function(
            posterior_mean_and_variance,
            lookahead_variance_reduction,
            integration_inputs,
            integration_log_weights,
        )
        next_inputs, acquisition_function_value = get_next_inputs_batch(
            rng, acquisition_function, sample_initial_inputs, batch_size
        )
        next_outputs = posterior_log_density_batch(next_inputs)
        data = {
            "inputs": np.concatenate((data["inputs"], next_inputs)),
            "outputs": np.concatenate((data["outputs"], next_outputs)),
        }
        if end_of_iteration_callback is not None:
            end_of_iteration_callback(
                iteration_index,
                posterior_mean_and_variance,
                data,
                next_inputs,
                acquisition_function_value,
            )
    return gaussian_process, data, parameters
