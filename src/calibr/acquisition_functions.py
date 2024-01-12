"""Acquisition functions for selecting new inputs points to evaluate model at."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.typing import ArrayLike


def get_maximum_variance_greedy_batch_acquisition_functions(
    neg_log_prior_density: Callable,
    gp_mean_and_variance: Callable,
    gp_lookahead_variance_reduction: Callable,
) -> tuple[Callable, Callable]:
    """
    Construct acquisition functions for greedy maximisation of variance.

    Selects next input points to evaluate log-likelihood at which maximise variance of
    an unnormalized posterior density approximation based on a Gaussian process emulator
    for the log-likelihood function. After an initial point is chosen, subsequent points
    in the batch are selected in a greedy fashion by maximising the variance given the
    already selected point(s), assuming the log-likelihood values at the already
    selected points are distributed according the Gaussian process predictive
    distribution.

    Args:
        neg_log_prior_density: Function evaluating negative logarithm of density of
            prior distribution on input space.
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for log-likelihood on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-likelihood input space at a test-point
            given one or 'pending' input points, when outputs associated with pending
            inputs are assumed to follow the posterior predictive distribution under
            the Gaussian process.

    Returns:
        Tuple of functions, with first function the acquisition function to minimize
        to select initial point in batch, and second function the acquisition function
        to maximize (over first argument) for subsequent points, with already chosen
        input points passed as the second argument.
    """

    def initial_acquisition_function(new_input: ArrayLike) -> Array:
        mean, variance = gp_mean_and_variance(new_input)
        return (
            2 * neg_log_prior_density(new_input)
            - 2 * (mean + variance)
            - jnp.log1p(-jnp.exp(-variance))
        )

    def acquisition_function(new_input: ArrayLike, pending_inputs: ArrayLike) -> Array:
        mean, variance = gp_mean_and_variance(new_input)
        lookahead_variance_reduction = gp_lookahead_variance_reduction(
            new_input, pending_inputs
        )
        lookahead_variance = variance - lookahead_variance_reduction
        return (
            2 * neg_log_prior_density(new_input)
            - 2 * (mean + variance)
            - jnp.log1p(-jnp.exp(-lookahead_variance))
        )

    return initial_acquisition_function, acquisition_function


def get_maximum_interquantile_range_greedy_batch_acquisition_functions(
    neg_log_prior_density: Callable,
    gp_mean_and_variance: Callable,
    gp_lookahead_variance_reduction: Callable,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> tuple[Callable, Callable]:
    """
    Construct acquisition functions for greedy maximisation of interquantile range.

    Selects next input points to evaluate log-likelihood at which maximise interquantile
    range of an unnormalized posterior density approximation based on a Gaussian process
    emulator for the log-likelihood function. After an initial point is chosen,
    subsequent points in the batch are selected in a greedy fashion by maximising the
    interquantile range given the already selected point(s), assuming the log-likelihood
    values at the already selected points are distributed according the Gaussian process
    predictive distribution.

    Args:
        neg_log_prior_density: Function evaluating negative logarithm of density of
            prior distribution on input space.
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for log-likelihood on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-likelihood input space at a test-point
            given one or 'pending' input points, when outputs associated with pending
            inputs are assumed to follow the posterior predictive distribution under
            the Gaussian process.
        quantile_interval: Lower and upper quantiles specifying inter-quantile range
            to optimize.

    Returns:
        Tuple of functions, with first function the acquisition function to minimize
        to select initial point in batch, and second function the acquisition function
        to maximize (over first argument) for subsequent points, with already chosen
        input points passed as the second argument.
    """
    lower = jsp.special.ndtri(quantile_interval[0])
    upper = jsp.special.ndtri(quantile_interval[1])

    def initial_acquisition_function(new_input: ArrayLike) -> Array:
        mean, variance = gp_mean_and_variance(new_input)
        standard_deviation = variance**0.5
        return (
            neg_log_prior_density(new_input)
            - mean
            - upper * standard_deviation
            - jnp.log1p(-jnp.exp(standard_deviation * (lower - upper)))
        )

    def acquisition_function(new_input: ArrayLike, pending_inputs: ArrayLike) -> Array:
        mean, variance = gp_mean_and_variance(new_input)
        standard_deviation = variance**0.5
        lookahead_variance_reduction = gp_lookahead_variance_reduction(
            new_input, pending_inputs
        )
        lookahead_variance = variance - lookahead_variance_reduction
        return (
            neg_log_prior_density(new_input)
            - mean
            - upper * standard_deviation
            - jnp.log1p(-jnp.exp(lookahead_variance * (lower - upper)))
        )

    return initial_acquisition_function, acquisition_function


def get_expected_integrated_variance_acquisition_function(
    neg_log_prior_density: Callable,
    gp_mean_and_variance: Callable,
    gp_lookahead_variance_reduction: Callable,
    quadrature_inputs: ArrayLike,
    quadrature_log_weights: ArrayLike,
) -> Callable:
    """
    Construct acquisition function for minimising expected integrated variance.

    Selects next input points to evaluate log-likelihood at which minimizes the
    expectation of the integral over the input space of the variance of an unnormalized
    posterior density approximation based on a Gaussian process emulator
    for the log-likelihood function, with the expectation being over the posterior
    predictive distribution on the unnormalized posterior under the Gaussian process.

    Args:
        neg_log_prior_density: Function evaluating negative logarithm of density of
            prior distribution on input space.
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for log-likelihood on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-likelihood input space at a test-point
            given one or 'pending' input points, when outputs associated with pending
            inputs are assumed to follow the posterior predictive distribution under
            the Gaussian process.
        quadrature_inputs: Quadrature points to approximate integrals over input space.
        quadrature_log_weights: Logarithm of weights associated with each of points
            in `quadrature_inputs`.

    Returns:
        The acquisition function to minimize to select new input point(s).
    """
    neg_log_prior_density_quadrature_inputs = jax.vmap(neg_log_prior_density)(
        quadrature_inputs
    )
    mean_quadrature_inputs, variance_quadrature_inputs = jax.vmap(gp_mean_and_variance)(
        quadrature_inputs
    )

    def acquisition_function(new_inputs: ArrayLike) -> Array:
        lookahead_variance_reduction_quadrature_inputs = jax.vmap(
            gp_lookahead_variance_reduction, (0, None)
        )(quadrature_inputs, new_inputs)
        # We neglect the initial constant wrt θ* term in
        # Lᵛₜ(θ*) = ∫ π²(θ) exp(2mₜ(θ) + s²ₜ(θ)) (exp(s²ₜ(θ)) - exp(τ²ₜ(θ; θ*))) dθ
        # and use
        # -log ∫ π²(θ) exp(2mₜ(θ) + s²ₜ(θ) + τ²ₜ(θ; θ*)) dθ
        # corresponding to the negative logarithm of the negation of the second term
        # in the expected integrated variance design criterion.
        # This appears to give a more numerically stable objective function.
        return -jsp.special.logsumexp(
            quadrature_log_weights
            - 2 * neg_log_prior_density_quadrature_inputs
            + 2 * mean_quadrature_inputs
            + variance_quadrature_inputs
            + lookahead_variance_reduction_quadrature_inputs
        )

    return acquisition_function


def get_integrated_median_interquantile_range_acquisition_function(
    neg_log_prior_density: Callable,
    gp_mean_and_variance: Callable,
    gp_lookahead_variance_reduction: Callable,
    quadrature_inputs: ArrayLike,
    quadrature_log_weights: ArrayLike,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> Callable:
    """
    Construct acquisition function for minimising integrated median interquantile range.

    Selects next input points to evaluate log-likelihood at which minimizes the integral
    over the input space of the median interquantile range of an unnormalized posterior
    density approximation based on a Gaussian process emulator for the log-likelihood
    function, with the median being over the posterior predictive distribution on the
    unnormalized posterior under the Gaussian process.

    Args:
        neg_log_prior_density: Function evaluating negative logarithm of density of
            prior distribution on input space.
        gp_mean_and_variance: Function evaluating mean and variance of Gaussian process
            emulator for log-likelihood on input space.
        gp_lookahead_variance_reduction: Function evaluating reduction in variance of
            Gaussian process emulator for log-likelihood input space at a test-point
            given one or 'pending' input points, when outputs associated with pending
            inputs are assumed to follow the posterior predictive distribution under
            the Gaussian process.
        quadrature_inputs: Quadrature points to approximate integrals over input space.
        quadrature_log_weights: Logarithm of weights associated with each of points
            in `quadrature_inputs`.
        quantile_interval: Lower and upper quantiles specifying inter-quantile range
            to optimize.

    Returns:
        The acquisition function to minimize to select new input point(s).
    """
    lower = jsp.special.ndtri(quantile_interval[0])
    upper = jsp.special.ndtri(quantile_interval[1])
    neg_log_prior_density_quadrature_inputs = jax.vmap(neg_log_prior_density)(
        quadrature_inputs
    )
    mean_quadrature_inputs, variance_quadrature_inputs = jax.vmap(gp_mean_and_variance)(
        quadrature_inputs
    )

    def acquisition_function(new_inputs: ArrayLike) -> Array:
        lookahead_variance_reduction_quadrature_inputs = jax.vmap(
            gp_lookahead_variance_reduction, (0, None)
        )(quadrature_inputs, new_inputs)
        lookahead_standard_deviation_quadrature_inputs = (
            variance_quadrature_inputs - lookahead_variance_reduction_quadrature_inputs
        ) ** 0.5
        return jsp.special.logsumexp(
            quadrature_log_weights
            + mean_quadrature_inputs
            - neg_log_prior_density_quadrature_inputs
            + upper * lookahead_standard_deviation_quadrature_inputs
            + jnp.log1p(
                -jnp.exp(
                    (lower - upper) * lookahead_standard_deviation_quadrature_inputs
                )
            )
        )

    return acquisition_function
