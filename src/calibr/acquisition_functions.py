from collections.abc import Callable
from types import ModuleType


def get_emulated_posterior_density_variance_acquisition_function(
    neg_log_prior_density: Callable,
    gp_predictive_mean_and_variance: Callable,
    numpy_module: ModuleType,
    scipy_module: ModuleType,
) -> Callable:
    def acquisition_function(inputs):
        mean, variance = gp_predictive_mean_and_variance(inputs)
        return (
            2 * neg_log_prior_density(inputs)
            - 2 * (mean + variance)
            - numpy_module.log1p(-numpy_module.exp(-variance))
        )

    return acquisition_function


def get_emulated_posterior_density_interquantile_range_acquisition_function(
    neg_log_prior_density: Callable,
    gp_predictive_mean_and_variance: Callable,
    numpy_module: ModuleType,
    scipy_module: ModuleType,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> Callable:
    lower = scipy_module.special.ndtri(quantile_interval[0])
    upper = scipy_module.special.ndtri(quantile_interval[1])

    def acquisition_function(inputs):
        mean, variance = gp_predictive_mean_and_variance(inputs)
        standard_deviation = variance**0.5
        return (
            neg_log_prior_density(inputs)
            - mean
            - upper * standard_deviation
            - numpy_module.log1p(
                -numpy_module.exp(standard_deviation * (lower - upper))
            )
        )

    return acquisition_function


def get_emulated_posterior_density_variance_greedy_batch_acquisition_function(
    neg_log_prior_density: Callable,
    gp_predictive_mean_and_variance: Callable,
    gp_lookahead_variance: Callable,
    numpy_module: ModuleType,
    scipy_module: ModuleType,
) -> Callable:
    def acquisition_function(inputs, pending_inputs):
        mean, variance = gp_predictive_mean_and_variance(inputs)
        lookahead_variance = gp_lookahead_variance(inputs, pending_inputs)
        return (
            2 * neg_log_prior_density(inputs)
            - 2 * (mean + variance)
            # TODO: check this equivalent to L394 in acquire_next_batch.m
            - numpy_module.log1p(-numpy_module.exp(-lookahead_variance))
        )

    return acquisition_function


def get_emulated_posterior_density_interquantile_range_greedy_batch_acquisition_function(
    neg_log_prior_density: Callable,
    gp_predictive_mean_and_variance: Callable,
    gp_lookahead_variance: Callable,
    numpy_module: ModuleType,
    scipy_module: ModuleType,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> Callable:
    lower = scipy_module.special.ndtri(quantile_interval[0])
    upper = scipy_module.special.ndtri(quantile_interval[1])

    def acquisition_function(inputs, pending_inputs):
        mean, variance = gp_predictive_mean_and_variance(inputs)
        standard_deviation = variance**0.5
        lookahead_variance, _ = gp_lookahead_variance(inputs, pending_inputs)
        return (
            neg_log_prior_density(inputs)
            - mean
            - upper * standard_deviation
            - numpy_module.log1p(
                -numpy_module.exp(lookahead_variance * (lower - upper))
            )
        )

    return acquisition_function


def get_emulated_posterior_density_expected_integrated_variance_acquisition_function(
    neg_log_prior_density: Callable,
    gp_predictive_mean_and_variance: Callable,
    gp_lookahead_variance: Callable,
    get_quadrature_points_and_weights: Callable,
    numpy_module: ModuleType,
    scipy_module: ModuleType,
) -> Callable:
    inputs_points, inputs_log_weights = get_quadrature_points_and_weights()
    neg_log_prior_density_input_points = neg_log_prior_density(inputs_points)
    mean_input_points, variance_input_points = gp_predictive_mean_and_variance(
        inputs_points
    )
    constant_term_input_points = (
        2 * mean_input_points
        + variance_input_points
        - 2 * neg_log_prior_density_input_points
    )

    def acquisition_function(inputs):
        _, lookahead_variance_diff = gp_lookahead_variance(inputs, inputs_points)
        return -scipy_module.special.logsumexp(
            inputs_log_weights + constant_term_input_points + lookahead_variance_diff
        )

    return acquisition_function


def get_emulated_posterior_density_integrated_median_interquantile_range_acquisition_function(
    neg_log_prior_density: Callable,
    gp_predictive_mean_and_variance: Callable,
    gp_lookahead_variance: Callable,
    get_quadrature_points_and_weights: Callable,
    numpy_module: ModuleType,
    scipy_module: ModuleType,
    quantile_interval: tuple[float, float] = (0.25, 0.75),
) -> Callable:
    lower = scipy_module.special.ndtri(quantile_interval[0])
    upper = scipy_module.special.ndtri(quantile_interval[1])
    inputs_points, inputs_log_weights = get_quadrature_points_and_weights()
    neg_log_prior_density_input_points = neg_log_prior_density(inputs_points)
    mean_input_points, _ = gp_predictive_mean_and_variance(inputs_points)
    constant_term_input_points = mean_input_points - neg_log_prior_density_input_points

    def acquisition_function(inputs):
        lookahead_variance, _ = gp_lookahead_variance(inputs, inputs_points)
        lookahead_standard_deviation = lookahead_variance**0.5
        # TODO: check negation vs. implementation in acq_imiqr in acquire_next_batch.m
        return -scipy_module.special.logsumexp(
            inputs_log_weights
            + constant_term_input_points
            + upper * lookahead_standard_deviation
            + numpy_module.log1p(
                -numpy_module.exp((lower - upper) * lookahead_standard_deviation)
            )
        )

    return acquisition_function
