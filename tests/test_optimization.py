"""Tests for optimization module."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np
import pytest
from numpy.typing import ArrayLike

from calibr.optimization import (
    GlobalMinimizer,
    ObjectiveFunction,
    basin_hopping,
    hessian_vector_product,
    minimize_with_restarts,
)

jax.config.update("jax_enable_x64", val=True)
jax.config.update("jax_platform_name", "cpu")


class ObjectiveFunctionTestCase(NamedTuple):
    """Objective function test case for optimization plus associated functions."""

    objective_function: ObjectiveFunction
    hvp_objective_function: Callable[[ArrayLike, ArrayLike], jax.Array]


OBJECTIVE_FUNCTION_TEST_CASES = {
    "quadratic_function": ObjectiveFunctionTestCase(
        lambda x: (x**2).sum() / 2,
        lambda _, v: v,
    ),
    "quartic_function": ObjectiveFunctionTestCase(
        lambda x: (x**4).sum() / 12,
        lambda x, v: x**2 * v,
    ),
    "rosenbrock_function": ObjectiveFunctionTestCase(
        lambda x: 100 * ((x[1:] - x[:-1] ** 2) ** 2).sum() + ((1 - x[:-1]) ** 2).sum(),
        lambda x, v: 200 * np.pad(v[1:] - 2 * x[:-1] * v[:-1], (1, 0))
        + np.pad(
            400 * (3 * x[:-1] ** 2 * v[:-1] - x[:-1] * v[1:] - x[1:] * v[:-1])
            + 2 * v[:-1],
            (0, 1),
        ),
    ),
}


@pytest.mark.parametrize("dimension", [1, 2, 5])
@pytest.mark.parametrize(
    "objective_function_test_case",
    list(OBJECTIVE_FUNCTION_TEST_CASES.values()),
    ids=list(OBJECTIVE_FUNCTION_TEST_CASES.keys()),
)
def test_hessian_vector_product(
    rng: np.random.Generator,
    dimension: int,
    objective_function_test_case: ObjectiveFunctionTestCase,
) -> None:
    """Check JAX hessian_vector_product implementation against manual derivations."""
    hvp = hessian_vector_product(objective_function_test_case.objective_function)
    x, v = rng.standard_normal((2, dimension))
    assert np.allclose(
        hvp(x, v), objective_function_test_case.hvp_objective_function(x, v)
    )


@pytest.mark.parametrize("dimension", [2, 5])
@pytest.mark.parametrize("global_minimizer", [minimize_with_restarts, basin_hopping])
@pytest.mark.parametrize(
    "objective_function_test_case",
    list(OBJECTIVE_FUNCTION_TEST_CASES.values()),
    ids=list(OBJECTIVE_FUNCTION_TEST_CASES.keys()),
)
def test_global_minimizer(
    rng: np.random.Generator,
    dimension: int,
    global_minimizer: GlobalMinimizer,
    objective_function_test_case: ObjectiveFunctionTestCase,
) -> None:
    """Check global minimizer functions find at least a local minimimum."""
    objective_function = objective_function_test_case.objective_function
    minima, objective_at_minima = global_minimizer(
        objective_function,
        lambda r: r.standard_normal(dimension),
        rng,
        minimize_tol=1e-8,
    )
    assert np.allclose(objective_at_minima, objective_function(minima))
    gradient_at_minima = jax.grad(objective_function)(minima)
    hessian_at_minima = jax.hessian(objective_function)(minima)
    assert np.allclose(gradient_at_minima, np.zeros(dimension), rtol=1e-6, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(hessian_at_minima) > 0)
