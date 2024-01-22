"""Functions for minimization of objective functions."""

from collections.abc import Callable
from heapq import heappush

import jax
from scipy.optimize import minimize


class ConvergenceError(Exception):
    """Error raised when optimizer fails to converge within given computation budget."""


def minimize_with_restarts(
    objective_function: Callable[[jax.Array], jax.Array],
    sample_initial_state: Callable[[], jax.Array],
    number_minima_to_find: int = 5,
    maximum_minimize_calls: int = 100,
    minimize_method: str = "Newton-CG",
    logging_function: Callable[[str], None] = lambda _: None,
) -> tuple[jax.Array, float]:
    """Minimize a differentiable objective function with random restarts.

    Iteratively calls `scipy.optimize.minimize` to attempt to find a minimum of an
    objective function until a specified number of candidate minima are successfully
    found, with the initial state for each `minimize` called being randomly sampled
    using a user provided function. The candidate minima with the minimum value for
    the objective function is returned along with the corresponding objective function
    value.

    Args:
        objective_function: Differentiable scalar-valued function of a single flat
            vector argument to be minimized. Assumed to be specified using JAX
            primitives such that its gradient and Hessian can be computed using JAX's
            automatic differentiation support, and to be suitable for just-in-time
            compilation.
        sample_initial_state: Callable with zero arguments, which when called returns
            a random initial state for optimization of appropriate dimension.
        number_minima_to_find: Number of candidate minima of objective function to try
            to find.
        maximum_minimize_calls: Maximum number of times to try calling
            `scipy.optimize.minimize` to find candidate minima. If insufficient
            candidates are found within this number of calls then a `ConvergenceError`
            exception is raised.
        minimize_method: String specifying one of optimization methods which can be
            passed to `method` argument of `scipy.optimize.minimize`.
        logging_function: Function to use to optionally log status messages during
            minimization. Defaults to a no-op function which discards messages.

    Returns:
        Tuple with first entry the state corresponding to the best minima candidate
        found and the second entry the corresponding objective function value.
    """
    minima_found: list[tuple[jax.Array, int, jax.Array]] = []
    minimize_calls = 0
    while (
        len(minima_found) < number_minima_to_find
        and minimize_calls < maximum_minimize_calls
    ):
        logging_function(f"Starting minimize call {minimize_calls + 1}")
        results = minimize(
            jax.jit(objective_function),
            x0=sample_initial_state(),
            jac=jax.jit(jax.grad(objective_function)),
            hess=jax.jit(jax.hessian(objective_function)),
            method=minimize_method,
        )
        minimize_calls += 1
        if results.success:
            logging_function(f"Found minima with value {results.fun}")
            # Add minima to minima_found maintaining heap invariant such that first
            # entry in minima_found is always best solution so far (with counter used
            # to break ties between solutions with equal values for objective)
            heappush(minima_found, (results.fun, len(minima_found), results.x))
    if len(minima_found) < number_minima_to_find:
        msg = (
            f"Did not find required {number_minima_to_find} minima in "
            f"{maximum_minimize_calls} minimize calls."
        )
        raise ConvergenceError(msg)
    # Heap property means first entry in minima_found will correspond to solution
    # with minimum acquisition function value
    (
        min_objective_function,
        _,
        state,
    ) = minima_found[0]
    logging_function(f"Best minima found has value {min_objective_function}")
    return state, float(min_objective_function)
