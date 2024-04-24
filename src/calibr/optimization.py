"""Functions for minimization of objective functions."""

from collections.abc import Callable
from heapq import heappush
from typing import Protocol, TypeAlias

import jax
from jax.typing import ArrayLike
from numpy import ndarray
from numpy.random import Generator
from scipy.optimize import basinhopping as _basin_hopping
from scipy.optimize import minimize as _minimize


class ConvergenceError(Exception):
    """Error raised when optimizer fails to converge within given computation budget."""


#: Type alias for scalar-valued objective function of array argument
ObjectiveFunction: TypeAlias = Callable[[ArrayLike], float]

#: Type alias for function sampling initial optimization state given random generator
InitialStateSampler: TypeAlias = Callable[[Generator], ndarray]


class GlobalMinimizer(Protocol):
    """Function which attempts to find global minimum of a scalar objective function."""

    def __call__(
        self,
        objective_function: ObjectiveFunction,
        sample_initial_state: InitialStateSampler,
        rng: Generator,
    ) -> tuple[jax.Array, float]:
        """
        Minimize a differentiable objective function.

        Args:
            objective_function: Differentiable scalar-valued function of a single flat
                vector argument to be minimized. Assumed to be specified using JAX
                primitives such that its gradient and Hessian can be computed using
                JAX's automatic differentiation support, and to be suitable for
                just-in-time compilation.
            sample_initial_state: Callable with one argument, which when passed a NumPy
                random number generator returns a random initial state for optimization
                of appropriate dimension.
            rng: Seeded NumPy random number generator.

        Returns:
            Tuple with first entry the state corresponding to the minima point and the
            second entry the corresponding objective function value.
        """


def hessian_vector_product(
    scalar_function: ObjectiveFunction,
) -> Callable[[ArrayLike, ArrayLike], jax.Array]:
    """
    Construct function to compute Hessian-vector product for scalar-valued function.

    Args:
        scalar_function: Scalar-valued objective function.

    Returns:
        Hessian-vector product function.
    """

    def hvp(x: ArrayLike, v: ArrayLike) -> jax.Array:
        return jax.jvp(jax.grad(scalar_function), (x,), (v,))[1]

    return hvp


def minimize_with_restarts(
    objective_function: ObjectiveFunction,
    sample_initial_state: InitialStateSampler,
    rng: Generator,
    *,
    number_minima_to_find: int = 5,
    maximum_minimize_calls: int = 100,
    minimize_method: str = "Newton-CG",
    minimize_max_iterations: int | None = None,
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
        sample_initial_state: Callable with one argument, which when passed a NumPy
            random number generator returns a random initial state for optimization of
            appropriate dimension.
        rng: Seeded NumPy random number generator.
        number_minima_to_find: Number of candidate minima of objective function to try
            to find.
        maximum_minimize_calls: Maximum number of times to try calling
            `scipy.optimize.minimize` to find candidate minima. If insufficient
            candidates are found within this number of calls then a `ConvergenceError`
            exception is raised.
        minimize_method: String specifying one of local optimization methods which can
            be passed to `method` argument of `scipy.optimize.minimize`.
        minimize_max_iterations: Maximum number of iterations in inner local
            minimization.
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
        results = _minimize(
            jax.jit(objective_function),
            x0=sample_initial_state(rng),
            jac=jax.jit(jax.grad(objective_function)),
            hessp=jax.jit(hessian_vector_product(objective_function)),
            method=minimize_method,
            options={"maxiter": minimize_max_iterations},
        )
        minimize_calls += 1
        if results.success:
            logging_function(f"Found minima with value {results.fun}")
            # Add minima to minima_found maintaining heap invariant such that first
            # entry in minima_found is always best solution so far (with counter used
            # to break ties between solutions with equal values for objective)
            heappush(minima_found, (results.fun, len(minima_found), results.x))
        else:
            logging_function(f"Minimization unsuccessful - {results.message}")
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


def basin_hopping(
    objective_function: ObjectiveFunction,
    sample_initial_state: InitialStateSampler,
    rng: Generator,
    *,
    num_iterations: int = 5,
    minimize_method: str = "Newton-CG",
    minimize_max_iterations: int | None = None,
) -> tuple[jax.Array, float]:
    """Minimize a differentiable objective function with SciPy basin-hopping algorithm.

    The basin-hopping algorithm nests an inner local minimization using the
    `scipy.optimize.minimize` method within an outer global stepping algorithm which
    perturbs the current state with a random displacement and accepts or rejects this
    proposal using a Metropolis criterion.

    Args:
        objective_function: Differentiable scalar-valued function of a single flat
            vector argument to be minimized. Assumed to be specified using JAX
            primitives such that its gradient and Hessian can be computed using JAX's
            automatic differentiation support, and to be suitable for just-in-time
            compilation.
        sample_initial_state: Callable with one argument, which when passed a NumPy
            random number generator returns a random initial state for optimization of
            appropriate dimension.
        rng: Seeded NumPy random number generator.
        num_iterations: Number of basin-hopping iterations, with number of inner
            `scipy.optimize.minimize` calls being `num_iterations + 1`.
        minimize_method: String specifying one of local optimization methods which can
            be passed to `method` argument of `scipy.optimize.minimize`.
        minimize_max_iterations: Maximum number of iterations in inner local
            minimization.

    Returns:
        Tuple with first entry the state corresponding to the best minima candidate
        found and the second entry the corresponding objective function value.
    """
    results = _basin_hopping(
        jax.jit(objective_function),
        x0=sample_initial_state(rng),
        niter=num_iterations,
        minimizer_kwargs={
            "method": minimize_method,
            "jac": jax.jit(jax.grad(objective_function)),
            "hessp": jax.jit(hessian_vector_product(objective_function)),
            "options": {"maxiter": minimize_max_iterations},
        },
        seed=rng,
    )
    return results.x, float(results.fun)
