"""
Gross-Pitaevskii Equation with a Disordered (Anderson Localization) Potential

Setting of Section 'Numerical experiments', subsection 'Anderson localization
in two dimensions': Omega = (-6, 6)^2, beta = 10, Omega split into 40 x 40
equal squares, V equal to 1 or eps^{-2} (eps = 0.03) on each square,
polynomial degree k = 4, N = 1599, 3199 or 6399 interior nodes per dimension,
TT ranks 20, 40, 50, 60, 80, Thomas--Fermi initial guess.

Settings of this script:
- The disorder is a single fixed draw, jax.random.bernoulli(PRNGKey(1),
  (40, 40)) (seed=1): about 48.9% of the squares take the value eps^{-2}. It
  does not depend on N, so all grids use the same potential.
- The runs stop when the norm of the Riemannian gradient falls below 1e-3
  (OPTIMIZER_KWARGS below), instead of the 1e-6 of the other experiments.
- The "S+V" preconditioner uses the constant V_0 = 0.5 + 0.5 / eps^2, i.e.
  v_i = 0.25 + 0.25 / eps^2 per dimension: the mean of the two values of V.
- The potential of the Thomas--Fermi initial guess (V_tt_fun) differs from
  the one used in the optimization (V_functions) at some nodes lying on square
  boundaries, where the two assign the node to different squares (about 0.8%
  of the nodes for N = 1599). This affects only the initial guess.

Since the CLI defaults (--dim 3, --beta 1000) override the defaults of
get_experiment_config, --dim 2 and --beta 10 must be passed.

Usage (settings of the paper):
    python paperexp_anderson.py --method tt --dim 2 --beta 10 --N 1599 --k 4 --tt_rank 40 --max_tt_rank_u2 40 --precond S+V --pymanopt_optimizer ConjugateGradient
    python paperexp_anderson.py --method full_H1 --dim 2 --beta 10 --N 1599 --k 4 --pymanopt_optimizer ConjugateGradient
    python paperexp_anderson.py --method full_au --dim 2 --beta 10 --N 1599 --k 4 --precond S+V --pymanopt_optimizer ConjugateGradient
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
from pathlib import Path

import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
from jax import config

# Configure JAX: GPU backend and double precision (paragraph 'Software,
# parameters, and hardware.')
dev = "gpu"
config.update("jax_platform_name", dev)
config.update("jax_enable_x64", True)


from experiments.experiment_run import main_exp


def get_experiment_config(
    dim=2, beta=10, L=6, n_squares=40, epsilon=0.03, seed=1, **kwargs
):
    """Get experiment configuration for GPE with the Anderson potential.

    The potential is ``V(x, y) = Q[i(x), j(y)]``, with ``i(x)`` the index of
    the square containing ``x`` and ``Q`` a random ``n_squares x n_squares``
    matrix with entries 1 or ``epsilon**-2``. It is passed to the optimization
    as a sum of ``n_squares`` separable terms ``1[x in square i] * Q[i, j(y)]``
    (TT rank ``n_squares``), and to the higher-order energy as a DG0 function.

    Args:
        dim: Spatial dimension (only 2 is implemented).
        beta: Nonlinearity parameter.
        L: The domain is (-L, L)^dim.
        n_squares: Number of squares per dimension.
        epsilon: V takes the values 1 and ``epsilon**-2``.
        seed: Seed of the PRNG key of the disorder.
        **kwargs: Ignored (``main_exp`` also passes N and k).

    Returns:
        Configuration dictionary for ``experiment_setup.create_setup_functions``.
    """

    if dim == 2:

        a, b = -L, L

        # Set up random potential: Q[i, j] = epsilon**-2 with probability 1/2,
        # else 1. One fixed draw (seed=1: ~48.9% of the squares at
        # epsilon**-2), independent of the grid size N.
        rng = jax.random.PRNGKey(seed)
        squares_limits = jnp.linspace(a, b, n_squares + 1)
        Q = jax.random.bernoulli(rng, shape=(n_squares, n_squares)) / (epsilon**2)
        Q = jnp.where(Q == 0, 1, Q)

        def segment_indicator(x, i, limits):
            """
            Returns the indicator function for the i-th segment [limits[i], limits[i+1]).
            The output is a 0/1 integer array.
            """
            # Define the lower and upper bounds for the i-th segment
            lower_bound = limits[i]
            upper_bound = limits[i + 1]

            # Calculate the boolean result: (x >= lower) AND (x < upper)
            is_in_segment = (x >= lower_bound) & (x < upper_bound)

            # Convert the True/False boolean array to 1/0 integer array
            return is_in_segment.astype(int)

        def create_segment_indicator_func(i, limits):
            """Factory that returns a function for the i-th indicator."""

            def indicator_func(x):
                # i is captured here with its current, correct value
                return segment_indicator(x, i, limits)

            return indicator_func

        def create_Q_lookup_func(i):
            """Factory that returns a function for the i-th Q lookup."""

            def Q_lookup_func(y):
                # i is captured here with its current, correct value
                idx = np.floor((y - a) / (b - a) * n_squares).astype(int)
                return Q[i, idx]

            return Q_lookup_func

        exp_config = {
            "dim": dim,
            "beta": beta,
            "a": a,
            "b": b,
            "V_vector": None,
            "V_functions": [
                (
                    create_segment_indicator_func(i, squares_limits),
                    create_Q_lookup_func(i),
                )
                for i in range(n_squares)
            ],
            "V_fenicsx_fun": (
                "DG",
                lambda x: Q[
                    np.floor((x[0] - a) / (b - a) * n_squares).astype(int),
                    np.floor((x[1] - a) / (b - a) * n_squares).astype(int),
                ],
            ),
            # Used only for the Thomas--Fermi initial guess. The square index of
            # x is computed by floor, whereas V_functions compares x with
            # squares_limits: at some nodes on square boundaries the two differ
            # (floating-point rounding).
            "V_tt_fun": lambda xy: Q[
                jnp.floor((xy[:, 0] - a) / (b - a) * n_squares).astype(int),
                jnp.floor((xy[:, 1] - a) / (b - a) * n_squares).astype(int),
            ],
            # S+V preconditioner: constant V_0 = v_1 + v_2 = 0.5 + 0.5 / eps^2,
            # the mean of the two values of V.
            "V0_fun": [
                lambda x: (0.25 + 0.25 / epsilon**2) * jnp.ones_like(x),
                lambda y: (0.25 + 0.25 / epsilon**2) * jnp.ones_like(y),
            ],
        }
    elif dim == 3:
        raise NotImplementedError("3D not implemented for Anderson experiment.")
    else:
        raise ValueError("Unsupported dimension. Only 2D and 3D are supported.")

    current_dev = jax.default_backend()
    f64f32_msg = jnp.array([1.0, 2.0, 3.0]).dtype

    msg = f"| Running on {current_dev.upper()} device with {f64f32_msg} precision. |"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))

    exp_config["name"] = f"anderson_{current_dev}_{f64f32_msg}"
    return exp_config


# Custom experiment-specific configurations (optional). The Anderson runs stop
# at Riemannian gradient norm 1e-3 (not 1e-6 as in the other experiments);
# the other stopping criteria are those of experiment_run.
OPTIMIZER_KWARGS = {
    "min_gradient_norm": 1e-3,
}
RGRAD_KWARGS_TT = None
RGRAD_KWARGS_FULL = None


if __name__ == "__main__":
    # To use custom kwargs, pass them to main_exp:
    main_exp(
        get_experiment_config,
        default_optimizer_kwargs=OPTIMIZER_KWARGS,
        default_rgrad_kwargs_tt=RGRAD_KWARGS_TT,
        default_rgrad_kwargs_full=RGRAD_KWARGS_FULL,
    )
