"""
Gross-Pitaevskii Equation with Lattice Potential Experiment Script

Setting of Section 'Numerical experiments', subsection 'Lattice potential':
    V(x) = |x|^2 + A + A sin(omega x_1) sin(omega x_2) sin(omega x_3),
with A = 40 and omega = 10 pi, on Omega = (-6, 6)^3, beta = 1000, polynomial
degree k = 4, N = 399 or 799 interior nodes per dimension, TT ranks 5, 10,
15, Thomas--Fermi initial guess. The potential is passed as a sum of 5
separable terms. The "S+V" preconditioner uses V_0(x) = sum_i (x_i^2 + A/3),
i.e. v_i(x_i) = x_i^2 + 40/3: the harmonic part plus the mean value A of the
lattice part split over the three dimensions.

Usage (settings of the paper):
    python paperexp_lattice.py --method tt --dim 3 --beta 1000 --N 399 --k 4 --tt_rank 10 --max_tt_rank_u2 10 --precond S+V --pymanopt_optimizer ConjugateGradient
    python paperexp_lattice.py --method full_H1 --dim 3 --beta 1000 --N 399 --k 4 --pymanopt_optimizer ConjugateGradient
    python paperexp_lattice.py --method full_au --dim 3 --beta 1000 --N 399 --k 4 --precond S+V --pymanopt_optimizer ConjugateGradient
    python paperexp_lattice.py --config config.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
from pathlib import Path

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

import math

import ufl

from experiments.experiment_run import main_exp


def get_experiment_config(dim=3, beta=1000, L=6, omega=10 * math.pi, A=40, **kwargs):
    """Get experiment configuration for GPE with lattice potential.

    Args:
        dim: Spatial dimension (3 in the paper).
        beta: Nonlinearity parameter.
        L: The domain is (-L, L)^dim.
        omega: Frequency of the lattice part.
        A: Amplitude (and offset) of the lattice part.
        **kwargs: Ignored (``main_exp`` also passes N and k).

    Returns:
        Configuration dictionary for ``experiment_setup.create_setup_functions``
        (V0_fun with v_i = x_i^2 + A/dim).
    """

    if dim == 3:
        exp_config = {
            "dim": dim,
            "beta": beta,
            "a": -L,
            "b": L,
            "V_functions": [
                (lambda x: x**2, jnp.ones_like, jnp.ones_like),
                (jnp.ones_like, lambda x: x**2, jnp.ones_like),
                (jnp.ones_like, jnp.ones_like, lambda x: x**2),
                (lambda x: A * jnp.ones_like(x), jnp.ones_like, jnp.ones_like),
                (
                    lambda x: A * jnp.sin(omega * x),
                    lambda x: jnp.sin(omega * x),
                    lambda x: jnp.sin(omega * x),
                ),
            ],
            "V_fenicsx_fun": lambda x: x[0] ** 2
            + x[1] ** 2
            + x[2] ** 2
            + A
            + A * ufl.sin(omega * x[0]) * ufl.sin(omega * x[1]) * ufl.sin(omega * x[2]),
            "V_tt_fun": lambda xy: xy[:, 0] ** 2
            + xy[:, 1] ** 2
            + xy[:, 2] ** 2
            + A
            + A
            * jnp.sin(omega * xy[:, 0])
            * jnp.sin(omega * xy[:, 1])
            * jnp.sin(omega * xy[:, 2]),
            "V0_fun": [
                lambda x: x**2 + A / 3,
                lambda y: y**2 + A / 3,
                lambda z: z**2 + A / 3,
            ],
        }
    else:
        raise ValueError(f"Unsupported dimension: {dim}.")

    current_dev = jax.default_backend()
    f64f32_msg = jnp.array([1.0, 2.0, 3.0]).dtype

    msg = f"| Running on {current_dev.upper()} device with {f64f32_msg} precision. |"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))

    exp_config["name"] = f"lattice_{current_dev}_{f64f32_msg}"
    return exp_config


if __name__ == "__main__":
    main_exp(get_experiment_config)
