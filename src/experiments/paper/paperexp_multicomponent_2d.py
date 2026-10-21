#!/usr/bin/env python3
"""
2D Multicomponent Gross-Pitaevskii Equation Experiment Script

Setting of Section 'Numerical experiments', subsection 'Multicomponent
Gross--Pitaevskii equation': p = 3 components on Omega = (0, 1)^2, beta = 10,
interaction matrix K with kappa_11 = 0.5, kappa_22 = 5, kappa_33 = 10 and
kappa_ij = 1 otherwise, masses N_1 = N_2 = N_3 = 1 (alphas), polynomial
degree k = 4, N = 1023 or 2047 interior nodes per dimension, TT ranks 5, 10,
15, 20 in the spatial modes and rank p = 3 in the component mode (Remark 'TT
rank in the component mode').

Potential (the same for all components). With eps = 2^-6 (64 x 64 squares),
square indices i(x) = ceil(x / eps), f1(x) = 1 if i(x) is even and 0
otherwise, f2(y) = 1 if i(y) is odd and 0 otherwise:
    V(x, y) = 2^12 (1 - f1(x) f2(y)) + trap(x) + trap(y),
    trap(t) = 10^6 (2 t - 1)^40.
Hence V - trap = 0 on the squares with even x-index and odd y-index, one
quarter of the 64 x 64 squares, and 2^12 on the other three quarters (it is
not a checkerboard, which would have value 0 on half of the squares). The
"S+V" preconditioner uses v_1(x) = trap(x) + 2^11 (1 - f1(x)) and
v_2(y) = trap(y) + 2^11 (1 - f2(y)).

The script uses a constant initial guess (default_init="constant") and, in the
TT format, route II of Section 'The multicomponent Gross--Pitaevskii equation'
(subsection 'Riemannian gradient II: change of metric') with an adaptive CG
tolerance: relative tolerance min(||g||^theta, kappa) with kappa = 0.01,
theta = 1, where ||g|| is the Frobenius norm of the projected right-hand side
(``tol: None`` disables the fixed tolerance 1e-10 of experiment_run). The full
format uses the fixed tolerance 1e-10.

default_init only applies with --config; on the command line pass
--init constant. Since the CLI defaults (--dim 3, --beta 1000) override the
defaults of get_experiment_config, --dim 2 and --beta 10 must be passed.

Usage (settings of the paper; R-GD is --pymanopt_optimizer SteepestDescent):
    python paperexp_multicomponent_2d.py --method tt --dim 2 --beta 10 --N 2047 --k 4 --tt_rank 10 --max_tt_rank_u2 10 --precond S+V --pymanopt_optimizer ConjugateGradient --init constant
    python paperexp_multicomponent_2d.py --method full_au --dim 2 --beta 10 --N 2047 --k 4 --precond S+V --pymanopt_optimizer ConjugateGradient --init constant
    python paperexp_multicomponent_2d.py --config config.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

# Configure JAX: GPU backend and double precision (paragraph 'Software,
# parameters, and hardware.')
dev = "gpu"
config.update("jax_platform_name", dev)
config.update("jax_enable_x64", True)


from experiments.experiment_run import main_exp


def get_experiment_config(dim=2, beta=10, L=None, **kwargs):
    """Get experiment configuration for 2D multicomponent GPE.

    Domain is (0,1)^2 (L is ignored). The potential is
    ``2^12 (1 - f1(x) f2(y)) + trap(x) + trap(y)``, see the module docstring:
    V - trap vanishes on one quarter of the 64 x 64 squares.

    Args:
        dim: Spatial dimension (must be 2).
        beta: Nonlinearity parameter.
        L: Ignored.
        **kwargs: Ignored (``main_exp`` also passes N and k).

    Returns:
        Configuration dictionary for ``experiment_setup.create_setup_functions``,
        including the multicomponent keys ``p``, ``alphas`` and
        ``kappa_matrix``.
    """
    if dim != 2:
        raise ValueError("paperexp_multicomponent_2d only supports dim=2")

    a, b = 0, 1

    # Potential parameters: 2^le squares per dimension of edge eps = 2^-le,
    # potential height 2^(2 le) = 2^12
    le = 6
    num_squares = 2**le
    epsilon = 1 / num_squares
    pot_bet = 2 ** (2 * le)

    def idx(x):
        return jnp.ceil(x / epsilon).astype(jnp.int32)

    def idx_numpy(x):
        return np.ceil(x / epsilon).astype(np.int32)

    piecewise_f1 = lambda x1: (idx(x1) % 2 == 0).astype(jnp.float64)
    piecewise_f2 = lambda x2: (idx(x2) % 2 != 0).astype(jnp.float64)
    trap = lambda x: (2 * x - 1) ** 40 * 1e6

    piecewise_f1_np = lambda x: (idx_numpy(x) % 2 == 0).astype(np.float64)
    piecewise_f2_np = lambda x: (idx_numpy(x) % 2 != 0).astype(np.float64)

    # V as sum of rank-1 terms:
    # V(x,y) = (sqrt(pot_bet)*1)*(sqrt(pot_bet)*1) - (sqrt(pot_bet)*f1(x))*(sqrt(pot_bet)*f2(y)) + trap(x) + trap(y)
    #        = pot_bet*(1 - f1(x)*f2(y)) + trap(x) + trap(y)
    V_functions = [
        (
            lambda x: jnp.sqrt(pot_bet) * jnp.ones_like(x),
            lambda x: jnp.sqrt(pot_bet) * jnp.ones_like(x),
        ),
        (
            lambda x: -jnp.sqrt(pot_bet) * piecewise_f1(x),
            lambda x: jnp.sqrt(pot_bet) * piecewise_f2(x),
        ),
        (trap, jnp.ones_like),
        (jnp.ones_like, trap),
    ]

    # FEniCSx potential: V(x,y) = trap(x) + trap(y) + pot_bet*(1 - f1(x)*f2(y))
    V_fenicsx_fun = [
        lambda x: trap(x[0]) + trap(x[1]) + pot_bet,
        (
            "DG",
            0,
            lambda x: -pot_bet * piecewise_f1_np(x[0]) * piecewise_f2_np(x[1]),
        ),
    ]

    # Full potential as a scalar function of (x,y) pairs (shape [M, 2])
    V_tt_fun = (
        lambda xy: pot_bet
        - pot_bet * piecewise_f1(xy[:, 0]) * piecewise_f2(xy[:, 1])
        + trap(xy[:, 0])
        + trap(xy[:, 1])
    )

    # Preconditioner separable potential V_0(x, y) = v_1(x) + v_2(y)
    V0_fun = [
        lambda x: trap(x) + pot_bet / 2 * (1 - piecewise_f1(x)),
        lambda y: trap(y) + pot_bet / 2 * (1 - piecewise_f2(y)),
    ]

    # Multicomponent parameters
    p = 3
    alphas = jnp.array([1.0, 1.0, 1.0])

    kappa_matrix = np.ones((p, p))
    kappa_matrix[0, 0] = 0.5
    kappa_matrix[1, 1] = 5.0
    kappa_matrix[2, 2] = 10.0
    kappa_matrix = jnp.array(kappa_matrix)

    current_dev = jax.default_backend()
    f64f32_msg = jnp.array([1.0, 2.0, 3.0]).dtype

    msg = f"| Running on {current_dev.upper()} device with {f64f32_msg} precision. |"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))

    exp_config = {
        "dim": dim,
        "beta": beta,
        "a": a,
        "b": b,
        "V_functions": V_functions,
        "V_fenicsx_fun": V_fenicsx_fun,
        "V_tt_fun": V_tt_fun,
        "V0_fun": V0_fun,
        "p": p,
        "alphas": alphas,
        "kappa_matrix": kappa_matrix,
        "name": f"multicomponent_2d_{current_dev}_{f64f32_msg}",
    }

    return exp_config


if __name__ == "__main__":
    main_exp(
        get_experiment_config,
        default_rgrad_kwargs_tt={"tol": None, "kappa": 0.01, "theta": 1.0},
        default_init="constant",
    )
    # main_exp(get_experiment_config)
