"""Construction of the pymanopt problems, optimizers and initial guesses.

``create_setup_functions`` turns an experiment configuration (domain,
potential, nonlinearity, optional multicomponent data) into two factories:

- ``setup_tt_problem``: the a_u Sobolev gradient flow on the manifold of
  unit-norm (or, with ``p`` components, oblique) fixed TT-rank tensors, with the
  a_u metric and the a_u Riemannian gradient of Section 'Riemannian
  optimization on the manifold of unit-norm fixed TT-rank functions' (route II
  of Section 'The multicomponent Gross--Pitaevskii equation' in the
  multicomponent case) and the TT preconditioners of Section 'Preconditioning
  in the TT format';
- ``setup_full_problem``: the a_u or H1 Sobolev gradient flow on full tensors,
  with the preconditioners of Section 'Preconditioning in the full tensor
  format'.

Both use the spectral element discretization of Section 'Discretization by
spectral elements' and log, at every iteration, the energy re-evaluated with
the higher-order quadrature of Remark 'Evaluating the energy with a
higher-order quadrature rule'. ``save_results``/``load_results`` pickle the
pymanopt results.
"""

import gc
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Absolute path to the results directory: $TTGPE_RESULTS_DIR if set, else
# repo_root/results. experiment_run.main_exp overrides it with --results_dir.
RESULTS_DIR = Path(
    os.environ.get("TTGPE_RESULTS_DIR", Path(__file__).parent.parent.parent / "results")
).resolve()

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pymanopt
import ttax
from jax import config
from pymanopt.optimizers.optimizer import OptimizerResult

from core.energy import (
    EnergyHighOrder,
    EnergyHighOrder_TT,
    FullEnergyFunctional,
    setup_S_precond,
    setup_SplusV_precond,
    thomas_fermi_full_init,
    thomas_fermi_tt_init,
    tt_energyfactory,
)
from core.fixed_tt_rank import (
    FixedRankPoint,
    FixedRankTangentVector,
    FixedTTRankEmbedded,
)
from core.problem import (
    get_inner_au_tangent,
    setup_1Dmatrices,
    setup_matrices,
    setup_multicomponent_matrices,
)
from core.tangent import cg_tree_math_point_dependent
from core.tensor_full import Sphere_Mass
from core.tt_utils import partial_flat_inner_last, tt_from_fun

rng = jax.random.PRNGKey(0)

# Fallback optimizer options. When the experiments are run through
# experiment_run.main_exp, its DEFAULT_OPTIMIZER_KWARGS override all of these
# except max_cost_evaluations.
DEFAULT_OPTIMIZER_KWARGS = {
    "max_time": 3600 * 4,
    "max_iterations": 1000,
    "min_gradient_norm": 1e-7,
    "min_step_size": 1e-11,
    "max_cost_evaluations": 9999999,
    "verbosity": 2,
    "log_verbosity": 1,
}

# Optimizers (paragraphs 'Riemannian Gradient Descent (R-GD)', 'Riemannian
# Nonlinear Conjugate Gradient (R-NLCG)' and 'Step size.'). No line searcher is
# passed, so pymanopt's defaults are used:
# - SteepestDescent (R-GD): BackTrackingLineSearcher, Armijo backtracking with
#   sufficient decrease 1e-4, contraction factor 0.5, at most 25 backtracking
#   steps; the initial trial step is 2 * (f_k - f_{k-1}) / df_k times the
#   optimism factor 2 (1 / ||d|| at the first iteration).
# - ConjugateGradient (R-NLCG): Hestenes--Stiefel rule
#   beta = max(0, <g_k, y_k> / <y_k, T(xi_{k-1})>), with y_k = g_k - T(g_{k-1}),
#   T the transporter and inner products in the manifold metric (a_u, or H1
#   for full_H1); Powell restarts are disabled (pymanopt's orth_value=inf) and
#   the direction is reset to -g_k when it is not a descent direction. Line
#   search: AdaptiveLineSearcher, Armijo backtracking with sufficient
#   decrease 0.5, contraction factor 0.5, at most 10 backtracking steps; the
#   initial trial step is the previous accepted step (doubled unless exactly
#   one backtracking step was needed).
# A step is rejected (step size 0) if it does not decrease the cost.
OPTIMIZER_FROM_STRING = {
    "SteepestDescent": pymanopt.optimizers.SteepestDescent,
    "ConjugateGradient": partial(
        pymanopt.optimizers.ConjugateGradient, beta_rule="HestenesStiefel"
    ),
    "TrustRegions": pymanopt.optimizers.TrustRegions,
}


def create_setup_functions(config: dict):
    """Return setup functions for the TT and full tensor optimization problems.

    The domain is ``Omega = (a, b)^dim`` and the energy is that of Section
    'Problem formulation' (or its multicomponent version) with interaction
    parameter ``beta``.

    Args:
        config: Experiment configuration with keys
            dim: spatial dimension.
            beta: nonlinearity parameter.
            a, b: domain bounds.
            V_functions: potential used in the optimization, as a list of
                tuples of 1D functions (separable terms, see
                ``core.problem.setup_matrices``). Default: ``V = 1``.
            V_vector: optional batch TT vector replacing the potential built
                from ``V_functions``.
            V_fenicsx_fun: potential used by the higher-order energy
                (``EnergyHighOrder``/``EnergyHighOrder_TT``).
            V_tt_fun: potential as a function of points of shape
                ``[M, dim]``, used only for the Thomas--Fermi initial guess.
            V0_fun: list of ``dim`` 1D functions ``v_i`` of the separable
                potential ``V_0 = v_1(x_1) + ... + v_d(x_d)`` of the "S+V"
                preconditioner (Section 'Preconditioning in the full tensor
                format'). If None, the TT problem is not preconditioned.
            p, alphas, kappa_matrix: number of components, their masses and
                the interaction matrix K (multicomponent case only).

    Returns:
        Tuple ``(setup_tt_problem, setup_full_problem)`` of functions.
    """

    # Extract parameters at the beginning
    dim = config["dim"]
    beta = config["beta"]
    a = config["a"]
    b = config["b"]
    V_functions = config.get("V_functions", [[jnp.ones_like] * dim])
    _V_vector = config.get("V_vector", None)
    V_fenicsx_fun = config["V_fenicsx_fun"]
    f0 = config.get("f0", None)
    V_tt_fun = config.get("V_tt_fun", None)
    V0_fun = config.get("V0_fun", None)
    # Multicomponent parameters
    p = config.get("p", None)
    alphas = config.get("alphas", None)
    kappa_matrix = config.get("kappa_matrix", None)

    def setup_tt_problem(
        N,
        k,
        tt_rank,
        rng,
        optimizer_kwargs={},
        rgrad_kwargs={"maxiter": 100, "tol": 1e-10, "precond": "S+V"},
        setup_highorder=True,
        precompile=True,
        max_tt_rank_u2=None,
        pymanopt_optimizer="SteepestDescent",
        init="thomas_fermi",
    ):
        """Set up a TT-rank-constrained optimization problem.

        The manifold is ``FixedTTRankEmbedded`` with identity mass, whose
        metric is replaced by the a_u inner product. The Riemannian gradient
        is computed with ``gradE_au_Hu`` (single component, via the operator
        H_u of subsection 'The $a_u$ Riemannian gradient') or with
        ``gradE_au_PAP_N`` (multicomponent, route II of subsection 'Riemannian
        gradient II: change of metric'). In the multicomponent case the TT
        rank in the component mode is ``p`` (Remark 'TT rank in the component
        mode').

        Initial guesses (``init``):
            "thomas_fermi": ``u^2 = (mu - V)_+ / (2 beta)`` with ``V =
                V_tt_fun`` (the positive part is floored at a few machine
                epsilons in ``core.energy``), with ``mu`` found by bisection so
                that the discrete L2 norm is 1; approximated in TT format by
                cross approximation at the target ranks.
            "constant": ``u = 1`` (rank 1, padded with zero cores to the
                target ranks).
            "random": a random point of the manifold.
        The nodal values are multiplied by the square roots of the quadrature
        weights (rescaled basis) and normalized. In the multicomponent case
        the scalar guess is copied to all ``p`` components and each
        component is normalized to its mass.

        Args:
            N: Number of interior grid points per dimension.
            k: Polynomial degree.
            tt_rank: TT rank(s) of the spatial modes.
            rng: Random number generator key.
            optimizer_kwargs: Optimizer options (merged over
                ``DEFAULT_OPTIMIZER_KWARGS``).
            rgrad_kwargs: Riemannian gradient options: CG ``maxiter`` and
                ``tol`` (or ``kappa``/``theta`` for the adaptive tolerance of
                route II), ``precond`` ("S+V", "S" or None) and
                ``precond_kwargs`` (e.g. ``rank_expsum``).
            setup_highorder: Whether to set up the high-order energy
                functional.
            precompile: Whether to evaluate cost and gradient once at a random
                point to trigger compilation.
            max_tt_rank_u2: If not None, ``u * u`` is rounded to this TT rank
                in the metric and in the gradient.
            pymanopt_optimizer: Key of ``OPTIMIZER_FROM_STRING``.
            init: Initial guess, "thomas_fermi", "constant" or "random".

        Returns:
            Tuple ``(problem, optimizer, u0, tt_functions, E_highorder, rng)``:
            the pymanopt problem and optimizer, the initial point
            (``FixedRankPoint``), the dictionary of TT energy functions from
            ``tt_energyfactory``, the high-order energy (or None) and the
            updated RNG key.
        """
        # Energy
        S, M, V_vector_tmp, msh_1d = setup_matrices(dim, N, a, b, V_functions, k)
        if _V_vector is None:
            V_vector = V_vector_tmp
        else:
            V_vector = _V_vector

        # Multicomponent: extend matrices with component dimension
        if p is not None:
            S, M, V_vector, K = setup_multicomponent_matrices(
                p, S, M, V_vector, kappa_matrix
            )
        else:
            K = None

        _rgrad_kwargs = rgrad_kwargs.copy()
        precond = _rgrad_kwargs.pop("precond", None)
        # Setup preconditioner for TT
        M_1d, S_1d, msh_1d, _ = setup_1Dmatrices(N, a, b, k)
        if V0_fun is None or precond is None:
            Prec_PAP_Mr = None
            _rgrad_kwargs.pop("precond_kwargs", None)
        elif precond == "S+V":
            V0 = [f(msh_1d) for f in V0_fun]
            Vfull = None  # Dummy to avoid unnecessary memory allocation
            M_1d_inv_sqrt = jnp.diag(1 / jnp.sqrt(jnp.diag(M_1d.todense())))
            S_1d_rescaled = M_1d_inv_sqrt @ S_1d @ M_1d_inv_sqrt
            E_full = FullEnergyFunctional(
                dim,
                S_1d_rescaled,
                M_1d,
                Vfull,
                beta,
                K=kappa_matrix,
                alphas=alphas,
                p=p,
            )
            Prec_PAP_Mr = setup_SplusV_precond(
                E_full,
                M_1d,
                V0,
                factors_only=True,
                p=p,
                **_rgrad_kwargs.pop("precond_kwargs", {}),
            )
        elif precond == "S":
            Vfull = None  # Dummy to avoid unnecessary memory allocation
            M_1d_inv_sqrt = jnp.diag(1 / jnp.sqrt(jnp.diag(M_1d.todense())))
            S_1d_rescaled = M_1d_inv_sqrt @ S_1d @ M_1d_inv_sqrt
            E_full = FullEnergyFunctional(
                dim,
                S_1d_rescaled,
                M_1d,
                Vfull,
                beta,
                K=kappa_matrix,
                alphas=alphas,
                p=p,
            )
            Prec_PAP_Mr = setup_S_precond(
                E_full,
                M_1d,
                factors_only=True,
                p=p,
                **_rgrad_kwargs.pop("precond_kwargs", {}),
            )
        else:
            raise ValueError
        tt_functions = tt_energyfactory(
            S,
            M,
            V_vector,
            beta,
            Prec_PAP_Mr=Prec_PAP_Mr,
            Prec_PAP_N=None,
            K=K,
            alphas=alphas,
        )

        if setup_highorder:
            E_highorder = EnergyHighOrder_TT(
                a, b, beta, V_functions, N, dim, k, K=kappa_matrix, p=p, alphas=alphas
            )
        else:
            E_highorder = None
        bilinear_form = jax.jit(tt_functions["bilinear_form"])

        # Manifold
        [key, rng] = jax.random.split(rng, 2)
        if p is not None:
            tt_sz = [N] * dim + [p]
            tt_rank_list = [tt_rank] * (dim - 1) + [p]
            manifold = FixedTTRankEmbedded(
                tt_sz, tt_rank_list, "identity", key, p=p, alphas=alphas
            )
        else:
            tt_sz = [N] * dim
            manifold = FixedTTRankEmbedded(tt_sz, tt_rank, "identity", key)

        # Cost
        E_jit = jax.jit(tt_functions["energy"])

        @pymanopt.function.jax(manifold)
        def cost(point):
            return E_jit(point.u)

        if max_tt_rank_u2 is None:
            # Inner product
            manifold.inner_product = get_inner_au_tangent(bilinear_form)

            # Gradient
            if p is None:
                gradE_withinfo = tt_functions["gradE_au_Hu_withinfo"]
            else:
                gradE_withinfo = tt_functions["gradE_au_PAP_N_withinfo"]
            gradE_withinfo_jit = jax.jit(
                ttax.fuse(lambda u: gradE_withinfo(u, **_rgrad_kwargs))
            )

            _cg_logger = {"iters": 0}

            def gradE_jit(u):
                rgrad, cg_info = gradE_withinfo_jit(u)
                if isinstance(cg_info, dict):
                    _cg_logger["iters"] = cg_info["iters"]
                    # print(f"CG iters: {cg_info['iters']}")
                else:
                    # print(f"CG iters: {[info['iters'] for info in cg_info]}")
                    _cg_logger["iters"] = sum([info["iters"] for info in cg_info])
                return rgrad

            @pymanopt.function.jax(manifold)
            def riemannian_gradient(point):
                return FixedRankTangentVector(gradE_jit(point.u))

        else:
            # Inner product with u^2
            manifold.inner_product = get_inner_au_tangent(bilinear_form, u2=True)

            # Gradient with u^2
            if p is None:
                gradE_withinfo_u2 = tt_functions["gradE_au_Hu_withinfo_u2"]
            else:
                gradE_withinfo_u2 = tt_functions["gradE_au_PAP_N_withinfo_u2"]
            gradE_withinfo_u2_jit = jax.jit(
                ttax.fuse(lambda u, u2: gradE_withinfo_u2(u, u2, **_rgrad_kwargs))
            )

            _cg_logger = {"iters": 0}

            def gradE_u2_jit(u, u2):
                rgrad, cg_info = gradE_withinfo_u2_jit(u, u2)
                if isinstance(cg_info, dict):
                    # print(f"CG iters: {cg_info['iters']}")
                    _cg_logger["iters"] = cg_info["iters"]
                else:
                    # print(f"CG iters: {[info['iters'] for info in cg_info]}")
                    _cg_logger["iters"] = sum([info["iters"] for info in cg_info])
                return rgrad

            @pymanopt.function.jax(manifold)
            def riemannian_gradient(point):
                return FixedRankTangentVector(gradE_u2_jit(point.u, point.u2))

        # Pymanopt problem
        problem = pymanopt.Problem(
            manifold, cost=cost, riemannian_gradient=riemannian_gradient
        )

        # Run once to compile
        if precompile:
            print("Precompiling cost and gradient...")
            point = manifold.random_point()
            if max_tt_rank_u2 is not None:
                point._max_tt_rank_u2 = max_tt_rank_u2
            problem.cost(point)
            print("Cost compiled.")
            problem.riemannian_gradient(point)
            print("Gradient compiled.")
            gc.collect()

        if E_highorder is not None:
            print("[NOTE] Saving exact energy")
        else:
            print("[WARNING] **Not** saving exact energy")

        # Pymanopt optimizer. log_callable is called by pymanopt at every
        # iteration; its entries are added to the iteration log:
        # - "lmd": a_u(u, u) at the current iterate;
        # - "E_exact": energy re-evaluated with the higher-order quadrature
        #   rule (Remark 'Evaluating the energy with a higher-order quadrature
        #   rule'), used only for plotting, never by the optimizer;
        # - "cg_iters": CG iterations of the last gradient evaluation.
        # pymanopt also stores the wall-clock time spent in this callable as
        # "log_time", which the analysis scripts subtract from the run time.
        def log_callable(*args, **kwargs):
            gc.collect()
            u = kwargs["point"].u
            lmd = bilinear_form(u, u, u)
            log_dict = {"lmd": lmd}
            if E_highorder is not None:
                E_exact = E_highorder(u)
                log_dict["E_exact"] = E_exact
            log_dict["cg_iters"] = _cg_logger["iters"]

            gc.collect()
            return log_dict

        # Line search and CG beta rule: see the comment on OPTIMIZER_FROM_STRING.
        optimizer_kwargs = {**DEFAULT_OPTIMIZER_KWARGS, **optimizer_kwargs}
        optimizer = OPTIMIZER_FROM_STRING[pymanopt_optimizer](
            log_callable=log_callable, **optimizer_kwargs
        )

        # Pymanopt initial point
        M_sqrt = ttax.ops.tensor_to_vector(
            ttax.TT([jnp.expand_dims(jnp.sqrt(jnp.diag(M_1d.todense())), [0, 2])] * dim)
        )
        norm_fn = lambda u: ttax.flat_inner(
            ttax.multiply(M_sqrt, u), ttax.multiply(M_sqrt, u)
        )
        scalar_target_ranks = (
            manifold._tt_rank[:-1] if p is not None else manifold._tt_rank
        )

        # f0 overrides init for backward compatibility
        _init = init if f0 is None else "f0"

        if _init == "random":
            print("[INIT] Random initialization.")
            u0 = manifold.random_point()
            if max_tt_rank_u2 is not None:
                u0._max_tt_rank_u2 = max_tt_rank_u2
        else:
            if _init == "thomas_fermi":
                print("[INIT] Thomas-Fermi initialization.")
                cores = thomas_fermi_tt_init(
                    V_tt_fun,
                    beta,
                    [msh_1d] * dim,
                    target_ranks=scalar_target_ranks,
                    norm_fn=norm_fn,
                    eps=1e-16,
                    rng=np.random.default_rng(1),
                )
            elif _init == "constant":
                print("[INIT] Constant (rank-1, zero-padded) initialization.")
                cores = tt_from_fun(
                    lambda xy: jnp.ones(xy.shape[0]),
                    [msh_1d] * dim,
                    target_ranks=scalar_target_ranks,
                    eps=1e-16,
                    rng=np.random.default_rng(1),
                )
            else:
                raise ValueError(
                    f"Unknown init type: {_init!r}. Choose from 'thomas_fermi', 'random', 'constant'."
                )

            if p is not None:
                # Extend scalar TT to p components
                scalar_u = ttax.multiply(
                    ttax.ops.tensor_to_vector(ttax.TT(cores)), M_sqrt
                )
                scalar_cores = list(ttax.ops.vector_to_tensor(scalar_u).tt_cores)
                scalar_cores[-1] = jnp.repeat(scalar_cores[-1], p, axis=-1)
                scalar_cores.append(jnp.expand_dims(jnp.eye(p, p), axis=2))
                u0_tensor = ttax.ops.tensor_to_vector(ttax.TT(scalar_cores))
                print(
                    f"Norm before normalization: {partial_flat_inner_last(u0_tensor, u0_tensor)}"
                )
            else:
                u0_tensor = ttax.multiply(
                    ttax.ops.tensor_to_vector(ttax.TT(cores)), M_sqrt
                )
                print(
                    f"Norm before normalization: {ttax.flat_inner(u0_tensor, u0_tensor)}"
                )

            u0 = FixedRankPoint(u0_tensor, max_tt_rank_u2=max_tt_rank_u2)
            u0 = manifold._normalize_point(u0)
        # print("Gradient at different points check")
        # point = manifold.random_point()
        # problem.riemannian_gradient(point)
        # problem.riemannian_gradient(u0)

        return problem, optimizer, u0, tt_functions, E_highorder, rng

    def setup_full_problem(
        N,
        k,
        metric,
        rng,
        optimizer_kwargs={},
        rgrad_kwargs={"maxiter": 200, "tol": 1e-10, "precond": None},
        setup_highorder=True,
        precompile=True,
        pymanopt_optimizer="SteepestDescent",
        init="thomas_fermi",
    ):
        """Set up a full tensor optimization problem.

        The manifold is ``Sphere_Mass`` (unit sphere, or oblique manifold if
        ``p`` is given) of dense coefficient tensors in the rescaled basis.
        With ``metric="H1"`` the metric is ``(v, S w)`` and the gradient is
        the H1 Sobolev gradient (systems with S solved by fast
        diagonalization); with ``metric="au"`` the metric is a_u and the
        gradient is computed by preconditioned CG. The "S" preconditioner is
        set up if requested (or for H1), otherwise "S+V" with ``V0_fun`` if
        available. Initial guesses are as in ``setup_tt_problem``, without TT
        approximation.

        Args:
            N: Number of interior grid points per dimension.
            k: Polynomial degree.
            metric: Metric to use ("H1" or "au").
            rng: Random number generator key.
            optimizer_kwargs: Optimizer options (merged over
                ``DEFAULT_OPTIMIZER_KWARGS``).
            rgrad_kwargs: Riemannian gradient options (CG ``maxiter``,
                ``tol``, ``precond``).
            setup_highorder: Whether to set up the high-order energy
                functional.
            precompile: Whether to evaluate cost and gradient once at a random
                point to trigger compilation.
            pymanopt_optimizer: Key of ``OPTIMIZER_FROM_STRING``.
            init: Initial guess, "thomas_fermi", "constant" or "random".

        Returns:
            Tuple ``(problem_full, optimizer_full, u0_full, E_full,
            E_highorder, rng)``: the pymanopt problem and optimizer, the
            initial point (array), the ``FullEnergyFunctional``, the high-order
            energy (or None) and the updated RNG key.
        """
        M_1d, S_1d, msh_1d, _ = setup_1Dmatrices(N, a, b, k)
        M_1d_inv_sqrt = jnp.diag(1 / jnp.sqrt(jnp.diag(M_1d.todense())))
        S_1d_rescaled = M_1d_inv_sqrt @ S_1d @ M_1d_inv_sqrt

        _, _, V_vector_tmp, _ = setup_matrices(dim, N, a, b, V_functions, k)
        if _V_vector is None:
            V_vector_sc = V_vector_tmp
        else:
            V_vector_sc = _V_vector
        # Build single-component Vfull (before multicomponent extension)
        batch_size = V_vector_sc.shape[0]
        Vfull = jnp.zeros_like(
            ttax.full(ttax.ops.vector_to_tensor(V_vector_sc.batch_loc[0,]))
        )
        for i in range(batch_size):
            Vfull += ttax.full(ttax.ops.vector_to_tensor(V_vector_sc.batch_loc[i,]))

        E_full = FullEnergyFunctional(
            dim, S_1d_rescaled, M_1d, Vfull, beta, K=kappa_matrix, alphas=alphas, p=p
        )
        if rgrad_kwargs.get("precond", None) == "S" or metric == "H1":
            E_full.Sinv_setup()
        elif V0_fun is not None:
            E_full.SplusV_setup([f(msh_1d) for f in V0_fun])
        E_full.setup_preconditioners()
        if setup_highorder:
            E_highorder = EnergyHighOrder(
                a,
                b,
                beta,
                V_fenicsx_fun,
                msh_1d,
                dim,
                k,
                K=kappa_matrix,
                p=p,
                alphas=alphas,
            )
        else:
            E_highorder = None

        [key, rng] = jax.random.split(rng, 2)
        tt_sz = [N] * dim
        manifold_full = Sphere_Mass(tt_sz, None, key, p=p, alphas=alphas)

        @pymanopt.function.jax(manifold_full)
        @jax.jit
        def cost_full(point):
            return E_full(point)

        if metric == "H1":  # H1 Sobolev gradient flow
            manifold_full.inner_product = jax.jit(
                lambda point, ta, tb: jnp.vdot(ta, E_full.S_fun(tb))
            )

            @pymanopt.function.jax(manifold_full)
            # @jax.jit
            def riemannian_gradient_full(u):
                return E_full.riemannian_gradient_H1(u)

        elif metric == "au":
            manifold_full.inner_product = jax.jit(E_full.bilinear_form)

            _cg_logger = {"iters": 0}
            cg_jitted = jax.jit(
                cg_tree_math_point_dependent, static_argnames=["A", "M", "return_info"]
            )

            @pymanopt.function.jax(manifold_full)
            # @jax.jit
            def riemannian_gradient_full(u):
                gc.collect()
                rgrad, info_cg = E_full.riemannian_gradient_au_withaux(
                    u, cg=cg_jitted, **rgrad_kwargs
                )
                rgrad = jax.tree.map(lambda x: x.block_until_ready(), rgrad)
                gc.collect()
                if isinstance(info_cg, dict):
                    _cg_logger["iters"] = info_cg["iters"]
                else:
                    _cg_logger["iters"] = sum([info["iters"] for info in info_cg])
                return rgrad

        else:
            raise ValueError("Unknown metric")

        # Pymanopt problem
        problem_full = pymanopt.Problem(
            manifold_full, cost=cost_full, riemannian_gradient=riemannian_gradient_full
        )
        # Run once to compile
        if precompile:
            point = manifold_full.random_point()
            problem_full.cost(point)
            problem_full.riemannian_gradient(point)
            gc.collect()

        if E_highorder is not None:
            print("[NOTE] Saving exact energy")
        else:
            print("[WARNING] **Not** saving exact energy")

        # Pymanopt optimizer. Same per-iteration log as in setup_tt_problem
        # ("lmd", "E_exact" from the higher-order quadrature rule, "cg_iters"
        # for the a_u metric); pymanopt adds the logging time as "log_time".
        def log_callable_full(*args, **kwargs):
            gc.collect()
            u = kwargs["point"]
            lmd = E_full.bilinear_form(u, u, u)
            log_dict = {"lmd": lmd}
            if E_highorder is not None:
                E_exact = E_highorder(u)
                log_dict["E_exact"] = E_exact
            if metric == "au":
                log_dict["cg_iters"] = _cg_logger["iters"]

            gc.collect()

            return log_dict

        # Line search and CG beta rule: see the comment on OPTIMIZER_FROM_STRING.
        optimizer_kwargs = {**DEFAULT_OPTIMIZER_KWARGS, **optimizer_kwargs}
        optimizer_full = OPTIMIZER_FROM_STRING[pymanopt_optimizer](
            log_callable=log_callable_full, **optimizer_kwargs
        )

        # Pymanopt initial point
        # M_inv is always spatial (shape [N]*dim), never includes component axis
        if init == "random":
            print("[INIT] Random initialization.")
            u0_full = manifold_full.random_point()
        elif init == "thomas_fermi":
            print("[INIT] Thomas-Fermi initialization.")
            norm_fn_full = lambda f_arr: jnp.vdot(f_arr, f_arr / E_full.M_inv)
            u0_scalar = thomas_fermi_full_init(
                V_tt_fun, beta, [msh_1d] * dim, norm_fn_full
            )
            u0_scalar = u0_scalar / jnp.sqrt(E_full.M_inv)
            if p is not None:
                # Stack p copies of scalar init and normalize each component
                u0_full = jnp.stack([u0_scalar] * p, axis=-1)
                u0_full = manifold_full._normalize(u0_full)
            else:
                print(f"Norm before normalization: {jnp.vdot(u0_scalar, u0_scalar)}")
                u0_full = u0_scalar / jnp.sqrt(jnp.vdot(u0_scalar, u0_scalar))
        elif init == "constant":
            print("[INIT] Constant initialization.")
            u0_scalar = jnp.ones([N] * dim) / jnp.sqrt(E_full.M_inv)
            if p is not None:
                # Stack p copies of scalar init and normalize each component
                u0_full = jnp.stack([u0_scalar] * p, axis=-1)
                u0_full = manifold_full._normalize(u0_full)
            else:
                print(f"Norm before normalization: {jnp.vdot(u0_scalar, u0_scalar)}")
                u0_full = u0_scalar / jnp.sqrt(jnp.vdot(u0_scalar, u0_scalar))
        else:
            raise ValueError(
                f"Unknown init type: {init!r}. Choose from 'thomas_fermi', 'random', 'constant'."
            )

        return problem_full, optimizer_full, u0_full, E_full, E_highorder, rng

    return setup_tt_problem, setup_full_problem


def save_results(results, folder):
    """Pickle the results to ``RESULTS_DIR/folder/<timestamp>.pkl``.

    A ``FixedRankPoint`` stored as final point of an ``OptimizerResult`` is
    replaced by its TT vector before pickling.

    Args:
        results: Dictionary containing the results to save (key "res": the
            pymanopt ``OptimizerResult``).
        folder: Destination folder, relative to ``RESULTS_DIR``.
    """
    # Ensure the results directory exists
    dest = RESULTS_DIR / folder
    dest.mkdir(parents=True, exist_ok=True)
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = str(dest / f"{timestamp}.pkl")

    if isinstance(results["res"], OptimizerResult) and isinstance(
        results["res"].point, FixedRankPoint
    ):
        results["res"].point = results["res"].point.u

    # Save the results to a file
    with open(filename, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(folder, timestamp=None):
    """Load results saved by ``save_results``.

    Args:
        folder: Folder where the results are stored (absolute, or relative to
            ``RESULTS_DIR``).
        timestamp: Timestamp of the file to load. If None, loads the latest
            file.

    Returns:
        Dictionary containing the loaded results.
    """
    if not os.path.isabs(folder):
        folder = str(RESULTS_DIR / folder)
    if timestamp is None:
        # Load the latest file in the folder
        files = [f for f in os.listdir(folder) if f.endswith(".pkl")]
        files.sort()
        filename = os.path.join(folder, files[-1])
    else:
        filename = os.path.join(folder, f"{timestamp}.pkl")

    # Enable float64 before unpickling: with x64 disabled, JAX silently
    # truncates the stored float64 arrays to float32 when they are restored.
    jax.config.update("jax_enable_x64", True)
    with open(filename, "rb") as f:
        results = pickle.load(f)

    return results
