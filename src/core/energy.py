"""Energy functionals, a_u Riemannian gradients and preconditioners.

Companion code of "Riemannian optimization on low-rank tensor train manifolds
for the Gross--Pitaevskii equation". Contents:

- `tt_energyfactory`: discrete energy, bilinear form a_u and the a_u Riemannian
  gradients on the manifold of unit-norm (multicomponent: fixed-mass) fixed
  TT-rank tensors; Sec. 'The a_u Riemannian gradient', Sec. 'Discretization by
  spectral elements' and Sec. 'The multicomponent Gross--Pitaevskii equation'.
- `FullEnergyFunctional`: the same quantities in the full tensor format, with
  the fast-diagonalization preconditioners of Sec. 'Preconditioning in the full
  tensor format'.
- `EnergyHighOrder`, `EnergyHighOrder_TT`: post-processing evaluation of the
  energy with a higher-order quadrature, Remark 'Evaluating the energy with a
  higher-order quadrature rule'.
- Thomas--Fermi initial guesses (`thomas_fermi_tt_init`,
  `thomas_fermi_full_init`).
- `setup_expsum_precond`, `setup_SplusV_precond`, `setup_S_precond`: TT
  preconditioners via exponential sums, Sec. 'Preconditioning in the TT format'.

Conventions. Arrays and TT tensors are coefficient tensors in the rescaled SEM
basis of Sec. 'Discretization by spectral elements': the quadrature mass matrix
is the identity, the discrete L^2 inner product is the Frobenius one, and the
quadrature weights only enter the quartic term, through the reciprocal-weight
tensor M_rec (`M_inv` in the code). In the multicomponent case the last tensor
mode indexes the p components. A tangent vector to the fixed TT-rank manifold
M_r at u is stored in "delta form", i.e. as the list of (gauged) core
variations returned by `ttax.project`; `deltas_to_tangent` turns it into a TT
tensor.
"""

import functools
import inspect
import string
import sys
import time
import warnings
from copy import deepcopy
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import Callable

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import basix
import jax.numpy as jnp
import numpy as np
import ttax
import ufl
from dolfinx import fem, mesh
from jax import jit
from mpi4py import MPI
from ttax.riemannian import deltas_to_tangent

from core.exp_sum import extract_coefficients, to_filename
from core.problem import setup_1Dmatrices
from core.tangent import (
    cg_tree_math_point_dependent,
    lincomb_deltas,
    vdot_deltas,
    zeroslike_deltas,
)
from core.tensor_full import (
    apply_single_axis,
    apply_transformations,
    sum_apply_transformation,
)
from core.tt_utils import (
    get_component,
    partial_flat_inner_last,
    partial_flat_inner_last_tangent_deltas,
    rescale_last_component,
    rescale_last_component_deltas,
    tt_from_fun,
)


def fuse(_func=None, *, argnums=None):
    """Decorator to fuse function computations using ttax.fuse.

    Keyword and default arguments are bound to positions before calling the
    fused function, so the decorated function keeps its original signature.

    Args:
        _func: Function to be fused (when used without parentheses).
        argnums: Argument positions to fuse.

    Returns:
        Decorated function with fused computations.
    """

    def decorator(func):
        fused_func = ttax.fuse(func, argnums=argnums)
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # Pass only the values, as positional arguments
            return fused_func(*bound.args)

        return wrapper

    if _func is not None and inspect.isfunction(_func):
        return decorator(_func)
    else:
        return decorator


def tt_energyfactory(
    S, M, V, beta, Prec_PAP_Mr=None, Prec_PAP_N=None, K=None, alphas=None
):
    """Factory function to create energy and gradient functions for tensor train optimization.

    The discrete energy and bilinear form are those of Sec. 'Discretization by
    spectral elements' (single component) and of Sec. 'Discretization' of
    'The multicomponent Gross--Pitaevskii equation' (K is not None), with

        A_u = S + diag(V) + 2 beta diag(M_rec * u2),   u2 = u * u,

    where, in the multicomponent case, u2 is replaced by K(u * u). Most
    returned functions come in two variants: `f(..., u, ...)` and
    `f_u2(..., u, u2, ...)`, where `u2` is a precomputed (possibly
    rank-truncated) approximation of the Hadamard square u * u; the plain
    variant computes u2 = ttax.multiply(u, u). Tangent vectors are in delta
    form (see the module docstring).

    Args:
        S: Stiffness operator in TT format (as a TT matrix); S ⊗ I_p in the
            multicomponent case.
        M: Mass operator in TT format (as a TT tensor, since it is a diagonal TT matrix).
            This is the tensor of quadrature weights; only its entry-wise
            reciprocal M_rec = 1/M (`M_inv`) is used, in the quartic term.
        V: Potential operator in TT format (as a TT tensor, since it is a diagonal TT matrix).
            Batched TT tensor of potential values at the nodes; the batch
            entries are summed.
        beta: Nonlinearity coefficient.
        Prec_PAP_Mr: Preconditioner for PAP_Mr operator (default: identity).
            Called as Prec_PAP_Mr(deltas, u, u2) and returning deltas, e.g. the
            output of `setup_SplusV_precond`.
        Prec_PAP_N: Preconditioner for PAP_N operator (default: identity).
            Currently always replaced by P^N Prec_PAP_Mr P^N (see the comment
            where Prec_PAP_N is defined below).
        K: TT matrix I ⊗ ... ⊗ I ⊗ K applying the p x p interaction matrix K
            to the component mode (multicomponent case), or None for the
            single-component problem.
        alphas: List of component-wise norms (default: None).
            In this function alphas[j] is the prescribed mass N_j = ||u_j||^2 of
            component j (multicomponent case only).

    Returns:
        Dictionary containing energy, gradient, and projection functions.
        The entries "gradE_au" and "gradE_au_u2" are aliases for the gradient
        used by default: gradE_au_Hu (Sec. 'The a_u Riemannian gradient') when
        K is None, gradE_au_PAP_N (route II, Sec. 'Riemannian gradient II:
        change of metric') otherwise.
    """
    # Compute the M operator
    if isinstance(M, ttax.TTMatrix):
        M_cores = M.tt_cores
        M_inv = ttax.TTMatrix([1 / core for core in M_cores])
    else:
        M_inv = 1 / M

    if K is None:
        # ===== ENERGY =====
        # @fuse
        def energy_u2(u, u2):
            """Compute energy functional E(u).

            E(u) = 1/2 <u, S u> + 1/2 <V * u, u> + beta/2 <u2, M_rec * u2>,
            Sec. 'Discretization by spectral elements'.

            Args:
                u: TT representation of the state.
                u2: Hadamard square u * u in TT format (possibly truncated).

            Returns:
                Energy value.
            """
            utSu = ttax.flat_inner(u, S @ u)
            utVu = jnp.sum(ttax.flat_inner(ttax.multiply(V, u), u))
            u2tMsqrtu2 = ttax.flat_inner(u2, ttax.multiply(M_inv, u2))
            return 0.5 * utSu + 0.5 * utVu + 0.5 * beta * u2tMsqrtu2

        # ====== BILINEAR FORM ======
        @fuse
        def bilinear_form_u2(u2, v, w):
            """Compute bilinear form a_u(v, w).

            a_u(v, w) = <S v, w> + <V * v, w> + 2 beta <u2, v * M_rec * w>,
            Sec. 'Discretization by spectral elements'.

            Args:
                u2: Hadamard square u * u of the base point u, in TT format.
                v: First argument in TT format.
                w: Second argument in TT format.

            Returns:
                Bilinear form value.
            """
            vtSw = ttax.flat_inner(S @ v, w)
            vtVw = jnp.sum(ttax.flat_inner(ttax.multiply(V, v), w))
            vtMsqrtu2w = ttax.flat_inner(
                u2, ttax.multiply(v, ttax.multiply(M_inv, w))
            )  # FIXME: THIS CAN BE FUSED, BUT IF I MOVE THE MASS, E.G, TO THE SECOND TERM, IT DOES NOT ANYMORE
            return vtSw + vtVw + 2 * beta * vtMsqrtu2w

    else:
        # ===== ENERGY =====
        # @fuse
        def energy_u2(u, u2):
            """Compute energy functional E(u).

            Multicomponent energy
            E(u) = 1/2 <u, S u> + 1/2 <V * u, u> + beta/2 <K u2, M_rec * u2>,
            Sec. 'Discretization' (multicomponent Gross--Pitaevskii equation).

            Args:
                u: TT representation of the state.
                u2: Hadamard square u * u in TT format (possibly truncated).

            Returns:
                Energy value.
            """
            utSu = ttax.flat_inner(u, S @ u)
            utVu = jnp.sum(ttax.flat_inner(ttax.multiply(V, u), u))
            u2tMsqrtu2 = ttax.flat_inner(K @ u2, ttax.multiply(M_inv, u2))
            return 0.5 * utSu + 0.5 * utVu + 0.5 * beta * u2tMsqrtu2

        # ====== BILINEAR FORM ======
        @fuse
        def bilinear_form_u2(u2, v, w):
            """Compute bilinear form a_u(v, w).

            Multicomponent bilinear form
            a_u(v, w) = <S v, w> + <V * v, w> + 2 beta <K u2, v * M_rec * w>,
            Sec. 'Discretization' (multicomponent Gross--Pitaevskii equation).

            Args:
                u2: Hadamard square u * u of the base point u, in TT format.
                v: First argument in TT format.
                w: Second argument in TT format.

            Returns:
                Bilinear form value.
            """
            vtSw = ttax.flat_inner(S @ v, w)
            vtVu = jnp.sum(ttax.flat_inner(ttax.multiply(V, v), w))
            vtMsqrtu2w = ttax.flat_inner(
                K @ u2, ttax.multiply(v, ttax.multiply(M_inv, w))
            )  # FIXME: THIS CAN BE FUSED, BUT IF I MOVE THE MASS, E.G, TO THE SECOND TERM, IT DOES NOT ANYMORE
            return vtSw + vtVu + 2 * beta * vtMsqrtu2w

    def energy(u):
        """Compute the discrete energy E(u), with u2 = u * u computed exactly.

        Args:
            u: State in TT format.

        Returns:
            Energy value.
        """
        return energy_u2(u, ttax.multiply(u, u))

    @fuse
    def bilinear_form(u, v, w):
        """Compute a_u(v, w), with u2 = u * u computed exactly.

        Args:
            u: Base point in TT format.
            v: First argument in TT format.
            w: Second argument in TT format.

        Returns:
            Bilinear form value.
        """
        return bilinear_form_u2(ttax.multiply(u, u), v, w)

    @fuse
    def _PA_Mr_u2(v, u, u2):
        """Apply the operator (Proj_u * A_u * v) with TT input and delta output,
        where Proj_u is the projection onto the tangent space of Mr at u.

        Proj_u is the Frobenius-orthogonal TT tangent-space projection. The
        result is obtained by automatic differentiation of the linear
        functional z -> a_u(v, z) at u.

        Args:
            v: Input vector in TT format.
            u: Base point in TT format.
            u2: Hadamard square u * u in TT format.

        Returns:
            Result in delta form.
        """
        return ttax.grad(lambda z: bilinear_form_u2(u2, v, z))(u)

    @fuse
    def _PA_Mr(v, u):
        """Same as `_PA_Mr_u2` with u2 = u * u (exposed as "PA_Mr")."""
        return _PA_Mr_u2(v, u, ttax.multiply(u, u))

    # PA_Mr = jax.jit(_PA_Mr)
    PA_Mr_u2 = _PA_Mr_u2
    PA_Mr = _PA_Mr

    @fuse
    def PA_Mr_tang_u2(v, u, u2):
        """Same as `_PA_Mr_u2`, but returns the tangent vector as a TT tensor.

        Args:
            v: Input vector in TT format.
            u: Base point in TT format.
            u2: Hadamard square u * u in TT format.

        Returns:
            Proj_u A_u v as a TT tensor.
        """
        return deltas_to_tangent(_PA_Mr_u2(v, u, u2), u)

    @fuse
    def PA_Mr_tang(v, u):
        """Same as `PA_Mr_tang_u2` with u2 = u * u."""
        return PA_Mr_tang_u2(v, u, ttax.multiply(u, u))

    @fuse(argnums=[1, 2])
    def PAP_Mr_deltas_u2(v_delta, u, u2):
        """Apply the operator (Proj_u * A_u * v) with delta input/output,
        where Proj_u is the projection onto the tangent space of Mr at u.

        Since v is a tangent vector, this is the operator
        PAP_Mr = Proj_u A_u Proj_u whose systems are solved by CG, Sec.
        'Discretization by spectral elements' and Sec. 'Preconditioning'.

        Args:
            v_delta: Input vector in delta form.
            u: Base point in TT format.
            u2: Square of u in TT format.

        Returns:
            Result in delta form.
        """
        return _PA_Mr_u2(deltas_to_tangent(v_delta, u), u, u2)

    @fuse(argnums=[1])
    def PAP_Mr_deltas(v_delta, u):
        """Same as `PAP_Mr_deltas_u2` with u2 = u * u."""
        return PAP_Mr_deltas_u2(v_delta, u, ttax.multiply(u, u))

    # Conjugate gradient method for PAP_Mr. The stopping criterion of
    # cg_tree_math_point_dependent is the relative residual ||r|| <= tol ||b||
    # in the Euclidean norm of the delta representation.
    if Prec_PAP_Mr is None:
        Prec_PAP_Mr = lambda x, u, u2: x  # Identity preconditioner
    cg_PAP_Mr = partial(
        cg_tree_math_point_dependent,
        A=PAP_Mr_deltas_u2,
        M=Prec_PAP_Mr,
        return_info=True,
    )

    # ===== PROJECTORS =====
    @fuse(argnums=[0, 1, 2])
    def Hu_u2(x, u, u2, maxiter=200, tol=1e-10):
        """Apply H_u(x) [See paper for definition].
        Solves the linear equation using CG.

        H_u x is the Galerkin inverse of A_u onto the tangent space of M_r;
        since the mass matrix is the identity in the rescaled basis,
        H_u x = (PAP_Mr)^+ Proj_u x, computed by CG with PAP_Mr preconditioned
        by Prec_PAP_Mr. Sec. 'The a_u Riemannian gradient', paragraph
        'Computation of Proj^M_u and H_u'.

        Args:
            x: Input vector in TT format.
            u: Base point in TT format.
            u2: Hadamard square u * u in TT format.
            maxiter: Maximum CG iterations.
            tol: Convergence tolerance.
                Relative residual tolerance of CG (Euclidean norm of the
                deltas).

        Returns:
            Tuple of (H_u(x) in delta form, projected x in delta form, CG info).
        """
        deltas_proj_x = ttax.project(x, u)
        x0 = zeroslike_deltas(deltas_proj_x)
        Hu_x_deltas, info_cg = cg_PAP_Mr(
            u=u, u2=u2, b=deltas_proj_x, x0=x0, maxiter=maxiter, tol=tol
        )
        return Hu_x_deltas, deltas_proj_x, info_cg

    @fuse(argnums=[0, 1])
    def Hu(x, u, maxiter=200, tol=1e-10):
        """Same as `Hu_u2` with u2 = u * u."""
        return Hu_u2(x, u, ttax.multiply(u, u), maxiter, tol)

    @fuse(argnums=[0, 1, 2])
    def project_au_Mr_u2(x, u, u2, maxiter=200, tol=1e-10):
        """Compute a_u-orthogonal projection onto tangent space to Mr at u.
        Solves the linear equation using CG.

        Proj^M_u x = (PAP_Mr)^+ Proj_u A_u x, Sec. 'The a_u Riemannian
        gradient', paragraph 'Computation of Proj^M_u and H_u'.

        Args:
            x: Input vector in TT format.
            u: Base point in TT format.
            u2: Hadamard square u * u in TT format.
            maxiter: Maximum CG iterations.
            tol: Convergence tolerance.
                Relative residual tolerance of CG.

        Returns:
            Projected vector in delta form.
        """
        deltas_proj_Au_x = PA_Mr(x, u)
        x0 = zeroslike_deltas(deltas_proj_Au_x)
        proj_au_Mr_x_deltas, info_cg = cg_PAP_Mr(
            u=u, u2=u2, b=deltas_proj_Au_x, x0=x0, maxiter=maxiter, tol=tol
        )
        return proj_au_Mr_x_deltas

    @fuse(argnums=[0, 1])
    def project_au_Mr(x, u, maxiter=200, tol=1e-10):
        """Same as `project_au_Mr_u2` with u2 = u * u."""
        return project_au_Mr_u2(x, u, ttax.multiply(u, u), maxiter, tol)

    # Proj^N_u below is the Frobenius-orthogonal projection onto the tangent
    # space of N_r = M_r ∩ (unit sphere | oblique manifold); it is Proj_u
    # followed by a rank-one (single component) or rank-p (multicomponent)
    # correction, Sec. 'Geometric tools ...' (paragraph '$L^2$-orthogonal
    # projection and transporter.') and Sec. 'Riemannian gradient II: change
    # of metric'.
    if K is None:

        def _PAP_N_body(v_deltas, u, u2, u_delta):
            """Proj^N_u A_u Proj^N_u on deltas, with Proj^N_u v = v - <v, u> u.

            `u_delta` is the delta form of u itself (ttax.project(u, u)).
            """
            v = lincomb_deltas(1, v_deltas, -vdot_deltas(v_deltas, u_delta), u_delta)
            res_delta = PAP_Mr_deltas_u2(v, u, u2)
            res_delta = lincomb_deltas(
                1, res_delta, -vdot_deltas(res_delta, u_delta), u_delta
            )
            return res_delta

        @fuse(argnums=[1, 2])
        def PAP_N_deltas_u2(v_deltas, u, u2):
            """Apply the operator (Proj^N_u * A_u * Proj^N_u * v) in delta form,
            where Proj^N_u is the projection onto the tangent space of Nr at u.

            Args:
                v_deltas: Input vector in delta form.
                u: Base point in TT format.
                u2: Hadamard square u * u in TT format.

            Returns:
                Result in delta form.
            """
            u_delta = ttax.project(u, u)
            return _PAP_N_body(v_deltas, u, u2, u_delta)

    else:

        def _P_N_deltas(v_deltas, u, u_delta):
            """Proj^N_u v = v - u diag<u, v> N^{-1} on deltas (multicomponent).

            The component-wise inner products and the rescaling only touch the
            last core, Sec. 'Implementation details'. `u_delta` is the delta
            form of u itself (ttax.project(u, u)).
            """
            coeff = partial_flat_inner_last_tangent_deltas(v_deltas, u_delta) / alphas
            orthogonalization_deltas = rescale_last_component_deltas(u_delta, coeff)
            res_delta = lincomb_deltas(
                1,
                v_deltas,
                -1,
                orthogonalization_deltas,
            )
            return res_delta

        def _PAP_N_body(v_deltas, u, u2, u_delta):
            """Proj^N_u A_u Proj^N_u on deltas (multicomponent)."""
            res_delta = _P_N_deltas(v_deltas, u, u_delta)
            res_delta = PAP_Mr_deltas_u2(res_delta, u, u2)
            res_delta = _P_N_deltas(res_delta, u, u_delta)
            return res_delta

        @fuse(argnums=[1, 2])
        def PAP_N_deltas_u2(v_deltas, u, u2):
            """Apply the operator (Proj^N_u * A_u * Proj^N_u * v) in delta form,
            where Proj^N_u is the projection onto the tangent space of Nr at u.

            Multicomponent operator PAP_N of route II, Sec. 'Riemannian
            gradient II: change of metric'.

            Args:
                v_deltas: Input vector in delta form.
                u: Base point in TT format.
                u2: Hadamard square u * u in TT format.

            Returns:
                Result in delta form.
            """
            u_delta = ttax.project(u, u)
            return _PAP_N_body(v_deltas, u, u2, u_delta)

    @fuse(argnums=[1])
    def PAP_N_deltas(v_deltas, u):
        """Same as `PAP_N_deltas_u2` with u2 = u * u."""
        return PAP_N_deltas_u2(v_deltas, u, ttax.multiply(u, u))

    # Conjugate gradient method for PAP_N
    if Prec_PAP_N is None and Prec_PAP_Mr is None:
        Prec_PAP_N = lambda x, u, u2: x  # Identity preconditioner
    elif Prec_PAP_Mr is not None:
        # The provided preconditioner approximates P_Mr (S+V)^{-1} (it projects
        # the result onto the tangent space of Mr). The PAP_N operator instead
        # sandwiches A_u between projections onto the tangent space of N, so the
        # matching preconditioner must project onto N: P_N (S+V)^{-1} P_N.
        _Prec_PAP_Mr_provided = Prec_PAP_Mr

        # FIXME: BROKEN FOR THE SINGLE-COMPONENT CASE. `_P_N_deltas` IS BOUND
        # ONLY IN THE MULTICOMPONENT BRANCH BELOW/ABOVE (`else:` OF `if K is
        # None`), SO WITH K IS NONE AND A PRECONDITIONER SUPPLIED ANY CALL TO
        # THE EXPORTED `gradE_au_PAP_N*` RAISES NameError. UNREACHABLE FROM THE
        # EXPERIMENTS TODAY (THE SELECTOR SENDS p IS NONE TO `gradE_au_Hu`), BUT
        # THESE ARE PUBLIC ENTRIES OF THE RETURNED DICT. FIX: HOIST A
        # SINGLE-COMPONENT `_P_N_deltas` -- THE `v - <v, u_delta> u_delta`
        # RANK-1 UPDATE ALREADY INLINED IN `_PAP_N_body` -- OUT OF THE BRANCH.
        # NOTE: Prec_PAP_Mr is never None at this point (it is replaced by the
        # identity above), so this branch is always taken and a user-supplied
        # Prec_PAP_N is ignored. Preconditioner of route II:
        # Proj^N_u B^{-1} Proj^N_u, Sec. 'Riemannian gradient II: change of
        # metric'.
        def Prec_PAP_N(deltas, u, u2):
            u_delta = ttax.project(u, u)
            deltas = _P_N_deltas(deltas, u, u_delta)
            z = _Prec_PAP_Mr_provided(deltas, u, u2)
            z = _P_N_deltas(z, u, u_delta)
            return z

    cg_PAP_N = partial(cg_tree_math_point_dependent, A=PAP_N_deltas_u2, M=Prec_PAP_N)

    if K is None:

        @fuse(argnums=[0, 1, 2])
        def project_au_N_u2(v, u, u2, maxiter=200, tol=1e-10):
            """Compute a_u-orthogonal projection onto tangent space to N at u, using
            H_u. Solves the linear equation using CG.

            Proj^N_u v = Proj^M_u v - (u, Proj^M_u v) / (u, H_u u) H_u u, Sec.
            'The a_u Riemannian gradient', paragraph 'Computation of
            Proj^N_u'. Requires two CG solves (single component only).

            Args:
                v: Input vector in TT format.
                u: Base point in TT format.
                u2: Hadamard square u * u in TT format.
                maxiter: Maximum CG iterations.
                tol: Convergence tolerance.
                    Relative residual tolerance of CG.

            Returns:
                Natural projection in delta form.
            """
            deltas_proj_v = project_au_Mr_u2(v, u, u2, maxiter, tol)
            Hu_u_deltas, deltas_proj_Mu, info_cg = Hu_u2(u, u, u2, maxiter, tol)

            coeff = vdot_deltas(deltas_proj_Mu, deltas_proj_v) / vdot_deltas(
                deltas_proj_Mu, Hu_u_deltas
            )
            PuNv = lincomb_deltas(1, deltas_proj_v, -coeff, Hu_u_deltas)

            return PuNv

        @fuse(argnums=[0, 1])
        def project_au_N(v, u, maxiter=200, tol=1e-10):
            """Same as `project_au_N_u2` with u2 = u * u."""
            return project_au_N_u2(v, u, ttax.multiply(u, u), maxiter, tol)

    else:
        warnings.warn("project_au_N not implemented for K != None")
        project_au_N_u2, project_au_N = None, None

    # ===== GRADIENTS =====
    # Riemannian gradient with Frobenius metric
    # (gradient on M_r, mass constraint not enforced; delta form)
    riemannian_gradient_embedded = ttax.fuse(ttax.grad(energy))

    # Riemannian gradient with Au metric
    if K is None:

        @fuse(argnums=[0, 1])
        def gradE_au_Hu_withinfo_u2(u, u2, maxiter=200, tol=1e-10):
            """Compute the a_u Riemannian gradient using the H_u operator, and return CG info.

            grad E(u) = Proj^N_u u = u - (u, u) / (u, H_u u) H_u u, using
            Proj^M_u u = u; one CG solve. a_u Riemannian gradient, Sec. 'The
            a_u Riemannian gradient'.

            Args:
                u: Current point in TT format.
                u2: Hadamard square u * u in TT format.
                maxiter: Maximum CG iterations.
                tol: Convergence tolerance.
                    Relative residual tolerance of CG (default 1e-10, as in
                    the paper runs).

            Returns:
                Tuple of (gradient in delta form, CG info).
            """
            print("gradE_au_Hu to be compiled...")
            deltas_proj_u = ttax.project(u, u)  # FIXME: AVOID PROJECTING EACH TIME

            Hu_u_deltas, deltas_proj_Mu, info_cg = Hu_u2(u, u, u2, maxiter, tol)

            coeff = vdot_deltas(deltas_proj_Mu, deltas_proj_u) / vdot_deltas(
                deltas_proj_Mu, Hu_u_deltas
            )
            PuNv = lincomb_deltas(1, deltas_proj_u, -coeff, Hu_u_deltas)

            return PuNv, info_cg

    else:

        @fuse(argnums=[0, 1])
        def gradE_au_Hu_withinfo_u2(u, u2, maxiter=200, tol=1e-10):
            """Compute the a_u Riemannian gradient using the H_u operator, and return CG info.

            Route I, Sec. 'Riemannian gradient I: the operator H_φ':
            grad E(u) = u - H_u(u Sigma), Sigma = diag(sigma), where G sigma = b
            with G[:, j] = diag<H_u u_j, u> (u_j: j-th frame, all components
            but j set to zero) and b = diag<u, u>. Costs p + 1 CG solves.

            Args:
                u: Current point in TT format.
                u2: Hadamard square u * u in TT format.
                maxiter: Maximum CG iterations.
                tol: Convergence tolerance.
                    Relative residual tolerance of each CG solve.

            Returns:
                Tuple of (gradient in delta form, CG info).
                The CG info is a list with one entry per solve (p + 1).
            """
            print("gradE_au_Hu to be compiled...")
            deltas_proj_u = ttax.project(u, u)  # FIXME: AVOID PROJECTING EACH TIME
            info_cg_list = []
            p = u.tt_cores[-1].shape[1]
            G = jnp.zeros((p, p))
            for i in range(p):
                ui = get_component(u, i)
                Hu_ui_deltas, deltas_proj_Mui, info_cg = Hu_u2(ui, u, u2, maxiter, tol)
                G = G.at[:, i].set(
                    partial_flat_inner_last_tangent_deltas(Hu_ui_deltas, deltas_proj_u)
                )
                info_cg_list.append(info_cg)

            b = partial_flat_inner_last_tangent_deltas(deltas_proj_u, deltas_proj_u)
            sigma = jnp.linalg.solve(G, b)
            usigma = rescale_last_component(u, sigma)
            Hu_usigma_deltas, deltas_proj_Musigma, info_cg = Hu_u2(
                usigma, u, u2, maxiter, tol
            )
            info_cg_list.append(info_cg)
            grad = lincomb_deltas(1, deltas_proj_u, -1, Hu_usigma_deltas)

            return grad, info_cg_list

    @fuse(argnums=[0])
    def gradE_au_Hu_withinfo(u, maxiter=200, tol=1e-10):
        """Same as `gradE_au_Hu_withinfo_u2` with u2 = u * u."""
        return gradE_au_Hu_withinfo_u2(u, ttax.multiply(u, u), maxiter=maxiter, tol=tol)

    @fuse(argnums=[0, 1])
    def gradE_au_Hu_u2(u, u2, maxiter=200, tol=1e-10):
        """Same as `gradE_au_Hu_withinfo_u2`, but returns only the gradient."""
        return gradE_au_Hu_withinfo_u2(u, u2, maxiter, tol)[0]

    @fuse(argnums=[0])
    def gradE_au_Hu(*args, **kwargs):
        """Same as gradE_au_Hu_withinfo, but returns only the gradient.

        Args:
            Same as gradE_au_Hu_withinfo.

        Returns:
            Gradient in delta form.
        """
        return gradE_au_Hu_withinfo(*args, **kwargs)[0]

    if K is None:

        @fuse(argnums=[0, 1])
        def gradE_au_PAP_N_withinfo_u2(
            u, u2, maxiter=200, kappa=0.1, theta=1, tol=None
        ):
            """Compute Riemannian gradient using the PAP_N operator, and return CG info.

            Change of metric (single-component case of route II, Sec.
            'Riemannian gradient II: change of metric'): solve
            PAP_N g = Proj^N_u(A_u u) by preconditioned CG, where
            Proj^N_u(A_u u) = Proj_u(A_u u) - <u, Proj_u(A_u u)> u is the
            Frobenius Riemannian gradient on N_r.

            Args:
                u: Current point in TT format.
                u2: Hadamard square u * u in TT format.
                maxiter: Maximum CG iterations.
                kappa: Tolerance parameter.
                theta: Exponent for adaptive tolerance (as in Riemannian Trust Region).
                    The relative CG tolerance is min(||b||^theta, kappa),
                    with b the right-hand side.
                tol: If not None, overrides kappa and theta (theta = 0), so
                    that the relative CG tolerance is min(1, tol).

            Returns:
                Tuple of (gradient in delta form, list of CG info dicts).
            """
            print("gradE_au_PAP_N to be compiled...")
            if tol is not None:
                print("Using tol instead of kappa and theta")
                kappa = tol
                theta = 0.0
            deltas_proj_u = ttax.project(u, u)
            grad_Mr_deltas = PAP_Mr_deltas_u2(deltas_proj_u, u, u2)
            coeff = vdot_deltas(deltas_proj_u, grad_Mr_deltas)
            grad_N_deltas = lincomb_deltas(1, grad_Mr_deltas, -coeff, deltas_proj_u)
            x0 = zeroslike_deltas(grad_N_deltas)
            gradnorm = jnp.sqrt(vdot_deltas(grad_N_deltas, grad_N_deltas))

            def _PAP_N_cached(v_deltas, u_, u2_):
                return _PAP_N_body(v_deltas, u_, u2_, deltas_proj_u)

            cg_PAP_N_cached = partial(
                cg_tree_math_point_dependent,
                A=_PAP_N_cached,
                M=Prec_PAP_N,
                return_info=True,
            )
            gradE_au_deltas, info_cg = cg_PAP_N_cached(
                u=u,
                u2=u2,
                b=grad_N_deltas,
                x0=x0,
                maxiter=maxiter,
                tol=jnp.min(jnp.array([gradnorm**theta, kappa])),
            )

            return gradE_au_deltas, [info_cg]

    else:

        @fuse(argnums=[0, 1])
        def gradE_au_PAP_N_withinfo_u2(
            u, u2, maxiter=200, kappa=0.1, theta=1, tol=None
        ):
            """Compute Riemannian gradient using the PAP_N operator, and return CG info.

            Route II, Sec. 'Riemannian gradient II: change of metric': solve
            PAP_N g = Proj^N_u(A_u u) by preconditioned CG (a single solve,
            independent of p), where
            Proj^N_u(A_u u) = Proj_u(A_u u) - u diag<u, Proj_u(A_u u)> N^{-1}
            is the Frobenius Riemannian gradient on N_r.

            Args:
                u: Current point in TT format.
                u2: Hadamard square u * u in TT format.
                maxiter: Maximum CG iterations.
                kappa: Tolerance parameter.
                theta: Exponent for adaptive tolerance (as in Riemannian Trust Region).
                    The relative CG tolerance is min(||b||^theta, kappa),
                    with b the right-hand side.
                tol: If not None, overrides kappa and theta (theta = 0), so
                    that the relative CG tolerance is min(1, tol).

            Returns:
                Tuple of (gradient in delta form, list of CG info dicts).
            """
            print("gradE_au_PAP_N to be compiled...")
            if tol is not None:
                print("Using tol instead of kappa and theta")
                kappa = tol
                theta = 0.0
            deltas_proj_u = ttax.project(u, u)
            grad_Mr_deltas = PAP_Mr_deltas_u2(deltas_proj_u, u, u2)
            coeff = (
                partial_flat_inner_last_tangent_deltas(deltas_proj_u, grad_Mr_deltas)
                / alphas
            )
            deltas_proj_u_scaled = rescale_last_component_deltas(deltas_proj_u, coeff)
            grad_N_deltas = lincomb_deltas(1, grad_Mr_deltas, -1, deltas_proj_u_scaled)

            x0 = zeroslike_deltas(grad_N_deltas)
            gradnorm = jnp.sqrt(vdot_deltas(grad_N_deltas, grad_N_deltas))

            def _PAP_N_cached(v_deltas, u_, u2_):
                return _PAP_N_body(v_deltas, u_, u2_, deltas_proj_u)

            cg_PAP_N_cached = partial(
                cg_tree_math_point_dependent,
                A=_PAP_N_cached,
                M=Prec_PAP_N,
                return_info=True,
            )
            gradE_au_deltas, info_cg = cg_PAP_N_cached(
                u=u,
                u2=u2,
                b=grad_N_deltas,
                x0=x0,
                maxiter=maxiter,
                # Adaptive relative CG tolerance min(||grad_N||^theta, kappa)
                # (Euclidean norm of the deltas of the right-hand side). The
                # multicomponent runs of the paper use kappa = 0.01, theta = 1
                # (set in paperexp_multicomponent_2d.py, tol=None), not the
                # fixed 1e-10 of the single-component and full-format solves.
                tol=jnp.min(jnp.array([gradnorm**theta, kappa])),
            )

            return gradE_au_deltas, [info_cg]

    @fuse(argnums=[0, 1])
    def gradE_au_PAP_N_u2(u, u2, maxiter=200, kappa=0.1, theta=1, tol=None):
        """Same as `gradE_au_PAP_N_withinfo_u2`, but returns only the gradient."""
        return gradE_au_PAP_N_withinfo_u2(
            u, u2, maxiter=maxiter, kappa=kappa, theta=theta, tol=tol
        )[0]

    @fuse(argnums=[0])
    def gradE_au_PAP_N_withinfo(u, maxiter=200, kappa=0.1, theta=1, tol=None):
        """Same as `gradE_au_PAP_N_withinfo_u2` with u2 = u * u."""
        return gradE_au_PAP_N_withinfo_u2(
            u, ttax.multiply(u, u), maxiter=maxiter, kappa=kappa, theta=theta, tol=tol
        )

    @fuse(argnums=[0])
    def gradE_au_PAP_N(u, maxiter=200, kappa=0.1, theta=1, tol=None):
        """Same as `gradE_au_PAP_N_u2` with u2 = u * u (gradient only)."""
        return gradE_au_PAP_N_u2(
            u, ttax.multiply(u, u), maxiter=maxiter, kappa=kappa, theta=theta, tol=tol
        )

    # @fuse
    def gradE_au_slow(u, *args, **kwargs):
        """Compute Riemannian gradient using explicit projection of u onto the
        tangent space of N at u.

        Calls project_au_N(u, u) (two CG solves); single component only, since
        project_au_N is None when K is not None.

        Args:
            u: Current point in TT format.
            args, kwargs: Additional arguments for projection.

        Returns:
            Gradient in delta form.
        """
        return project_au_N(u, u, *args, **kwargs)

    # ====== SANITY CHECKS ======
    def Au_mv(v, u):
        """Compute matrix-vector product A_u * v explicitly (for testing).

        Uses the single-component formula (u * u, not K(u * u)).

        Args:
            v: Input vector in TT format.
            u: Base point in TT format.

        Returns:
            Result in TT format.
        """
        Sv = S @ v
        Vv_batch = ttax.multiply(V, v)
        Vv_batch_sum = functools.reduce(
            ttax.ops.add,
            [Vv_batch.batch_loc[i,] for i in range(Vv_batch.tt_cores[0].shape[0])],
        )
        nonlinear_term = (
            2 * beta * ttax.multiply(ttax.multiply(u, u), ttax.multiply(M_inv, v))
        )
        return Sv + Vv_batch_sum + nonlinear_term

    def PAP_Mr_explicit(z, u):
        """Apply the operator (Proj^Mr_u * A_u * Proj^Mr_u * v) explicitly (for testing).

        Args:
            z: Input vector in TT format.
            u: Base point in TT format.

        Returns:
            Result in TT format.
        """
        tmp = deltas_to_tangent(ttax.project(z, u), u)
        tmp = Au_mv(tmp, u)
        tmp = deltas_to_tangent(ttax.project(tmp, u), u)
        return tmp

    tt_functions = {
        "energy": energy,
        "energy_u2": energy_u2,
        "bilinear_form": bilinear_form,
        "bilinear_form_u2": bilinear_form_u2,
        "PA_Mr": PA_Mr,
        "PA_Mr_u2": PA_Mr_u2,
        "PA_Mr_tang": PA_Mr_tang,
        "PA_Mr_tang_u2": PA_Mr_tang_u2,
        "PAP_Mr_deltas": PAP_Mr_deltas,
        "PAP_Mr_deltas_u2": PAP_Mr_deltas_u2,
        "cg_PAP_Mr": cg_PAP_Mr,
        "riemannian_gradient_embedded": riemannian_gradient_embedded,
        "gradE_au_Hu_withinfo": gradE_au_Hu_withinfo,
        "gradE_au_Hu_withinfo_u2": gradE_au_Hu_withinfo_u2,
        "gradE_au_Hu": gradE_au_Hu,
        "gradE_au_Hu_u2": gradE_au_Hu_u2,
        "gradE_au_PAP_N": gradE_au_PAP_N,
        "gradE_au_PAP_N_u2": gradE_au_PAP_N_u2,
        "gradE_au_PAP_N_withinfo": gradE_au_PAP_N_withinfo,
        "gradE_au_PAP_N_withinfo_u2": gradE_au_PAP_N_withinfo_u2,
        "gradE_au_slow": gradE_au_slow,
        "project_au_Mr": project_au_Mr,
        "project_au_Mr_u2": project_au_Mr_u2,
        "project_au_N": project_au_N,
        "project_au_N_u2": project_au_N_u2,
        "PAP_N_deltas": PAP_N_deltas,
        "PAP_N_deltas_u2": PAP_N_deltas_u2,
        "cg_PAP_N": cg_PAP_N,
        "Hu": Hu,
        "Hu_u2": Hu_u2,
        "Au_mv": Au_mv,
        "PAP_Mr_explicit": PAP_Mr_explicit,
        "gradE_au": (
            gradE_au_Hu if K is None else gradE_au_PAP_N
        ),  # Alias for convenience
        "gradE_au_u2": (
            gradE_au_Hu_u2 if K is None else gradE_au_PAP_N_u2
        ),  # Alias for convenience
    }

    return tt_functions


class FullEnergyFunctional:
    """Energy functional in full tensor format.

    Same discrete energy and bilinear form as `tt_energyfactory`, on full
    arrays of shape [n]*dim (or [n]*dim + [p] in the multicomponent case), and
    the fast-diagonalization preconditioners S^{-1} and (S + V_0)^{-1} of
    Sec. 'Preconditioning in the full tensor format'.

    Args:
        dim: Spatial dimension.
        S_1d: 1D stiffness matrix.
            In the rescaled basis (diag(m)^{-1/2} S diag(m)^{-1/2}).
        M_1d: 1D mass matrix.
            Lumped (diagonal) 1D mass matrix; its diagonal gives the
            quadrature weights m, and M_inv is the reciprocal-weight tensor
            M_rec.
        V: Potential (TT format or full tensor).
            Values at the nodes, shared by all components. May be None when
            the object is only used to set up preconditioners; then `M` and
            `M_inv` are not defined and only `factors_only=True` setups work.
        beta: Nonlinearity coefficient.
        K: p x p interaction matrix (multicomponent case), or None.
        alphas: Component-wise L^2 norms ||u_j|| (multicomponent case). Unlike
            in `tt_energyfactory`, they enter squared (as masses alphas**2) in
            `riemannian_gradient_au_withaux`, consistently with the full-format
            manifold, which normalizes each component to norm alphas[j].
        p: Number of components (multicomponent case), or None.
    """

    def __init__(self, dim, S_1d, M_1d, V, beta, K=None, alphas=None, p=None):
        # Mass
        mdiag = jnp.diag(M_1d.todense())
        tensorizing_string = ",".join(string.ascii_lowercase[:dim])
        if V is not None:
            self.M = jnp.einsum(tensorizing_string, *tuple([mdiag] * dim))
            self.M_inv = jnp.einsum(tensorizing_string, *tuple([1 / mdiag] * dim))
        # Stiffness
        self.S = []
        for i in range(dim):
            # Only one non-identity Kronecker factor per term; identities are
            # stored as None so apply_transformations skips them (no full-size
            # temporary, fewer flops).
            row = [None] * dim
            row[i] = S_1d  # FIXME: AVOID DENSE CONVERSION
            self.S.append(row)
        # Multicomponent: extend each S Kronecker-factor list with an identity
        # (stored as None) for the component axis so that apply_transformations
        # does not index out of bounds when u.ndim == dim+1.
        if K is not None:
            self.S = [row + [None] for row in self.S]
        # V and beta
        if ttax.is_tt_matrix(V):
            V = ttax.full(ttax.ops.vector_to_tensor(V))
        self.V = V
        self.beta = beta
        self.K = K
        self.alphas = None if alphas is None else jnp.asarray(alphas)
        self.p = p

        # Save 1d matrics
        self._tensorizing_string = tensorizing_string
        self._S1d = S_1d
        # Dense 1D stiffness used by the single-axis apply in Au (avoids the
        # zeros_like + identity einsums of sum_apply_transformation).
        self._S1d_dense = (
            jnp.asarray(S_1d.todense())
            if hasattr(S_1d, "todense")
            else jnp.asarray(S_1d)
        )
        # FIXME: WRITE-ONLY, NEVER READ ANYWHERE IN THE REPO -- LEFTOVER OF THE
        # PRE-RESCALING MASS. NOTE THAT `self.M_inv` ABOVE IS *NOT* IN THIS
        # CATEGORY AND MUST BE KEPT: IT IS THE RECIPROCAL-WEIGHT TENSOR OF THE
        # RESCALED BASIS (M_rec IN THE PAPER), USED BY THE QUARTIC TERM. ONLY
        # `self.M` IS A BARE `zeros_like` SHAPE TEMPLATE.
        self._M1d = M_1d
        self.dim = dim

        # Setup empty precond_dict
        self.precond_dict = {None: lambda x, u, u2: x}

        return

    def __call__(self, u):
        """Evaluate energy functional.

        E(u) = 1/2 <u, S u + V * u + beta M_rec * u^3> (single component), or
        1/2 <u, S u + V * u> + beta/2 sum(M_rec * (u^2 K) * u^2)
        (multicomponent), Sec. 'Discretization by spectral elements' and Sec.
        'Discretization'.

        Args:
            u: State vector in full tensor format. For K≠None, shape is
               [N]*dim + [p] where p is the number of components.

        Returns:
            Energy value.
        """
        if self.K is None:
            Eu = 0.5 * jnp.vdot(
                self.S_fun(u) + self.V * u + self.beta * self.M_inv * (u**3),
                u,
            )
        else:
            u2 = u**2
            K_u2 = u2 @ self.K
            Eu = 0.5 * jnp.vdot(
                self.S_fun(u) + self.V[..., None] * u, u
            ) + 0.5 * self.beta * jnp.sum(self.M_inv[..., None] * K_u2 * u2)
        return Eu

    def Au(self, v, u, u2):
        """Compute A_u * v.

        A_u = S + diag(V) + 2 beta diag(M_rec * u2), with u2 replaced by
        u2 K (mixing along the component axis) in the multicomponent case.

        Args:
            v: Input vector in full format.
            u: Base point in full format.
            u2: u**2 in full format.

        Returns:
            Result in full format.
        """
        if self.K is None:
            # Elementwise part fuses into a single full-size buffer under jit.
            Au_v = self.V * v + 2 * self.beta * self.M_inv * u2 * v
            for i in range(self.dim):
                Au_v = Au_v + apply_single_axis(v, self._S1d_dense, i)
        else:
            K_u2 = u2 @ self.K
            Au_v = (
                self.V[..., None] * v + 2 * self.beta * self.M_inv[..., None] * K_u2 * v
            )
            # Component axis (last) is identity -> only loop over spatial axes.
            for i in range(self.dim):
                Au_v = Au_v + apply_single_axis(v, self._S1d_dense, i)
        return Au_v

    def Sinv_setup(self, factors_only=False):
        """Setup inverse stiffness operator using eigendecomposition (for fast
        diagonalization).

        S = (W ⊗ ... ⊗ W) D (W ⊗ ... ⊗ W)^T with W^T S_1d W = diag(lmd_1d),
        Sec. 'Preconditioning in the full tensor format'. Sets `U` = [W]*dim,
        `Uinv` = [W^T]*dim, `lmd_1d` and, unless factors_only, the full
        eigenvalue tensor `lmd` = D.

        Args:
            factors_only: If True, only compute factorization without full tensor.
        """
        eigendecomp = jnp.linalg.eigh(self._S1d)
        lmd_1d, U = eigendecomp.eigenvalues, eigendecomp.eigenvectors
        e = jnp.ones_like(lmd_1d)

        if not factors_only:
            lmd_tensor = jnp.zeros_like(self.M)
            for i in range(self.dim):
                lmd_tensor += jnp.einsum(
                    self._tensorizing_string,
                    *tuple([e] * i + [lmd_1d] + [e] * (self.dim - i - 1)),
                )
        self.U = [U] * self.dim
        self.Uinv = [U.T] * self.dim
        self.lmd_1d = lmd_1d
        if not factors_only:
            self.lmd = lmd_tensor

        return

    def S_fun(self, v):
        """Apply stiffness operator.

        Args:
            v: Input vector in full format.

        Returns:
            S * v in full format.
        """
        return sum_apply_transformation(v, self.S)

    def Sinv_fun(self, v):
        """Apply inverse stiffness operator using fast diagonalization.

        Requires `Sinv_setup(factors_only=False)`. In the multicomponent case
        it applies S^{-1} ⊗ I_p.

        Args:
            v: Input vector in full format.

        Returns:
            S^{-1} * v in full format.
        """
        if self.K is not None:
            # Component axis identity -> None (skipped by apply_transformations).
            Sinv_v = apply_transformations(v, self.Uinv + [None])
            Sinv_v = Sinv_v / self.lmd[..., None]
            Sinv_v = apply_transformations(Sinv_v, self.U + [None])
        else:
            Sinv_v = apply_transformations(v, self.Uinv)
            Sinv_v = Sinv_v / self.lmd
            Sinv_v = apply_transformations(Sinv_v, self.U)
        return Sinv_v

    def SplusV_setup(self, V0, factors_only=False):
        """Setup inverse (S + V) operator using eigendecomposition (for fast
        diagonalization).

        Preconditioner B = S + V_0 with V_0(x) = v_1(x_1) + ... + v_d(x_d),
        from the eigendecompositions W_i^T (S_1d + diag(v_i)) W_i =
        diag(lambda_i), Sec. 'Preconditioning in the full tensor format'. Sets
        `U_SV`, `Uinv_SV`, `lmd_1d_SV` and, unless factors_only, the full
        eigenvalue tensor `lmd_SV` and its reciprocal.

        Args:
            V0: Potential values for each dimension of the approximate potential
                V0 [see paper for details].
                List of dim arrays v_i evaluated at the 1D nodes.
            factors_only: If True, only compute factorization without full tensor.
        """
        Delta = self._S1d  # FIXME: AVOID DENSE CONVERSION

        if not factors_only:
            lmd_tensor = jnp.zeros_like(self.M)

        self.U_SV = []
        self.Uinv_SV = []
        self.lmd_1d_SV = []

        for i in range(self.dim):
            eigendecomp = jnp.linalg.eigh(Delta + jnp.diag(V0[i]))
            lmd_1d, U = eigendecomp.eigenvalues, eigendecomp.eigenvectors
            e = jnp.ones_like(lmd_1d)

            # Update lambda tensor
            if not factors_only:
                lmd_tensor += jnp.einsum(
                    self._tensorizing_string,
                    *tuple([e] * i + [lmd_1d] + [e] * (self.dim - i - 1)),
                )

            # Update decomposition
            self.U_SV.append(U)
            self.Uinv_SV.append(U.T)
            self.lmd_1d_SV.append(lmd_1d)

        if not factors_only:
            self.lmd_SV = lmd_tensor
            # Precompute reciprocal to avoid a division at every preconditioner call.
            self._inv_lmd_SV = 1.0 / lmd_tensor

        return

    def SplusV_inv_fun(self, v):
        """Apply inverse (S + V) operator using fast diagonalization.

        Applies B^{-1} = (S + V_0)^{-1}; requires
        `SplusV_setup(V0, factors_only=False)`. In the multicomponent case it
        applies (S + V_0)^{-1} ⊗ I_p, with no identity shift (unlike the TT
        version, see `setup_expsum_precond`).

        Args:
            v: Input vector in full format.

        Returns:
            (S + V)^{-1} * v in full format.
        """
        if self.K is not None:
            # Component axis identity -> None (skipped by apply_transformations).
            Sinv_v = apply_transformations(v, self.Uinv_SV + [None])
            Sinv_v = Sinv_v * self._inv_lmd_SV[..., None]
            Sinv_v = apply_transformations(Sinv_v, self.U_SV + [None])
        else:
            Sinv_v = apply_transformations(v, self.Uinv_SV)
            Sinv_v = Sinv_v * self._inv_lmd_SV
            Sinv_v = apply_transformations(Sinv_v, self.U_SV)
        return Sinv_v

    def bilinear_form(self, u, v, w):
        """Compute bilinear form a_u(v, w).

        Args:
            u: Base point in full format.
            v: First argument in full format.
            w: Second argument in full format.

        Returns:
            Bilinear form value.
        """
        return jnp.vdot(self.Au(v, u, u * u), w)

    def riemannian_gradient_H1(self, u):
        """Compute Riemannian gradient with H^1 metric.
        Uses fast diagonalization to apply S^{-1}.

        rgrad = S^{-1} A_u u - (u, S^{-1} A_u u) / (u, S^{-1} u) S^{-1} u
        (H^1 Sobolev gradient on the unit sphere). Requires
        `Sinv_setup(factors_only=False)`; single component only.

        Args:
            u: Current point in full format.

        Returns:
            Gradient in full format.
        """
        if self.K is not None:
            raise NotImplementedError(
                "riemannian_gradient_H1 is not implemented for K≠None "
                "in FullEnergyFunctional"
            )
        GxAu_u = self.Sinv_fun(self.Au(u, u, u * u))
        Gx_u = self.Sinv_fun(u)
        rgrad = GxAu_u - (jnp.vdot(u, GxAu_u) / jnp.vdot(u, Gx_u)) * Gx_u
        return rgrad

    def setup_preconditioners(self):
        """Setup dictionary of available preconditioners.

        Keys None (identity), "S" (S^{-1}) and "S+V" ((S + V_0)^{-1}); the
        latter two require `Sinv_setup` and `SplusV_setup` with
        factors_only=False.
        """
        precond_dict = {
            None: (lambda x, u, u2: x),
            "S": (lambda x, u, u2: self.Sinv_fun(x)),
            "S+V": (lambda x, u, u2: self.SplusV_inv_fun(x)),
        }
        self.precond_dict = precond_dict

    def riemannian_gradient_au_withaux(
        self, u, maxiter=200, tol=1e-10, precond=None, cg=cg_tree_math_point_dependent
    ):
        """Compute Riemannian gradient with a_u metric (with auxiliary info).
        Uses CG to solve the linear system.

        One preconditioned CG solve for A_u^{-1} u, then
        rgrad = u - (u, u) / (u, A_u^{-1} u) A_u^{-1} u (single component,
        a_u-orthogonal projection onto the tangent space of the sphere, Sec.
        'The a_u Riemannian gradient'), or
        rgrad = u - A_u^{-1}(u) Sigma, Sigma = diag(alphas^2 / diag<u, A_u^{-1} u>)
        (multicomponent, Sec. 'Riemannian gradient I: the operator H_φ').

        Args:
            u: Current point in full format.
            maxiter: Maximum CG iterations.
            tol: Convergence tolerance.
                Relative residual tolerance of CG in the Euclidean norm
                (default 1e-10, as in the paper runs).
            precond: Preconditioner choice (None, 'S', or 'S+V').
            cg: Conjugate gradient solver function.

        Returns:
            Tuple of (gradient in full format, CG info).
        """
        GxAu_u = u

        if precond not in self.precond_dict.keys():
            raise ValueError("precond must be None, 'S' or 'S+V'")

        Gx_u, info_cg = cg(
            u,
            u * u,
            self.Au,
            u,
            jnp.zeros_like(u),
            M=self.precond_dict[precond],
            maxiter=maxiter,
            tol=tol,
            return_info=True,
        )

        if self.K is not None:
            sigma = self.alphas**2 / jnp.einsum("...d,...d->d", u, Gx_u)
            # A_u and the S+V preconditioner act component-wise (block-diagonal
            # in the component index), so A_u^{-1}(u sigma) = sigma A_u^{-1} u;
            # one solve suffices (Sec. 'Implementation details').
            Gx_usigma = Gx_u * sigma
            rgrad = GxAu_u - Gx_usigma
            return rgrad, info_cg
        else:
            rgrad = GxAu_u - (jnp.vdot(u, GxAu_u) / jnp.vdot(u, Gx_u)) * Gx_u
            return rgrad, info_cg

    def riemannian_gradient_au(self, *args, **kwargs):
        """Same as riemannian_gradient_au_withaux, but returns only the gradient.

        Args:
            Same as riemannian_gradient_au_withaux.

        Returns:
            Gradient in full format.
        """
        return self.riemannian_gradient_au_withaux(*args, **kwargs)[0]


class EnergyHighOrder:
    """Energy functional using high-order FEM with FEniCSx.

    High-order energy, Remark 'Evaluating the energy with a higher-order
    quadrature rule', for full-format states: the same Q_k GLL spectral
    element space on [a, b]^dim, integrated with a quadrature of order
    `quadrature_order` instead of the GLL rule.

    Args:
        a, b: Domain bounds.
        beta: Nonlinear term coefficient.
        V: Potential specification. Supported forms are:
            - exact callable/UFL-like term,
            - tuple (space_type, function),
            - tuple (space_type, degree, function),
            - list mixing any of the above, summed into one potential.
            space_type is "DG" or "Lagrange"; tuple terms are interpolated
            into that space (degree defaults to k).
        msh_1d: 1D mesh points.
            The n = k E - 1 interior GLL nodes of the 1D mesh.
        dim: Spatial dimension (2 or 3).
        k: Polynomial order.
        quadrature_order: Quadrature order (default: 4*k).
        K: Coupling matrix (pxp array) for multicomponent systems (default: None).
        p: Number of components; required when K is not None.
        alphas: Component masses ||u_j||^2 (multicomponent case).
    """

    def __init__(
        self,
        a,
        b,
        beta,
        V,
        msh_1d,
        dim,
        k,
        quadrature_order=None,
        K=None,
        p=None,
        alphas=None,
    ):
        print("Initializing high-order energy functional with FEniCSx...")
        self.beta = beta
        self.V = V
        self.dim = dim

        N = len(msh_1d)
        ncells = int((N + 1) / k)
        if quadrature_order is None:
            quadrature_order = 4 * k

        # Define matrix for rescaling
        M_1d_lumped, _, _, _ = setup_1Dmatrices(N, a, b, k)
        mdiag = jnp.diag(M_1d_lumped.todense())
        tensorizing_string = ",".join(string.ascii_lowercase[:dim])
        self.M_sqrt_lumped = jnp.einsum(
            tensorizing_string, *tuple([jnp.sqrt(mdiag)] * dim)
        )

        # Define space
        if dim == 2:
            self._cell_name = "quadrilateral"
            self._lagrange_variant = basix.LagrangeVariant.gll_warped
            element = basix.ufl.element(
                "Lagrange", self._cell_name, k, self._lagrange_variant
            )
            domain = mesh.create_rectangle(
                comm=MPI.COMM_WORLD,
                points=[jnp.asarray([a, a]), jnp.asarray([b, b])],
                n=[ncells, ncells],
                cell_type=mesh.CellType.quadrilateral,
            )
            boundary = (
                lambda x: np.isclose(x[0], a)
                | np.isclose(x[0], b)
                | np.isclose(x[1], a)
                | np.isclose(x[1], b)
            )

        elif dim == 3:
            self._cell_name = "hexahedron"
            self._lagrange_variant = basix.LagrangeVariant.gll_warped
            element = basix.ufl.element(
                "Lagrange", self._cell_name, k, self._lagrange_variant
            )
            domain = mesh.create_box(
                comm=MPI.COMM_WORLD,
                points=[jnp.asarray([a, a, a]), jnp.asarray([b, b, b])],
                n=[ncells, ncells, ncells],
                cell_type=mesh.CellType.hexahedron,
            )
            boundary = (
                lambda x: np.isclose(x[0], a)
                | np.isclose(x[0], b)
                | np.isclose(x[1], a)
                | np.isclose(x[1], b)
                | np.isclose(x[2], a)
                | np.isclose(x[2], b)
            )
        else:
            raise NotImplementedError("Implemented only dim==2 and dim==3")

        # Store domain and function space
        self.domain = domain
        self.H = fem.functionspace(domain, element)
        self._state_degree = k

        # Precompute nodes and boundary conditions
        print("Precomputing nodes and boundary conditions...")
        nodes = self.H.tabulate_dof_coordinates()[:, :dim]
        bc_dofs = fem.locate_dofs_geometrical(self.H, boundary)
        self.freedofs = np.setdiff1d(np.arange(len(nodes)), bc_dofs)

        # Setup quadrature
        metadata = {"quadrature_degree": quadrature_order}
        self.dx = ufl.Measure("dx", domain=domain, metadata=metadata)

        # Precompute reordering of nodes to match TT order
        # Use binary search (searchsorted) to map float coordinates to 1D integer
        # indices: O(N_freedofs * dim * log(N_1d)) vs the snapping loop which is
        # O(N_freedofs * dim * N_1d).
        print("Precomputing node reordering to match TT order...")
        msh_sorted = np.sort(msh_1d)
        free_nodes = nodes[self.freedofs, :]  # (N_freedofs, dim)
        # Insertion points via binary search
        int_idx = np.searchsorted(msh_sorted, free_nodes)  # (N_freedofs, dim)
        np.clip(int_idx, 0, len(msh_sorted) - 1, out=int_idx)
        # Nearest-neighbour correction: searchsorted gives the right or left
        # boundary of the bracket; pick whichever grid point is closer.
        left = np.maximum(int_idx - 1, 0)
        use_left = np.abs(free_nodes - msh_sorted[left]) < np.abs(
            free_nodes - msh_sorted[int_idx]
        )
        int_idx[use_left] = left[use_left]
        perm = np.lexsort([int_idx[:, i] for i in reversed(range(dim))])
        self.perm_inv = np.empty_like(perm)
        self.perm_inv[perm] = np.arange(len(perm))

        # Precompute spatial coordinates, nodes and potential
        print("Precomputing spatial coordinates, nodes and potential...")
        self.nodes = self.H.tabulate_dof_coordinates()[:, : self.dim]
        self.x = ufl.SpatialCoordinate(domain)
        self._V_interpolated_terms = []
        self.V_fenicsx = self._build_potential_expression(self.V)

        # Precompile FEniCSx forms (avoids FFCx recompilation at each __call__)
        print("Compiling FEniCSx forms for high-order energy functional...")
        self._u_func = fem.Function(self.H)
        _u = self._u_func
        self._norm_form = fem.form(_u**2 * self.dx)
        _grad_u = ufl.grad(_u)
        self._energy_form = fem.form(
            0.5 * ufl.dot(_grad_u, _grad_u) * self.dx
            + self.beta / 2 * (_u**4) * self.dx
            + 0.5 * self.V_fenicsx * (_u**2) * self.dx
        )
        print("FEniCSx forms compiled.")

        self.K = K
        self.p = p
        self.alphas = jnp.asarray(alphas) if alphas is not None else None
        if K is not None:
            # Multicomponent: p scalar functions on the same scalar H, coupled via UFL
            self._u_funcs = [fem.Function(self.H) for _ in range(p)]
            _us = self._u_funcs
            self._norm_forms = [fem.form(_us[i] ** 2 * self.dx) for i in range(p)]
            E_terms = []
            for i in range(p):
                E_terms.append(
                    0.5 * ufl.dot(ufl.grad(_us[i]), ufl.grad(_us[i])) * self.dx
                )
                E_terms.append(0.5 * self.V_fenicsx * _us[i] ** 2 * self.dx)
                for j in range(p):
                    E_terms.append(
                        float(K[i, j])
                        * self.beta
                        / 2
                        * _us[i] ** 2
                        * _us[j] ** 2
                        * self.dx
                    )
            self._energy_form = fem.form(functools.reduce(lambda a, b: a + b, E_terms))
            print("FEniCSx multicomponent forms compiled.")

    def _build_interpolated_potential_term(self, space_type, potential_fn, degree=None):
        """Interpolate `potential_fn` into a DG or Lagrange space of `degree`."""
        if space_type == "DG":
            degree = self._state_degree if degree is None else int(degree)
            print(f"Using DG({degree}) potential space for V_fenicsx...")
            space = fem.functionspace(self.domain, ("DG", degree))
        elif space_type == "Lagrange":
            degree = self._state_degree if degree is None else int(degree)
            if degree == self._state_degree:
                print("Using Lagrange potential space for V_fenicsx...")
                space = self.H
            else:
                print(f"Using Lagrange({degree}) potential space for V_fenicsx...")
                element = basix.ufl.element(
                    "Lagrange",
                    self._cell_name,
                    degree,
                    self._lagrange_variant,
                )
                space = fem.functionspace(self.domain, element)
        else:
            raise ValueError(f"Unsupported interpolated potential space: {space_type}")

        coefficient = fem.Function(space)
        coefficient.interpolate(potential_fn)
        self._V_interpolated_terms.append(coefficient)
        return coefficient

    def _build_single_potential_term(self, term):
        """Turn one potential term (see the class docstring) into a UFL term."""
        if isinstance(term, tuple):
            if len(term) == 2 and isinstance(term[0], str):
                space_type, potential_fn = term
                degree = None
            elif (
                len(term) == 3
                and isinstance(term[0], str)
                and isinstance(term[1], Integral)
            ):
                space_type, degree, potential_fn = term
            else:
                raise ValueError(
                    "Potential tuple terms must be (space_type, function) or "
                    "(space_type, degree, function)"
                )

            if not callable(potential_fn):
                raise TypeError("Interpolated potential terms must provide a callable")

            return self._build_interpolated_potential_term(
                space_type, potential_fn, degree=degree
            )

        if callable(term):
            return term(self.x)

        return term

    def _build_potential_expression(self, potential):
        """Build the UFL potential, summing the terms if a list is given."""
        if isinstance(potential, list):
            if not potential:
                raise ValueError("Potential term list cannot be empty")

            terms = [self._build_single_potential_term(term) for term in potential]
            total = terms[0]
            for term in terms[1:]:
                total = total + term
            return total

        return self._build_single_potential_term(potential)

    def __call__(self, u_full):
        """Compute energy using high-order quadrature.

        The coefficients in the rescaled basis are mapped to nodal values
        (division by the square roots of the GLL weights), the state is
        renormalized with the high-order quadrature to unit mass (mass
        alphas[j] per component if K is not None), and the energy
        (multicomponent: with the coupling K) is assembled.

        Args:
            u_full: State as a full (dense) vector or array. For K≠None,
                shape must be broadcastable to [N]*dim + [p].

        Returns:
            Energy value.
        """
        print("Evaluating high-order energy functional...")
        start_time = time.time()
        if self.K is None:
            u = self._u_func
            u.x.array[:] = 0.0
            u.x.array[self.freedofs] = np.asarray(
                u_full / self.M_sqrt_lumped
            ).flatten()[self.perm_inv]
            u.x.array[:] /= np.sqrt(fem.assemble_scalar(self._norm_form))
            E_highorder = fem.assemble_scalar(self._energy_form)
            E_highorder = self.domain.comm.allreduce(E_highorder, op=MPI.SUM)
        else:
            # Multicomponent: u_full has shape [N]*dim + [p] (component = last axis)
            u_shaped = np.asarray(u_full).reshape(
                list(self.M_sqrt_lumped.shape) + [self.p]
            )
            u_divided = u_shaped / self.M_sqrt_lumped[..., np.newaxis]
            u_flat = u_divided.reshape(-1, self.p)
            for i in range(self.p):
                self._u_funcs[i].x.array[:] = 0.0
                self._u_funcs[i].x.array[self.freedofs] = u_flat[:, i][self.perm_inv]
            for i in range(self.p):
                norm_i_sq = self.domain.comm.allreduce(
                    fem.assemble_scalar(self._norm_forms[i]), op=MPI.SUM
                )
                self._u_funcs[i].x.array[:] *= np.sqrt(
                    float(self.alphas[i]) / norm_i_sq
                )
            E_highorder = self.domain.comm.allreduce(
                fem.assemble_scalar(self._energy_form), op=MPI.SUM
            )
        print("...done in {:.2f} seconds.".format(time.time() - start_time))
        return E_highorder


def _extract_spatial_component(u_ttmatrix, i):
    """Extract component i from a multicomponent TT column-vector (TTMatrix with
    n_col=1) as a spatial-only TT column-vector.

    The component dimension is the last core (shape (r, p, 1, 1)).
    This folds the component selection into the preceding spatial core so the
    returned TTMatrix has only the spatial dimensions.

    Args:
        u_ttmatrix: TTMatrix column-vector with shape (n_q, ..., n_q, p).
        i: Component index to extract.

    Returns:
        TTMatrix column-vector with shape (n_q, ..., n_q) (spatial only).
    """
    cores = list(u_ttmatrix.tt_cores)
    comp_core = cores[-1]  # (r, p, 1, 1)
    sel = comp_core[:, i, 0, 0]  # (r,)
    prev_core = cores[-2]  # (r_prev, n_q, 1, r)  — n_col=1 for column vec
    # Contract: new_last[a, n, 1, 1] = sum_b prev_core[a, n, 0, b] * sel[b]
    new_last = jnp.einsum("anb,b->an", prev_core[:, :, 0, :], sel)[
        :, :, None, None
    ]  # (r_prev, n_q, 1, 1)
    return ttax.TTMatrix(cores[:-2] + [new_last])


class EnergyHighOrder_TT:
    """Energy functional using high-order FEM in TT format.

    High-order energy, Remark 'Evaluating the energy with a higher-order
    quadrature rule': the TT state in the rescaled basis is mapped to nodal
    values (Hadamard product with the reciprocal square roots of the GLL
    weights), then to values at the quadrature nodes by the rank-1 extension
    TT matrix `T`; the quadrature weights `W` multiply the nonlinear and
    potential terms, and the gradient term uses the exact stiffness matrix
    sum_k M ⊗ ... ⊗ S_1d ⊗ ... ⊗ M (exact 1D mass M, Lagrange basis).
    Operators are extended by an identity (ones) core on the component mode
    in the multicomponent case.

    Args:
        a, b: Domain bounds.
        beta: Nonlinear term coefficient.
        V_functions_or_vector: List of potential functions for each dimension or TT object.
            A list of rank-1 terms, each a list of dim 1D callables whose
            product gives the term (the terms are summed), evaluated at the
            quadrature nodes; or a TT object of potential values at the
            quadrature nodes. Shared by all components.
        N: Number of 1D mesh points.
            Number n of interior 1D GLL nodes.
        dim: Spatial dimension.
        k: Polynomial order.
        quadrature_order: Quadrature order (default: 4*k).
        K: Coupling matrix (pxp array) for multicomponent systems (default: None).
        p: Number of components; required when K is not None.
        alphas: Component masses ||u_j||^2 (multicomponent case).
    """

    def __init__(
        self,
        a,
        b,
        beta,
        V_functions_or_vector,
        N,
        dim,
        k,
        quadrature_order=None,
        K=None,
        p=None,
        alphas=None,
    ):
        self.beta = beta
        self.V_functions_or_vector = V_functions_or_vector
        self.dim = dim

        ncells = int((N + 1) / k)
        if quadrature_order is None:
            quadrature_order = 4 * k

        # Define 1d matrices
        M_1d_lumped, S_1d, msh_1d, M_1d_exact = setup_1Dmatrices(N, a, b, k)
        M_sqrt_inv_lumped = ttax.ops.tensor_to_vector(
            ttax.TT(
                [jnp.expand_dims(1 / jnp.sqrt(jnp.diag(M_1d_lumped.todense())), [0, 2])]
                * dim
            )
        )
        self.M_sqrt_inv_lumped = M_sqrt_inv_lumped
        M_1d_exact = M_1d_exact.todense()  # FIXME: AVOID DENSE CONVERSION
        S_1d = S_1d.todense()  # FIXME: AVOID DENSE CONVERSION

        # Define 1d mesh and function space
        element = basix.ufl.element(
            "Lagrange", "interval", k, basix.LagrangeVariant.gll_warped
        )
        domain = mesh.create_interval(
            comm=MPI.COMM_WORLD,
            points=[a, b],
            nx=ncells,
        )
        boundary = lambda x: np.isclose(x[0], a) | np.isclose(x[0], b)
        H = fem.functionspace(domain, element)

        # Precompute nodes and boundary conditions
        nodes = H.tabulate_dof_coordinates()[:, :dim]
        bc_dofs = fem.locate_dofs_geometrical(H, boundary)
        freedofs = np.setdiff1d(np.arange(len(nodes)), bc_dofs)

        # Get quadrature rule for the reference interval
        quad_points, quad_weights = basix.make_quadrature(
            basix.CellType.interval, quadrature_order
        )
        # Define quadrature points, weights, and values of basis functions at quadrature points
        trial_fn = ufl.TrialFunction(H)
        basis_expr = fem.Expression(trial_fn, quad_points)
        spatial_coords = ufl.SpatialCoordinate(domain)
        coords_expr = fem.Expression(spatial_coords, quad_points)
        num_cells = domain.topology.index_map(domain.topology.dim).size_local
        cell_quad_coords = coords_expr.eval(
            domain, np.arange(num_cells, dtype=np.int32)
        ).reshape(num_cells, len(quad_points), -1)
        cell_detJ = fem.Expression(ufl.JacobianDeterminant(domain), quad_points)
        cell_detJ_vals = cell_detJ.eval(
            domain, np.arange(num_cells, dtype=np.int32)
        ).reshape(num_cells, len(quad_points), -1)
        cell_basis_vals = basis_expr.eval(
            domain, np.arange(num_cells, dtype=np.int32)
        ).reshape(num_cells, len(quad_points), -1)
        points_all = np.zeros((cell_quad_coords.size,))
        weights_all = np.zeros((cell_quad_coords.size,))
        basis_all = np.zeros((cell_quad_coords.size, len(nodes)))
        for i in range(num_cells):
            geom_dofs = H.dofmap.cell_dofs(i)
            quad_idx = range(i * len(quad_points), (i + 1) * len(quad_points))
            points_all[quad_idx] = cell_quad_coords[i, :, 0]
            weights_all[quad_idx] = cell_detJ_vals[i, :, 0] * quad_weights.reshape(-1)
            basis_all[np.ix_(quad_idx, geom_dofs)] = cell_basis_vals[i, :, :]

        # Compute permutation for node reordering
        arr = nodes[freedofs, :]
        for c in msh_1d:
            arr[np.abs(arr - c) < 1e-13] = c
        perm = np.lexsort([arr[:, i] for i in reversed(range(1))])
        perm_inv = np.empty_like(perm)
        perm_inv[perm] = np.arange(len(perm))
        basis_all = basis_all[:, freedofs][:, perm]

        # Check points are sorted
        np.testing.assert_allclose(
            points_all, np.sort(points_all), rtol=0.0, atol=1e-16
        )

        # Evaluate potential at quadrature points
        if isinstance(self.V_functions_or_vector, list):
            V_functions = self.V_functions_or_vector
            V_addends = [
                ttax.ops.tensor_to_vector(
                    ttax.TT([f(jnp.expand_dims(points_all, [0, 2])) for f in funs_1d])
                )
                for funs_1d in V_functions
            ]
            cores = [
                jnp.stack([V_core.tt_cores[i] for V_core in V_addends], axis=0)
                for i in range(len(V_addends[0].tt_cores))
            ]
            V = ttax.TTMatrix(cores)
        elif ttax.is_tt_object(self.V_functions_or_vector):
            V = self.V_functions_or_vector
        else:
            raise ValueError(
                "V_functions_or_vector must be a list of functions or TT object"
            )

        self.V = V

        # Define extension operator
        T_cores = [jnp.expand_dims(basis_all, [0, 3])] * dim
        T = ttax.TTMatrix(T_cores)

        # Define weights as operator
        W_cores = [jnp.expand_dims(weights_all, [0, 2])] * dim
        W = ttax.ops.tensor_to_vector(ttax.TT(W_cores))

        # Define exact stiffness operator
        O = jnp.zeros_like(M_1d_exact)
        B1 = jnp.expand_dims(jnp.stack([S_1d, M_1d_exact], axis=2), 0)
        B2_up = jnp.stack([M_1d_exact, O], axis=2)
        B2_low = jnp.stack([S_1d, M_1d_exact], axis=2)
        B2 = jnp.stack([B2_up, B2_low], axis=0)
        B3 = jnp.expand_dims(jnp.stack([M_1d_exact, S_1d]), 3)
        S = ttax.TTMatrix([B1] + [B2] * (dim - 2) + [B3])

        self.K = K
        self.p = p
        self.alphas = alphas
        if K is not None:
            # Multicomponent: extend all TT operators with an identity/ones
            # component core so that they act independently on each component.
            ones_p = jnp.ones(p)
            I_p = jnp.eye(p)
            # Keep original (spatial-only) W for component-wise inner products
            self.W_spatial = W
            # Extend M_sqrt_inv_lumped with a ones component core
            self.M_sqrt_inv_lumped = ttax.TTMatrix(
                list(self.M_sqrt_inv_lumped.tt_cores) + [jnp.ones((1, p, 1, 1))]
            )
            # Extend T with an identity component core
            T = ttax.TTMatrix(T_cores + [jnp.expand_dims(I_p, [0, 3])])
            # Extend W with a ones component core
            W = ttax.TTMatrix(list(W.tt_cores) + [jnp.ones((1, p, 1, 1))])
            # Extend S with an identity component core
            S = ttax.TTMatrix(list(S.tt_cores) + [jnp.expand_dims(I_p, [0, 3])])
        else:
            self.W_spatial = None

        self.T = T
        self.W = W
        self.S = S
        self.partial_flat_inner_last_jit = jit(partial_flat_inner_last)

        return

    def __call__(self, u_TT: ttax.TT):
        """Compute energy using high-order quadrature in TT format.

        The state is renormalized with the high-order quadrature (unit mass,
        or mass alphas[j] per component) before the energy is evaluated.

        Args:
            u_TT: State in TT format. For K≠None, must include a component
                dimension as the last TT core.

        Returns:
            Energy value.
        """
        u_TT = ttax.multiply(
            self.M_sqrt_inv_lumped, u_TT
        )  # Renormalize with lumped mass
        u_quad = self.T @ u_TT

        # Renormalize with exact mass (quadrature weights)
        if self.K is None:
            # Single component: normalize to unit mass
            normalization = 1 / jnp.sqrt(
                ttax.flat_inner(u_quad, ttax.multiply(u_quad, self.W))
            )
            u_quad *= normalization  # Need also to renormalize with exact mass!
        else:
            # Multicomponent: normalize to alphas mass per component
            normalization = jnp.sqrt(self.alphas) / jnp.sqrt(
                self.partial_flat_inner_last_jit(u_quad, ttax.multiply(u_quad, self.W))
            )
            u_quad = rescale_last_component(u_quad, normalization)
            uTT_normalized = rescale_last_component(u_TT, normalization)

        u2_quad = ttax.multiply(u_quad, u_quad)
        if self.K is None:
            E = (
                self.beta / 2 * ttax.flat_inner(u2_quad, ttax.multiply(u2_quad, self.W))
                + 0.5 * jnp.sum(ttax.flat_inner(self.V, ttax.multiply(u2_quad, self.W)))
                + 0.5
                * ttax.flat_inner(u_TT * normalization, self.S @ u_TT * normalization)
            )
        else:
            # Extract per-component spatial TTs for the nonlinear and potential terms
            u2_spatials = [
                _extract_spatial_component(u2_quad, i) for i in range(self.p)
            ]
            # Gram matrix G[i,j] = integral u_i^2 * u_j^2 * w
            G = jnp.stack(
                [
                    jnp.stack(
                        [
                            ttax.flat_inner(
                                u2_spatials[i],
                                ttax.multiply(u2_spatials[j], self.W_spatial),
                            )
                            for j in range(self.p)
                        ]
                    )
                    for i in range(self.p)
                ]
            )
            E_nl = self.beta / 2 * jnp.sum(self.K * G)
            E_pot = 0.5 * sum(
                jnp.sum(
                    ttax.flat_inner(
                        self.V, ttax.multiply(u2_spatials[i], self.W_spatial)
                    )
                )
                for i in range(self.p)
            )
            E = (
                E_nl
                + E_pot
                + 0.5 * ttax.flat_inner(uTT_normalized, self.S @ uTT_normalized)
            )
        return E




def _thomas_fermi_fun_from_mu(V: Callable, beta: float, mu: float):
    """Return Thomas-Fermi function for a given chemical potential mu.

    The Thomas--Fermi profile solves the Gross--Pitaevskii equation without the
    kinetic term: u(x)^2 = (mu - V(x))_+ / (2 beta). The positive part is
    floored at 4 * machine epsilon, so u(x) >= 2 sqrt(eps) > 0 everywhere.

    Args:
        V: Potential function.
        beta: Nonlinearity coefficient.
        mu: Chemical potential.

    Returns:
        Callable x -> u(x).
    """
    return lambda x: jnp.sqrt(
        jnp.maximum((mu - V(x)) / (2 * beta), 4 * jnp.finfo(jnp.float64).eps)
    )


def _bisect_mu_for_norm(eval_norm_fn, dim, beta, tol=1e-8, max_iter=50):
    """Find mu such that eval_norm_fn(mu) == 1 via bisection.

    Assumes eval_norm_fn is increasing in mu. The bracket is built from a
    closed-form, dimension-dependent starting value (doubled until the norm
    exceeds 1, then halved until it is below 1), and is then bisected. If the
    norm at the final midpoint is below 0.5, the upper end of the bracket is
    returned instead.

    Args:
        eval_norm_fn: Callable (mu: float) -> float returning the discrete norm.
        dim: Spatial dimension (1, 2, or 3), used for the analytical initial bracket.
        beta: Nonlinearity coefficient, used for the analytical initial bracket.
        tol: Relative tolerance on mu for convergence.
        max_iter: Maximum number of bisection iterations.

    Returns:
        Converged mu value.
    """
    if dim == 1:
        mu_analytical = 0.5 * (3 * beta / 2) ** (2 / 3)
    elif dim == 2:
        mu_analytical = 0.5 * (4 * beta / np.pi) ** (1 / 2)
    else:
        mu_analytical = 0.5 * (15 * beta / (4 * np.pi)) ** (2 / 5)

    # Ensure mu_high gives norm > 1 (analytical mu is for continuous L²;
    # on a discrete grid with large N the norm is >> 1).
    mu_high = mu_analytical
    while eval_norm_fn(mu_high) <= 1.0:
        mu_high *= 2.0

    # Find mu_low such that norm < 1
    mu_low = mu_high / 2.0
    while eval_norm_fn(mu_low) >= 1.0:
        mu_low /= 2.0

    # Bisection
    for _ in range(max_iter):
        if (mu_high - mu_low) / mu_high < tol:
            break
        mu_mid = 0.5 * (mu_low + mu_high)
        evalued = eval_norm_fn(mu_mid)
        evalued_min = eval_norm_fn(mu_low)
        evalued_max = eval_norm_fn(mu_high)
        print(
            "norm at low, mid, high: {:.4f}, {:.4f}, {:.4f}".format(
                evalued_min, evalued, evalued_max
            )
        )
        if evalued > 1.0:
            mu_high = mu_mid
        else:
            mu_low = mu_mid

    mu_final = 0.5 * (mu_low + mu_high)
    print(
        "Bisection converged to mu = {:.6e} with norm = {:.6e}".format(
            mu_final, eval_norm_fn(mu_final)
        )
    )
    if eval_norm_fn(mu_final) < 0.5:
        return mu_high
    else:
        return mu_final


def thomas_fermi_tt_init(
    V: Callable,
    beta: float,
    meshes_1d: list,
    target_ranks,
    norm_fn: Callable,
    tol: float = 1e-8,
    max_iter: int = 50,
    **tt_kwargs,
):
    """Compute Thomas-Fermi TT initialization with mu chosen so that the
    discretized TT vector already satisfies norm_fn(u) == 1.

    Uses bisection on the chemical potential mu.

    Thomas--Fermi initial guess u^2 = (mu - V)_+ / (2 beta) (see
    `_thomas_fermi_fun_from_mu`), approximated in TT format by `tt_from_fun`,
    with mu found by bisection so that the discrete solution has unit mass as
    measured by norm_fn. The TT approximations evaluated during the
    bisection use a fixed rng (seed 1); the final one uses the caller's rng,
    if given.

    Args:
        V: Potential function accepting an array of shape (n_pts, dim).
        beta: Nonlinearity coefficient.
        meshes_1d: List of 1D meshes, one per dimension.
        target_ranks: TT target rank (passed to tt_from_fun).
        norm_fn: Callable (ttax_vector) -> float returning the discrete norm of u.
            Bisection targets norm_fn(u) == 1.
        tol: Relative tolerance on mu for bisection convergence.
        max_iter: Maximum number of bisection iterations.
        **tt_kwargs: Extra keyword arguments forwarded to tt_from_fun.

    Returns:
        TT cores (list of numpy arrays) at the converged mu.
    """
    dim = len(meshes_1d)
    # Pop user rng to avoid duplicate-kwarg error inside eval_norm, which
    # always uses a fixed rng for reproducibility across bisection steps.
    user_rng = tt_kwargs.pop("rng", None)

    def eval_norm(mu):
        f = _thomas_fermi_fun_from_mu(V, beta, mu)
        cores = tt_from_fun(
            f,
            meshes_1d,
            target_ranks=target_ranks,
            rng=np.random.default_rng(1),
            **tt_kwargs,
        )
        u = ttax.ops.tensor_to_vector(ttax.TT(cores))
        return float(norm_fn(u))

    mu_final = _bisect_mu_for_norm(eval_norm, dim, beta, tol, max_iter)
    f_final = _thomas_fermi_fun_from_mu(V, beta, mu_final)
    final_kwargs = dict(tt_kwargs)
    if user_rng is not None:
        final_kwargs["rng"] = user_rng
    return tt_from_fun(f_final, meshes_1d, target_ranks=target_ranks, **final_kwargs)


def thomas_fermi_full_init(
    V: Callable,
    beta: float,
    meshes_1d: list,
    norm_fn: Callable,
    tol: float = 1e-8,
    max_iter: int = 50,
):
    """Compute Thomas-Fermi full-tensor initialization with mu found by bisection.

    Evaluates the Thomas-Fermi function on the full tensor-product grid and
    bisects on mu until norm_fn(f_arr) == 1.

    Thomas--Fermi initial guess u^2 = (mu - V)_+ / (2 beta) (see
    `_thomas_fermi_fun_from_mu`), with mu chosen so that the discrete
    solution has unit mass as measured by norm_fn.

    Args:
        V: Potential function accepting an array of shape (n_pts, dim).
        beta: Nonlinearity coefficient.
        meshes_1d: List of 1D meshes, one per dimension.
        norm_fn: Callable (f_arr: jnp.ndarray) -> float returning the discrete
            norm of f. Bisection targets norm_fn(f) == 1.
        tol: Relative tolerance on mu for bisection convergence.
        max_iter: Maximum number of bisection iterations.

    Returns:
        Full tensor array of shape [N_0, N_1, ...] with norm_fn(f_arr) ≈ 1.
    """
    dim = len(meshes_1d)
    grids = np.meshgrid(*meshes_1d, indexing="ij")
    xy = np.stack([g.reshape(-1) for g in grids], axis=-1)
    shape = [len(m) for m in meshes_1d]

    def eval_norm(mu):
        f = _thomas_fermi_fun_from_mu(V, beta, mu)
        f_arr = f(jnp.array(xy)).reshape(shape)
        return float(norm_fn(f_arr))

    mu_final = _bisect_mu_for_norm(eval_norm, dim, beta, tol, max_iter)
    f_final = _thomas_fermi_fun_from_mu(V, beta, mu_final)
    return f_final(jnp.array(xy)).reshape(shape)


def setup_expsum_precond(rank_expsum, U, Uinv, lmd_1d, batch_expsum=None, p=None):
    """Setup preconditioner using exponential sum approximation.

    TT preconditioner via exponential sums, Sec. 'Preconditioning in the TT
    format': B^{-1} = W diag(1 / Lambda) W^T is approximated by
    W diag(Lambda_k) W^T, where Lambda(i_1, ..., i_d) = lambda_1(i_1) + ... +
    lambda_d(i_d) and
    Lambda_k = sum_i (omega_i / lmin) exp(-alpha_i / lmin * lambda_1) ⊗ ... ⊗
    exp(-alpha_i / lmin * lambda_d), with (omega_i, alpha_i) the tabulated
    exponential-sum coefficients for 1/x on [1, lmax / lmin] (read from
    src/external/EXP_SUM/1_x_rel/Rel1_x_n). The returned function applies
    Proj_u W diag(Lambda_k) W^T to a tangent vector.

    Args:
        rank_expsum: Number of terms in the exponential sum approximation.
        U: Transformation matrices (TT format).
            Rank-1 TT matrix W = W_1 ⊗ ... ⊗ W_d of eigenvectors.
        Uinv: Inverse transformation matrices (TT format).
            Rank-1 TT matrix W^T.
        lmd_1d: List of eigenvalue vectors for each dimension.
        batch_expsum: Batching strategy (default: None).
            None: Lambda_k is formed as a single rank-k TT tensor and applied
            at once. 1: the k terms are applied one at a time, each projected
            onto the tangent space and accumulated in delta form (the split
            form of Sec. 'Preconditioning in the TT format'). Other values are
            not implemented.
        p: Number of components (multicomponent case), or None. In the
            multicomponent case U and Uinv are extended in place by an
            identity core on the component mode.

    Returns:
        Preconditioner function.
        Prec_PAP_Mr(deltas, x, x2) -> deltas, as expected by
        `tt_energyfactory` (x2 is unused).
    """
    ### EXPONENTIAL SUM APPROXIMATION ###
    lmin = sum(map(jnp.min, lmd_1d))
    lmax = sum(map(jnp.max, lmd_1d))
    # Multicomponent TT preconditioner (Remark 'Preconditioner'): the component
    # mode is treated as one more Kronecker summand whose 1D eigenvalues are all
    # equal to 1 (the shift of lmin/lmax here and the extra core
    # exp(-alpha_i / lmin * ones(p)) below), while W, W^T get identity factors.
    # The eigenvalue tensor is therefore lambda_1 + ... + lambda_d + 1 and the
    # preconditioner approximates (S + V_0 + I)^{-1} ⊗ I_p (setup_S_precond:
    # (S + I)^{-1} ⊗ I_p), not (S + V_0)^{-1} ⊗ I_p as in the full format
    # (FullEnergyFunctional.SplusV_inv_fun).
    if p is not None:
        lmin, lmax = (
            lmin + 1,
            lmax + 1,
        )  # Account for identity component in multicomponent case
    # print(f"Exponential sum approximation: lmin={lmin:.4e}, lmax={lmax:.4e}")
    Mp = lmax / lmin

    # Get the directory where this file is located, then construct path to EXP_SUM
    _energy_dir = Path(__file__).parent
    _expsum_base = _energy_dir.parent / "external" / "EXP_SUM" / "1_x_rel" / "Rel1_x_n"

    try:
        omega, alpha, xi = extract_coefficients(
            str(_expsum_base / to_filename(Mp, rank_expsum)[0])
        )
    except:
        omega, alpha, xi = extract_coefficients(
            str(_expsum_base / to_filename(Mp, rank_expsum)[1])
        )
    # Define exponential sum approximation
    addend_list = []
    for i in range(rank_expsum):
        cores = [
            jnp.expand_dims(jnp.exp(-alpha[i] / lmin * diag), [0, 2]) for diag in lmd_1d
        ]
        if p is not None:
            cores.append(
                jnp.expand_dims(jnp.exp(-alpha[i] / lmin * jnp.ones(p)), [0, 2])
            )

        addend = (omega[i] / lmin) * ttax.ops.tensor_to_vector(ttax.TT(deepcopy(cores)))
        addend_list.append(addend)

    if p is not None:
        U_cores = U.tt_cores
        U_cores.append(jnp.expand_dims(jnp.eye(p), [0, 3]))
        U = ttax.TTMatrix(U_cores)

        Uinv_cores = Uinv.tt_cores
        Uinv_cores.append(jnp.expand_dims(jnp.eye(p), [0, 3]))
        Uinv = ttax.TTMatrix(Uinv_cores)

    ### DEFINE THE PRECONDITIONER ###
    if batch_expsum is None:
        Lambda_inv = addend_list[0]
        for l in addend_list[1:]:
            Lambda_inv += l
        S_inv = lambda v: U @ (ttax.multiply(Lambda_inv, Uinv @ v))
        Prec_PAP_Mr = lambda deltas, x, x2: ttax.project(
            S_inv(ttax.riemannian.deltas_to_tangent(deltas, x)), x
        )
    elif batch_expsum == 1:

        def Prec_PAP_Mr(deltas, x, x2):
            tang = ttax.riemannian.deltas_to_tangent(deltas, x)
            tang = Uinv @ tang
            res = zeroslike_deltas(deltas)
            for l in addend_list:
                res = lincomb_deltas(
                    1, res, 1, ttax.project(U @ ttax.multiply(l, tang), x)
                )
            return res

    else:
        raise NotImplementedError("Batching not implemented yet for batch_expsum>1")

    return Prec_PAP_Mr


def setup_SplusV_precond(
    E_full: FullEnergyFunctional,
    M_1d,  # FIXME: UNUSED IN THIS BODY -- LEFTOVER OF THE PRE-RESCALING MASS. REMOVING IT MEANS UPDATING ALL CALL SITES.
    V0,
    rank_expsum,
    batch_expsum=None,
    factors_only=False,
    p=None,
):
    """Setup (S + V)^{-1} preconditioner using exponential sum.

    TT preconditioner via exponential sums, Sec. 'Preconditioning in the TT
    format', for B = S + V_0: the 1D eigendecompositions are computed by
    `E_full.SplusV_setup` and passed to `setup_expsum_precond`. With p not
    None it approximates (S + V_0 + I)^{-1} ⊗ I_p (see the comment in
    `setup_expsum_precond`).

    Args:
        E_full: Full energy functional object.
            Built with the 1D stiffness matrix in the rescaled basis.
        M_1d: 1D mass matrix.
        V0: Potential values for each dimension.
            List of dim arrays v_i at the 1D nodes, V_0 = v_1 + ... + v_d.
        rank_expsum: Number of terms in the exponential sum approximation.
        batch_expsum: Batching strategy (default: None).
        factors_only: If True, only compute factorization without full tensor.
        p: Number of components (multicomponent case), or None.

    Returns:
        Preconditioner function.
    """
    # Setup full case
    E_full.SplusV_setup(V0, factors_only=factors_only)
    U = ttax.TTMatrix([jnp.expand_dims(c, [0, 3]) for c in E_full.U_SV])
    Uinv = ttax.TTMatrix([jnp.expand_dims(c, [0, 3]) for c in E_full.Uinv_SV])
    lmd_1d = E_full.lmd_1d_SV
    return setup_expsum_precond(rank_expsum, U, Uinv, lmd_1d, batch_expsum, p)


def setup_S_precond(
    E_full: FullEnergyFunctional,
    M_1d,  # FIXME: UNUSED IN THIS BODY -- LEFTOVER OF THE PRE-RESCALING MASS. REMOVING IT MEANS UPDATING ALL CALL SITES.
    rank_expsum,
    batch_expsum=None,
    factors_only=False,
    p=None,
):
    """Setup S^{-1} preconditioner using exponential sum.

    Same as `setup_SplusV_precond` with V_0 = 0 (the "S" preconditioner of
    Sec. 'Numerical experiments'). With p not None it approximates
    (S + I)^{-1} ⊗ I_p.

    Args:
        E_full: Full energy functional object.
            Built with the 1D stiffness matrix in the rescaled basis.
        M_1d: 1D mass matrix.
        rank_expsum: Number of terms in the exponential sum approximation.
        batch_expsum: Batching strategy (default: None).
        factors_only: If True, only compute factorization without full tensor.
        p: Number of components (multicomponent case), or None.

    Returns:
        Preconditioner function.
    """
    # Setup full case
    E_full.Sinv_setup(factors_only=factors_only)
    U = ttax.TTMatrix([jnp.expand_dims(c, [0, 3]) for c in E_full.U])
    Uinv = ttax.TTMatrix([jnp.expand_dims(c, [0, 3]) for c in E_full.Uinv])
    lmd_1d = [E_full.lmd_1d] * E_full.dim
    return setup_expsum_precond(rank_expsum, U, Uinv, lmd_1d, batch_expsum, p)
