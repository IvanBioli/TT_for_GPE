"""Preconditioned conjugate gradient and arithmetic on tangent vectors.

Tangent vectors of the TT manifold are stored in the delta parametrization
(Section 'The manifold of unit-norm fixed TT-rank functions'), i.e. as a list
of delta cores, a pytree. The linear systems with the projected operator
``P A_u P`` that define the Riemannian gradient (Section 'The $a_u$ Riemannian
gradient') are solved by preconditioned CG directly on these pytrees, using
``tree_math`` so that the iteration is traced and compiled once by ``jax``.
The same CG is used in the full format, where the "vectors" are dense arrays.
"""

import functools

import jax
import jax.numpy as jnp
import tree_math as tm
import tree_math.numpy as tnp
from jax import lax


def _cg_tree_math_core(
    A,
    b,
    x0,
    M,
    maxiter,
    tol,
    atol,
    point_dependent=False,
    u=None,
    u2=None,
    return_info=False,
):
    """Preconditioned conjugate gradient loop on ``tree_math`` vectors.

    Runs the standard PCG recursion inside ``jax.lax.while_loop`` and stops
    when the Euclidean (Frobenius) norm of the unpreconditioned residual
    satisfies ``||r|| <= max(tol * ||b||, atol)`` or after ``maxiter``
    iterations.

    Args:
        A: Linear operator, called as ``A(x)`` or, if ``point_dependent``,
            as ``A(x, u, u2)``.
        b: Right-hand side.
        x0: Initial guess.
        M: Preconditioner (approximate inverse of ``A``), with the same
            calling convention as ``A``.
        maxiter: Maximum number of CG iterations.
        tol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        point_dependent: If True, ``u`` and ``u2`` are passed to ``A`` and
            ``M`` (the base point and, optionally, its Hadamard square).
        u: Base point forwarded to ``A`` and ``M``.
        u2: Precomputed ``u * u`` forwarded to ``A`` and ``M``.
        return_info: If True, also return the final residual norm and the
            number of iterations.

    Returns:
        The approximate solution, or a tuple ``(x, info)`` with
        ``info = {"res": ||r||, "iters": k}`` if ``return_info`` is True.
    """
    A = tm.unwrap(A)
    M = tm.unwrap(M)
    atol2 = tnp.maximum(tol**2 * (b @ b), atol**2)

    def cond_fun(value):
        x, r, gamma, p, k = value
        return (r @ r > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p, u, u2) if point_dependent else A(p)
        alpha = gamma / (p.conj() @ Ap)
        x_ = x + alpha * p
        r_ = r - alpha * Ap
        z_ = M(r_, u, u2) if point_dependent else M(r_)
        gamma_ = r_.conj() @ z_
        beta_ = gamma_ / gamma
        p_ = z_ + beta_ * p
        return x_, r_, gamma_, p_, k + 1

    r0 = b - (A(x0, u, u2) if point_dependent else A(x0))
    p0 = z0 = M(r0, u, u2) if point_dependent else M(r0)
    gamma0 = r0 @ z0
    initial_value = (x0, r0, gamma0, p0, 0)
    x_final, r_final, _, _, k_final = lax.while_loop(cond_fun, body_fun, initial_value)

    if return_info:
        info = {"res": jnp.sqrt(r_final @ r_final), "iters": k_final}
        # print(f"Iters: {info['iters']}\t Residual: {info['res']:.3e}")
        return x_final, info
    else:
        return x_final


@functools.partial(tm.wrap, vector_argnames=["b", "x0"])
def cg_tree_math(A, b, x0, M, maxiter, tol, atol=0.0, return_info=False):
    """Preconditioned CG for an operator ``A(x)`` on pytrees.

    Equivalent to ``jax.scipy.sparse.linalg.cg``, written with ``tree_math``.
    See ``_cg_tree_math_core`` for the stopping criterion.

    Args:
        A: Symmetric positive definite operator ``x -> A(x)``.
        b: Right-hand side (pytree).
        x0: Initial guess (pytree with the structure of ``b``).
        M: Preconditioner ``r -> M(r)``.
        maxiter: Maximum number of iterations.
        tol: Relative residual tolerance.
        atol: Absolute residual tolerance (default: 0).
        return_info: If True, also return ``{"res", "iters"}``.

    Returns:
        The approximate solution, optionally with the info dictionary.
    """
    print("Compiling cg_tree_math...")
    return _cg_tree_math_core(
        A,
        b,
        x0,
        M,
        maxiter,
        tol,
        atol,
        point_dependent=False,
        u=None,
        u2=None,
        return_info=return_info,
    )


@functools.partial(tm.wrap, vector_argnames=["u", "u2", "b", "x0"])
def cg_tree_math_point_dependent(
    u, u2, A, b, x0, M, maxiter, tol, atol=0.0, return_info=False
):
    """Preconditioned CG for an operator ``A(x, u, u2)`` depending on a point.

    Same as ``cg_tree_math``, but the operator and the preconditioner also
    receive the base point ``u`` and ``u2`` (e.g. ``u * u``). Passing them as
    arguments, rather than closing over them, lets the jitted solver be reused
    across iterations of the optimizer.

    Args:
        u: Base point (e.g. the current iterate).
        u2: Auxiliary point-dependent data (e.g. ``u * u``), may be None.
        A: Operator ``(x, u, u2) -> A_u(x)``.
        b: Right-hand side (pytree).
        x0: Initial guess.
        M: Preconditioner ``(r, u, u2) -> M_u(r)``.
        maxiter: Maximum number of iterations.
        tol: Relative residual tolerance.
        atol: Absolute residual tolerance (default: 0).
        return_info: If True, also return ``{"res", "iters"}``.

    Returns:
        The approximate solution, optionally with the info dictionary.
    """
    print("Compiling cg_tree_math_point_dependent...")
    return _cg_tree_math_core(
        A,
        b,
        x0,
        M,
        maxiter,
        tol,
        atol,
        point_dependent=True,
        u=u,
        u2=u2,
        return_info=return_info,
    )


# Functions acting directly on the tree map, i.e. on lists of delta cores
# representing tangent vectors at the same base point (vector-space operations
# and the Frobenius inner product / norm of the delta parametrization).
add_deltas = lambda d1, d2: jax.tree.map(lambda x, y: x + y, d1, d2)
sub_deltas = lambda d1, d2: jax.tree.map(lambda x, y: x - y, d1, d2)
scale_deltas = lambda alpha, d: jax.tree.map(lambda x: alpha * x, d)
lincomb_deltas = lambda alpha, d1, beta, d2: jax.tree.map(
    lambda x, y: alpha * x + beta * y, d1, d2
)
vdot_deltas = lambda d1, d2: jnp.sum(jnp.array(jax.tree.map(jnp.vdot, d1, d2)))
norm_deltas = lambda d: jnp.sqrt(vdot_deltas(d, d))
zeroslike_deltas = lambda d: jax.tree.map(jnp.zeros_like, d)
