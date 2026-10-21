"""Assembly of the discrete operators in TT format.

Implements the spectral element discretization of Section 'Discretization by
spectral elements': continuous piecewise polynomials of degree ``k`` with the
Lagrange basis on Gauss--Lobatto--Legendre (GLL) nodes, and GLL quadrature on
the same nodes, so that the 1D mass matrix is diagonal. The basis is rescaled
by the square roots of the quadrature weights, so that the mass matrix becomes
the identity and the stiffness matrix becomes ``diag(m)^{-1/2} S diag(m)^{-1/2}``.
The d-dimensional stiffness matrix is assembled as a TT matrix of rank 2 and
the potential as a sum of rank-1 TT tensors. ``setup_multicomponent_matrices``
appends the component mode as in subsection 'Discretization' of Section 'The
multicomponent Gross--Pitaevskii equation'.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import basix
import jax
import jax.numpy as jnp
import numpy as np
import ttax
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from core.fixed_tt_rank import FixedRankPoint, FixedRankTangentVector


def setup_1Dmatrices(N, a, b, k):
    """Setup 1D mass and stiffness matrices using FEniCSx. We employ a spectral element
    method with Lagrange polynomials on Gauss-Lobatto points as basis functions,
    and employ the same Gauss-Lobatto points as quadrature points. This makes
    the mass matrix diagonal.

    The interval ``(a, b)`` is split into ``(N + 1) / k`` elements, which gives
    ``N`` interior nodes (``n = kE - 1`` in Section 'Discretization by spectral
    elements'). Boundary degrees of freedom are removed (homogeneous Dirichlet
    conditions) and the matrices are ordered by increasing node coordinate.

    Args:
        N: Number of interior grid points (``N + 1`` must be divisible by
            ``k``).
        a: Left domain boundary.
        b: Right domain boundary.
        k: Polynomial degree.

    Returns:
        Tuple ``(M_1d, S_1d, msh_1d, M_1d_exact)``: the diagonal GLL-quadrature
        mass matrix and the stiffness matrix (scipy sparse, Lagrange basis,
        ``N x N``), the interior node coordinates, and the mass matrix
        assembled with a quadrature of degree ``2k`` (not diagonal).
    """
    # Assemble 1D matrices
    ncells = int((N + 1) / k)
    element_1d = basix.ufl.element(
        "Lagrange", "interval", k, basix.LagrangeVariant.gll_warped
    )
    domain_1d = mesh.create_interval(
        comm=MPI.COMM_WORLD,
        points=jnp.asarray([a, b]),
        nx=ncells,
    )
    V_1d = fem.functionspace(domain_1d, element_1d)

    nodes_1d = V_1d.tabulate_dof_coordinates()[:, 0]
    p_1d = np.argsort(nodes_1d)
    msh_1d = nodes_1d[p_1d][1:-1]

    metadata_1d = {"quadrature_rule": "gll", "quadrature_degree": 2 * k - 1}
    dx_1d = ufl.Measure("dx", domain=domain_1d, metadata=metadata_1d)

    u = ufl.TrialFunction(V_1d)
    v = ufl.TestFunction(V_1d)
    gradgrad = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_1d
    S_1d = fem.assemble_matrix(fem.form(gradgrad)).to_scipy()[:, p_1d][p_1d, :]
    S_1d = S_1d[1:-1, 1:-1]
    M_1d = fem.assemble_matrix(fem.form(ufl.dot(u, v) * dx_1d)).to_scipy()[:, p_1d][
        p_1d, :
    ]
    M_1d = M_1d[1:-1, 1:-1]
    np.testing.assert_allclose(M_1d.todense(), np.diag(np.diag(M_1d.todense())))

    metadata_1d_accurate = {"quadrature_degree": 2 * k}
    dx_1d_accurate = ufl.Measure("dx", domain=domain_1d, metadata=metadata_1d_accurate)
    M_1d_exact = fem.assemble_matrix(
        fem.form(ufl.dot(u, v) * dx_1d_accurate)
    ).to_scipy()[:, p_1d][p_1d, :]
    M_1d_exact = M_1d_exact[1:-1, 1:-1]

    return M_1d, S_1d, msh_1d, M_1d_exact


def setup_matrices(dim, N, a, b, V_functions, k):
    """Setup multi-dimensional matrices in TT format.

    All operators are expressed in the rescaled basis ``phi_i / sqrt(m_i)`` of
    Section 'Discretization by spectral elements', in which the mass matrix
    is the identity.

    Args:
        dim: Spatial dimension.
        N: Number of interior grid points per dimension.
        a: Left domain boundary.
        b: Right domain boundary.
        V_functions: Potential as a sum of separable terms: a list whose
            entries are tuples ``(f_1, ..., f_dim)`` of 1D functions, each
            tuple representing the rank-1 term ``f_1(x_1) * ... * f_dim(x_dim)``.
            If None, no potential tensor is built.
        k: Polynomial degree.

    Returns:
        Tuple ``(S, M, V, msh_1d)``:
            S: stiffness matrix ``S_1d (x) I (x) ... + ... + I (x) ... (x) S_1d``
                as a TT matrix of rank 2, with the rescaled ``S_1d``.
            M: tensor of the GLL quadrature weights ``m_{i_1} ... m_{i_d}``,
                as a rank-1 TT vector (or a scalar if all weights are equal).
            V: batch of rank-1 TT vectors, one per entry of ``V_functions``,
                holding the nodal values of each term (None if
                ``V_functions`` is None).
            msh_1d: interior node coordinates.
    """
    M_1d, S_1d, msh_1d, _ = setup_1Dmatrices(N, a, b, k)
    S_1d, M_1d = S_1d.todense(), M_1d.todense()  # FIXME: AVOID DENSE CONVERSION

    # Rescale matrices to make mass the identity
    M_1d_inv_sqrt = jnp.diag(1 / jnp.sqrt(jnp.diag(M_1d)))
    S_1d = M_1d_inv_sqrt @ S_1d @ M_1d_inv_sqrt
    I_1d = jnp.eye(S_1d.shape[0])

    # Define tensors
    O = jnp.zeros_like(M_1d)
    B1 = jnp.expand_dims(jax.numpy.stack([S_1d, I_1d], axis=2), 0)
    B2_up = jax.numpy.stack([I_1d, O], axis=2)
    B2_low = jax.numpy.stack([S_1d, I_1d], axis=2)
    B2 = jax.numpy.stack([B2_up, B2_low], axis=0)
    B3 = jnp.expand_dims(jax.numpy.stack([I_1d, S_1d]), 3)
    S = ttax.TTMatrix([B1] + [B2] * (dim - 2) + [B3])

    mdiag = jnp.diag(M_1d)
    if np.allclose(mdiag, mdiag[0], atol=1e-10):
        M = mdiag[0] ** dim
    else:
        M = ttax.ops.tensor_to_vector(ttax.TT([jnp.expand_dims(mdiag, [0, 2])] * dim))

    if V_functions is None:
        V = None
    else:

        # msh_1d = jnp.array([-6] + msh_1d.tolist() + [6])

        V_addends = [
            ttax.ops.tensor_to_vector(
                ttax.TT([f(jnp.expand_dims(msh_1d, [0, 2])) for f in funs_1d])
            )
            for funs_1d in V_functions
        ]
        cores = [
            jnp.stack([V_core.tt_cores[i] for V_core in V_addends], axis=0)
            for i in range(len(V_addends[0].tt_cores))
        ]
        V = ttax.TTMatrix(cores)

    return S, M, V, msh_1d


def setup_multicomponent_matrices(p, S, M, V_vector, kappa_matrix):
    """Extend the single-component operators with a component mode of size p.

    Implements the extension of subsection 'Discretization' of Section 'The
    multicomponent Gross--Pitaevskii equation': ``S_p = S (x) I_p``, the
    weight tensor and every potential term are extended by the all-ones vector
    in the component mode (all components share the same potential), and the
    coupling ``K_p = I (x) ... (x) I (x) K`` is a rank-1 TT matrix whose last
    core is the interaction matrix.

    Args:
        p: Number of components.
        S: Stiffness TT matrix from ``setup_matrices``. Its core list is
            extended in place.
        M: Weight tensor from ``setup_matrices`` (TT vector or scalar).
        V_vector: Batch TT vector of potential terms from ``setup_matrices``.
        kappa_matrix: Interaction matrix ``K`` of shape ``(p, p)``.

    Returns:
        Tuple ``(S_new, M_new, V_vector_new, K)`` of TT operators on tensors of
        shape ``[N, ..., N, p]``.
    """
    # Add identity component to S
    cores_S = S.tt_cores
    identity_core = jnp.expand_dims(jnp.eye(p, p), axis=[0, 3])  # Shape (1, p, p, 1)
    cores_S.append(identity_core)
    S_new = ttax.TTMatrix(cores_S)

    # Add identity component to M, i.e., all ones
    if isinstance(M, ttax.TTMatrix):
        cores_M = ttax.ops.vector_to_tensor(M).tt_cores
        ones_core = jnp.expand_dims(jnp.ones(p), [0, 2])
        cores_M.append(ones_core)
        M_new = ttax.ops.tensor_to_vector(ttax.TT(cores_M))
    else:
        M_new = M

    # Add all ones last component to V_vector
    V_addends = []
    batch_size = V_vector.shape[0]
    for i in range(batch_size):
        V_vector_i = ttax.ops.vector_to_tensor(V_vector.batch_loc[i,])
        V_vector_i_cores = V_vector_i.tt_cores
        ones_core = jnp.expand_dims(jnp.ones(p), [0, 2])
        V_vector_i_cores.append(ones_core)
        V_addends.append(ttax.ops.tensor_to_vector(ttax.TT(V_vector_i_cores)))

    cores = [
        jnp.stack([V_core.tt_cores[i] for V_core in V_addends], axis=0)
        for i in range(len(V_addends[0].tt_cores))
    ]
    V_vector_new = ttax.TTMatrix(cores)

    # TT Matrix with all identities and kappa_matrix as last core
    K_cores = []
    for i in range(len(V_addends[0].tt_cores) - 1):
        N = V_addends[0].tt_cores[i].shape[1]
        K_cores.append(jnp.expand_dims(jnp.eye(N), axis=[0, 3]))
    K_cores.append(jnp.expand_dims(kappa_matrix, axis=[0, 3]))
    K = ttax.TTMatrix(K_cores)

    return S_new, M_new, V_vector_new, K


def get_inner_au_tangent(bilinear_form, u2=False):
    """Create the a_u Riemannian metric on the TT manifold.

    The returned function evaluates ``a_u(v, w)`` (Section 'Riemannian
    optimization on the manifold of unit-norm fixed TT-rank functions',
    subsection 'Geometric tools') for tangent vectors given in delta form, after
    embedding them as TT tensors.

    Args:
        bilinear_form: Bilinear form function, called as
            ``bilinear_form(u, v, w)`` (or ``bilinear_form(u2, v, w)``).
        u2: If True, pass the cached (possibly rounded) ``u * u`` of the point,
            ``point.u2``, instead of ``u`` to ``bilinear_form``.

    Returns:
        Function ``(point, tangent_vector_a, tangent_vector_b) -> a_u(v, w)``
        to be used as ``manifold.inner_product``.
    """

    if not u2:

        def inner_product_au(
            point: FixedRankPoint,
            tangent_vector_a: FixedRankTangentVector,
            tangent_vector_b: FixedRankTangentVector,
        ):
            u = point.u
            v = ttax.riemannian.deltas_to_tangent(tangent_vector_a.tree, u)
            w = ttax.riemannian.deltas_to_tangent(tangent_vector_b.tree, u)
            return bilinear_form(u, v, w)

        return inner_product_au

    else:

        def inner_product_au2(
            point: FixedRankPoint,
            tangent_vector_a: FixedRankTangentVector,
            tangent_vector_b: FixedRankTangentVector,
        ):
            u = point.u
            u2 = point.u2
            v = ttax.riemannian.deltas_to_tangent(tangent_vector_a.tree, u)
            w = ttax.riemannian.deltas_to_tangent(tangent_vector_b.tree, u)
            return bilinear_form(u2, v, w)

        return inner_product_au2
