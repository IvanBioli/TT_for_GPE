"""Utilities for TT tensors: cross approximation and multicomponent helpers.

``tt_from_fun`` builds the TT cores of a function sampled on a tensor grid
(used for the initial guesses). The other functions implement the
operations on the component mode needed in the multicomponent case (Section
'The multicomponent Gross--Pitaevskii equation', subsection 'Implementation
details'): for a multicomponent tensor of shape ``[n, ..., n, p]`` they act
on the last core only.
"""

from typing import Callable, List, Union

import jax.numpy as jnp
import numpy as np
import tt
import tt.core
import tt.core.vector
import ttax
from ttax import I_OR_IJ, TTEinsum, TTMatrix, to_function


def tt_from_fun(
    f: Callable,
    meshes_1d: List[np.ndarray],
    target_ranks: Union[None, List[int], np.ndarray] = None,
    **kwargs
):
    """Approximate a function on a tensor grid in TT format by cross approximation.

    Uses ``tt.multifuncrs2`` (ttpy) on the coordinate tensors of the grid. If
    ``target_ranks`` is given, the result is rounded to the smallest interior
    target rank and then padded with zero blocks so that its TT ranks equal
    ``target_ranks`` exactly (the represented tensor is unchanged).

    Args:
        f: Function of the grid points, called by ``tt.multifuncrs2`` with an
            array of shape ``[M, dim]`` and returning ``M`` values.
        meshes_1d: List of ``dim`` one-dimensional grids.
        target_ranks: Optional TT ranks ``[1, r_1, ..., r_{dim-1}, 1]``.
        **kwargs: Passed to ``tt.multifuncrs2`` (e.g. ``eps``, ``rng``).

    Returns:
        List of TT cores (numpy arrays of shape ``(r_{i-1}, n_i, r_i)``).
    """
    # Construct tensors from meshes in 1d
    meshes_1d_core = [np.expand_dims(msh_1d, [0, 2]) for msh_1d in meshes_1d]
    ones_1d_core = [np.ones_like(msh_1d_core) for msh_1d_core in meshes_1d_core]
    dim = len(meshes_1d_core)
    coords = []
    for i in range(dim):
        coords += [
            ones_1d_core[: max(i, 0)]
            + meshes_1d_core[i : i + 1]
            + ones_1d_core[min(i + 1, dim) :]
        ]
    coords = [tt.vector.from_list(c) for c in coords]

    # Cross approximation
    if target_ranks is not None:
        target_ranks = np.array(target_ranks)
        if target_ranks.size == 1:
            target_ranks = np.array([1] + [target_ranks[1]] * (dim - 1) + [1])
        kwargs["rmax"] = int(np.min(target_ranks[1:-1]))
    tt_fun = tt.multifuncrs2(coords, f, verb=0, **kwargs)
    if target_ranks is not None:
        tt_fun = tt_fun.round(rmax=kwargs["rmax"])

    # Increase ranks if need be (pad with zero cores: adds zero to the tensor)
    if target_ranks is not None:
        diff = target_ranks - tt_fun.r
        diff[0] = diff[-1] = 1
        if np.any(diff[1:-1] > 0):
            shape = tt_fun.n
            zero_cores = []
            for i in range(len(shape)):
                zero_cores += [np.zeros((diff[i], shape[i], diff[i + 1]))]
            tt_fun = tt_fun + tt.vector.from_list(zero_cores)

    return tt.vector.to_list(tt_fun)


@ttax.fuse
def partial_flat_inner_last(a, b):
    """Component-wise inner products of two multicomponent TT vectors.

    Contracts all modes except the last (component) one, i.e. computes the
    diagonal of the matrix of component-wise inner products
    ``[(a_j, b_j)]_{j=1..p}`` of Section 'The multicomponent Gross--Pitaevskii
    equation'.

    Args:
        a: TT vector (TT matrix with column size 1) of shape ``[..., p]``.
        b: TT vector with the same mode sizes as ``a``.

    Returns:
        Array of length ``p`` (size of the last mode).
    """
    print("Compiling partial_flat_inner_last...")
    # 1. Define the same cumulative einsum strategy
    tt_einsum = TTEinsum(
        inputs=[["a", I_OR_IJ, "b"], ["c", I_OR_IJ, "d"], ["a", "c"]],
        output=["b", "d"],
        how_to_apply="cumulative",
    )
    func = to_function(tt_einsum)

    # 2. Run the function to get the list of intermediate contractions
    # all_res is a list of tensors of shape (rank_a, rank_b)
    all_res = func(a, b)

    # 3. Retrieve the accumulated tensor BEFORE the last core
    # shape: (rank_prev_a, rank_prev_b)
    left_block = all_res[-2]

    # 4. Get the actual last cores of the TT-tensors
    # Standard TT core shape: (rank_in, n_physical, rank_out)
    # For the last core, rank_out is usually 1.
    core_a_last = a.tt_cores[-1]
    core_b_last = b.tt_cores[-1]

    # 5. Contract the left_block with the last cores, keeping the physical index 'n'
    # 'ab' = left_block (rank_a, rank_b)
    # 'an...' = core_a (rank_a, n, 1, 1) -> 'an' effectively
    # 'bn...' = core_b (rank_b, n, 1, 1) -> 'bn' effectively
    # Output 'n' -> The vector of partial inner products

    # We slice [:, :, 0] to remove the final rank-1 dimension for cleaner einsum
    res_vector = jnp.einsum(
        "ab,an,bn->n", left_block, core_a_last[:, :, 0, 0], core_b_last[:, :, 0, 0]
    )

    return res_vector


def partial_flat_inner_last_tangent_deltas(u_deltas, deltas_v):
    """Component-wise inner products of two tangent vectors from their deltas.

    Contracts only the last delta cores. This equals the component-wise inner
    products of the two tangent vectors when the base point is
    left-orthogonal and one of the two is the base point itself (whose delta
    representation is zero except in the last core), as in the lemma of
    subsection 'Implementation details' of Section 'The multicomponent
    Gross--Pitaevskii equation'. Its cost is that of one contraction of the
    last cores.

    Args:
        u_deltas: Delta cores of the first tangent vector (typically the
            base point).
        deltas_v: Delta cores of the second tangent vector.

    Returns:
        Array of length ``p`` (size of the last mode).
    """
    print("Compiling partial_flat_inner_last_tangent_deltas...")
    last_delta_u = u_deltas[-1]
    last_delta_v = deltas_v[-1]
    last_term = jnp.einsum("an...,an...->n", last_delta_u, last_delta_v)
    return last_term


def rescale_last_component(u: TTMatrix, scale_factors: jnp.ndarray) -> TTMatrix:
    """Rescale each component of a multicomponent TT vector.

    Multiplies the ``j``-th slice of the last core by ``scale_factors[j]``,
    i.e. computes ``u Sigma`` with ``Sigma = diag(scale_factors)``.

    Args:
        u: TT vector (TT matrix with column size 1) of shape ``[..., p]``.
        scale_factors: 1D array of length ``p`` with one factor per component.

    Returns:
        Rescaled TT vector.
    """
    cores = u.tt_cores.copy()
    cores[-1] = cores[-1] * jnp.expand_dims(scale_factors, axis=(0, 2, 3))
    return TTMatrix(cores)


def rescale_last_component_deltas(deltas: List, scale_factors: jnp.ndarray) -> List:
    """Rescale each component of a tangent vector given by its deltas.

    Multiplies the ``j``-th slice of the last delta core by
    ``scale_factors[j]``. Applied to the deltas of the base point (zero except
    in the last core) this gives the deltas of ``u Sigma``.

    Args:
        deltas: List of delta cores of a multicomponent tangent vector.
        scale_factors: 1D array of length ``p`` with one factor per component.

    Returns:
        List of delta cores with the last one rescaled.
    """
    deltas_rescaled = deltas.copy()
    deltas_rescaled[-1] = deltas[-1] * jnp.expand_dims(scale_factors, axis=(0, 2, 3))
    return deltas_rescaled


def get_component(u: TTMatrix, component_index: int) -> TTMatrix:
    """Zero all components of a multicomponent TT vector except one.

    Keeps the shape ``[..., p]`` and sets to zero all slices of the last core
    except ``component_index`` (the frame of the ``j``-th component in
    subsection 'Implementation details' of Section 'The multicomponent
    Gross--Pitaevskii equation').

    Args:
        u: TT vector (TT matrix with column size 1) of shape ``[..., p]``.
        component_index: Index of the component to keep (0-based).

    Returns:
        TT vector of the same shape with a single nonzero component.
    """
    cores = u.tt_cores.copy()
    last = jnp.zeros_like(cores[-1])
    last = last.at[:, component_index : component_index + 1, :, :].set(
        cores[-1][:, component_index : component_index + 1, :, :]
    )
    cores[-1] = last
    return TTMatrix(cores)
