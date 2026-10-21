"""Pymanopt manifold of unit-norm fixed TT-rank tensors.

Implements the manifold of Section 'Finite-dimensional discretization and
manifolds', subsection 'The manifold of unit-norm fixed TT-rank functions':
the intersection of the manifold of tensors of fixed TT rank with the unit
sphere, in the rescaled basis in which the discrete L2 inner product is the
Frobenius one. With ``p`` components it implements the manifold of
subsection 'The manifold of multicomponent unit-norm fixed TT-rank functions'
(fixed TT rank intersected with the oblique manifold, i.e. one mass
constraint per component).

Points are TT vectors wrapped in ``FixedRankPoint``; tangent vectors are kept
in the delta parametrization (gauge conditions with respect to the
left-orthogonalized base point). The geometric tools are those of subsection
'Geometric tools' of Section 'Riemannian optimization on the manifold of
unit-norm fixed TT-rank functions': the L2-orthogonal projection onto the
tangent space (also used as transporter) and the retraction by TT rounding
followed by normalization. The Riemannian metric defined here is the
Frobenius one; ``experiments.experiment_setup`` replaces ``inner_product`` by
the a_u metric.
"""

import sys
from functools import cached_property
from pathlib import Path
from typing import List, Literal

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import tree_math as tm
import ttax
import ttax.autodiff
from pymanopt.manifolds.manifold import RiemannianSubmanifold
from ttax.riemannian import TTMatrix, TTTensOrMat, deltas_to_tangent

from core.tangent import (
    zeroslike_deltas,
)
from core.tt_utils import (
    partial_flat_inner_last,
    partial_flat_inner_last_tangent_deltas,
    rescale_last_component,
    rescale_last_component_deltas,
)


class FixedRankPoint:
    """Point on the manifold of fixed-rank TT tensors.

    Wraps a TT vector and caches quantities reused within one iteration.
    Attribute access not defined here is forwarded to the wrapped TT object.

    Args:
        u: Tensor train representation.
        max_tt_rank_u2: If not None, ``u2`` (the Hadamard square ``u * u``) is
            rounded to this TT rank.
    """

    def __init__(self, u: TTTensOrMat, max_tt_rank_u2=None):
        self._u = u
        self._max_tt_rank_u2 = max_tt_rank_u2

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped instance
        return getattr(self._u, name)

    @property
    def u(self):
        """The wrapped TT vector."""
        return self._u

    @property
    def max_tt_rank_u2(self):
        """TT rank used to round ``u2`` (None: no rounding)."""
        return self._max_tt_rank_u2

    @cached_property
    def delta(self):
        """The point itself as a tangent vector at the point (delta form)."""
        return FixedRankTangentVector(ttax.project(self._u, self._u))

    @cached_property
    def u2(self):
        """Hadamard square ``u * u``, rounded to ``max_tt_rank_u2`` if set."""
        u2_fullrank = ttax.multiply(self._u, self._u)
        if self._max_tt_rank_u2 is None:
            return u2_fullrank
        else:
            # print("Rounding u^2 ...")
            return ttax.round(u2_fullrank, self._max_tt_rank_u2)


FixedRankTangentVector = tm.Vector
"""Tangent vector on the manifold of fixed-rank TT tensors. It is represented in
delta form, i.e., it satisfies the gauge conditions with respect to the base
point. Deltas are stored as a tree_math Vector."""


class FixedTTRankEmbedded(RiemannianSubmanifold):
    """Riemannian manifold of fixed-rank TT tensors with unit mass (if mass is
    not None) and embedded geometry.

    With ``mass=None`` this is the manifold of fixed TT rank alone. With
    ``mass="identity"`` (the case used in the experiments, since the mass
    matrix is the identity in the rescaled basis) it is its intersection with
    the unit sphere, or, if ``p`` is given, with the set of tensors whose
    ``p`` components (slices along the last mode) have squared norms
    ``alphas``.

    Args:
        tt_sz: Size of tensor in each dimension (the last one is ``p`` in the
            multicomponent case).
        tt_rank: TT rank (integer or list of ``len(tt_sz) - 1`` ranks).
        mass: None, ``"identity"``, or a TT matrix used in the norm (default:
            None). Only ``"identity"`` is supported if ``p`` is given.
        rng: Random number generator key (default: None).
        p: Number of components, or None for the single-component case.
        alphas: Prescribed squared norms (masses) of the ``p`` components
            (default: all ones).
    """

    def __init__(
        self,
        tt_sz: List[int],
        tt_rank: int | List[int],
        mass: None | TTMatrix | Literal["identity"] = None,
        rng: None | jnp.ndarray = None,
        p: None | int = None,
        alphas: None | jnp.ndarray = None,
    ):
        # Number of components
        self.p = p
        if alphas is None and p is not None:
            alphas = jnp.ones(p)
        if p is None and alphas is not None:
            raise ValueError(
                "If alphas is provided, p (number of multicomponents) must also be provided."
            )
        self.alphas = alphas

        # Size and rank of TT-tensors
        self._tt_sz = tt_sz
        if isinstance(tt_rank, int):
            tt_rank = [tt_rank] * (len(tt_sz) - 1)
        if len(tt_rank) != len(tt_sz) - 1:
            raise ValueError(
                f"It must hold len(tt_rank) = len(tt_sz) - 1. Got len(tt_rank) = {len(tt_rank)}, len(tt_sz) = {len(tt_sz)}."
            )
        tt_rank = [1] + tt_rank + [1]
        self._tt_rank = tt_rank

        if rng is None:
            self._rng = jax.random.PRNGKey(0)
        else:
            self._rng = rng

        # Check mass
        if mass is not None and mass != "identity" and not isinstance(mass, TTMatrix):
            raise ValueError(
                f"mass must be None, 'identity', or a TTMatrix. Got {type(mass)}."
            )
        if mass != "identity" and p is not None:
            raise NotImplementedError(
                "Mass matrices other than identity are not implemented yet for the multi-component case."
            )

        self._mass = mass  # Mass matrix for normalization of the norm
        name = f"Embedded manifold of {self._tt_sz} tensors of TT-rank {self._tt_rank}"  # Name of the manifold
        dimension = np.sum(
            [
                n * rmu * rmu_prec
                for (n, rmu, rmu_prec) in zip(
                    self._tt_sz, self._tt_rank[:-1], self._tt_rank[1:]
                )
            ]
        ) - np.sum(np.array(tt_rank[1:-1]) ** 2)
        super().__init__(name, dimension, point_layout=1)

        # Pre-compile projection (needed for orthogonalization of tangent vectors)
        self.partial_flat_inner_last = jax.jit(partial_flat_inner_last)
        self.partial_flat_inner_last_tangent_deltas = jax.jit(
            partial_flat_inner_last_tangent_deltas
        )
        self._project_fn = jax.jit(ttax.fuse(ttax.project))
        self.random_tangent_vector(self.random_point())
        self._project_fn(self.random_point().u, self.random_point().u)

    @property
    def typical_dist(self):
        """Typical distance (used by pymanopt); set to the dimension."""
        return self.dim

    def inner_product(
        self,
        point: FixedRankPoint,
        tangent_vector_a: FixedRankTangentVector,
        tangent_vector_b: FixedRankTangentVector,
    ):
        """Compute inner product between two tangent vectors.

        Frobenius (discrete L2) inner product, computed directly on the deltas
        if the mass is None or the identity.

        Args:
            point: Base point on manifold.
            tangent_vector_a: First tangent vector.
            tangent_vector_b: Second tangent vector.

        Returns:
            Inner product value.
        """
        if self._mass is None or self._mass == "identity":
            return tangent_vector_a.conj() @ tangent_vector_b
        else:
            vector_a = deltas_to_tangent(tangent_vector_a.tree, point.u)
            vector_b = deltas_to_tangent(tangent_vector_b.tree, point.u)
            return ttax.flat_inner(vector_a, ttax.multiply(self._mass, vector_b))

    def projection(self, point: FixedRankPoint, vector: TTTensOrMat):
        """Project vector onto tangent space at point.

        L2-orthogonal projection: projection onto the tangent space of the
        fixed-rank manifold followed by the removal of the component along
        the point (component-wise if ``p`` is given), see paragraph
        '$L^2$-orthogonal projection and transporter.'.

        Args:
            point: Base point on manifold.
            vector: Vector to project, as a TT vector.

        Returns:
            Projected tangent vector (delta form).
        """
        proj_Mr_tangent = FixedRankTangentVector(ttax.project(vector, point.u))
        return self._orthogonalize_tangent_vector(point, proj_Mr_tangent)

    def retraction(self, point: FixedRankPoint, tangent_vector: FixedRankTangentVector):
        """Retract tangent vector to manifold, i.e. compute the retraction of
        R_point(point + tangent_vector).

        As in paragraph 'Retraction.': the TT vector ``point + tangent_vector``
        is rounded to the fixed TT ranks and then normalized (component-wise
        if ``p`` is given).

        Args:
            point: Base point on manifold.
            tangent_vector: Tangent vector to retract.

        Returns:
            New point on manifold.
        """
        retraction_Mr = ttax.round(
            deltas_to_tangent((point.delta + tangent_vector).tree, point.u),
            self._tt_rank,
        )
        return self._normalize_point(
            FixedRankPoint(retraction_Mr, point.max_tt_rank_u2)
        )

    def norm(self, point: FixedRankPoint, tangent_vector: FixedRankTangentVector):
        """Compute norm of tangent vector, induced by ``inner_product``.

        Args:
            point: Base point on manifold.
            tangent_vector: Tangent vector.

        Returns:
            Norm value.
        """
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def random_point(self):
        """Generate random point on manifold.

        Returns:
            Random normalized point.
        """
        [key, self._rng] = jax.random.split(self._rng, 2)
        u = ttax.ops.tensor_to_vector(
            ttax.random.tensor(
                key, self._tt_sz, tt_rank=self._tt_rank, dtype=jnp.float64
            )
        )
        return self._normalize_point(FixedRankPoint(u))

    def to_tangent_space(self, point: FixedRankPoint, vector: FixedRankTangentVector):
        """Convert vector representation to tangent space representation (delta
        representation).

        Args:
            point: Base point on manifold.
            vector: Vector in delta form.

        Returns:
            Vector in tangent space.
        """
        return self.projection(point, deltas_to_tangent(vector.tree, point.u))

    def random_tangent_vector(self, point: FixedRankPoint):
        """Generate random tangent vector at point.

        Draws Gaussian deltas, enforces the gauge conditions and projects out
        the component along the point. The vector is divided twice by the
        squared Frobenius norm of its deltas (before and after the
        orthogonalization), so it is not normalized to unit norm.

        Args:
            point: Base point on manifold.

        Returns:
            Random tangent vector (delta form).
        """
        left = ttax.decompositions.orthogonalize(point.u)
        deltas = []
        for cc in left.tt_cores:
            [key, self._rng] = jax.random.split(self._rng, 2)
            deltas += [jax.random.normal(key, cc.shape)]
        deltas = ttax.autodiff._enforce_gauge_conditions(deltas, left)
        vector = FixedRankTangentVector(deltas)
        vector = vector / (vector.conj() @ vector)
        vector = self._orthogonalize_tangent_vector(point, vector)
        vector = vector / (vector.conj() @ vector)
        return vector

    def embedding(self, point: FixedRankPoint, tangent_vector: FixedRankTangentVector):
        """Embed tangent vector into ambient space, i.e. convert from tangent
        space representation (delta representation) to vector representation.

        Args:
            point: Base point on manifold.
            tangent_vector: Tangent vector in delta form.

        Returns:
            Vector in TT format.
        """
        return deltas_to_tangent(tangent_vector.tree, point.u)

    def transport(
        self,
        point_a: FixedRankPoint,
        point_b: FixedRankPoint,
        tangent_vector_a: FixedRankTangentVector,
    ):
        """Vector transport from point_a to point_b.

        L2-orthogonal projection of the embedded tangent vector onto the
        tangent space at ``point_b`` (paragraph '$L^2$-orthogonal projection
        and transporter.').

        Args:
            point_a: Source point.
            point_b: Target point.
            tangent_vector_a: Tangent vector at point_a.

        Returns:
            Transported tangent vector at point_b.
        """
        return self.projection(point_b, self.embedding(point_a, tangent_vector_a))

    def zero_vector(self, point: FixedRankPoint):
        """Create zero tangent vector at point.

        Args:
            point: Base point on manifold.

        Returns:
            Zero tangent vector.
        """
        return FixedRankTangentVector(zeroslike_deltas(point.u.tt_cores))

    def _normalize_point(self, point: FixedRankPoint):
        """Normalize point with respect to mass matrix.

        In the multicomponent case, rescales the last core so that the
        squared norm of the ``j``-th component equals ``alphas[j]``; the
        component norms are read from the last core of ``point.delta``.

        Args:
            point: Point to normalize.

        Returns:
            Normalized point.
        """
        if self._mass is None:
            return point
        elif self.p is None:
            if self._mass == "identity":
                return FixedRankPoint(
                    point.u * (1 / jnp.sqrt(ttax.flat_inner(point.u, point.u))),
                    point.max_tt_rank_u2,
                )
            else:
                return FixedRankPoint(
                    point.u
                    * (
                        1
                        / jnp.sqrt(
                            ttax.flat_inner(point.u, ttax.multiply(self._mass, point.u))
                        )
                    ),
                    point.max_tt_rank_u2,
                )
        else:  # Multi-component case with identity mass
            component_mass = self.partial_flat_inner_last_tangent_deltas(
                point.delta.tree, point.delta.tree
            )
            norm_factor = jnp.sqrt(self.alphas / component_mass)
            normalized_u = rescale_last_component(point.u, norm_factor)
            return FixedRankPoint(normalized_u, point.max_tt_rank_u2)

    def _orthogonalize_tangent_vector(
        self, point: FixedRankPoint, vector: FixedRankTangentVector
    ):
        """Orthogonalize tangent vector with respect to point.

        Removes from ``vector`` its L2 component along the point, i.e. maps the
        tangent space of the fixed-rank manifold onto that of the intersection
        manifold. In the multicomponent case this is done component-wise
        (``v_j - (v_j, u_j) / alphas[j] * u_j``) on the last delta core.

        Args:
            point: Base point on manifold.
            vector: Tangent vector to orthogonalize.

        Returns:
            Orthogonalized tangent vector.
        """
        if self._mass is None:
            return vector
        if self.p is None:
            if self._mass == "identity":
                return vector - (vector.conj() @ point.delta) * point.delta
            else:
                return (
                    vector
                    - ttax.flat_inner(
                        deltas_to_tangent(vector.tree, point.u),
                        ttax.multiply(self._mass, point.u),
                    )
                    * point.delta
                )
        else:  # Multi-component case
            if self._mass != "identity":
                raise NotImplementedError(
                    "Orthogonalization of tangent vector for multi-component case with non-identity mass is not implemented yet."
                )
            else:
                # Multi-component case with identity mass
                component_mass = self.partial_flat_inner_last_tangent_deltas(
                    point.delta.tree, vector.tree
                )
                orthogonality_components = component_mass / self.alphas
                orthogonalization_deltas = rescale_last_component_deltas(
                    point.delta.tree, orthogonality_components
                )
                return vector - FixedRankTangentVector(orthogonalization_deltas)
