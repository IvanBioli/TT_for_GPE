"""Full-format (dense tensor) counterparts of the TT manifold tools.

Used for the "full" baselines of Section 'Numerical experiments' (subsection
'Comparison of the full and TT formats'): the unit sphere (or, with ``p``
components, the oblique manifold) in the space of dense coefficient tensors,
and helpers that apply Kronecker-structured operators mode by mode.
"""

import string
from typing import List

import jax
import jax.numpy as jnp
from pymanopt.manifolds.manifold import RiemannianSubmanifold


def apply_single_axis(u: jnp.ndarray, T: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Contract matrix T (new_d, old_d) with tensor u along a single axis.

    Args:
        u: Input tensor.
        T: Transformation matrix of shape (new_dim, old_dim).
        axis: Axis of u to contract.

    Returns:
        Transformed tensor (one full-size output, no extra temporaries).
    """
    ndim = u.ndim
    input_chars = string.ascii_lowercase[:ndim]
    new_char = chr(ord("a") + ndim)
    output_chars = input_chars[:axis] + new_char + input_chars[axis + 1 :]
    einsum_str = f"{input_chars},{new_char}{input_chars[axis]}->{output_chars}"
    return jnp.einsum(einsum_str, u, T)


def apply_transformations(u: jnp.ndarray, transforms: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Apply transformation matrices along each axis of tensor u.

    Args:
        u: Input tensor of shape (d0, d1, ..., dn)
        transforms: List of transformation matrices [T0, T1, ..., Tn]
                   where Ti has shape (new_di, di). A ``None`` entry is treated
                   as the identity and skipped, avoiding a full-size copy and the
                   associated flops for that axis.

    Returns:
        Transformed tensor
    """
    result = u
    ndim = u.ndim

    # Apply transformations in reverse order (last axis first)
    for axis in range(ndim - 1, -1, -1):
        T = transforms[axis]
        if T is None:  # identity -> skip (no full-size temporary)
            continue
        result = apply_single_axis(result, T, axis)

    return result


def sum_apply_transformation(u: jnp.ndarray, transforms_list: List[List[jnp.ndarray]]):
    """Apply multiple transformation lists and sum results.

    Accumulates into a single buffer (no ``zeros_like`` allocation) and relies on
    ``apply_transformations`` skipping ``None`` (identity) entries.

    Args:
        u: Input tensor.
        transforms_list: List of transformation lists.

    Returns:
        Sum of transformed tensors.
    """
    result = None
    for t in transforms_list:
        term = apply_transformations(u, t)
        result = term if result is None else result + term

    return result


def inner_product_first_components(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute inner product along the first physical dimensions of two tensors.

    Args:
        u: First tensor of shape (..., p)
        v: Second tensor of shape (..., p)

    Returns:
        Inner product along the first dimensions, resulting in a tensor of shape (p)
    """
    res_vector = jnp.einsum("...i,...i->i", u, v)
    return res_vector


class Sphere_Mass(RiemannianSubmanifold):
    """Sphere manifold with mass-weighted metric.

    Unit sphere ``{u : (u, M u) = 1}`` of dense tensors. If ``p`` is given,
    the tensors have shape ``shape + [p]`` and each component (slice along the
    last axis) is normalized separately to the Euclidean norm ``alphas[j]``;
    the mass is not used in that case. The projection is used as vector
    transport and the retraction is ``u + v`` followed by normalization.

    Args:
        shape: Shape of tensors on the sphere.
        mass: Mass matrix or list of matrices (default: None). A list is
            applied mode-wise (Kronecker product), anything else is
            multiplied entry-wise; None means the identity.
        rng: Random number generator key (default: None).
        p: Number of components, or None for the single-component case.
        alphas: Prescribed norms of the ``p`` components.
    """

    def __init__(
        self,
        shape: List[int],
        mass: None | List[jnp.ndarray] = None,
        rng: None | jnp.ndarray = None,
        p: None | int = None,
        alphas: None | jnp.ndarray = None,
    ):
        # p and alphas for multicomponent case
        self.p = p
        self.alphas = alphas

        # Name and dimensions
        if p is None:
            name = f"Sphere manifold of shape {shape} tensors"
            self._shape = jnp.asarray(shape)
            dimension = int(jnp.prod(self._shape) - 1)
            super().__init__(name, dimension)
        else:
            name = f"Oblique manifold of shape {shape} tensors with p={p} components"
            self._shape = jnp.asarray(shape + [p])
            dimension = int(jnp.prod(self._shape) * p - p)
            super().__init__(name, dimension)

        if rng is None:
            self._rng = jax.random.PRNGKey(0)
        else:
            self._rng = rng

        # Mass matrix for normalization of the norm
        self._mass = mass
        if self._mass is None:
            self._M = lambda x: x
        elif isinstance(self._mass, list):
            self._M = jax.jit(lambda x: apply_transformations(x, self._mass))
        else:
            self._M = jax.jit(lambda x: self._mass * x)

    @property
    def typical_dist(self):
        """Typical distance (used by pymanopt); ``sqrt(dim)``."""
        return jnp.sqrt(self.dim)

    def _M_inner_product(self, tensor_a, tensor_b):
        """Compute mass-weighted inner product.

        Args:
            tensor_a: First tensor.
            tensor_b: Second tensor.

        Returns:
            Inner product value.
        """
        return jnp.vdot(tensor_a, self._M(tensor_b))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        """Compute inner product between tangent vectors.

        Args:
            point: Base point on manifold.
            tangent_vector_a: First tangent vector.
            tangent_vector_b: Second tangent vector.

        Returns:
            Inner product value.
        """
        return self._M_inner_product(tangent_vector_a, tangent_vector_b)

    def norm(self, point, tangent_vector):
        """Compute norm of tangent vector.

        Args:
            point: Base point on manifold.
            tangent_vector: Tangent vector.

        Returns:
            Norm value.
        """
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def projection(self, point, vector):
        """Project vector onto tangent space at point.

        Removes the M-component along ``point``; in the multicomponent case,
        component-wise ``v_j - (u_j, v_j) / alphas[j]**2 * u_j`` (Euclidean
        inner products).

        Args:
            point: Base point on manifold.
            vector: Vector to project.

        Returns:
            Projected tangent vector.
        """
        if self.p is None:
            return vector - self._M_inner_product(point, vector) * point
        else:
            # Multicomponent case: project each component separately
            projected = (
                vector
                - inner_product_first_components(point, vector)
                / (self.alphas**2)
                * point
            )
            return projected

    to_tangent_space = projection

    def retraction(self, point, tangent_vector):
        """Retract tangent vector to manifold, i.e., computes the retraction of
        R_point(point + tangent_vector).

        Args:
            point: Base point on manifold.
            tangent_vector: Tangent vector to retract.

        Returns:
            New point on manifold.
        """
        return self._normalize(point + tangent_vector)

    def random_point(self):
        """Generate random point on manifold.

        Returns:
            Random normalized point.
        """
        [key, self._rng] = jax.random.split(self._rng, 2)
        point = jax.random.normal(key, self._shape)
        return self._normalize(point)

    def random_tangent_vector(self, point):
        """Generate random tangent vector at point.

        Args:
            point: Base point on manifold.

        Returns:
            Random normalized tangent vector.
        """
        [key, self._rng] = jax.random.split(self._rng, 2)
        vector = jax.random.normal(key, self._shape)
        return self._normalize(self.projection(point, vector))

    def transport(self, point_a, point_b, tangent_vector_a):
        """Vector transport from point_a to point_b, by projection.

        Args:
            point_a: Source point.
            point_b: Target point.
            tangent_vector_a: Tangent vector at point_a.

        Returns:
            Transported tangent vector at point_b.
        """
        return self.projection(point_b, tangent_vector_a)

    def zero_vector(self, point):
        """Create zero tangent vector at point.

        Args:
            point: Base point on manifold.

        Returns:
            Zero tangent vector.
        """
        return jnp.zeros(self._shape)

    def _normalize(self, array):
        """Normalize array with respect to mass matrix.

        In the multicomponent case each component is rescaled to Euclidean
        norm ``alphas[j]``.

        Args:
            array: Array to normalize.

        Returns:
            Normalized array.
        """
        if self.p is None:
            return array / jnp.sqrt(self._M_inner_product(array, array))
        else:
            # Multicomponent case: normalize each component separately
            norms = jnp.sqrt(inner_product_first_components(array, array))
            return array / norms * self.alphas
