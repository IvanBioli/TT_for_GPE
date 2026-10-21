"""Tests for the pytree arithmetic on tangent vectors in core.tangent.

Checks that the helper functions on lists of delta cores (sum, difference,
scaling, linear combination, inner product, norm, zeros) agree with the
corresponding core-by-core computations and with ``tree_math.Vector``
arithmetic, for tangent vectors of random TT tensors.
"""

import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import tree_math as tm
import ttax

from core.tangent import (
    add_deltas,
    lincomb_deltas,
    norm_deltas,
    scale_deltas,
    sub_deltas,
    vdot_deltas,
    zeroslike_deltas,
)


class TestTreeMapOps(unittest.TestCase):
    """Delta-core arithmetic vs core-by-core and tree_math results."""

    @classmethod
    def setUpClass(cls):
        """Build two delta representations of random rank-2 TT tensors."""
        cls.tt_sz = (10, 10, 10)
        cls.tt_rank = 2
        rng = jax.random.PRNGKey(0)

        def get_projected_vector(key):
            tensor = ttax.random.tensor(
                key, cls.tt_sz, tt_rank=cls.tt_rank, dtype=jnp.float64
            )
            vector = ttax.ops.tensor_to_vector(tensor)
            deltas = ttax.project(vector, vector)  # project onto itself for consistency
            return deltas, tm.Vector(deltas)

        key1, key2, key3 = jax.random.split(rng, 3)
        cls.d1, cls.tm_v1 = get_projected_vector(key1)
        cls.d2, cls.tm_v2 = get_projected_vector(key2)
        cls.alpha = 1.2345678
        cls.beta = 2.34567891

    def assertTreeAllClose(self, t1, t2, tol=1e-10):
        """Assert that two lists of arrays agree core by core up to ``tol``."""
        diffs = [jnp.linalg.norm(a - b) for a, b in zip(t1, t2)]
        self.assertTrue(np.all(np.array(diffs) < tol), f"Diffs: {diffs}")

    def test_sum(self):
        """Sum of deltas."""
        expected = [x + y for x, y in zip(self.d1, self.d2)]
        self.assertTreeAllClose(add_deltas(self.d1, self.d2), expected)
        self.assertTreeAllClose((self.tm_v1 + self.tm_v2).tree, expected)

    def test_diff(self):
        """Difference of deltas."""
        expected = [x - y for x, y in zip(self.d1, self.d2)]
        self.assertTreeAllClose(sub_deltas(self.d1, self.d2), expected)
        self.assertTreeAllClose((self.tm_v1 - self.tm_v2).tree, expected)

    def test_scale(self):
        """Scaling of deltas."""
        expected = [self.alpha * x for x in self.d1]
        self.assertTreeAllClose(scale_deltas(self.alpha, self.d1), expected)
        self.assertTreeAllClose((self.alpha * self.tm_v1).tree, expected)

    def test_lincomb(self):
        """Linear combination of deltas."""
        expected = [self.alpha * x + self.beta * y for x, y in zip(self.d1, self.d2)]
        self.assertTreeAllClose(
            lincomb_deltas(self.alpha, self.d1, self.beta, self.d2), expected
        )
        self.assertTreeAllClose(
            (self.alpha * self.tm_v1 + self.beta * self.tm_v2).tree, expected
        )

    def test_vdot(self):
        """Frobenius inner product of deltas."""
        expected = jnp.sum(
            jnp.array([jnp.vdot(x, y) for x, y in zip(self.d1, self.d2)])
        )
        self.assertAlmostEqual(
            float(vdot_deltas(self.d1, self.d2)), float(expected), places=10
        )
        self.assertAlmostEqual(
            float(self.tm_v1.conj() @ self.tm_v2), float(expected), places=10
        )

    def test_norm(self):
        """Frobenius norm of deltas."""
        expected = jnp.sum(
            jnp.sqrt(
                jnp.sum(jnp.array([jnp.linalg.vector_norm(x) ** 2 for x in self.d1]))
            )
        )
        self.assertAlmostEqual(float(norm_deltas(self.d1)), float(expected), places=10)

    def test_zeros(self):
        """Zero deltas with the same structure."""
        expected = [jnp.zeros_like(x) for x in self.d1]
        self.assertTreeAllClose(zeroslike_deltas(self.d1), expected)


if __name__ == "__main__":
    unittest.main()
