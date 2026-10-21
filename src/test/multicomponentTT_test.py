"""Tests for the multicomponent case in the TT format.

On a 2D harmonic-potential problem with p = 4 components (N=99, k=2,
beta=10, non-trivial interaction matrix K and component masses alphas),
checks the manifold of subsection 'The manifold of multicomponent unit-norm
fixed TT-rank functions' (component-wise normalization to squared norm
``alphas[i]``, tangent vectors, retraction), the component-wise inner
products computed from the last core (subsection 'Implementation details'),
the projected operators with cached ``u * u``, and the a_u Riemannian
gradient computed by route II ('Riemannian gradient II: change of metric')
and via H_u.
"""

import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import ttax
from jax import config
from ttax.riemannian import deltas_to_tangent

from core.energy import tt_energyfactory
from core.fixed_tt_rank import (
    FixedRankPoint,
    FixedTTRankEmbedded,
    partial_flat_inner_last,
)
from core.problem import setup_matrices, setup_multicomponent_matrices
from core.tangent import zeroslike_deltas
from core.tt_utils import partial_flat_inner_last_tangent_deltas


class MulticomponentTTManifoldTests(unittest.TestCase):
    """Tests for FixedTTRankEmbedded manifold with multicomponent case (p > 1)."""

    @classmethod
    def setUpClass(cls):
        """2D harmonic problem with p=4 components, manifolds and TT energy."""
        config.update("jax_enable_x64", True)

        # Problem parameters
        cls.dim = 2
        cls.N = 99
        cls.a, cls.b = -6, 6
        cls.k = 2
        cls.p = 4  # Number of components
        cls.tt_rank = 2

        # Define potential
        V_functions = [
            (lambda x: x**2, jnp.ones_like),
            (jnp.ones_like, lambda x: x**2),
        ]

        # Single component matrices
        cls.S_single, cls.M_single, cls.V_single, cls.msh_1d = setup_matrices(
            cls.dim, cls.N, cls.a, cls.b, V_functions, cls.k
        )

        # Multicomponent kappa matrix
        kappa_matrix = np.ones((cls.p, cls.p))
        kappa_matrix[0, 0] = 0.5
        kappa_matrix[1, 1] = 5
        kappa_matrix[2, 2] = 10
        kappa_matrix = jnp.array(kappa_matrix)

        # Multicomponent matrices
        cls.S, cls.M, cls.V_vector, cls.K = setup_multicomponent_matrices(
            cls.p, cls.S_single, cls.M_single, cls.V_single, kappa_matrix
        )

        # Define non-trivial alphas for component masses
        cls.alphas = jnp.array([1.0, 5.0, 0.5, 2.0])

        # Manifold setup
        cls.tt_sz = [cls.N] * cls.dim + [cls.p]
        cls.tt_rank_list = [cls.tt_rank] * (cls.dim - 1) + [min(cls.tt_rank, cls.p)]
        cls.tt_rank_list_complete = [1] + cls.tt_rank_list + [1]
        cls.rng = jax.random.PRNGKey(42)

        cls.manifold = FixedTTRankEmbedded(
            cls.tt_sz, cls.tt_rank_list, "identity", cls.rng, cls.p, alphas=cls.alphas
        )
        cls.manifold_mr = FixedTTRankEmbedded(
            cls.tt_sz, cls.tt_rank_list, None, cls.rng, None, None
        )

        # Energy functions
        cls.beta = 10.0
        cls.functions = tt_energyfactory(
            cls.S, cls.M, cls.V_vector, cls.beta, K=cls.K, alphas=cls.alphas
        )

    @classmethod
    def random_unit_vector(cls, rank):
        """Random point of the multicomponent manifold with the given rank."""
        cls.rng, key = jax.random.split(cls.rng)
        tt_sz = [cls.N] * cls.dim + [cls.p]
        tt_rank_list = [rank] * (cls.dim - 1) + [min(rank, cls.p)]
        manifold = FixedTTRankEmbedded(
            tt_sz, tt_rank_list, "identity", cls.rng, cls.p, alphas=cls.alphas
        )
        return manifold.random_point().u

    @classmethod
    def random_tangent_vector(cls, rank, u):
        """Random tangent vector at u (component-wise L2-orthogonal to u), embedded."""
        cls.rng, key = jax.random.split(cls.rng)
        tt_sz = [cls.N] * cls.dim + [cls.p]
        tt_rank_list = [rank] * (cls.dim - 1) + [min(rank, cls.p)]
        manifold = FixedTTRankEmbedded(
            tt_sz, tt_rank_list, "identity", key, cls.p, alphas=cls.alphas
        )
        return deltas_to_tangent(
            manifold.random_tangent_vector(FixedRankPoint(u)).tree, u
        )

    @classmethod
    def random_tangent_vector_sphere(cls, rank, u):
        """Same as ``random_tangent_vector``."""
        cls.rng, key = jax.random.split(cls.rng)
        tt_sz = [cls.N] * cls.dim + [cls.p]
        tt_rank_list = [rank] * (cls.dim - 1) + [min(rank, cls.p)]
        manifold = FixedTTRankEmbedded(
            tt_sz, tt_rank_list, "identity", key, cls.p, alphas=cls.alphas
        )
        return deltas_to_tangent(
            manifold.random_tangent_vector(FixedRankPoint(u)).tree, u
        )

    def _get_component(self, u, i):
        """Extract the i-th component from a multicomponent TT tensor."""
        return ttax.ops.tensor_to_vector(ttax.ops.vector_to_tensor(u)[:, :, i])

    def _component_M_norm_squared(self, u, i):
        """Compute the M-norm squared of the i-th component."""
        u_i = self._get_component(u, i)
        return ttax.flat_inner(u_i, u_i)

    def test_1_random_point_component_norms(self):
        """Test that the M-norm of each component equals the corresponding alpha."""
        print("\n[TEST] Random point component M-norms: Expect norms equal to alphas")

        point = self.manifold.random_point()
        u = point.u

        for i in range(self.p):
            norm_sq = self._component_M_norm_squared(u, i)
            expected = self.alphas[i]
            print(
                f"\t Component {i}: ‖u_i‖_M² = {norm_sq:.10f}, alpha[{i}] = {expected}"
            )
            self.assertAlmostEqual(float(norm_sq), float(expected), places=10)

    def test_2_partial_flat_inner_last(self):
        """Test that partial_flat_inner_last computes component-wise M-inner products correctly."""
        print("\n[TEST] partial_flat_inner_last: Expect component-wise inner products")

        u = self.manifold.random_point().u

        # Compute using partial_flat_inner_last
        Mu = u
        partial_inner = partial_flat_inner_last(u, Mu)

        # Compute component-wise for verification
        component_inner = []
        for i in range(self.p):
            inner_i = self._component_M_norm_squared(u, i)
            component_inner.append(inner_i)

        component_inner = jnp.array(component_inner)

        print(f"\t partial_flat_inner_last result: {partial_inner}")
        print(f"\t Component-wise computation:     {component_inner}")

        diff = jnp.linalg.norm(partial_inner - component_inner) / jnp.linalg.norm(
            component_inner
        )
        print(f"\t Difference norm: {diff}")
        self.assertLess(float(diff), 1e-10)

    def test_21_partial_flat_inner_last_tangent_deltas(self):
        """Test that partial_flat_inner_last_tangent_deltas computes component-wise M-inner products correctly."""
        print(
            "\n[TEST] partial_flat_inner_last_tangent_deltas: Expect component-wise inner products"
        )

        u = self.manifold_mr.random_point().u
        u_deltas = ttax.project(u, u)

        v = self.random_tangent_vector(self.tt_rank, u)
        deltas_v = ttax.project(v, u)

        partial_inner_deltas = partial_flat_inner_last_tangent_deltas(
            u_deltas, deltas_v
        )
        partial_inner = partial_flat_inner_last(u, v)

        print(f"\t partial_flat_inner_last result: {partial_inner}")
        print(
            f"\t partial_flat_inner_last_tangent_deltas result: {partial_inner_deltas}"
        )

        diff = jnp.linalg.norm(partial_inner - partial_inner_deltas)
        rel_diff = jnp.linalg.norm(
            partial_inner - partial_inner_deltas
        ) / jnp.linalg.norm(partial_inner_deltas)
        print(f"\t Difference norm: {diff}")
        print(f"\t Relative difference norm: {rel_diff}")
        self.assertLess(float(diff), 1e-10)

    def test_3_tangent_vector_preserves_point_norms(self):
        """Test that creating a random tangent vector doesn't modify the M-norms of the point."""
        print(
            "\n[TEST] Tangent vector creation preserves point M-norms: Expect unchanged norms"
        )

        point = self.manifold.random_point()
        u = point.u

        # Record M-norms before creating tangent vector
        norms_before = [
            float(self._component_M_norm_squared(u, i)) for i in range(self.p)
        ]

        # Create random tangent vector (this should not modify the point)
        v = self.manifold.random_tangent_vector(point)

        # Record M-norms after
        norms_after = [
            float(self._component_M_norm_squared(u, i)) for i in range(self.p)
        ]

        print(f"\t M-norms before tangent creation: {norms_before}")
        print(f"\t M-norms after tangent creation:  {norms_after}")

        for i in range(self.p):
            self.assertAlmostEqual(norms_before[i], norms_after[i], places=10)
            self.assertAlmostEqual(norms_after[i], float(self.alphas[i]), places=10)

    def test_4_tangent_vector_M_orthogonal_to_point(self):
        """Test that each component of a tangent vector is M-orthogonal to the corresponding component of the point."""
        print(
            "\n[TEST] Tangent vector M-orthogonality: Expect (v_i, u_i)_M = 0 for all i"
        )

        point = self.manifold.random_point()
        u = point.u

        # Create random tangent vector
        v_delta = self.manifold.random_tangent_vector(point)
        v = deltas_to_tangent(v_delta.tree, u)

        for i in range(self.p):
            u_i = self._get_component(u, i)
            v_i = self._get_component(v, i)
            inner = ttax.flat_inner(u_i, v_i)
            print(f"\t Component {i}: (u_i, v_i)_M = {inner:.2e}")
            self.assertAlmostEqual(float(inner), 0.0, places=10)

    def test_5_partial_flat_inner_with_tangent(self):
        """Test partial_flat_inner_last with a tangent vector embedded in TT format."""
        print(
            "\n[TEST] partial_flat_inner_last with tangent: Expect (u_i, v_i)_M = 0 for all i"
        )

        point = self.manifold.random_point()
        u = point.u

        # Create random tangent vector and embed it
        v_delta = self.manifold.random_tangent_vector(point)
        v = deltas_to_tangent(v_delta.tree, u)

        # Compute using partial_flat_inner_last
        Mv = v
        partial_inner_uv = partial_flat_inner_last(u, Mv)

        print(f"\t partial_flat_inner_last(u, M*v): {partial_inner_uv}")

        for i in range(self.p):
            self.assertAlmostEqual(float(partial_inner_uv[i]), 0.0, places=10)

    def test_6_multiple_random_points_normalization(self):
        """Test that multiple random points all have correct component norms."""
        print(
            "\n[TEST] Multiple random points normalization: All should have correct alphas"
        )

        n_points = 5
        all_pass = True

        for j in range(n_points):
            point = self.manifold.random_point()
            u = point.u
            norms = [self._component_M_norm_squared(u, i) for i in range(self.p)]
            errors = [
                abs(float(norms[i]) - float(self.alphas[i])) for i in range(self.p)
            ]
            max_error = max(errors)

            if max_error > 1e-10:
                all_pass = False
                print(f"\t Point {j}: FAIL - max error = {max_error:.2e}")
            else:
                print(f"\t Point {j}: PASS - norms = {[f'{n:.6f}' for n in norms]}")

        self.assertTrue(all_pass)

    def test_7_retraction_preserves_normalization(self):
        """Test that retraction preserves the component-wise M-norms."""
        print(
            "\n[TEST] Retraction preserves M-norms: After retraction, norms should equal alphas"
        )

        point = self.manifold.random_point()
        v = self.manifold.random_tangent_vector(point)

        # Perform retraction with a small step
        step_size = 0.1
        scaled_v = step_size * v
        new_point = self.manifold.retraction(point, scaled_v)

        # Check component norms of the new point
        for i in range(self.p):
            norm_sq = self._component_M_norm_squared(new_point.u, i)
            expected = self.alphas[i]
            print(
                f"\t Component {i}: ‖u_new_i‖_M² = {norm_sq:.10f}, alpha[{i}] = {expected}"
            )
            self.assertAlmostEqual(float(norm_sq), float(expected), places=10)

    def test_8_PAP_Mr_deltas_u2(self):
        """Test PAP_Mr_deltas_u2: consistency with PAP_Mr_deltas and self-adjointness."""
        print(
            "\n[TEST] PAP_Mr_deltas_u2: Expect consistency with PAP_Mr_deltas and self-adjointness"
        )

        PAP_Mr_deltas = self.functions["PAP_Mr_deltas"]
        PAP_Mr_deltas_u2 = self.functions["PAP_Mr_deltas_u2"]

        point = self.manifold.random_point()
        u = point.u
        u2 = ttax.multiply(u, u)

        v_delta = self.manifold.random_tangent_vector(point)
        w_delta = self.manifold.random_tangent_vector(point)
        v_deltas = v_delta.tree
        w_deltas = w_delta.tree

        # Test consistency: PAP_Mr_deltas_u2(v, u, u2) == PAP_Mr_deltas(v, u)
        result_u2 = PAP_Mr_deltas_u2(v_deltas, u, u2)
        result = PAP_Mr_deltas(v_deltas, u)
        diff = deltas_to_tangent(result_u2, u) - deltas_to_tangent(result, u)
        rel_diff = ttax.ops.norm(diff) / ttax.ops.norm(deltas_to_tangent(result, u))
        print(f"\t Consistency (rel. diff): {rel_diff:.2e}")
        self.assertLess(float(rel_diff), 1e-10)

        # Test self-adjointness: <PAP_Mr(v), w> = <v, PAP_Mr(w)>
        PAP_v = deltas_to_tangent(PAP_Mr_deltas_u2(v_deltas, u, u2), u)
        PAP_w = deltas_to_tangent(PAP_Mr_deltas_u2(w_deltas, u, u2), u)
        v_emb = deltas_to_tangent(v_deltas, u)
        w_emb = deltas_to_tangent(w_deltas, u)
        lhs = ttax.flat_inner(PAP_v, w_emb)
        rhs = ttax.flat_inner(v_emb, PAP_w)
        print(f"\t <PAP_Mr(v), w> = {lhs:.10f}")
        print(f"\t <v, PAP_Mr(w)> = {rhs:.10f}")
        self.assertAlmostEqual(float(lhs), float(rhs), places=8)

    def test_9_PAP_N(self):
        """Test PAP_N_deltas: CG convergence, per-component normality, and
        <PAP_N(v), w> = a_u(v, w) for w in tangent space of N."""
        print("\n[TEST] PAP_N: Expect CG convergence and (x_i, u_i)_M = 0 for all i")

        PAP_N_deltas = self.functions["PAP_N_deltas"]
        cg_PAP_N = self.functions["cg_PAP_N"]

        point = self.manifold.random_point()
        u = point.u
        u2 = ttax.multiply(u, u)

        # Use a manifold tangent vector (already in N-sphere tangent space)
        v_delta = self.manifold.random_tangent_vector(point)
        v_deltas = v_delta.tree

        # Build RHS: b = PAP_N * v
        b = PAP_N_deltas(v_deltas, u)
        x0 = zeroslike_deltas(b)

        # Solve PAP_N * x = b with CG
        x_result = cg_PAP_N(u=u, u2=u2, b=b, x0=x0, maxiter=200, tol=1e-10)

        # Check relative residual
        v_emb = deltas_to_tangent(v_deltas, u)
        x_emb = deltas_to_tangent(x_result, u)
        res = v_emb - x_emb
        rel_error = ttax.ops.norm(res) / ttax.ops.norm(deltas_to_tangent(b, u))
        print(f"\t Relative residual: {rel_error:.2e}")
        self.assertLess(float(rel_error), 1e-10)

        # Check per-component normality: (x_i, u_i)_M = 0 for all i
        component_inners = partial_flat_inner_last(x_emb, u)
        print(f"\t Component inners (x_i, u_i)_M: {component_inners}")
        for i in range(self.p):
            self.assertAlmostEqual(float(component_inners[i]), 0.0, places=8)

        # Check that <PAP_N(v), w> = a_u(v, w) for w in tangent space of N
        w_delta = self.manifold.random_tangent_vector(point)
        w_emb = deltas_to_tangent(w_delta.tree, u)
        PAP_v = deltas_to_tangent(PAP_N_deltas(v_deltas, u), u)
        bilinear_form = self.functions["bilinear_form"]
        lhs = ttax.flat_inner(PAP_v, w_emb)
        rhs = bilinear_form(u, v_emb, w_emb)
        print(f"\t <PAP_N(v), w> = {lhs:.10f}")
        print(f"\t a_u(v, w) = {rhs:.10f}")
        print(f"\t Difference = {lhs - rhs:.2e}")
        self.assertAlmostEqual(float(lhs), float(rhs), places=6)

    def test_91_PAP_N_cached(self):
        """Test PAP_N_deltas_u2: consistency with PAP_N_deltas when u2 is pre-computed."""
        print(
            "\n[TEST] PAP_N_cached: Expect PAP_N_deltas_u2(v, u, u*u) == PAP_N_deltas(v, u)"
        )

        PAP_N_deltas = self.functions["PAP_N_deltas"]
        PAP_N_deltas_u2 = self.functions["PAP_N_deltas_u2"]

        point = self.manifold.random_point()
        u = point.u
        u2 = ttax.multiply(u, u)

        v_delta = self.manifold.random_tangent_vector(point)
        v_deltas = v_delta.tree

        # Compute with pre-cached u2 and without
        result_cached = PAP_N_deltas_u2(v_deltas, u, u2)
        result_noncached = PAP_N_deltas(v_deltas, u)

        result_cached_emb = deltas_to_tangent(result_cached, u)
        result_noncached_emb = deltas_to_tangent(result_noncached, u)
        diff = result_cached_emb - result_noncached_emb
        rel_diff = ttax.ops.norm(diff) / ttax.ops.norm(result_noncached_emb)
        print(f"\t Consistency (rel. diff): {rel_diff:.2e}")
        self.assertLess(float(rel_diff), 1e-10)

        # Additionally verify the cached result is in the N-sphere tangent space
        component_inners = partial_flat_inner_last(result_cached_emb, u)
        print(f"\t Component inners of PAP_N result: {component_inners}")
        for i in range(self.p):
            self.assertAlmostEqual(float(component_inners[i]), 0.0, places=8)

    def test_92_gradE_au_PAP_N(self):
        """Route II gradient: (g, u) = 0 and a_u(g, v) = a_u(u, v) on the tangent space."""
        print("\n[TEST] Riemannian gradient in the au-metric: Expect equality")
        bilinear_form = self.functions["bilinear_form"]

        u = self.random_unit_vector(self.tt_rank)
        vN = self.random_tangent_vector_sphere(3 * self.tt_rank, u)
        lhs = bilinear_form(u, u, vN)

        PAPNgrad = self.functions["gradE_au_PAP_N"]

        grad_PAP_N = PAPNgrad(u, maxiter=200, kappa=1e-10, theta=0.0)
        grad_PAP_N = deltas_to_tangent(grad_PAP_N, u)
        innerM = ttax.flat_inner(u, grad_PAP_N)
        rhs = bilinear_form(u, grad_PAP_N, vN)
        print(f"\t\tPAP_N gradient")
        print(f"\t\t\t (grad_au_E(u),u)_L2:\t\t{innerM}")
        print(f"\t\t\t a_u(grad_au_E(u), vN)_L2:\t{rhs}")
        print(f"\t\t\t a_u(u, vN)_L2:\t\t\t{lhs}")
        self.assertAlmostEqual(innerM, 0.0, places=10)
        self.assertAlmostEqual(lhs, rhs, places=6)

        print("Testing JIT compilation...")
        PAPNgrad_jit = jax.jit(PAPNgrad)
        PAPNgrad_jit(u, maxiter=100, kappa=1e-10, theta=0.0)
        print("...successful")

    def test_93_gradE_au_Hu(self):
        """Test Riemannian gradient computed with gradE_au_Hu: normality and gradient identity."""
        print("\n[TEST] Riemannian gradient in the au-metric via Hu: Expect equality")
        bilinear_form = self.functions["bilinear_form"]

        u = self.random_unit_vector(self.tt_rank)
        vN = self.random_tangent_vector_sphere(3 * self.tt_rank, u)
        lhs = bilinear_form(u, u, vN)

        HuGrad = self.functions["gradE_au_Hu"]

        grad_Hu = HuGrad(u, maxiter=200, tol=1e-10)
        grad_Hu = deltas_to_tangent(grad_Hu, u)
        innerM = ttax.flat_inner(u, grad_Hu)
        rhs = bilinear_form(u, grad_Hu, vN)
        print(f"\t\tHu gradient")
        print(f"\t\t\t (grad_au_E(u),u)_L2:\t\t{innerM}")
        print(f"\t\t\t a_u(grad_au_E(u), vN)_L2:\t{rhs}")
        print(f"\t\t\t a_u(u, vN)_L2:\t\t\t{lhs}")
        self.assertAlmostEqual(innerM, 0.0, places=9)
        self.assertAlmostEqual(lhs, rhs, places=6)

        print("Testing JIT compilation...")
        HuGrad_jit = jax.jit(HuGrad)
        HuGrad_jit(u, maxiter=100, tol=1e-10)
        print("...successful")


if __name__ == "__main__":
    unittest.main()
