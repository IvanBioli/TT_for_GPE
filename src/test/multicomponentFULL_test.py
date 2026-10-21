"""Tests for the multicomponent case in the full (dense) format.

On a 2D harmonic-potential problem with p = 4 components (N=99, k=2,
beta=10, non-trivial interaction matrix K and component norms alphas),
checks the oblique manifold ``Sphere_Mass`` (component-wise normalization,
tangent vectors, retraction), the a_u bilinear form of
``FullEnergyFunctional`` against finite differences, the a_u Riemannian
gradient with and without the "S+V" preconditioner, and the symmetry of the
preconditioner. Here ``Sphere_Mass`` normalizes the components to norm
``alphas[i]``, i.e. squared norm ``alphas[i]**2``.
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

from core.energy import FullEnergyFunctional
from core.problem import setup_1Dmatrices, setup_matrices
from core.tensor_full import Sphere_Mass, inner_product_first_components


class MulticomponentFullTests(unittest.TestCase):
    """Tests for FullEnergyFunctional and Sphere_Mass with multicomponent support (K != None).

    These tests mirror the TT multicomponent tests in multicomponentTT_test.py but operate
    entirely in the full-tensor / dense format, using the Sphere_Mass manifold.
    """

    @classmethod
    def setUpClass(cls):
        """2D harmonic problem with p=4 components, the full energy and manifold."""
        config.update("jax_enable_x64", True)

        cls.dim = 2
        cls.N = 99
        cls.a, cls.b = -6, 6
        cls.k = 2
        cls.p = 4
        cls.beta = 10.0

        V_functions = [
            (lambda x: x**2, jnp.ones_like),
            (jnp.ones_like, lambda x: x**2),
        ]

        # 1D matrices
        M_1d, S_1d, msh_1d, _ = setup_1Dmatrices(cls.N, cls.a, cls.b, cls.k)
        M_1d_inv_sqrt = jnp.diag(1 / jnp.sqrt(jnp.diag(M_1d.todense())))
        S_1d_rescaled = M_1d_inv_sqrt @ S_1d @ M_1d_inv_sqrt

        # Single-component matrices (used for V_full)
        S_single, M_single, V_single, cls.msh_1d = setup_matrices(
            cls.dim, cls.N, cls.a, cls.b, V_functions, cls.k
        )

        # Full-format V: sum TT addends into a dense tensor
        Vfull = jnp.zeros([cls.N] * cls.dim)
        for i in range(V_single.shape[0]):
            Vfull = Vfull + ttax.full(ttax.ops.vector_to_tensor(V_single.batch_loc[i,]))
        cls.Vfull = Vfull

        # Kappa matrix and component norms
        kappa_matrix = np.ones((cls.p, cls.p))
        kappa_matrix[0, 0] = 0.5
        kappa_matrix[1, 1] = 5
        kappa_matrix[2, 2] = 10
        cls.kappa_matrix = jnp.array(kappa_matrix)
        cls.alphas = jnp.array([1.0, 5.0, 0.5, 2.0])

        # FullEnergyFunctional with multicomponent coupling K
        cls.E_full = FullEnergyFunctional(
            cls.dim,
            S_1d_rescaled,
            M_1d,
            Vfull,
            cls.beta,
            K=cls.kappa_matrix,
            p=cls.p,
            alphas=cls.alphas,
        )

        # Sphere_Mass manifold: shape = [N]*dim, p components with per-component norms
        cls.rng = jax.random.PRNGKey(42)
        cls.manifold = Sphere_Mass(
            shape=[cls.N] * cls.dim,
            mass=None,
            rng=cls.rng,
            p=cls.p,
            alphas=cls.alphas,
        )

        # V0: separable approximation of the potential for the S+V preconditioner.
        # The potential is V(x,y) = x^2 + y^2, so V0[i] = msh_1d^2 for each dimension.
        V0 = [jnp.array(msh_1d) ** 2] * cls.dim
        cls.E_full.SplusV_setup(V0)
        cls.E_full.setup_preconditioners()

    # ------------------------------------------------------------------
    # Manifold tests (analogous to test_1 … test_7 in multicomponentTT_test.py)
    # ------------------------------------------------------------------

    def test_1_random_point_component_norms(self):
        """Random point: ||u_i||^2 = alphas[i]^2 for all i."""
        print("\n[TEST] Random point component norms: Expect ||u_i||^2 = alphas[i]^2")
        u = self.manifold.random_point()
        norms_sq = inner_product_first_components(u, u)
        expected = self.alphas**2
        for i in range(self.p):
            print(
                f"\t Component {i}: ||u_i||^2 = {float(norms_sq[i]):.10f}, "
                f"alphas[{i}]^2 = {float(expected[i]):.10f}"
            )
            self.assertAlmostEqual(float(norms_sq[i]), float(expected[i]), places=10)

    def test_2_multiple_random_points_normalization(self):
        """All random points have correct per-component norms."""
        print("\n[TEST] Multiple random points normalization")
        n_points = 5
        all_pass = True
        for j in range(n_points):
            u = self.manifold.random_point()
            norms_sq = inner_product_first_components(u, u)
            expected = self.alphas**2
            errors = [
                abs(float(norms_sq[i]) - float(expected[i])) for i in range(self.p)
            ]
            max_err = max(errors)
            if max_err > 1e-10:
                all_pass = False
                print(f"\t Point {j}: FAIL — max error = {max_err:.2e}")
            else:
                print(
                    f"\t Point {j}: PASS — norms_sq = {[f'{float(norms_sq[i]):.6f}' for i in range(self.p)]}"
                )
        self.assertTrue(all_pass)

    def test_3_tangent_vector_orthogonal_to_point(self):
        """Tangent vector is per-component L2-orthogonal to the base point."""
        print(
            "\n[TEST] Tangent vector orthogonality: Expect <v_i, u_i>_L2 = 0 for all i"
        )
        u = self.manifold.random_point()
        v = self.manifold.random_tangent_vector(u)
        inners = inner_product_first_components(v, u)
        for i in range(self.p):
            print(f"\t Component {i}: <v_i, u_i>_L2 = {float(inners[i]):.2e}")
            self.assertAlmostEqual(float(inners[i]), 0.0, places=10)

    def test_4_retraction_preserves_normalization(self):
        """Retraction preserves per-component norms."""
        print("\n[TEST] Retraction preserves component norms")
        u = self.manifold.random_point()
        v = self.manifold.random_tangent_vector(u)
        u_new = self.manifold.retraction(u, 0.1 * v)
        norms_sq = inner_product_first_components(u_new, u_new)
        expected = self.alphas**2
        for i in range(self.p):
            print(
                f"\t Component {i}: ||u_new_i||^2 = {float(norms_sq[i]):.10f}, "
                f"expected = {float(expected[i]):.10f}"
            )
            self.assertAlmostEqual(float(norms_sq[i]), float(expected[i]), places=10)

    def test_5_tangent_vector_preserves_point_norms(self):
        """Creating a random tangent vector does not modify the base point."""
        print("\n[TEST] Tangent vector creation preserves point norms")
        u = self.manifold.random_point()
        norms_before = inner_product_first_components(u, u)
        _ = self.manifold.random_tangent_vector(u)
        norms_after = inner_product_first_components(u, u)
        for i in range(self.p):
            self.assertAlmostEqual(
                float(norms_before[i]), float(norms_after[i]), places=12
            )
            self.assertAlmostEqual(
                float(norms_after[i]), float(self.alphas[i] ** 2), places=10
            )

    # ------------------------------------------------------------------
    # Energy / gradient tests (analogous to test_92 / test_93 in TT file)
    # ------------------------------------------------------------------

    def test_6_Au_finite_difference(self):
        """bilinear_form(u, u, v) == directional derivative of E at u in direction v."""
        print("\n[TEST] Au finite-difference check")
        rng1 = jax.random.PRNGKey(123)
        u = self.manifold.random_point()

        rng2 = jax.random.PRNGKey(456)
        v_raw = jax.random.normal(rng2, u.shape)
        v_raw = v_raw / jnp.sqrt(jnp.vdot(v_raw, v_raw))

        eps = 1e-5
        dE_fd = (
            float(self.E_full(u + eps * v_raw)) - float(self.E_full(u - eps * v_raw))
        ) / (2 * eps)
        dE_analytic = float(self.E_full.bilinear_form(u, u, v_raw))
        rel_diff = abs(dE_fd - dE_analytic) / (abs(dE_analytic) + 1e-15)
        print(f"\t FD:       {dE_fd:.10f}")
        print(f"\t Analytic: {dE_analytic:.10f}")
        print(f"\t Rel diff: {rel_diff:.2e}")
        self.assertLess(rel_diff, 1e-4)

    def test_7_gradient_au_normality_and_identity(self):
        """riemannian_gradient_au: normality and gradient identity.

        Normality: <grad_au_i, u_i>_L2 = 0 for all i.
        Gradient identity: a_u(grad_au, v_N) = a_u(u, v_N) for v_N in T_u N.
        """
        print("\n[TEST] Riemannian gradient a_u: normality and gradient identity")

        u = self.manifold.random_point()

        # Build a tangent vector v_N ∈ T_u N using the manifold projection
        rng = jax.random.PRNGKey(99)
        v_raw = jax.random.normal(rng, u.shape)
        v_N = self.manifold.projection(u, v_raw)

        # Setup preconditioners for the energy functional (if needed)
        # self.E_full.SplusV_setup
        # self.E_full.setup_preconditioners()

        grad_au, info_cg = self.E_full.riemannian_gradient_au_withaux(
            u, maxiter=200, tol=1e-10
        )

        # Normality: per-component <grad_au_i, u_i>_L2 = 0
        comp_inners = inner_product_first_components(grad_au, u)
        print(f"\t Per-component <grad_au_i, u_i>_L2: {np.array(comp_inners)}")
        for i in range(self.p):
            self.assertAlmostEqual(
                float(comp_inners[i]),
                0.0,
                places=8,
                msg=f"Component {i} normality failed",
            )

        # Gradient identity: a_u(grad_au, v_N) = a_u(u, v_N)
        lhs = float(self.E_full.bilinear_form(u, grad_au, v_N))
        rhs = float(self.E_full.bilinear_form(u, u, v_N))
        print(f"\t a_u(grad_au, v_N) = {lhs:.10f}")
        print(f"\t a_u(u, v_N)       = {rhs:.10f}")
        print(f"\t Difference:         {lhs - rhs:.2e}")
        self.assertAlmostEqual(lhs, rhs, places=6)

    def test_91_SplusV_inv_symmetry_multicomponent(self):
        """SplusV_inv_fun is self-adjoint for multicomponent v — verifies it handles K≠None."""
        print(
            "\n[TEST] SplusV_inv_fun symmetry: <P(u), v> == <u, P(v)> for multicomponent tensors"
        )
        rng1 = jax.random.PRNGKey(555)
        rng2 = jax.random.PRNGKey(666)
        u = jax.random.normal(rng1, [self.N] * self.dim + [self.p])
        v = jax.random.normal(rng2, [self.N] * self.dim + [self.p])
        Pu = self.E_full.SplusV_inv_fun(u)
        Pv = self.E_full.SplusV_inv_fun(v)
        lhs = float(jnp.vdot(Pu, v))
        rhs = float(jnp.vdot(u, Pv))
        print(f"\t <P(u), v> = {lhs:.10f}")
        print(f"\t <u, P(v)> = {rhs:.10f}")
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_93_gradient_au_precondSplusV_normality_and_identity(self):
        """Preconditioned ('S+V') gradient: normality and gradient identity.

        Uses (S+V0)^{-1} as preconditioner inside riemannian_gradient_au_withaux.
        Requires SplusV_setup() to have been called (done in setUpClass).
        """
        print(
            "\n[TEST] Preconditioned (S+V) Riemannian gradient: normality and identity"
        )
        u = self.manifold.random_point()
        rng = jax.random.PRNGKey(77)
        v_raw = jax.random.normal(rng, u.shape)
        v_N = self.manifold.projection(u, v_raw)

        grad_au, _ = self.E_full.riemannian_gradient_au_withaux(
            u, maxiter=200, tol=1e-10, precond="S+V"
        )

        # Normality: per-component <grad_i, u_i>_L2 = 0
        comp_inners = inner_product_first_components(grad_au, u)
        print(f"\t Per-component <grad_SV_i, u_i>: {np.array(comp_inners)}")
        for i in range(self.p):
            self.assertAlmostEqual(
                float(comp_inners[i]),
                0.0,
                places=8,
                msg=f"Component {i} normality failed",
            )

        # Gradient identity: a_u(grad_au, v_N) = a_u(u, v_N)
        lhs = float(self.E_full.bilinear_form(u, grad_au, v_N))
        rhs = float(self.E_full.bilinear_form(u, u, v_N))
        print(f"\t a_u(grad_SV, v_N) = {lhs:.10f}")
        print(f"\t a_u(u, v_N)       = {rhs:.10f}")
        print(f"\t Difference:         {lhs - rhs:.2e}")
        self.assertAlmostEqual(lhs, rhs, places=6)


if __name__ == "__main__":
    unittest.main()
