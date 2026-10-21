"""Tests for the TT energy functions of core.energy.tt_energyfactory.

On a small 3D harmonic-potential problem (N=15, k=2, beta=1000, TT rank 2),
checks the identities underlying Sections 'Discretization by spectral
elements' and 'The $a_u$ Riemannian gradient': the gradient/bilinear-form
relation a_u(u, v) = E'(u)[v], the symmetry of A_u, the agreement of the
projected operator P A_u P computed explicitly and by automatic
differentiation (paragraph 'Automatic differentiation of the TT cores.'), the
CG solves with P A_u P, the a_u-orthogonal projections and the operator H_u.
A second test case checks that
``EnergyHighOrder`` (Remark 'Evaluating the energy with a higher-order
quadrature rule') accepts the potential as a sum of terms.
"""

import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

# jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
import numpy as np
import ttax
from ttax.riemannian import deltas_to_tangent, tangent_to_deltas

from core.energy import EnergyHighOrder, tt_energyfactory
from core.problem import setup_1Dmatrices, setup_matrices
from core.tangent import cg_tree_math, zeroslike_deltas


class TTEnergyFunctionalTests(unittest.TestCase):
    """Identities satisfied by the TT energy, bilinear form and gradients."""

    @classmethod
    def setUpClass(cls):
        """Assemble the 3D harmonic problem and the TT energy functions."""
        config = jax.config
        config.update("jax_enable_x64", True)

        cls.dim, cls.N, cls.beta = 3, 15, 1000
        cls.tt_sz = [cls.N] * cls.dim
        cls.tt_rank = 2
        a, b = -6, 6
        V_functions = [
            (lambda x: x**2, jnp.ones_like, jnp.ones_like),
            (jnp.ones_like, lambda x: x**2, jnp.ones_like),
            (jnp.ones_like, jnp.ones_like, lambda x: x**2),
        ]
        S, M, V_vector, _ = setup_matrices(cls.dim, cls.N, a, b, V_functions, k=2)

        cls.S = S
        cls.M = M
        cls.V_vector = V_vector
        cls.functions = tt_energyfactory(cls.S, cls.M, cls.V_vector, cls.beta)
        cls.rng = jax.random.PRNGKey(42)

    @classmethod
    def random_unit_vector(cls, rank):
        """Random TT vector of the given rank with unit Frobenius norm."""
        cls.rng, key = jax.random.split(cls.rng)
        u = ttax.random.tensor(key, cls.tt_sz, tt_rank=rank, dtype=jnp.float64)
        u = ttax.ops.tensor_to_vector(u)
        norm = jnp.sqrt(ttax.flat_inner(u, u))
        return u * (1 / norm)

    @classmethod
    def random_tangent_vector(cls, rank, u):
        """Random vector of the tangent space of the fixed-rank manifold at u."""
        cls.rng, key = jax.random.split(cls.rng)
        v = ttax.random.tensor(key, cls.tt_sz, tt_rank=rank, dtype=jnp.float64)
        return deltas_to_tangent(ttax.project(ttax.ops.tensor_to_vector(v), u), u)

    @classmethod
    def random_tangent_vector_sphere(cls, rank, u):
        """Random tangent vector at u, L2-orthogonal to u."""
        cls.rng, key = jax.random.split(cls.rng)
        v = ttax.random.tensor(key, cls.tt_sz, tt_rank=rank, dtype=jnp.float64)
        v = ttax.ops.tensor_to_vector(v)
        v = v - ttax.flat_inner(u, v) * u
        return deltas_to_tangent(ttax.project(v, u), u)

    def test_1_gradient_vs_bilinear_form(self):
        """Frobenius Riemannian gradient: <grad E(u), v> = a_u(u, v) on T_u M_r."""
        print("\n[TEST] Riemannian gradient vs Bilinear Form: Expect equality")
        a = self.functions["bilinear_form"]
        riemannian_gradient = self.functions["riemannian_gradient_embedded"]
        u = self.random_unit_vector(self.tt_rank)
        gradE_deltas = riemannian_gradient(u)
        gradE = deltas_to_tangent(gradE_deltas, u)

        v = self.random_tangent_vector(self.tt_rank, u)
        lhs = a(u, u, v)
        rhs = ttax.flat_inner(gradE, v)
        print(f"\ta(u, u, v):\t{lhs}")
        print(f"\t<gradE(u), v>:\t{rhs}")
        self.assertAlmostEqual(lhs, rhs, places=10)

        print("Testing FUSION and JIT compilation...")
        riemannian_gradient = jax.jit(ttax.fuse(riemannian_gradient))
        gradE_deltas = riemannian_gradient(u)
        print("...successful")

    def test_2_bilinear_vs_matvec_symmetry(self):
        """a_u(v, w) = <w, A_u v> = <v, A_u w>."""
        print("\n[TEST] Bilinear Form vs Matrix-Vector product: Expect equality")
        a = self.functions["bilinear_form"]
        Au_mv = self.functions["Au_mv"]

        u = self.random_unit_vector(self.tt_rank)
        v = self.random_tangent_vector(self.tt_rank, u)
        w = self.random_tangent_vector(self.tt_rank, u)

        lhs = a(u, v, w)
        rhs1 = ttax.flat_inner(w, Au_mv(v, u))
        rhs2 = ttax.flat_inner(v, Au_mv(w, u))
        print(f"\ta(u,v,w):\t{lhs}")
        print(f"\t<w, A_u*v>:\t{rhs1}")
        print(f"\t<v, A_u*w>:\t{rhs2}")
        self.assertAlmostEqual(lhs, rhs1, places=10)
        self.assertAlmostEqual(lhs, rhs2, places=10)

        print("Testing FUSION and JIT compilation...")
        a_jit = jax.jit(ttax.fuse(a))
        a_jit(u, v, w)
        print("...successful")

    def test_3_PAP_explicit_vs_gradients(self):
        """P A_u P computed explicitly equals its autodiff version on tangent vectors."""
        print("\n[TEST] PAP Explicit vs Gradients: Expect all differences ≈ 0")
        u = self.random_unit_vector(self.tt_rank)
        v = self.random_tangent_vector(self.tt_rank, u)
        PAP_Mr_explicit = self.functions["PAP_Mr_explicit"]
        PA_Mr_tang = self.functions["PA_Mr_tang"]
        PAP_explicit = PAP_Mr_explicit(v, u)
        PAP_grad = PA_Mr_tang(v, u)

        diff = ttax.ops.norm(PAP_explicit - PAP_grad)
        print(f"\t‖explicit - PAP_grad‖:\t\t{diff}")
        self.assertAlmostEqual(diff, 0.0, places=10)

        print("Testing JIT compilation...")

        PA_Mr_tangent_jit = jax.jit(PA_Mr_tang)
        PA_Mr_tangent_jit(v, u)

        PA_Mr = self.functions["PA_Mr"]
        PA_Mr_jit = jax.jit(PA_Mr)
        PA_Mr_jit(v, u)

        # FIXME: Fusion does not work with PAP_Mr_deltas because v_deltas is not a TT vector.
        PAP_Mr_deltas = self.functions["PAP_Mr_deltas"]
        v_deltas = ttax.project(v, u)
        PAP_Mr_deltas_jit = jax.jit(PAP_Mr_deltas)
        PAP_Mr_deltas_jit(v_deltas, u)
        print("...successful")

    def test_4_PAP_with_non_tangent_vector(self):
        """For a non-tangent input, the autodiff P A_u P matches only after projection."""
        print("\n[TEST] PAP with Non-Tangent Vector: Only PAP_grad_proj should match")
        self.rng, key = jax.random.split(self.rng)
        u = self.random_unit_vector(self.tt_rank)
        v_nt = ttax.random.tensor(key, self.tt_sz, 3 * self.tt_rank, dtype=jnp.float64)
        v_nt = ttax.ops.tensor_to_vector(v_nt)

        PAP_Mr_explicit = self.functions["PAP_Mr_explicit"]
        PA_Mr_tang = self.functions["PA_Mr_tang"]
        explicit = PAP_Mr_explicit(v_nt, u)
        PAP_grad_proj = PA_Mr_tang(deltas_to_tangent(ttax.project(v_nt, u), u), u)
        PAP_grad_unproj = PA_Mr_tang(v_nt, u)

        norm_diff_proj = ttax.ops.norm(explicit - PAP_grad_proj)
        norm_diff_unproj = ttax.ops.norm(explicit - PAP_grad_unproj)

        print(f"\t‖explicit - PAP_grad‖:\t\t{norm_diff_unproj}")
        print(f"\t‖explicit - PAP_grad_proj‖:\t{norm_diff_proj}")
        self.assertAlmostEqual(norm_diff_proj, 0.0, places=10)
        self.assertGreater(norm_diff_unproj, 1)

    def test_5_PAP_Mr_linear_system_CG_solver(self):
        """CG with P A_u P recovers v from b = P A_u P v."""
        print("\n[TEST] PAP Linear Solve (CG): Relative residual error should be ≈ 0")
        u = self.random_unit_vector(self.tt_rank)
        v = self.random_tangent_vector(3 * self.tt_rank, u)
        v_delta = tangent_to_deltas(v)
        PAP_Mr_deltas = self.functions["PAP_Mr_deltas"]
        b = PAP_Mr_deltas(v_delta, u)
        x0 = zeroslike_deltas(v_delta)

        A = lambda v: PAP_Mr_deltas(v, u)
        x_result = cg_tree_math(A, b, x0, M=lambda x: x, maxiter=200, tol=1e-10)
        res = deltas_to_tangent(v_delta, u) - deltas_to_tangent(x_result, u)
        rel_error = ttax.ops.norm(res) / ttax.ops.norm(deltas_to_tangent(b, u))
        print(f"\tRelative error:\t{rel_error}")
        self.assertLess(rel_error, 1e-10)

        cg_PAP_Mr = self.functions["cg_PAP_Mr"]
        x_result, _ = cg_PAP_Mr(
            u=u, u2=ttax.multiply(u, u), b=b, x0=x0, maxiter=200, tol=1e-10
        )
        res = deltas_to_tangent(v_delta, u) - deltas_to_tangent(x_result, u)
        rel_error = ttax.ops.norm(res) / ttax.ops.norm(deltas_to_tangent(b, u))
        print(f"\tRelative error:\t{rel_error}")
        self.assertLess(rel_error, 1e-10)

    def test_6_project_au_Mr(self):
        """project_au_Mr is a_u-orthogonal: a_u(z, v) = a_u(P z, v) on T_u M_r."""
        print("\n[TEST] project_au_Mr: Expect a_u(z,v) ≈ a_u(project_au_Mr(z),v)")
        u = self.random_unit_vector(self.tt_rank)
        v = self.random_tangent_vector(3 * self.tt_rank, u)
        z = self.random_unit_vector(3 * self.tt_rank)

        project_au_Mr = self.functions["project_au_Mr"]
        bilinear_form = self.functions["bilinear_form"]

        proj_au_Mr_z_deltas = project_au_Mr(z, u)
        proj_au_Mr_z = deltas_to_tangent(proj_au_Mr_z_deltas, u)
        lhs = bilinear_form(u, z, v)
        rhs = bilinear_form(u, proj_au_Mr_z, v)
        print(f"\ta_u(z, v):\t{lhs}")
        print(f"\ta_u(proj(z),v):\t{rhs}")
        self.assertAlmostEqual(lhs, rhs, places=7)

        print("Testing JIT compilation...")

        project_au_Mr_jit = jax.jit(project_au_Mr)
        project_au_Mr_jit(z, u)
        print("...successful")

    def test_7_Hu(self):
        """H_u satisfies a_u(H_u v, w) = (v, w) for w in T_u M_r."""
        print("\n[TEST] Hu: Expect a_u(H_u(v),w) ≈ (v, w)_L2")
        u = self.random_unit_vector(self.tt_rank)
        v = self.random_tangent_vector(3 * self.tt_rank, u)
        w = self.random_tangent_vector(3 * self.tt_rank, u)

        Hu = self.functions["Hu"]
        bilinear_form = self.functions["bilinear_form"]

        Hu_v_deltas, _, _ = Hu(v, u)
        Hu_v = deltas_to_tangent(Hu_v_deltas, u)
        lhs = ttax.flat_inner(v, w)
        rhs = bilinear_form(u, Hu_v, w)
        print(f"\t\t(v, w)_L2:\t{lhs}")
        print(f"\t\ta_u(Hu(v),w):\t{rhs}")
        self.assertAlmostEqual(lhs, rhs, places=7)

        print("Testing JIT compilation...")
        Hu_jit = jax.jit(Hu)
        Hu_jit(v, u)
        print("...successful")

    def test_8_project_au_N(self):
        """project_au_N maps into T_u N_r and is a_u-orthogonal."""
        print(
            "\n[TEST] project_au_N: Expect (project_au_N(z), u)_L2 ≈ 0 and a_u(z,vN) ≈ a_u(project_au_N(z),vN)"
        )
        u = self.random_unit_vector(self.tt_rank)
        vN = self.random_tangent_vector(3 * self.tt_rank, u)
        vN = vN - ttax.flat_inner(u, vN) * u
        z = self.random_unit_vector(3 * self.tt_rank)

        project_au_N = self.functions["project_au_N"]
        bilinear_form = self.functions["bilinear_form"]

        proj_au_N_z_deltas = project_au_N(z, u)
        proj_au_N_z = deltas_to_tangent(proj_au_N_z_deltas, u)
        lhs = bilinear_form(u, z, vN)
        rhs = bilinear_form(u, proj_au_N_z, vN)
        innerM = ttax.flat_inner(u, proj_au_N_z)
        print(f"\t\t(proj(z),u)_L2:\t\t{innerM}")
        print(f"\t\ta_u(z, vN):\t\t{lhs}")
        print(f"\t\ta_u(proj(z),vN):\t{rhs}")
        self.assertAlmostEqual(lhs, rhs, places=7)
        self.assertAlmostEqual(innerM, 0.0, places=10)

        print("Testing JIT compilation...")
        project_au_N_jit = jax.jit(project_au_N)
        project_au_N_jit(z, u)
        print("...successful")



class EnergyHighOrderPotentialCompositionTests(unittest.TestCase):
    """EnergyHighOrder with a potential given as a sum of terms."""

    @classmethod
    def setUpClass(cls):
        """Small 2D grid (N=7, k=1) and the constant tensor u = 1."""
        config = jax.config
        config.update("jax_enable_x64", True)

        cls.dim = 2
        cls.N = 7
        cls.k = 1
        cls.beta = 0.5
        cls.a, cls.b = -1.0, 1.0
        _, _, cls.msh_1d, _ = setup_1Dmatrices(cls.N, cls.a, cls.b, cls.k)
        cls.u_full = jnp.ones([cls.N] * cls.dim, dtype=jnp.float64)
        cls.quadrature_order = 8

    def _evaluate_energy(self, potential):
        """High-order energy of the constant tensor for the given potential."""
        energy = EnergyHighOrder(
            self.a,
            self.b,
            self.beta,
            potential,
            self.msh_1d,
            self.dim,
            self.k,
            quadrature_order=self.quadrature_order,
        )
        return float(energy(self.u_full))

    def test_92_composite_exact_terms_match_single_callable(self):
        """A potential given as a list of callables gives the same energy as their sum."""
        print("\n[TEST] EnergyHighOrder composite exact terms match single callable")
        single_term = lambda x: x[0] ** 2 + x[1] ** 2 + 1.0
        composite_terms = [lambda x: x[0] ** 2, lambda x: x[1] ** 2 + 1.0]

        energy_single = self._evaluate_energy(single_term)
        energy_composite = self._evaluate_energy(composite_terms)

        print(f"\t single term energy:\t{energy_single}")
        print(f"\t composite energy:\t{energy_composite}")
        self.assertAlmostEqual(energy_single, energy_composite, places=10)

    def test_93_composite_exact_and_dg0_match_single_callable(self):
        """A potential with a DG0 term gives the same energy as the single callable."""
        print(
            "\n[TEST] EnergyHighOrder composite exact + DG0 terms match single callable"
        )
        single_term = lambda x: x[0] ** 2 + x[1] ** 2 + 0.25
        composite_terms = [
            lambda x: x[0] ** 2 + x[1] ** 2,
            ("DG", 0, lambda x: np.full_like(x[0], 0.25, dtype=np.float64)),
        ]

        energy_single = self._evaluate_energy(single_term)
        energy_composite = self._evaluate_energy(composite_terms)

        print(f"\t single term energy:\t{energy_single}")
        print(f"\t composite energy:\t{energy_composite}")
        self.assertAlmostEqual(energy_single, energy_composite, places=10)


if __name__ == "__main__":
    unittest.main()
