import unittest
import numpy as np


class SolverDiagnosticsTest(unittest.TestCase):
    def test_create_grid_accepts_dtype_and_phase_for_precision_experiments(self):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        try:
            from JaxDFT.src.hamiltonian import create_grid

            grid = create_grid(1.0, [4.0, 4.0, 4.0], dtype=jnp.float64, phase=0.5)

            self.assertEqual(jnp.float64, grid.coords.dtype)
            self.assertAlmostEqual(-1.5, float(grid.coords[0, 0, 0, 0]))
            self.assertAlmostEqual(0.5, float(grid.grid_phase))
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_select_scf_residual_uses_requested_metric(self):
        from JaxDFT.src.solver import select_scf_residual

        self.assertEqual(1.0, select_scf_residual("max", 1.0, 2.0, 3.0))
        self.assertEqual(2.0, select_scf_residual("rms", 1.0, 2.0, 3.0))
        self.assertEqual(3.0, select_scf_residual("l2", 1.0, 2.0, 3.0))

    def test_energy_and_forces_default_return_shape_is_unchanged(self):
        import jax
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import create_grid
        from JaxDFT.src.io import load_pseudopotentials
        from JaxDFT.src.solver import energy_and_forces

        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        pseudo = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")[0]
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        result = energy_and_forces(grid, coords, [pseudo], 1, 0.3, 1e-4, jax.random.PRNGKey(0))

        self.assertEqual(2, len(result))

    def test_energy_and_forces_can_return_diagnostics(self):
        import jax
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import create_grid
        from JaxDFT.src.io import load_pseudopotentials
        from JaxDFT.src.solver import energy_and_forces

        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        pseudo = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")[0]
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        energy, forces, info = energy_and_forces(
            grid,
            coords,
            [pseudo],
            1,
            0.3,
            1e-4,
            jax.random.PRNGKey(0),
            return_info=True,
        )

        self.assertEqual((1, 3), tuple(forces.shape))
        self.assertIn("scf_iterations", info)
        self.assertIn("density_diff", info)
        self.assertIn("orbital_residual", info)
        self.assertIn("energy_components", info)
        self.assertIn("local_pseudopotential", info["energy_components"])
        self.assertIn("nonlocal_pseudopotential", info["energy_components"])
        self.assertIn("local_pseudopotential_by_atom", info["energy_components"])
        self.assertIn("local_pseudopotential_min", info)
        self.assertIn("local_pseudopotential_max", info)
        self.assertIn("local_pseudopotential_integral", info)
        self.assertIn("local_pseudopotential_integral_by_atom", info)
        self.assertIn("hartree_potential_min", info)
        self.assertIn("hartree_potential_max", info)
        self.assertIn("hartree_potential_integral", info)
        self.assertIn("n_bands", info)
        self.assertIn("occupations", info)
        self.assertIn("eigenvalues", info)
        self.assertIn("scf_converged", info)
        self.assertIn("orbital_iterations", info)
        self.assertIn("orbital_converged", info)
        self.assertIn("orbital_residuals", info)
        self.assertIn("orbital_max_iter", info)
        self.assertIn("orbital_tolerance", info)
        self.assertIn("energy_history", info)
        self.assertIn("energy_delta_last", info)
        self.assertIn("energy_delta_history", info)
        self.assertIn("energy_delta_last10_max", info)
        self.assertIn("density_converged", info)
        self.assertIn("energy_converged", info)
        self.assertIn("density_diff_history", info)
        self.assertIn("density_rms_diff", info)
        self.assertIn("density_l2_diff", info)
        self.assertIn("density_rms_diff_history", info)
        self.assertIn("density_l2_diff_history", info)
        self.assertIn("orbital_residual_history", info)
        self.assertIn("density_min", info)
        self.assertIn("density_integral", info)
        self.assertEqual(1, int(info["n_bands"]))
        self.assertEqual((1,), tuple(info["occupations"].shape))
        self.assertEqual((1,), tuple(info["eigenvalues"].shape))
        self.assertEqual((1,), tuple(info["orbital_residuals"].shape))
        self.assertAlmostEqual(0.0, float(info["energy_components"]["nonlocal_pseudopotential"]), places=7)
        self.assertEqual((1,), tuple(info["energy_components"]["local_pseudopotential_by_atom"].shape))
        self.assertEqual((1,), tuple(info["local_pseudopotential_integral_by_atom"].shape))

    def test_scf_converged_combines_density_and_energy_diagnostics(self):
        import jax
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import create_grid
        from JaxDFT.src.io import load_pseudopotentials
        from JaxDFT.src.solver import energy_and_forces

        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        pseudo = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")[0]
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        _, _, info = energy_and_forces(
            grid,
            coords,
            [pseudo],
            2,
            0.3,
            1.0,
            jax.random.PRNGKey(0),
            return_info=True,
            energy_tolerance=0.0,
        )

        self.assertTrue(bool(info["density_converged"]))
        self.assertFalse(bool(info["energy_converged"]))
        self.assertFalse(bool(info["scf_converged"]))

    def test_energy_and_forces_accepts_orbital_solver_controls(self):
        import jax
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import create_grid
        from JaxDFT.src.io import load_pseudopotentials
        from JaxDFT.src.solver import energy_and_forces

        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        pseudo = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")[0]
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        _, _, info = energy_and_forces(
            grid,
            coords,
            [pseudo],
            1,
            0.3,
            1e-4,
            jax.random.PRNGKey(0),
            return_info=True,
            orbital_max_iter=2,
            orbital_tolerance=1e-3,
            orbital_preconditioner="kinetic",
            orbital_preconditioner_shift=2.0,
            mixing_mode="pulay",
            anderson_regularization=1e-4,
            anderson_history=3,
            mixing_safeguard="density_diff",
            mixing_safeguard_factor=1.1,
            scf_convergence_metric="rms",
            pulay_residual_metric="kerker",
            pulay_kerker_k0=0.8,
            laplacian_order=6,
        )

        self.assertEqual(2, int(info["orbital_max_iter"]))
        self.assertAlmostEqual(1e-3, float(info["orbital_tolerance"]))
        self.assertEqual("kinetic", info["orbital_preconditioner"])
        self.assertAlmostEqual(2.0, float(info["orbital_preconditioner_shift"]))
        self.assertEqual("pulay", info["mixing_mode"])
        self.assertAlmostEqual(1e-4, float(info["anderson_regularization"]))
        self.assertEqual(3, int(info["anderson_history"]))
        self.assertEqual("density_diff", info["mixing_safeguard"])
        self.assertAlmostEqual(1.1, float(info["mixing_safeguard_factor"]))
        self.assertIn("mixing_fallback_count", info)
        self.assertEqual("rms", info["scf_convergence_metric"])
        self.assertEqual("kerker", info["pulay_residual_metric"])
        self.assertAlmostEqual(0.8, float(info["pulay_kerker_k0"]))
        self.assertEqual(6, int(info["laplacian_order"]))

    def test_safeguard_mixing_falls_back_when_density_diff_worsens(self):
        import jax.numpy as jnp

        from JaxDFT.src.solver import safeguarded_density_mixing

        rho = jnp.array([1.0, 1.0], dtype=jnp.float32)
        rho_new = jnp.array([2.0, 2.0], dtype=jnp.float32)
        anderson_candidate = jnp.array([-10.0, -10.0], dtype=jnp.float32)
        linear_candidate = rho + 0.3 * (rho_new - rho)

        mixed, did_fallback = safeguarded_density_mixing(
            rho,
            rho_new,
            anderson_candidate,
            linear_candidate,
            mode="density_diff",
            factor=1.0,
            previous_density_diff=0.5,
            current_density_diff=0.6,
        )

        self.assertTrue(bool(did_fallback))
        self.assertTrue(jnp.allclose(mixed, linear_candidate))

    def test_pulay_mixing_uses_history_and_preserves_affine_constraint(self):
        import jax.numpy as jnp

        from JaxDFT.src.solver import pulay_mixing

        rho_hist = jnp.zeros((2, 2), dtype=jnp.float32)
        f_hist = jnp.zeros((2, 2), dtype=jnp.float32)
        rho0 = jnp.array([1.0, 1.0], dtype=jnp.float32)
        rho1 = jnp.array([1.5, 0.5], dtype=jnp.float32)

        mixed0, rho_hist, f_hist, coeff0 = pulay_mixing(
            rho0,
            rho1,
            rho_hist,
            f_hist,
            mix_alpha=0.2,
            iter_idx=0,
            m=2,
        )

        self.assertTrue(jnp.allclose(mixed0, jnp.array([1.1, 0.9], dtype=jnp.float32)))
        self.assertTrue(jnp.allclose(coeff0, jnp.array([1.0, 0.0], dtype=jnp.float32)))

        mixed1, _, _, coeff1 = pulay_mixing(
            mixed0,
            jnp.array([1.0, 0.9], dtype=jnp.float32),
            rho_hist,
            f_hist,
            mix_alpha=0.2,
            iter_idx=1,
            m=2,
        )

        self.assertAlmostEqual(1.0, float(jnp.sum(coeff1)), places=6)
        self.assertTrue(jnp.all(jnp.isfinite(mixed1)))

    def test_pulay_mixing_stores_charge_neutral_residual(self):
        import jax.numpy as jnp

        from JaxDFT.src.solver import pulay_mixing

        rho_hist = jnp.zeros((2, 4), dtype=jnp.float32)
        f_hist = jnp.zeros((2, 4), dtype=jnp.float32)
        rho = jnp.ones((4,), dtype=jnp.float32)
        rho_new = jnp.array([1.2, 1.1, 1.0, 0.9], dtype=jnp.float32)

        _, _, f_hist_next, _ = pulay_mixing(
            rho,
            rho_new,
            rho_hist,
            f_hist,
            mix_alpha=0.2,
            iter_idx=0,
            m=2,
        )

        self.assertAlmostEqual(0.0, float(jnp.mean(f_hist_next[0])), places=6)

    def test_kerker_metric_keeps_residual_neutral_and_suppresses_long_wavelength(self):
        import jax.numpy as jnp

        from JaxDFT.src.solver import apply_density_residual_metric

        residual = jnp.array([1.0, 0.5, -0.5, -1.0], dtype=jnp.float32)
        euclidean = apply_density_residual_metric(
            residual,
            metric="euclidean",
            grid_shape=(4, 1, 1),
            spacing=1.0,
            kerker_k0=1.0,
        )
        kerker = apply_density_residual_metric(
            residual,
            metric="kerker",
            grid_shape=(4, 1, 1),
            spacing=1.0,
            kerker_k0=1.0,
        )

        self.assertAlmostEqual(0.0, float(jnp.mean(kerker)), places=6)
        self.assertLess(float(jnp.linalg.norm(kerker)), float(jnp.linalg.norm(euclidean)))

    def test_stabilize_density_clips_and_renormalizes_charge(self):
        import jax.numpy as jnp

        from JaxDFT.src.solver import stabilize_density

        rho = jnp.array([-1.0, 0.0, 2.0, 4.0], dtype=jnp.float32)
        stabilized = stabilize_density(rho, volume_element=0.25, n_electrons=2.0)

        self.assertGreaterEqual(float(jnp.min(stabilized)), 0.0)
        self.assertAlmostEqual(2.0, float(jnp.sum(stabilized) * 0.25), places=6)

    def test_orbital_residual_matches_physical_norm_for_uniform_grid_scaling(self):
        import jax
        import jax.numpy as jnp

        from JaxDFT.src.solver import solve_orbitals_subspace

        diag = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)

        def apply_h(x):
            return diag * x

        eigvals, eigvecs, info = solve_orbitals_subspace(
            apply_h,
            n_grid=4,
            n_bands=1,
            x_init=jnp.array([[1.0], [1.0], [0.0], [0.0]], dtype=jnp.float32),
            max_iter=1,
            key=jax.random.PRNGKey(0),
            return_info=True,
        )

        residual = apply_h(eigvecs[:, 0]) - eigvals[0] * eigvecs[:, 0]
        euclidean_norm = jnp.sqrt(jnp.sum(residual * residual))

        dv = 0.125
        physical_orbital = eigvecs[:, 0] / jnp.sqrt(dv)
        physical_residual = apply_h(physical_orbital) - eigvals[0] * physical_orbital
        physical_norm = jnp.sqrt(dv * jnp.sum(physical_residual * physical_residual))

        self.assertAlmostEqual(float(euclidean_norm), float(info["residuals"][0]), places=6)
        self.assertAlmostEqual(float(physical_norm), float(info["residuals"][0]), places=6)

    def test_solve_orbitals_subspace_accepts_kinetic_preconditioner(self):
        import jax
        import jax.numpy as jnp

        from JaxDFT.src.solver import kinetic_precondition_residuals, solve_orbitals_subspace

        diag = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)

        def apply_h(x):
            return diag * x

        _, _, info = solve_orbitals_subspace(
            apply_h,
            n_grid=4,
            n_bands=1,
            x_init=jnp.array([[1.0], [1.0], [0.0], [0.0]], dtype=jnp.float32),
            max_iter=1,
            key=jax.random.PRNGKey(0),
            return_info=True,
            preconditioner_fn=lambda r: kinetic_precondition_residuals(
                r,
                grid_shape=(4, 1, 1),
                spacing=1.0,
                shift=1.0,
            ),
        )

        self.assertTrue(np.isfinite(float(info["residual_norm"])))


if __name__ == "__main__":
    unittest.main()
