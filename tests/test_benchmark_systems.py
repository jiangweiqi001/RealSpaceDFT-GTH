import math
import unittest


class BenchmarkSystemsTest(unittest.TestCase):
    def test_fixed_benchmark_systems_are_defined(self):
        from JaxDFT.scripts.benchmark_systems import get_benchmark_systems

        systems = get_benchmark_systems()
        self.assertEqual(["H2", "H2O", "CO"], [system.name for system in systems])

        h2, h2o, co = systems
        self.assertEqual(("H", "H"), h2.symbols)
        self.assertAlmostEqual(abs(h2.coords_bohr[1][2] - h2.coords_bohr[0][2]), 1.4)

        self.assertEqual(("O", "H", "H"), h2o.symbols)
        oh1 = math.dist(h2o.coords_bohr[0], h2o.coords_bohr[1])
        oh2 = math.dist(h2o.coords_bohr[0], h2o.coords_bohr[2])
        self.assertAlmostEqual(oh1, 1.8)
        self.assertAlmostEqual(oh2, 1.8)

        self.assertEqual(("C", "O"), co.symbols)
        self.assertAlmostEqual(abs(co.coords_bohr[1][2] - co.coords_bohr[0][2]), 2.132)

    def test_error_mha_is_jax_minus_reference_in_millihartree(self):
        from JaxDFT.scripts.benchmark_systems import error_mha

        self.assertAlmostEqual(error_mha(-1.25, -1.20), -50.0)

    def test_run_benchmark_reports_scf_iterations_from_jaxdft_info(self):
        import JaxDFT.scripts.benchmark_systems as benchmarks

        system = benchmarks.get_benchmark_systems()[0]
        original_reference = benchmarks.run_pyscf_reference
        original_reference_with_components = benchmarks.run_pyscf_reference_with_components
        original_jaxdft = benchmarks.run_jaxdft_energy
        try:
            benchmarks.run_pyscf_reference = lambda _: -1.0
            benchmarks.run_pyscf_reference_with_components = lambda _: (
                -1.0,
                {"e1": -2.0, "coul": 0.5, "exc": -0.1, "nuc": 0.6},
            )
            benchmarks.run_jaxdft_energy = lambda *args, **kwargs: (
                -1.01,
                (3, 3, 3),
                0.5,
                {
                    "scf_iterations": 4,
                    "density_diff": 1e-4,
                    "density_rms_diff": 2e-5,
                    "density_l2_diff": 3e-5,
                    "scf_convergence_residual": 3e-5,
                    "energy_delta_last": 0.11,
                    "energy_delta_history": [math.inf, 0.5, 0.4, 0.11],
                    "energy_delta_last10_max": 0.5,
                    "density_converged": True,
                    "energy_converged": False,
                    "orbital_residual": 2e-5,
                    "energy_components": {
                        "local_pseudopotential": -2.0,
                        "nonlocal_pseudopotential": 0.25,
                        "hartree": 0.75,
                        "local_pseudopotential_by_atom": [-1.2, -0.8],
                    },
                    "local_pseudopotential_min": -3.0,
                    "local_pseudopotential_max": 0.1,
                    "local_pseudopotential_integral": -4.0,
                    "local_pseudopotential_integral_by_atom": [-1.5, -2.5],
                    "hartree_potential_min": 0.0,
                    "hartree_potential_max": 1.5,
                    "hartree_potential_integral": 6.0,
                    "projector_overlap_diagnostics": [
                        {"overlap_error": [[0.01, 0.0], [0.0, -0.02]]}
                    ],
                    "orbital_iterations": 7,
                    "scf_converged": False,
                    "orbital_converged": True,
                    "eigenvalues": [-0.5],
                    "orbital_residuals": [2e-5],
                    "orbital_max_iter": 30,
                    "orbital_tolerance": 1e-5,
                    "orbital_preconditioner": "kinetic",
                    "orbital_preconditioner_shift": 2.0,
                    "mixing_mode": "pulay",
                    "pulay_residual_metric": "kerker",
                    "pulay_kerker_k0": 0.8,
                    "laplacian_order": 6,
                    "calculation_dtype": "float64",
                    "grid_phase": 0.5,
                    "energy_history": [0.0, -0.5, -0.9, -1.01],
                    "density_diff_history": [0.1, 0.05, 0.01, 1e-4],
                    "density_rms_diff_history": [0.02, 0.01, 0.005, 2e-5],
                    "density_l2_diff_history": [0.03, 0.02, 0.006, 3e-5],
                    "density_min": 1e-12,
                    "density_integral": 2.0,
                    "anderson_regularization": 1e-4,
                    "anderson_history": 3,
                    "mixing_safeguard": "density_diff",
                    "mixing_safeguard_factor": 1.1,
                    "mixing_fallback_count": 2,
                    "scf_convergence_metric": "l2",
                },
            )

            result = benchmarks.run_benchmark(
                system,
                0.5,
                4.0,
                10,
                0.3,
                1e-5,
                "unused",
                orbital_preconditioner="kinetic",
                orbital_preconditioner_shift=2.0,
                mixing_mode="pulay",
                pulay_residual_metric="kerker",
                pulay_kerker_k0=0.8,
                laplacian_order=6,
                calculation_dtype="float64",
                grid_phase=0.5,
                anderson_regularization=1e-4,
                anderson_history=3,
                mixing_safeguard="density_diff",
                mixing_safeguard_factor=1.1,
                scf_convergence_metric="l2",
                energy_tolerance=1e-7,
            )

            self.assertEqual(4, result.scf_iterations)
            self.assertAlmostEqual(1e-4, result.density_diff)
            self.assertAlmostEqual(2e-5, result.density_rms_diff)
            self.assertAlmostEqual(3e-5, result.density_l2_diff)
            self.assertAlmostEqual(0.11, result.energy_delta_last)
            self.assertEqual((math.inf, 0.5, 0.4, 0.11), result.energy_delta_history_last10)
            self.assertAlmostEqual(0.5, result.energy_delta_last10_max)
            self.assertTrue(result.density_converged)
            self.assertFalse(result.energy_converged)
            self.assertAlmostEqual(2e-5, result.orbital_residual)
            self.assertAlmostEqual(-2.0, result.local_pseudopotential_energy)
            self.assertAlmostEqual(0.25, result.nonlocal_pseudopotential_energy)
            self.assertEqual((-1.2, -0.8), result.local_pseudopotential_energy_by_atom)
            self.assertAlmostEqual(-3.0, result.local_pseudopotential_min)
            self.assertAlmostEqual(0.1, result.local_pseudopotential_max)
            self.assertAlmostEqual(-4.0, result.local_pseudopotential_integral)
            self.assertEqual((-1.5, -2.5), result.local_pseudopotential_integral_by_atom)
            self.assertAlmostEqual(0.75, result.hartree_energy)
            self.assertAlmostEqual(0.0, result.hartree_potential_min)
            self.assertAlmostEqual(1.5, result.hartree_potential_max)
            self.assertAlmostEqual(6.0, result.hartree_potential_integral)
            self.assertAlmostEqual(0.02, result.projector_overlap_max_error)
            self.assertAlmostEqual(-2.0, result.pyscf_e1)
            self.assertAlmostEqual(0.5, result.pyscf_coul)
            self.assertAlmostEqual(-0.1, result.pyscf_xc)
            self.assertAlmostEqual(0.6, result.pyscf_nuc)
            self.assertEqual(7, result.orbital_iterations)
            self.assertFalse(result.scf_converged)
            self.assertTrue(result.orbital_converged)
            self.assertEqual((-0.5,), result.eigenvalues)
            self.assertEqual((2e-5,), result.orbital_residuals)
            self.assertEqual(30, result.orbital_max_iter)
            self.assertAlmostEqual(1e-5, result.orbital_tolerance)
            self.assertEqual("kinetic", result.orbital_preconditioner)
            self.assertAlmostEqual(2.0, result.orbital_preconditioner_shift)
            self.assertEqual("pulay", result.mixing_mode)
            self.assertEqual("kerker", result.pulay_residual_metric)
            self.assertAlmostEqual(0.8, result.pulay_kerker_k0)
            self.assertEqual(6, result.laplacian_order)
            self.assertEqual("float64", result.calculation_dtype)
            self.assertAlmostEqual(0.5, result.grid_phase)
            self.assertEqual((0.0, -0.5, -0.9, -1.01), result.energy_history_last10)
            self.assertEqual((0.1, 0.05, 0.01, 1e-4), result.density_diff_history_last10)
            self.assertEqual((0.02, 0.01, 0.005, 2e-5), result.density_rms_diff_history_last10)
            self.assertEqual((0.03, 0.02, 0.006, 3e-5), result.density_l2_diff_history_last10)
            self.assertAlmostEqual(1e-12, result.density_min)
            self.assertAlmostEqual(2.0, result.density_integral)
            self.assertAlmostEqual(1e-4, result.anderson_regularization)
            self.assertEqual(3, result.anderson_history)
            self.assertEqual("density_diff", result.mixing_safeguard)
            self.assertAlmostEqual(1.1, result.mixing_safeguard_factor)
            self.assertEqual(2, result.mixing_fallback_count)
            self.assertEqual("l2", result.scf_convergence_metric)
            self.assertAlmostEqual(1e-7, result.energy_tolerance)
            self.assertAlmostEqual(-10.0, result.error_mha)
        finally:
            benchmarks.run_pyscf_reference = original_reference
            benchmarks.run_pyscf_reference_with_components = original_reference_with_components
            benchmarks.run_jaxdft_energy = original_jaxdft


if __name__ == "__main__":
    unittest.main()
