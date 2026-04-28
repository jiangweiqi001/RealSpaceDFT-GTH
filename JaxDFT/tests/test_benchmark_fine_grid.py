import csv
import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT_PATH = pathlib.Path("JaxDFT/scripts/benchmark_fine_grid.py")
SPEC = importlib.util.spec_from_file_location("benchmark_fine_grid", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class BenchmarkDefinitionTest(unittest.TestCase):
    def test_system_specs_match_expected_scan_values(self):
        self.assertEqual(benchmark.SYSTEM_SPECS["O"].geometry_param_name, "none")
        self.assertEqual(benchmark.SYSTEM_SPECS["O"].geometry_values, [0.0])
        self.assertEqual(benchmark.SYSTEM_SPECS["H2O"].geometry_values, [1.6, 1.8, 2.0, 2.4])
        self.assertEqual(benchmark.SYSTEM_SPECS["N2"].geometry_values, [1.8, 2.0, 2.2, 2.4, 2.8])

    def test_build_shift_vector_uses_actual_spacing(self):
        shift = benchmark.build_shift_vector(actual_spacing=0.32, shift_fraction=0.5, case_name="half_shift")
        self.assertEqual(shift, [0.16, 0.16, 0.16])
        center = benchmark.build_shift_vector(actual_spacing=0.32, shift_fraction=0.5, case_name="center")
        self.assertEqual(center, [0.0, 0.0, 0.0])

    def test_build_o_coords_returns_shifted_single_atom(self):
        coords = benchmark.build_coords("O", 0.0, shift=[0.1, 0.2, 0.3], angle_deg=104.5)
        self.assertEqual(coords, [[0.1, 0.2, 0.3]])

    def test_build_h2o_coords_returns_symmetric_geometry(self):
        coords = benchmark.build_coords("H2O", 2.0, shift=[0.0, 0.0, 0.0], angle_deg=104.5)
        self.assertEqual(len(coords), 3)
        self.assertAlmostEqual(coords[0][0], 0.0, places=7)
        self.assertAlmostEqual(coords[1][0], -coords[2][0], places=7)
        self.assertAlmostEqual(coords[1][2], coords[2][2], places=7)

    def test_build_n2_coords_returns_linear_dimer(self):
        coords = benchmark.build_coords("N2", 2.4, shift=[0.0, 0.0, 0.0], angle_deg=104.5)
        self.assertEqual(coords, [[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]])

    def test_build_jobs_covers_system_grid_shift_and_mode(self):
        jobs = benchmark.build_jobs(
            system_name="H2O",
            spacing_values=[0.4, 0.32],
            shift_fraction=0.5,
            benchmark_mode="standard",
            fine_grid_modes=["off", "auto"],
            projector_subgrids=[2, 3],
            projector_radii=[4.0, 6.0],
            box_size=9.6,
        )
        self.assertEqual(len(jobs), 4 * 2 * 2 * 2)
        first = jobs[0]
        self.assertEqual(first.system, "H2O")
        self.assertEqual(first.geometry_param_name, "oh_distance")
        self.assertIn(first.case, ("center", "half_shift"))
        self.assertIn(first.mode_label, ("off", "auto_local"))

    def test_projector_sweep_jobs_cover_radius_and_subgrid_grid(self):
        jobs = benchmark.build_jobs(
            system_name="N2",
            spacing_values=[0.4],
            shift_fraction=0.5,
            benchmark_mode="projector_sweep",
            fine_grid_modes=["off", "auto"],
            projector_subgrids=[2, 4],
            projector_radii=[4.0, 8.0],
            box_size=9.6,
        )
        expected_modes = 2 + 2 * 2
        self.assertEqual(len(jobs), 5 * 1 * 2 * expected_modes)
        labels = {job.mode_label for job in jobs}
        self.assertIn("off", labels)
        self.assertIn("auto_local", labels)
        self.assertIn("local_proj_sg2_r4.0", labels)
        self.assertIn("local_proj_sg4_r8.0", labels)

    def test_projector_patch_jobs_are_marked_experimental_in_notes(self):
        job = benchmark._projector_patch_job(
            system_name="N2",
            geometry_param_name="bond_length",
            geometry_param=2.4,
            spacing=0.4,
            box_size=9.6,
            case="half_shift",
            shift_fraction=0.5,
            projector_subgrid=2,
            projector_radius=4.0,
        )
        notes = benchmark.job_notes(job)
        self.assertIn("experimental", notes.lower())
        self.assertIn("projector", notes.lower())

    def test_default_csv_fieldnames_are_stable(self):
        self.assertEqual(
            benchmark.CSV_FIELDNAMES,
            [
                "system",
                "case",
                "geometry_param_name",
                "geometry_param",
                "spacing_bohr",
                "actual_spacing_bohr",
                "box_size_bohr",
                "shift_fraction",
                "mode_label",
                "fine_grid_mode",
                "local_subgrid",
                "local_mode",
                "local_patch_radius_factor",
                "projector_subgrid",
                "projector_mode",
                "projector_patch_radius_factor",
                "energy_jax_ha",
                "energy_pyscf_ha",
                "error_vs_pyscf_ha",
                "scf_steps",
                "runtime_sec",
                "notes",
            ],
        )

    def test_write_csv_uses_expected_columns(self):
        rows = [
            {
                "system": "O",
                "case": "center",
                "geometry_param_name": "none",
                "geometry_param": 0.0,
                "spacing_bohr": 0.4,
                "actual_spacing_bohr": 0.4,
                "box_size_bohr": 9.6,
                "shift_fraction": 0.0,
                "mode_label": "off",
                "fine_grid_mode": "off",
                "local_subgrid": 1,
                "local_mode": "cell_average",
                "local_patch_radius_factor": 6.0,
                "projector_subgrid": 1,
                "projector_mode": "cell_average",
                "projector_patch_radius_factor": 6.0,
                "energy_jax_ha": -1.0,
                "energy_pyscf_ha": -0.9,
                "error_vs_pyscf_ha": -0.1,
                "scf_steps": 12,
                "runtime_sec": 1.2,
                "notes": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "benchmark.csv"
            benchmark.write_csv(path, rows)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                loaded_rows = list(reader)
        self.assertEqual(reader.fieldnames, benchmark.CSV_FIELDNAMES)
        self.assertEqual(loaded_rows[0]["system"], "O")


if __name__ == "__main__":
    unittest.main()
