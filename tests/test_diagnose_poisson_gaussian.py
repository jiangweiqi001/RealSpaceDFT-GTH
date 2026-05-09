"""Tests for JaxDFT/scripts/diagnose_poisson_gaussian.py (P0 Poisson diagnostic)."""

import importlib.util
import json
import os
import subprocess
import sys
import unittest


def _load_diagnose_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "JaxDFT", "scripts", "diagnose_poisson_gaussian.py")
    spec = importlib.util.spec_from_file_location("diagnose_poisson_gaussian", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod, repo_root


class DiagnosePoissonGaussianTest(unittest.TestCase):
    def test_electron_count_near_charge_mild_grid(self):
        mod, _ = _load_diagnose_module()
        row = mod.compute_row(0.3, 12.0, 0.25, 1.0)
        self.assertAlmostEqual(row["charge"], 1.0, places=6)
        self.assertLess(abs(row["electron_count"] - 1.0), 0.02)

    def test_hartree_error_reasonable_mild_parameters(self):
        """Coarse but stable grid: error should stay modest (finite-box + discretization)."""
        mod, _ = _load_diagnose_module()
        row = mod.compute_row(0.3, 12.0, 0.25, 1.0)
        self.assertLess(abs(row["error_mHa"]), 500.0)

    def test_json_output_is_parseable(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(repo_root, "JaxDFT", "scripts", "diagnose_poisson_gaussian.py")
        proc = subprocess.run(
            [
                sys.executable,
                script,
                "--spacings",
                "0.4",
                "--box-sizes",
                "10",
                "--alphas",
                "0.5",
                "--charge",
                "1.0",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(1, len(data))
        self.assertIn("E_H_num", data[0])
        self.assertIn("error_mHa", data[0])
        self.assertIn("hartree_potential_min", data[0])


if __name__ == "__main__":
    unittest.main()
