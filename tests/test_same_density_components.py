"""Tests for same-density component diagnostic (PySCF rho on JaxDFT grid)."""

import importlib.util
import math
import json
import os
import subprocess
import sys
import unittest


def _have_pyscf() -> bool:
    try:
        import pyscf  # noqa: F401

        return True
    except ImportError:
        return False


def _load_diagnose_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "JaxDFT", "scripts", "diagnose_same_density_components.py")
    spec = importlib.util.spec_from_file_location("diagnose_same_density_components", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod, repo_root


class SameDensityJaxOperatorsTest(unittest.TestCase):
    def test_jax_operator_components_finite(self):
        import jax.numpy as jnp

        mod, repo_root = _load_diagnose_module()
        from JaxDFT.src.hamiltonian import create_grid
        from JaxDFT.src.io import load_pseudopotentials

        grid = create_grid(1.0, [6.0, 6.0, 6.0])
        ne = 2.0
        npts = grid.shape[0] * grid.shape[1] * grid.shape[2]
        rho0 = ne / (npts * float(grid.volume_element))
        rho = jnp.full(grid.shape, rho0, dtype=grid.coords.dtype)
        coords = jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=grid.coords.dtype)
        pseudo_dir = os.path.join(repo_root, "JaxDFT", "data", "gth_potentials")
        pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
        zion = jnp.array([p["zion"] for p in pseudos], dtype=grid.coords.dtype)
        rloc = jnp.array([p["rloc"] for p in pseudos], dtype=grid.coords.dtype)
        c = jnp.array([p["c"] for p in pseudos], dtype=grid.coords.dtype)
        out = mod.jax_operator_components_from_rho(rho, grid, coords, zion, rloc, c)
        self.assertTrue(math.isfinite(out["jax_hartree_same_density"]))
        self.assertTrue(math.isfinite(out["jax_local_pseudopotential_same_density"]))
        self.assertTrue(math.isfinite(out["jax_xc_same_density"]))


@unittest.skipUnless(_have_pyscf(), "PySCF not installed")
class SameDensityPyscfIntegrationTest(unittest.TestCase):
    def test_h2_rho_integral_near_two_electrons(self):
        mod, _ = _load_diagnose_module()
        from JaxDFT.scripts.benchmark_systems import get_benchmark_systems

        h2 = get_benchmark_systems()[0]
        self.assertEqual("H2", h2.name)
        pseudo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "JaxDFT",
            "data",
            "gth_potentials",
        )
        row = mod.run_same_density_diagnostic(
            h2,
            target_spacing=1.0,
            box_size=12.0,
            pseudo_dir=pseudo_dir,
            calculation_dtype="float32",
            grid_phase=0.0,
        )
        self.assertLess(abs(row["rho_electron_count_error"]), 0.15)
        self.assertLess(abs(row["delta_hartree_vs_pyscf_coul_mHa"]), 500.0)

    def test_script_json_stdout(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(repo_root, "JaxDFT", "scripts", "diagnose_same_density_components.py")
        proc = subprocess.run(
            [
                sys.executable,
                script,
                "--systems",
                "H2",
                "--target-spacing",
                "1.0",
                "--box-size",
                "12",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        self.assertEqual(1, len(data))
        self.assertEqual("H2", data[0]["system"])
        self.assertIn("jax_hartree_same_density", data[0])
        self.assertIn("pyscf_coul", data[0])


if __name__ == "__main__":
    unittest.main()
