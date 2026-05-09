import math
import unittest

import numpy as np


PSEUDO_DIR = "JaxDFT/data/gth_potentials"


class ProjectorDiagnosticsTest(unittest.TestCase):
    def test_c_o_local_gth_lda_parameters_match_pyscf_when_available(self):
        try:
            from pyscf import gto
        except ImportError:
            self.skipTest("PySCF is not installed")

        from JaxDFT.src.io import load_pseudopotentials

        local_by_symbol = {
            pseudo["symbol"]: pseudo
            for pseudo in load_pseudopotentials(["C", "O"], PSEUDO_DIR)
        }

        for symbol in ("C", "O"):
            pyscf_pp = gto.basis.load_pseudo("gth-lda", symbol)
            local = local_by_symbol[symbol]

            np.testing.assert_allclose(local["q"], sum(pyscf_pp[0]), rtol=0.0, atol=0.0)
            np.testing.assert_allclose(local["rloc"], pyscf_pp[1], rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(local["c"][:pyscf_pp[2]], pyscf_pp[3], rtol=0.0, atol=1e-12)

            nonempty_pyscf_channels = [channel for channel in pyscf_pp[5:] if channel[1] > 0]
            self.assertEqual(len(nonempty_pyscf_channels), len(local["projectors"]))
            for local_channel, pyscf_channel in zip(local["projectors"], nonempty_pyscf_channels):
                rp, n_proj, h = pyscf_channel
                self.assertEqual(n_proj, local_channel["h"].shape[0])
                np.testing.assert_allclose(local_channel["r"], rp, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(local_channel["h"], h, rtol=0.0, atol=1e-12)

    def test_carbon_and_oxygen_parser_skips_empty_projector_channels(self):
        from JaxDFT.src.io import load_pseudopotentials

        carbon, oxygen = load_pseudopotentials(["C", "O"], PSEUDO_DIR)

        for pseudo in (carbon, oxygen):
            self.assertEqual(1, len(pseudo["projectors"]))
            projector = pseudo["projectors"][0]
            self.assertEqual(0, projector["l"])
            self.assertEqual((1, 1), projector["h"].shape)
            self.assertGreater(float(projector["h"][0, 0]), 0.0)

    def test_parser_builds_symmetric_h_matrices_for_nonempty_channels(self):
        from JaxDFT.src.io import load_pseudopotentials

        carbon, oxygen = load_pseudopotentials(["C", "O"], PSEUDO_DIR)

        for pseudo in (carbon, oxygen):
            for projector in pseudo["projectors"]:
                h = projector["h"]
                self.assertEqual(h.shape[0], h.shape[1])
                np.testing.assert_allclose(h, h.T, rtol=0.0, atol=1e-12)

    def test_radial_projector_normalization_matches_gth_convention(self):
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import get_gth_projector

        r = jnp.linspace(0.0, 8.0, 20000)
        dr = float(r[1] - r[0])

        for l, i, rp in ((0, 1, 0.30455321), (1, 1, 0.25682890)):
            p = np.asarray(get_gth_projector(r, l, i, rp))
            radial_norm = np.trapezoid((p * p) * np.asarray(r * r), dx=dr)
            self.assertAlmostEqual(1.0, radial_norm, places=4)

    def test_carbon_s_projector_uses_y00_angular_weight(self):
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import create_grid, precompute_projectors
        from JaxDFT.src.io import load_pseudopotentials

        grid = create_grid(0.4, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        carbon = load_pseudopotentials(["C"], PSEUDO_DIR)
        _, _, coeffs = precompute_projectors(grid, coords, carbon)

        h00 = carbon[0]["projectors"][0]["h"][0, 0]
        self.assertEqual((1,), coeffs.shape)
        self.assertAlmostEqual(h00 / (4.0 * math.pi), float(coeffs[0]), places=6)

    def test_precomputed_nonlocal_operator_is_hermitian_for_carbon(self):
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import (
            apply_nonlocal_precomputed,
            create_grid,
            precompute_projectors,
        )
        from JaxDFT.src.io import load_pseudopotentials

        grid = create_grid(0.4, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        pseudo = load_pseudopotentials(["C"], PSEUDO_DIR)
        projectors = precompute_projectors(grid, coords, pseudo)
        self.assertIsNotNone(projectors)
        p_i, p_j, coeffs = projectors

        x = grid.coords[..., 0]
        y = grid.coords[..., 1]
        z = grid.coords[..., 2]
        psi = jnp.sin(0.7 * x) + 0.2 * jnp.cos(0.5 * y) + 0.1 * z
        phi = jnp.cos(0.3 * x) - 0.4 * jnp.sin(0.6 * z) + 0.05 * y

        v_psi = apply_nonlocal_precomputed(psi, p_i, p_j, coeffs, grid.volume_element)
        v_phi = apply_nonlocal_precomputed(phi, p_i, p_j, coeffs, grid.volume_element)

        left = float(jnp.sum(phi * v_psi) * grid.volume_element)
        right = float(jnp.sum(v_phi * psi) * grid.volume_element)
        self.assertTrue(math.isfinite(left))
        self.assertAlmostEqual(left, right, places=5)

    def test_projector_overlap_diagnostic_reports_carbon_s_normalization(self):
        import jax.numpy as jnp

        from JaxDFT.src.hamiltonian import create_grid, projector_overlap_diagnostics
        from JaxDFT.src.io import load_pseudopotentials

        grid = create_grid(0.2, [8.0, 8.0, 8.0])
        coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        carbon = load_pseudopotentials(["C"], PSEUDO_DIR)

        diagnostics = projector_overlap_diagnostics(grid, coords, carbon)

        self.assertEqual(1, len(diagnostics))
        diag = diagnostics[0]
        self.assertEqual("C", diag["symbol"])
        self.assertEqual(0, int(diag["l"]))
        self.assertEqual((1, 1), tuple(diag["overlap"].shape))
        self.assertAlmostEqual(1.0, float(diag["overlap"][0, 0]), delta=5e-2)


if __name__ == "__main__":
    unittest.main()
