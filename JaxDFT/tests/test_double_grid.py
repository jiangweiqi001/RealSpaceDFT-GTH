import unittest

import jax.numpy as jnp

from JaxDFT.src.hamiltonian import create_grid
from JaxDFT.src.hamiltonian import (
    apply_nonlocal_fine_integral,
    build_local_potential,
    get_gth_projector,
    gth_local_potential_value,
    precompute_projectors,
)
from JaxDFT.src.io import load_pseudopotentials
from JaxDFT.src.solver import energy_and_forces


class DoubleGridLocalPotentialTest(unittest.TestCase):
    def test_local_potential_uses_subgrid_cell_average(self):
        atom_coords = jnp.array([[0.17, -0.11, 0.09]], dtype=jnp.float32)
        grid_coords = jnp.array([[[[0.0, 0.0, 0.0]]]], dtype=jnp.float32)
        zion = jnp.array([1.0], dtype=jnp.float32)
        rloc = jnp.array([0.2], dtype=jnp.float32)
        c = jnp.array([[-4.1802368, 0.72507482, 0.0, 0.0]], dtype=jnp.float32)
        spacing = 0.48
        subgrid = 3

        sampled = build_local_potential(
            atom_coords, grid_coords, zion, rloc, c,
            spacing=spacing, local_subgrid=subgrid,
        )

        offsets_1d = (jnp.arange(subgrid, dtype=jnp.float32) + 0.5) / subgrid - 0.5
        offsets_1d = offsets_1d * spacing
        ox, oy, oz = jnp.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
        offsets = jnp.stack([ox, oy, oz], axis=-1).reshape(-1, 3)
        r = jnp.linalg.norm(offsets - atom_coords[0], axis=-1)
        expected = jnp.mean(gth_local_potential_value(r, zion[0], rloc[0], c[0]))

        self.assertAlmostEqual(float(sampled[0, 0, 0]), float(expected), places=6)

    def test_local_subgrid_one_matches_point_sample(self):
        atom_coords = jnp.array([[0.17, -0.11, 0.09]], dtype=jnp.float32)
        grid_coords = jnp.array([[[[0.0, 0.0, 0.0]]]], dtype=jnp.float32)
        zion = jnp.array([1.0], dtype=jnp.float32)
        rloc = jnp.array([0.2], dtype=jnp.float32)
        c = jnp.array([[-4.1802368, 0.72507482, 0.0, 0.0]], dtype=jnp.float32)

        sampled = build_local_potential(
            atom_coords, grid_coords, zion, rloc, c,
            spacing=0.48, local_subgrid=1,
        )

        r = jnp.linalg.norm(grid_coords[0, 0, 0] - atom_coords[0])
        expected = gth_local_potential_value(r, zion[0], rloc[0], c[0])
        self.assertAlmostEqual(float(sampled[0, 0, 0]), float(expected), places=6)

    def test_energy_and_forces_accepts_local_subgrid(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")

        energy, forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            local_subgrid=3,
        )

        self.assertTrue(jnp.isfinite(energy))
        self.assertEqual(forces.shape, coords.shape)

    def test_local_patch_mode_matches_cell_average_inside_patch(self):
        atom_coords = jnp.array([[0.17, -0.11, 0.09]], dtype=jnp.float32)
        grid_coords = jnp.array([[[[0.0, 0.0, 0.0]]]], dtype=jnp.float32)
        zion = jnp.array([1.0], dtype=jnp.float32)
        rloc = jnp.array([0.2], dtype=jnp.float32)
        c = jnp.array([[-4.1802368, 0.72507482, 0.0, 0.0]], dtype=jnp.float32)

        patch_sampled = build_local_potential(
            atom_coords, grid_coords, zion, rloc, c,
            spacing=0.48,
            local_subgrid=3,
            local_mode="patch",
            local_patch_radius_factor=4.0,
        )
        averaged = build_local_potential(
            atom_coords, grid_coords, zion, rloc, c,
            spacing=0.48,
            local_subgrid=3,
        )

        self.assertAlmostEqual(float(patch_sampled[0, 0, 0]), float(averaged[0, 0, 0]), places=6)

    def test_energy_and_forces_accepts_local_patch_mode(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")

        energy, forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            local_subgrid=3,
            local_mode="patch",
            local_patch_radius_factor=4.0,
        )

        self.assertTrue(jnp.isfinite(energy))
        self.assertEqual(forces.shape, coords.shape)


class DoubleGridProjectorTest(unittest.TestCase):
    def test_projector_uses_subgrid_cell_average(self):
        class Grid:
            pass

        grid = Grid()
        grid.coords = jnp.array([[[[0.0, 0.0, 0.0]]]], dtype=jnp.float32)
        grid.spacing = jnp.asarray(0.48, dtype=jnp.float32)
        atom_coords = jnp.array([[0.17, -0.11, 0.09]], dtype=jnp.float32)
        rp = 0.3
        pseudos = [{
            "projectors": [{
                "l": 0,
                "r": rp,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        subgrid = 3

        P_i, P_j, coeffs = precompute_projectors(
            grid, atom_coords, pseudos, projector_subgrid=subgrid,
        )

        offsets_1d = (jnp.arange(subgrid, dtype=jnp.float32) + 0.5) / subgrid - 0.5
        offsets_1d = offsets_1d * grid.spacing
        ox, oy, oz = jnp.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
        offsets = jnp.stack([ox, oy, oz], axis=-1).reshape(-1, 3)
        r = jnp.linalg.norm(offsets - atom_coords[0], axis=-1)
        expected = jnp.mean(get_gth_projector(r, 0, 1, rp))

        self.assertAlmostEqual(float(P_i[0, 0, 0, 0]), float(expected), places=6)
        self.assertAlmostEqual(float(P_j[0, 0, 0, 0]), float(expected), places=6)
        self.assertAlmostEqual(float(coeffs[0]), float(1.25 / (4.0 * jnp.pi)), places=6)

    def test_energy_and_forces_accepts_projector_subgrid(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")

        energy, forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            local_subgrid=3, projector_subgrid=3,
        )

        self.assertTrue(jnp.isfinite(energy))
        self.assertEqual(forces.shape, coords.shape)

    def test_projector_patch_mode_reduces_to_center_sample(self):
        grid = create_grid(1.0, [2.0, 2.0, 2.0])
        atom_coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        rp = 0.3
        pseudos = [{
            "projectors": [{
                "l": 0,
                "r": rp,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]

        data = precompute_projectors(
            grid,
            atom_coords,
            pseudos,
            projector_mode="patch",
            projector_subgrid=1,
            projector_patch_radius_factor=0.01,
        )

        expected = get_gth_projector(jnp.asarray(0.0, dtype=jnp.float32), 0, 1, rp)
        tag, _, _, P_i, P_j, coeffs, _, _ = data
        self.assertEqual(tag, "fine_integral")
        self.assertAlmostEqual(float(P_i[0, 0]), float(expected), places=6)
        self.assertAlmostEqual(float(P_j[0, 0]), float(expected), places=6)
        self.assertAlmostEqual(float(coeffs[0]), float(1.25 / (4.0 * jnp.pi)), places=6)

    def test_fine_integral_projector_applies_overlap_and_adjoint_scatter(self):
        grid = create_grid(1.0, [2.0, 2.0, 2.0])
        atom_coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        pseudos = [{
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]

        data = precompute_projectors(
            grid,
            atom_coords,
            pseudos,
            projector_mode="patch",
            projector_subgrid=1,
            projector_patch_radius_factor=0.01,
        )
        psi = jnp.ones(grid.shape, dtype=jnp.float32)
        v_nonlocal = apply_nonlocal_fine_integral(psi, *data[1:])

        p0 = get_gth_projector(jnp.asarray(0.0, dtype=jnp.float32), 0, 1, 0.3)
        expected_center = (1.25 / (4.0 * jnp.pi)) * p0 * p0
        self.assertAlmostEqual(float(v_nonlocal[1, 1, 1]), float(expected_center), places=5)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(v_nonlocal))), float(expected_center), places=5)

    def test_energy_and_forces_accepts_projector_patch_mode(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")

        energy, forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            local_subgrid=3,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=2.0,
        )

        self.assertTrue(jnp.isfinite(energy))
        self.assertEqual(forces.shape, coords.shape)


class UnifiedFineGridInterfaceTest(unittest.TestCase):
    def test_atom_patch_mode_matches_explicit_local_patch_settings(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")

        unified_energy, unified_forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            fine_grid_mode="atom_patch",
            fine_subgrid=2,
            fine_grid_radius_factor=2.0,
        )
        explicit_energy, explicit_forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            local_subgrid=2,
            local_mode="patch",
            local_patch_radius_factor=2.0,
            projector_subgrid=1,
            projector_mode="cell_average",
        )

        self.assertAlmostEqual(float(unified_energy), float(explicit_energy), places=6)
        self.assertEqual(unified_forces.shape, explicit_forces.shape)

    def test_auto_mode_matches_recommended_local_patch_settings(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")

        auto_energy, auto_forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            fine_grid_mode="auto",
        )
        explicit_energy, explicit_forces = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            local_subgrid=5,
            local_mode="patch",
            local_patch_radius_factor=4.0,
            projector_subgrid=1,
            projector_mode="cell_average",
        )

        self.assertAlmostEqual(float(auto_energy), float(explicit_energy), places=6)
        self.assertEqual(auto_forces.shape, explicit_forces.shape)

    def test_fine_grid_mode_off_keeps_point_sampling(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")

        off_energy, _ = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
            fine_grid_mode="off",
        )
        baseline_energy, _ = energy_and_forces(
            grid, coords, pseudos,
            max_iter=1, mix_alpha=0.3, tolerance=1e-5, key=None,
        )

        self.assertAlmostEqual(float(off_energy), float(baseline_energy), places=6)


if __name__ == "__main__":
    unittest.main()
