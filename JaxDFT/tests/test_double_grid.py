import unittest
import warnings

import jax.numpy as jnp

from JaxDFT.src.hamiltonian import create_grid
from JaxDFT.src.hamiltonian import (
    apply_nonlocal_fine_integral,
    build_patch_polynomial_reconstruction_data,
    build_fine_interpolation_data,
    build_local_potential,
    gather_fine_values,
    get_gth_projector,
    gth_local_potential_value,
    precompute_projectors,
    reconstruct_fine_wavefunction,
    reconstruct_patch_wavefunction,
    scatter_fine_values_adjoint,
    scatter_patch_wavefunction_adjoint,
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
    def test_fine_interpolation_reproduces_cubic_polynomial(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        x = grid.coords[..., 0]
        y = grid.coords[..., 1]
        z = grid.coords[..., 2]
        psi = 1.0 + 0.5 * x - 0.25 * y**2 + 0.125 * z**3 + 0.2 * x * y * z

        positions = jnp.array([
            [0.25, -0.10, 0.35],
            [-0.60, 0.40, -0.20],
        ], dtype=jnp.float32)
        flat_indices, weights, valid = build_fine_interpolation_data(grid, positions)
        gathered = gather_fine_values(psi, flat_indices, weights)

        expected = (
            1.0
            + 0.5 * positions[:, 0]
            - 0.25 * positions[:, 1] ** 2
            + 0.125 * positions[:, 2] ** 3
            + 0.2 * positions[:, 0] * positions[:, 1] * positions[:, 2]
        )

        self.assertTrue(bool(jnp.all(valid)))
        self.assertTrue(jnp.allclose(gathered, expected, atol=1e-6, rtol=1e-6))

    def test_fine_scatter_is_adjoint_of_fine_gather(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        positions = jnp.array([
            [0.15, -0.35, 0.20],
            [-0.55, 0.25, -0.10],
            [0.30, 0.40, -0.45],
        ], dtype=jnp.float32)
        flat_indices, weights, valid = build_fine_interpolation_data(grid, positions)

        psi = jnp.arange(jnp.prod(jnp.array(grid.shape)), dtype=jnp.float32).reshape(grid.shape) / 10.0
        fine_values = jnp.array([0.7, -1.1, 0.3], dtype=jnp.float32)
        fine_dv = jnp.asarray(0.125, dtype=jnp.float32)
        coarse_dv = jnp.asarray(grid.volume_element, dtype=jnp.float32)

        gathered = gather_fine_values(psi, flat_indices, weights)
        scattered = scatter_fine_values_adjoint(
            fine_values,
            psi.shape,
            flat_indices,
            weights,
            fine_dv,
            coarse_dv,
        )

        lhs = fine_dv * jnp.sum(gathered * fine_values * valid.astype(jnp.float32))
        rhs = coarse_dv * jnp.sum(psi * scattered)
        self.assertAlmostEqual(float(lhs), float(rhs), places=6)

    def test_reconstructed_fine_wavefunction_preserves_interpolated_patch_mass(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        x = grid.coords[..., 0]
        y = grid.coords[..., 1]
        z = grid.coords[..., 2]
        psi = 1.0 + 0.2 * x + 0.1 * y**2 - 0.05 * z**3
        positions = jnp.array([
            [0.25, -0.10, 0.35],
            [-0.60, 0.40, -0.20],
            [0.30, 0.10, -0.45],
        ], dtype=jnp.float32)
        flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
        fine_dv = jnp.asarray(0.125, dtype=jnp.float32)

        psi_fine = reconstruct_fine_wavefunction(psi, flat_indices, weights, fine_dv)
        rho_fine = gather_fine_values(psi * psi, flat_indices, weights)

        mass_from_wavefunction = fine_dv * jnp.sum(psi_fine * psi_fine)
        mass_from_density = fine_dv * jnp.sum(rho_fine)
        self.assertAlmostEqual(float(mass_from_wavefunction), float(mass_from_density), places=6)

    def test_patch_polynomial_reconstruction_reproduces_quadratic_function(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        atom_coord = jnp.array([0.2, -0.15, 0.1], dtype=jnp.float32)
        patch_positions = jnp.array([
            [0.10, -0.05, 0.20],
            [-0.20, 0.15, -0.10],
            [0.35, -0.25, 0.05],
        ], dtype=jnp.float32)
        sample_indices, eval_matrix = build_patch_polynomial_reconstruction_data(
            grid,
            atom_coord,
            patch_positions,
        )

        x = grid.coords[..., 0]
        y = grid.coords[..., 1]
        z = grid.coords[..., 2]
        psi = 1.0 + 0.3 * x - 0.2 * y + 0.1 * z + 0.05 * x * x - 0.07 * y * z + 0.08 * x * y

        coarse_samples = psi.reshape(-1)[sample_indices]
        reconstructed = eval_matrix @ coarse_samples
        expected = (
            1.0
            + 0.3 * patch_positions[:, 0]
            - 0.2 * patch_positions[:, 1]
            + 0.1 * patch_positions[:, 2]
            + 0.05 * patch_positions[:, 0] ** 2
            - 0.07 * patch_positions[:, 1] * patch_positions[:, 2]
            + 0.08 * patch_positions[:, 0] * patch_positions[:, 1]
        )
        self.assertTrue(jnp.allclose(reconstructed, expected, atol=5e-4, rtol=5e-4))

    def test_patch_polynomial_reconstruction_has_matching_adjoint(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        atom_coord = jnp.array([0.2, -0.15, 0.1], dtype=jnp.float32)
        patch_positions = jnp.array([
            [0.10, -0.05, 0.20],
            [-0.20, 0.15, -0.10],
            [0.35, -0.25, 0.05],
        ], dtype=jnp.float32)
        sample_indices, eval_matrix = build_patch_polynomial_reconstruction_data(
            grid,
            atom_coord,
            patch_positions,
        )
        sample_indices = sample_indices[None, :]
        eval_matrix = eval_matrix[None, :, :]
        psi = jnp.arange(jnp.prod(jnp.array(grid.shape)), dtype=jnp.float32).reshape(grid.shape) / 10.0
        fine_values = jnp.array([[0.7, -1.1, 0.3]], dtype=jnp.float32)
        fine_dv = jnp.asarray(0.125, dtype=jnp.float32)
        coarse_dv = jnp.asarray(grid.volume_element, dtype=jnp.float32)

        psi_patch = reconstruct_patch_wavefunction(psi, sample_indices, eval_matrix)
        scattered = scatter_patch_wavefunction_adjoint(
            fine_values,
            psi.shape,
            sample_indices,
            eval_matrix,
            fine_dv,
            coarse_dv,
        )
        lhs = fine_dv * jnp.sum(psi_patch * fine_values)
        rhs = coarse_dv * jnp.sum(psi * scattered)
        self.assertAlmostEqual(float(lhs), float(rhs), places=6)

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
        tag = data[0]
        P_i = data[3]
        P_j = data[4]
        coeffs = data[5]
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
        expected_coeff = (1.25 / (4.0 * jnp.pi)) * p0 * p0
        patch_sample_indices = data[8][0]
        patch_eval_matrix = data[9][0, 0]
        expected_flat = jnp.zeros((jnp.prod(jnp.array(grid.shape)),), dtype=jnp.float32)
        expected_flat = expected_flat.at[patch_sample_indices].add(expected_coeff * patch_eval_matrix)
        expected = expected_flat.reshape(grid.shape)

        self.assertTrue(jnp.allclose(v_nonlocal, expected, atol=1e-5, rtol=1e-5))

    def test_fine_integral_projector_is_linear_in_psi(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        atom_coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
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
            projector_subgrid=2,
            projector_patch_radius_factor=4.0,
        )
        psi_a = jnp.sin(grid.coords[..., 0]) + 0.2 * jnp.cos(grid.coords[..., 1])
        psi_b = 0.3 * grid.coords[..., 2] - 0.1 * grid.coords[..., 0] * grid.coords[..., 1]

        applied_a = apply_nonlocal_fine_integral(psi_a, *data[1:])
        applied_b = apply_nonlocal_fine_integral(psi_b, *data[1:])
        applied_sum = apply_nonlocal_fine_integral(psi_a + psi_b, *data[1:])

        self.assertTrue(
            jnp.allclose(applied_sum, applied_a + applied_b, atol=1e-6, rtol=1e-6)
        )

    def test_energy_and_forces_accepts_projector_patch_mode(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
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
        self.assertTrue(any("experimental" in str(w.message).lower() for w in caught))


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
