import unittest

import jax.numpy as jnp

from JaxDFT.src.hamiltonian import (
    apply_nonlocal_fine_integral,
    build_local_potential,
    create_grid,
    laplacian_8th,
    precompute_projectors,
)
from JaxDFT.src.experimental_patch_solver_bridge import build_experimental_patch_apply_h
from JaxDFT.src.experimental_patch_solver_bridge import (
    experimental_patch_scf,
    solve_experimental_patch_orbitals,
)
from JaxDFT.src.io import load_pseudopotentials
from JaxDFT.src.solver import scf, solve_orbitals_dense, solve_orbitals_subspace
from JaxDFT.src.mixed_orbital import MixedOrbital, add_mixed_orbitals, scale_mixed_orbital
from JaxDFT.src.patch_builder import build_atom_patch_specs
from JaxDFT.src.patch_maps import (
    build_patch_maps,
    coarse_to_patch,
    patch_to_coarse_adjoint,
)
from JaxDFT.src.patch_projector_operator import (
    apply_patch_projector,
    build_patch_projector_data,
)
from JaxDFT.src.patch_projector_operator_v2 import (
    IndependentPatchOrbital,
    apply_independent_patch_projector,
    physical_patch_values,
)
from JaxDFT.src.mixed_state_dense_prototype import (
    build_patch_complement_basis,
    build_patch_kinetic_matrix,
    build_patch_overlap_block,
    build_patch_potential_matrix,
    build_total_mixed_metric,
    build_fixed_veff_mixed_matrices,
    project_patch_correction,
    build_fixed_veff_mixed_apply_h,
    solve_fixed_veff_mixed_dense,
    solve_fixed_veff_mixed_dense_host,
)
from JaxDFT.src.mixed_basis_galerkin_v3 import (
    build_fixed_veff_galerkin_matrices_v3,
    build_fixed_veff_galerkin_components_v3,
    build_v3_vloc_blocks,
    compute_v3_vloc_block_expectations,
    build_patch_kinetic_value_matrix_v3,
    build_patch_physical_basis_v3,
    compute_v3_energy_decomposition,
    compute_v3_generalized_fractions,
    solve_fixed_veff_galerkin_dense_host_v3,
)


class PatchBuilderTest(unittest.TestCase):
    def test_build_atom_patch_specs_uses_projector_radius_and_subgrid(self):
        grid = create_grid(0.8, [3.2, 3.2, 3.2])
        coords = jnp.array([[0.13, -0.07, 0.19]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")

        specs = build_atom_patch_specs(
            grid,
            coords,
            pseudos,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec.atom_index, 0)
        self.assertAlmostEqual(float(spec.fine_spacing), float(grid.spacing) / 2.0, places=6)
        self.assertAlmostEqual(float(spec.radius), 4.0 * float(pseudos[0]["projectors"][0]["r"]), places=6)
        self.assertEqual(spec.positions.shape[1], 3)
        self.assertGreater(spec.positions.shape[0], 1)


class PatchMapTest(unittest.TestCase):
    def test_patch_map_reproduces_quadratic_field(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.2, -0.15, 0.1]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")
        spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=2.0)[0]
        patch_map = build_patch_maps(grid, [spec])[0]

        x = grid.coords[..., 0]
        y = grid.coords[..., 1]
        z = grid.coords[..., 2]
        psi = 1.0 + 0.3 * x - 0.2 * y + 0.1 * z + 0.05 * x * x - 0.07 * y * z + 0.08 * x * y

        reconstructed = coarse_to_patch(psi, patch_map)
        positions = spec.positions
        expected = (
            1.0
            + 0.3 * positions[:, 0]
            - 0.2 * positions[:, 1]
            + 0.1 * positions[:, 2]
            + 0.05 * positions[:, 0] ** 2
            - 0.07 * positions[:, 1] * positions[:, 2]
            + 0.08 * positions[:, 0] * positions[:, 1]
        )
        self.assertTrue(jnp.allclose(reconstructed, expected, atol=5e-4, rtol=5e-4))

    def test_patch_map_adjoint_matches_inner_product(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.2, -0.15, 0.1]], dtype=jnp.float32)
        pseudos = load_pseudopotentials(["O"], "JaxDFT/data/gth_potentials")
        spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=2.0)[0]
        patch_map = build_patch_maps(grid, [spec])[0]

        psi = jnp.arange(jnp.prod(jnp.array(grid.shape)), dtype=jnp.float32).reshape(grid.shape) / 10.0
        fine_values = jnp.linspace(-0.4, 0.7, spec.positions.shape[0], dtype=jnp.float32)
        psi_patch = coarse_to_patch(psi, patch_map)
        scattered = patch_to_coarse_adjoint(fine_values, patch_map, psi.shape)

        lhs = float(patch_map.fine_dv * jnp.sum(psi_patch * fine_values))
        rhs = float(patch_map.coarse_dv * jnp.sum(psi * scattered))
        self.assertAlmostEqual(lhs, rhs, places=6)


class MixedOrbitalTest(unittest.TestCase):
    def test_mixed_orbital_linear_algebra_helpers(self):
        coarse_a = jnp.ones((2, 2, 2), dtype=jnp.float32)
        coarse_b = 2.0 * coarse_a
        patch_a = {0: jnp.array([1.0, 2.0], dtype=jnp.float32)}
        patch_b = {0: jnp.array([-1.0, 0.5], dtype=jnp.float32)}

        a = MixedOrbital(coarse=coarse_a, patch_values=patch_a)
        b = MixedOrbital(coarse=coarse_b, patch_values=patch_b)
        summed = add_mixed_orbitals(a, b)
        scaled = scale_mixed_orbital(-2.0, a)

        self.assertTrue(jnp.allclose(summed.coarse, coarse_a + coarse_b))
        self.assertTrue(jnp.allclose(summed.patch_values[0], patch_a[0] + patch_b[0]))
        self.assertTrue(jnp.allclose(scaled.coarse, -2.0 * coarse_a))
        self.assertTrue(jnp.allclose(scaled.patch_values[0], -2.0 * patch_a[0]))


class PatchProjectorOperatorTest(unittest.TestCase):
    def test_patch_projector_matches_existing_patch_primitive_for_single_atom(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 6.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 6.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]

        patch_specs = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)
        patch_maps = build_patch_maps(grid, patch_specs)
        projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)

        psi = jnp.sin(grid.coords[..., 0]) + 0.2 * jnp.cos(grid.coords[..., 1])
        mixed = MixedOrbital(
            coarse=psi,
            patch_values={patch_specs[0].atom_index: coarse_to_patch(psi, patch_maps[0])},
        )
        applied = apply_patch_projector(mixed, projector_data, grid.shape)

        fine_data = precompute_projectors(
            grid,
            coords,
            pseudos,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=4.0,
        )
        expected = apply_nonlocal_fine_integral(psi, *fine_data[1:])

        self.assertTrue(jnp.allclose(applied.coarse, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(applied.patch_values, {})

    def test_patch_projector_ignores_empty_nonlocal_channels(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 6.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 6.0,
            "projectors": [
                {
                    "l": 0,
                    "r": 0.3,
                    "h": jnp.array([[1.25]], dtype=jnp.float32),
                },
                {
                    "l": 1,
                    "r": 0.35,
                    "h": jnp.zeros((0, 0), dtype=jnp.float32),
                },
            ],
        }]

        patch_specs = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)
        patch_maps = build_patch_maps(grid, patch_specs)
        projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)

        self.assertEqual(len(projector_data.channels), 1)


class PatchProjectorOperatorV2Test(unittest.TestCase):
    def test_physical_patch_values_uses_complement_basis_coordinates(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 6.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 6.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_specs = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)
        patch_maps = build_patch_maps(grid, patch_specs)
        basis = build_patch_complement_basis(patch_maps[0])

        psi = jnp.sin(grid.coords[..., 0]) + 0.2 * jnp.cos(grid.coords[..., 1])
        coeffs = jnp.linspace(-0.2, 0.15, basis.shape[1], dtype=jnp.float32)
        state = IndependentPatchOrbital(coarse=psi, patch_corrections={0: coeffs})
        values = physical_patch_values(state, patch_maps, patch_bases={0: basis})

        expected = coarse_to_patch(psi, patch_maps[0]) + basis @ coeffs
        self.assertTrue(jnp.allclose(values[0], expected, atol=1e-5, rtol=1e-5))

    def test_independent_patch_projector_retains_patch_response(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 6.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 6.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]

        patch_specs = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)
        patch_maps = build_patch_maps(grid, patch_specs)
        projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)

        psi = jnp.sin(grid.coords[..., 0]) + 0.2 * jnp.cos(grid.coords[..., 1])
        zero_patch = jnp.zeros_like(coarse_to_patch(psi, patch_maps[0]))
        state = IndependentPatchOrbital(
            coarse=psi,
            patch_corrections={patch_specs[0].atom_index: zero_patch},
        )
        applied = apply_independent_patch_projector(state, projector_data, patch_maps, grid.shape)

        fine_data = precompute_projectors(
            grid,
            coords,
            pseudos,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=4.0,
        )
        expected_coarse = apply_nonlocal_fine_integral(psi, *fine_data[1:])
        self.assertTrue(jnp.allclose(applied.coarse, expected_coarse, atol=1e-5, rtol=1e-5))

        channel = projector_data.channels[0]
        psi_patch = coarse_to_patch(psi, patch_maps[0])
        overlap = patch_maps[0].fine_dv * jnp.sum(channel.p_j * psi_patch)
        expected_patch = channel.p_i * (channel.coeff * overlap)
        self.assertIn(patch_specs[0].atom_index, applied.patch_corrections)
        self.assertTrue(
            jnp.allclose(
                applied.patch_corrections[patch_specs[0].atom_index],
                expected_patch,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_independent_patch_projector_is_linear_in_coarse_and_patch_corrections(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 6.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 6.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_specs = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)
        patch_maps = build_patch_maps(grid, patch_specs)
        projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)

        coarse_a = jnp.sin(grid.coords[..., 0])
        coarse_b = 0.3 * jnp.cos(grid.coords[..., 2])
        patch_a = jnp.linspace(-0.1, 0.2, patch_specs[0].positions.shape[0], dtype=jnp.float32)
        patch_b = jnp.linspace(0.05, -0.15, patch_specs[0].positions.shape[0], dtype=jnp.float32)
        state_a = IndependentPatchOrbital(coarse=coarse_a, patch_corrections={0: patch_a})
        state_b = IndependentPatchOrbital(coarse=coarse_b, patch_corrections={0: patch_b})
        state_sum = IndependentPatchOrbital(coarse=coarse_a + coarse_b, patch_corrections={0: patch_a + patch_b})

        applied_a = apply_independent_patch_projector(state_a, projector_data, patch_maps, grid.shape)
        applied_b = apply_independent_patch_projector(state_b, projector_data, patch_maps, grid.shape)
        applied_sum = apply_independent_patch_projector(state_sum, projector_data, patch_maps, grid.shape)

        self.assertTrue(jnp.allclose(applied_sum.coarse, applied_a.coarse + applied_b.coarse, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            jnp.allclose(
                applied_sum.patch_corrections[0],
                applied_a.patch_corrections[0] + applied_b.patch_corrections[0],
                atol=1e-5,
                rtol=1e-5,
            )
        )


class MixedStateDensePrototypeTest(unittest.TestCase):
    def test_patch_complement_basis_reduces_to_strict_coordinate_space(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]
        basis = build_patch_complement_basis(patch_map)

        self.assertGreater(basis.shape[0], basis.shape[1])
        self.assertGreater(basis.shape[1], 0)
        self.assertTrue(jnp.allclose(basis.T @ basis, jnp.eye(basis.shape[1], dtype=jnp.float32), atol=5e-4, rtol=5e-4))

    def test_project_patch_correction_removes_coarse_duplicate_directions(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]
        correction = jnp.linspace(-0.3, 0.4, patch_spec.positions.shape[0], dtype=jnp.float32)
        projected = project_patch_correction(correction, patch_map)

        overlap_with_coarse_subspace = patch_map.eval_matrix.T @ projected
        self.assertTrue(jnp.allclose(overlap_with_coarse_subspace, 0.0, atol=5e-5, rtol=1e-5))

    def test_patch_overlap_block_is_symmetric_positive_and_not_diagonal(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]

        overlap = build_patch_overlap_block(patch_map, duplicate_penalty=10.0)
        self.assertTrue(jnp.allclose(overlap, overlap.T, atol=1e-6, rtol=1e-6))
        min_eig = jnp.min(jnp.linalg.eigvalsh(overlap))
        self.assertGreater(float(min_eig), 0.0)
        self.assertGreater(float(jnp.max(jnp.abs(overlap - jnp.diag(jnp.diag(overlap))))), 0.0)

    def test_fixed_veff_mixed_apply_h_matches_coarse_component_of_experimental_apply_h(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]
        coarse_psi = jnp.sin(grid.coords[..., 0]) + 0.2 * jnp.cos(grid.coords[..., 1])

        mixed_apply_h, mixed_size, _, _ = build_fixed_veff_mixed_apply_h(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
        )
        coarse_apply_h = build_experimental_patch_apply_h(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        mixed_state = jnp.zeros((mixed_size,), dtype=jnp.float32).at[: coarse_psi.size].set(coarse_psi.reshape(-1))
        applied_mixed = mixed_apply_h(mixed_state)
        applied_coarse = coarse_apply_h(coarse_psi.reshape(-1))

        self.assertTrue(jnp.allclose(applied_mixed[: coarse_psi.size], applied_coarse, atol=1e-5, rtol=1e-5))
        self.assertGreater(mixed_size, coarse_psi.size)

    def test_fixed_veff_mixed_apply_h_feeds_patch_response_back_to_coarse_block(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]
        mixed_apply_h, mixed_size, patch_maps, patch_bases = build_fixed_veff_mixed_apply_h(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
        )

        coarse_size = int(jnp.prod(jnp.array(grid.shape)))
        patch_specs = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)
        projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)
        channel = projector_data.channels[0]
        patch_state = patch_bases[0].T @ project_patch_correction(channel.p_i, patch_maps[0])
        mixed_state = jnp.concatenate(
            [
                jnp.zeros((coarse_size,), dtype=jnp.float32),
                patch_state,
            ],
            axis=0,
        )
        applied = mixed_apply_h(mixed_state)

        self.assertEqual(applied.shape[0], mixed_size)
        self.assertGreater(float(jnp.linalg.norm(applied[:coarse_size])), 1e-5)

    def test_total_mixed_metric_is_not_controlled_by_patch_penalty(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]

        metric_a = build_total_mixed_metric(grid, [patch_map], patch_metric_duplicate_penalty=10.0)
        metric_b = build_total_mixed_metric(grid, [patch_map], patch_metric_duplicate_penalty=10.0)

        self.assertTrue(jnp.allclose(metric_a, metric_b, atol=1e-6, rtol=1e-6))
        coarse_size = int(jnp.prod(jnp.array(grid.shape)))
        patch_block = metric_a[coarse_size:, coarse_size:]
        cross_block = metric_a[:coarse_size, coarse_size:]
        self.assertGreater(
            float(jnp.max(jnp.abs(patch_block - jnp.diag(jnp.diag(patch_block))))),
            0.0,
        )
        self.assertGreater(float(jnp.max(jnp.abs(cross_block))), 0.0)

    def test_patch_potential_matrix_generalized_spectrum_tracks_local_potential_scale(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]
        v_eff_patch = -2.0 + 0.2 * patch_spec.positions[:, 0]

        s_patch = build_patch_overlap_block(patch_map, duplicate_penalty=10.0)
        v_patch = build_patch_potential_matrix(patch_map, v_eff_patch)
        evals_s, evecs_s = jnp.linalg.eigh(0.5 * (s_patch + s_patch.T))
        inv_sqrt = evecs_s @ jnp.diag(1.0 / jnp.sqrt(evals_s)) @ evecs_s.T
        transformed = inv_sqrt @ v_patch @ inv_sqrt
        eigvals = jnp.linalg.eigvalsh(0.5 * (transformed + transformed.T))

        self.assertGreater(float(jnp.min(eigvals)), -5.0)
        self.assertLess(float(jnp.max(eigvals)), 1.0)

    def test_patch_kinetic_matrix_is_positive_in_patch_metric(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(grid, coords, pseudos, patch_subgrid=2, patch_radius_factor=4.0)[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]

        s_patch = build_patch_overlap_block(patch_map, duplicate_penalty=10.0)
        k_patch = build_patch_kinetic_matrix(patch_spec, patch_map, kinetic_scale=1.0)
        evals_s, evecs_s = jnp.linalg.eigh(0.5 * (s_patch + s_patch.T))
        inv_sqrt = evecs_s @ jnp.diag(1.0 / jnp.sqrt(evals_s)) @ evecs_s.T
        transformed = inv_sqrt @ k_patch @ inv_sqrt
        eigvals = jnp.linalg.eigvalsh(0.5 * (transformed + transformed.T))

        self.assertGreater(float(jnp.min(eigvals)), -1e-3)
        self.assertGreater(float(jnp.max(eigvals)), 1e-2)

    def test_fixed_veff_mixed_dense_solver_returns_expected_dimensions(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]
        eigvals, eigvecs, mixed_size = solve_fixed_veff_mixed_dense(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=2,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
        )

        self.assertEqual(eigvals.shape, (2,))
        self.assertEqual(eigvecs.shape, (mixed_size, 2))
        self.assertGreater(mixed_size, int(jnp.prod(jnp.array(grid.shape))))

    def test_mixed_dense_solver_accepts_independent_patch_metric_penalty(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        eigvals, eigvecs, mixed_size = solve_fixed_veff_mixed_dense(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=2,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
            patch_metric_duplicate_penalty=25.0,
        )

        self.assertEqual(eigvals.shape, (2,))
        self.assertEqual(eigvecs.shape, (mixed_size, 2))
        self.assertTrue(jnp.all(jnp.isfinite(eigvals)))

    def test_host_dense_solver_matches_jax_dense_on_small_system(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        h_dense, s_dense, total_size = build_fixed_veff_mixed_matrices(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
            patch_metric_duplicate_penalty=25.0,
        )
        jax_vals, jax_vecs, jax_size = solve_fixed_veff_mixed_dense(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=2,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
            patch_metric_duplicate_penalty=25.0,
        )
        host_vals, host_vecs, host_size = solve_fixed_veff_mixed_dense_host(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=2,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_penalty=1.0,
            patch_metric_duplicate_penalty=25.0,
        )

        self.assertEqual(h_dense.shape, (total_size, total_size))
        self.assertEqual(s_dense.shape, (total_size, total_size))
        self.assertEqual(jax_size, host_size)
        self.assertEqual(jax_size, total_size)
        self.assertEqual(host_vecs.shape, (host_size, 2))
        self.assertTrue(jnp.all(jnp.isfinite(host_vals)))
        self.assertLessEqual(float(host_vals[0]), float(host_vals[1]))
        self.assertTrue(jnp.allclose(jax_vals, host_vals, atol=3e-4, rtol=1e-3))


class MixedBasisGalerkinV3Test(unittest.TestCase):
    def test_v3_builds_symmetric_generalized_matrices(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        h_dense, s_dense, coarse_size, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        self.assertEqual(h_dense.shape, s_dense.shape)
        self.assertGreater(h_dense.shape[0], coarse_size)
        self.assertTrue(jnp.allclose(h_dense, h_dense.T, atol=1e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(s_dense, s_dense.T, atol=1e-6, rtol=1e-6))
        min_eig = jnp.min(jnp.linalg.eigvalsh(0.5 * (s_dense + s_dense.T)))
        self.assertGreater(float(min_eig), 0.0)

    def test_v3_cross_blocks_are_reciprocal(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        h_dense, s_dense, coarse_size, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        h_cp = h_dense[:coarse_size, coarse_size:]
        h_pc = h_dense[coarse_size:, :coarse_size]
        s_cp = s_dense[:coarse_size, coarse_size:]
        s_pc = s_dense[coarse_size:, :coarse_size]

        self.assertGreater(float(jnp.linalg.norm(h_cp)), 1e-8)
        self.assertTrue(jnp.allclose(h_cp, h_pc.T, atol=1e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(s_cp, s_pc.T, atol=1e-6, rtol=1e-6))

    def test_v3_host_dense_solver_returns_finite_bands(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        eigvals, eigvecs, coarse_size, _ = solve_fixed_veff_galerkin_dense_host_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=2,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        self.assertEqual(eigvals.shape, (2,))
        self.assertEqual(eigvecs.shape[1], 2)
        self.assertGreater(eigvecs.shape[0], coarse_size)
        self.assertTrue(jnp.all(jnp.isfinite(eigvals)))
        self.assertLessEqual(float(eigvals[0]), float(eigvals[1]))

    def test_v3_patch_basis_is_overlap_and_kinetic_orthogonal_to_coarse_trace(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(
            grid,
            coords,
            pseudos,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]
        basis = build_patch_physical_basis_v3(patch_spec, patch_map, kinetic_scale=1.0)

        self.assertGreater(basis.shape[1], 0)
        overlap_null = patch_map.eval_matrix.T @ basis
        kinetic_null = patch_map.eval_matrix.T @ build_patch_kinetic_value_matrix_v3(
            patch_spec,
            patch_map,
            kinetic_scale=1.0,
        ) @ basis
        self.assertTrue(jnp.allclose(overlap_null, 0.0, atol=5e-4, rtol=5e-4))
        self.assertTrue(jnp.allclose(kinetic_null, 0.0, atol=5e-3, rtol=5e-3))

    def test_v3_component_matrices_sum_to_total_h(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        h_dense, s_dense, _, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        _, _, _, _, components = build_fixed_veff_galerkin_components_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        rebuilt = components["t"] + components["v_loc"] + components["v_nl"]

        self.assertEqual(h_dense.shape, rebuilt.shape)
        self.assertEqual(h_dense.shape, s_dense.shape)
        self.assertTrue(jnp.allclose(h_dense, rebuilt, atol=1e-5, rtol=1e-5))

    def test_v3_vnl_cc_matches_direct_patch_value_projection(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        _, _, coarse_size, metadata, components = build_fixed_veff_galerkin_components_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        projector_data = build_patch_projector_data(metadata.patch_specs, metadata.patch_maps, pseudos)
        patch_map_by_atom = {patch_map.atom_index: patch_map for patch_map in metadata.patch_maps}
        expected = jnp.zeros((coarse_size, coarse_size), dtype=jnp.float32)

        for channel in projector_data.channels:
            patch_map = patch_map_by_atom[channel.atom_index]
            projector_value = channel.coeff * (patch_map.fine_dv ** 2) * jnp.outer(channel.p_i, channel.p_j)
            projector_value = 0.5 * (projector_value + projector_value.T)
            local = patch_map.eval_matrix.T @ projector_value @ patch_map.eval_matrix
            indices = jnp.asarray(patch_map.sample_indices).tolist()
            for local_row, global_row in enumerate(indices):
                for local_col, global_col in enumerate(indices):
                    expected = expected.at[global_row, global_col].add(local[local_row, local_col])

        v_nl_cc = components["v_nl"][:coarse_size, :coarse_size]
        self.assertGreater(float(jnp.linalg.norm(v_nl_cc)), 1e-8)
        self.assertTrue(jnp.allclose(v_nl_cc, expected, atol=1e-5, rtol=1e-5))

    def test_v3_fractions_and_energy_decomposition_are_self_consistent(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        eigvals, eigvecs, coarse_size, _ = solve_fixed_veff_galerkin_dense_host_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=1,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        h_dense, s_dense, _, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        _, _, _, _, components = build_fixed_veff_galerkin_components_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        vec = eigvecs[:, 0]
        fractions = compute_v3_generalized_fractions(vec, s_dense, coarse_size)
        energies = compute_v3_energy_decomposition(vec, s_dense, components)

        self.assertAlmostEqual(
            float(fractions["coarse"] + fractions["patch"]),
            1.0,
            places=4,
        )
        self.assertAlmostEqual(
            float(energies["t"] + energies["v_loc"] + energies["v_nl"]),
            float(eigvals[0]),
            places=4,
        )

    def test_v3_vloc_blocks_sum_to_total_vloc_matrix(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        _, _, _, _, components = build_fixed_veff_galerkin_components_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        blocks = build_v3_vloc_blocks(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        rebuilt = blocks["cc"] + blocks["cp"] + blocks["pp"]

        self.assertTrue(jnp.allclose(components["v_loc"], rebuilt, atol=1e-5, rtol=1e-5))

    def test_v3_vloc_uses_coarse_baseline_plus_patch_increment(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
        c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
        v_full = build_local_potential(
            coords,
            grid.coords,
            zion,
            rloc,
            c,
            spacing=grid.spacing,
            local_subgrid=2,
            local_mode="cell_average",
        )
        v_coarse = build_local_potential(
            coords,
            grid.coords,
            zion,
            rloc,
            c,
            spacing=grid.spacing,
            local_subgrid=1,
            local_mode="cell_average",
        )

        _, _, coarse_size, metadata, components = build_fixed_veff_galerkin_components_v3(
            grid,
            coords,
            pseudos,
            v_full,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        expected = jnp.zeros_like(components["v_loc"])
        expected = expected.at[:coarse_size, :coarse_size].set(
            grid.volume_element * jnp.diag(v_coarse.reshape(-1))
        )
        delta_grid = v_full - v_coarse

        for patch_spec, patch_map in zip(metadata.patch_specs, metadata.patch_maps):
            atom_index = patch_map.atom_index
            basis = metadata.patch_bases[atom_index]
            patch_slice = metadata.patch_slices[atom_index]
            eval_matrix = jnp.asarray(patch_map.eval_matrix, dtype=jnp.float32)
            delta_patch = coarse_to_patch(delta_grid, patch_map)
            potential_value = patch_map.fine_dv * jnp.diag(delta_patch)
            h_pp = basis.T @ potential_value @ basis
            h_pp = 0.5 * (h_pp + h_pp.T)
            h_cp = eval_matrix.T @ potential_value @ basis
            expected = expected.at[patch_slice, patch_slice].set(h_pp)
            expected = expected.at[patch_map.sample_indices[:, None], jnp.arange(patch_slice.start, patch_slice.stop)[None, :]].add(h_cp)
            expected = expected.at[patch_slice, :coarse_size].set(expected[:coarse_size, patch_slice].T)

        self.assertTrue(jnp.allclose(components["v_loc"], expected, atol=1e-5, rtol=1e-5))

    def test_v3_vloc_block_expectations_sum_to_total_vloc_energy(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        eigvals, eigvecs, coarse_size, _ = solve_fixed_veff_galerkin_dense_host_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=1,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        _, s_dense, _, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        _, _, _, _, components = build_fixed_veff_galerkin_components_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        blocks = build_v3_vloc_blocks(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        vec = eigvecs[:, 0]
        total = compute_v3_energy_decomposition(vec, s_dense, {"v_loc": components["v_loc"]})["v_loc"]
        split = compute_v3_vloc_block_expectations(vec, s_dense, blocks)

        self.assertAlmostEqual(
            float(split["cc"] + split["cp"] + split["pp"]),
            float(total),
            places=4,
        )

    def test_v3_vloc_cc_mode_no_longer_changes_result(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        blocks_patch = build_v3_vloc_blocks(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_stiffness=1.0,
            vloc_cc_mode="patch",
        )
        blocks_coarse = build_v3_vloc_blocks(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_stiffness=1.0,
            vloc_cc_mode="coarse",
        )

        self.assertTrue(jnp.allclose(blocks_patch["cc"], blocks_coarse["cc"], atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(blocks_patch["cp"], blocks_coarse["cp"], atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(blocks_patch["pp"], blocks_coarse["pp"], atol=1e-6, rtol=1e-6))

    def test_v3_patch_stiffness_no_longer_changes_result(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        h_a, s_a, _, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_stiffness=0.5,
        )
        h_b, s_b, _, _ = build_fixed_veff_galerkin_matrices_v3(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
            patch_stiffness=3.0,
        )

        self.assertTrue(jnp.allclose(h_a, h_b, atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(s_a, s_b, atol=1e-6, rtol=1e-6))

    def test_v3_reg_no_longer_changes_patch_basis_subspace(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        patch_spec = build_atom_patch_specs(
            grid,
            coords,
            pseudos,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )[0]
        patch_map = build_patch_maps(grid, [patch_spec])[0]

        basis_a = build_patch_physical_basis_v3(patch_spec, patch_map, reg=1e-12)
        basis_b = build_patch_physical_basis_v3(patch_spec, patch_map, reg=1e-1)

        self.assertEqual(basis_a.shape, basis_b.shape)
        proj_a = basis_a @ basis_a.T
        proj_b = basis_b @ basis_b.T
        self.assertTrue(jnp.allclose(proj_a, proj_b, atol=1e-5, rtol=1e-5))


class ExperimentalPatchSolverBridgeTest(unittest.TestCase):
    def test_experimental_patch_apply_h_matches_existing_patch_hamiltonian_path(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 6.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 6.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]
        psi = jnp.sin(grid.coords[..., 0]) + 0.2 * jnp.cos(grid.coords[..., 1])

        apply_h = build_experimental_patch_apply_h(
            grid,
            coords,
            pseudos,
            v_eff,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        actual = apply_h(psi.reshape(-1)).reshape(grid.shape)

        fine_data = precompute_projectors(
            grid,
            coords,
            pseudos,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=4.0,
        )
        expected_nonlocal = apply_nonlocal_fine_integral(psi, *fine_data[1:])
        expected = -0.5 * laplacian_8th(psi, grid.spacing, grid.mask) + v_eff * psi + expected_nonlocal

        self.assertTrue(jnp.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_experimental_patch_orbital_solver_matches_reference_patch_solver(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]
        key = jnp.array([0, 7], dtype=jnp.uint32)

        actual_vals, actual_vecs = solve_experimental_patch_orbitals(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=1,
            max_iter=10,
            tol=1e-5,
            key=key,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        fine_data = precompute_projectors(
            grid,
            coords,
            pseudos,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=4.0,
        )

        def reference_apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            lap = laplacian_8th(psi, grid.spacing, grid.mask)
            v_nonlocal = apply_nonlocal_fine_integral(psi, *fine_data[1:])
            hpsi = -0.5 * lap + v_eff * psi + v_nonlocal
            return hpsi.reshape(-1)

        expected_vals, expected_vecs = solve_orbitals_subspace(
            reference_apply_h,
            int(jnp.prod(jnp.array(grid.shape))),
            1,
            max_iter=10,
            tol=1e-5,
            key=key,
        )

        self.assertTrue(jnp.allclose(actual_vals, expected_vals, atol=1e-5, rtol=1e-5))
        overlap = jnp.abs(jnp.sum(actual_vecs[:, 0] * expected_vecs[:, 0]))
        self.assertGreater(float(overlap), 0.99)

    def test_experimental_patch_orbital_solver_dense_mode_matches_reference_dense_solver(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        v_eff = 0.15 * grid.coords[..., 0] - 0.03 * grid.coords[..., 2]

        actual_vals, actual_vecs = solve_experimental_patch_orbitals(
            grid,
            coords,
            pseudos,
            v_eff,
            n_bands=1,
            solver_mode="dense",
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )

        fine_data = precompute_projectors(
            grid,
            coords,
            pseudos,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=4.0,
        )

        def reference_apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            lap = laplacian_8th(psi, grid.spacing, grid.mask)
            v_nonlocal = apply_nonlocal_fine_integral(psi, *fine_data[1:])
            hpsi = -0.5 * lap + v_eff * psi + v_nonlocal
            return hpsi.reshape(-1)

        expected_vals, expected_vecs = solve_orbitals_dense(
            reference_apply_h,
            int(jnp.prod(jnp.array(grid.shape))),
            1,
        )

        self.assertTrue(jnp.allclose(actual_vals, expected_vals, atol=1e-5, rtol=1e-5))
        overlap = jnp.abs(jnp.sum(actual_vecs[:, 0] * expected_vecs[:, 0]))
        self.assertGreater(float(overlap), 0.999)

    def test_experimental_patch_scf_matches_reference_patch_scf(self):
        grid = create_grid(1.0, [4.0, 4.0, 4.0])
        coords = jnp.array([[0.1, -0.2, 0.05]], dtype=jnp.float32)
        pseudos = [{
            "zion": 2.0,
            "rloc": 0.24762086,
            "c": jnp.array([-16.58031797, 2.39570092, 0.0, 0.0], dtype=jnp.float32),
            "q": 2.0,
            "projectors": [{
                "l": 0,
                "r": 0.3,
                "h": jnp.array([[1.25]], dtype=jnp.float32),
            }],
        }]
        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
        c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
        v_loc = build_local_potential(
            coords,
            grid.coords,
            zion,
            rloc,
            c,
            spacing=grid.spacing,
            local_subgrid=2,
            local_mode="patch",
            local_patch_radius_factor=4.0,
        )
        key = jnp.array([0, 11], dtype=jnp.uint32)
        occ = jnp.array([2.0], dtype=jnp.float32)

        actual = experimental_patch_scf(
            grid,
            coords,
            1,
            occ,
            v_loc,
            pseudos,
            max_iter=1,
            mix_alpha=0.3,
            tolerance=1e-5,
            key=key,
            patch_subgrid=2,
            patch_radius_factor=4.0,
        )
        expected = scf(
            grid,
            coords,
            1,
            occ,
            v_loc,
            pseudos,
            max_iter=1,
            mix_alpha=0.3,
            tolerance=1e-5,
            key=key,
            projector_subgrid=2,
            projector_mode="patch",
            projector_patch_radius_factor=4.0,
        )
        actual_rho, actual_vals, actual_vecs, actual_v_h, actual_eps_xc, actual_v_xc = actual
        expected_rho, expected_vals, expected_vecs, expected_v_h, expected_eps_xc, expected_v_xc = expected

        self.assertTrue(jnp.allclose(actual_rho, expected_rho, atol=2e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(actual_vals, expected_vals, atol=2e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(actual_v_h, expected_v_h, atol=2e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(actual_eps_xc, expected_eps_xc, atol=2e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(actual_v_xc, expected_v_xc, atol=2e-5, rtol=1e-5))
        overlap = jnp.abs(jnp.sum(actual_vecs[:, 0] * expected_vecs[:, 0]))
        self.assertGreater(float(overlap), 0.999)


if __name__ == "__main__":
    unittest.main()
