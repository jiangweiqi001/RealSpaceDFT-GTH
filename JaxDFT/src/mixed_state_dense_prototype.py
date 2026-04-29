import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg

from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch, patch_to_coarse_adjoint
from .patch_projector_operator import build_patch_projector_data
from .patch_projector_operator_v2 import (
    IndependentPatchOrbital,
    apply_independent_patch_projector,
)
from .hamiltonian import laplacian_8th


def build_patch_complement_basis(patch_map, reg=1e-6):
    a = np.asarray(patch_map.eval_matrix, dtype=np.float64)
    u, s, _ = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(s > reg))
    basis = u[:, rank:]
    return jnp.asarray(basis, dtype=jnp.float32)


def _patch_lengths(patch_maps, patch_bases=None):
    if patch_bases is None:
        return [int(patch_map.eval_matrix.shape[0]) for patch_map in patch_maps]
    return [int(patch_bases[patch_map.atom_index].shape[1]) for patch_map in patch_maps]


def mixed_state_size(grid_shape, patch_maps, patch_bases=None):
    coarse_size = int(jnp.prod(jnp.array(grid_shape)))
    return coarse_size + sum(_patch_lengths(patch_maps, patch_bases=patch_bases))


def build_patch_duplicate_projector(patch_map, reg=1e-6):
    a = jnp.asarray(patch_map.eval_matrix, dtype=jnp.float32)
    u, s, _ = jnp.linalg.svd(a, full_matrices=False)
    keep = s > reg
    u_kept = u[:, keep]
    projector = u_kept @ u_kept.T
    return 0.5 * (projector + projector.T)


def build_patch_orthogonal_projector(patch_map, reg=1e-6):
    projector = build_patch_duplicate_projector(patch_map, reg=reg)
    identity = jnp.eye(projector.shape[0], dtype=jnp.float32)
    return identity - projector


def project_patch_correction(correction, patch_map, reg=1e-6):
    q = build_patch_orthogonal_projector(patch_map, reg=reg)
    return q @ jnp.asarray(correction, dtype=jnp.float32)


def build_patch_overlap_block(patch_map, duplicate_penalty=10.0, reg=1e-6):
    p_dup = build_patch_duplicate_projector(patch_map, reg=reg)
    q = build_patch_orthogonal_projector(patch_map, reg=reg)
    identity = jnp.eye(q.shape[0], dtype=jnp.float32)
    return patch_map.fine_dv * (q + duplicate_penalty * p_dup) + reg * identity


def build_patch_potential_matrix(patch_map, v_eff_patch, reg=1e-6):
    q = build_patch_orthogonal_projector(patch_map, reg=reg)
    values = jnp.asarray(v_eff_patch, dtype=jnp.float32)
    weighted = jnp.diag(values) * patch_map.fine_dv
    matrix = q @ weighted @ q
    return 0.5 * (matrix + matrix.T)


def build_patch_kinetic_matrix(patch_spec, patch_map, kinetic_scale=1.0, reg=1e-6):
    positions = jnp.asarray(patch_spec.positions, dtype=jnp.float32)
    spacing = float(patch_spec.fine_spacing)
    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1)
    neighbor_mask = jnp.logical_and(dist2 > 0.0, jnp.abs(dist2 - spacing * spacing) < 1e-5)
    adjacency = neighbor_mask.astype(jnp.float32)
    # Dirichlet patch kinetic: points near the patch boundary still see the
    # missing neighbors as zero-valued exterior nodes, so the diagonal stays 6.
    lap = 6.0 * jnp.eye(adjacency.shape[0], dtype=jnp.float32) - adjacency
    q = build_patch_orthogonal_projector(patch_map, reg=reg)
    projected_lap = q @ lap @ q
    matrix = 0.5 * kinetic_scale * patch_map.fine_dv * projected_lap / (spacing * spacing)
    return 0.5 * (matrix + matrix.T)


def build_patch_stiffness_matrix(patch_spec, patch_map, stiffness_scale=1.0, reg=1e-6):
    matrix = build_patch_kinetic_matrix(
        patch_spec,
        patch_map,
        kinetic_scale=stiffness_scale,
        reg=reg,
    )
    return 0.5 * (matrix + matrix.T)


def build_patch_penalty_matrix(patch_map, patch_penalty, reg=1e-6):
    q = build_patch_orthogonal_projector(patch_map, reg=reg)
    matrix = patch_penalty * patch_map.fine_dv * q
    return 0.5 * (matrix + matrix.T)


def build_total_mixed_metric(
    grid,
    patch_maps,
    patch_bases=None,
    patch_metric_duplicate_penalty=10.0,
    reg=1e-6,
):
    coarse_size = int(jnp.prod(jnp.array(grid.shape)))
    total_size = coarse_size + sum(_patch_lengths(patch_maps, patch_bases=patch_bases))
    total_metric = jnp.zeros((total_size, total_size), dtype=jnp.float32)
    total_metric = total_metric.at[:coarse_size, :coarse_size].set(
        grid.volume_element * jnp.eye(coarse_size, dtype=jnp.float32)
    )
    offset = coarse_size
    for patch_map in patch_maps:
        patch_block_values = build_patch_overlap_block(
            patch_map,
            duplicate_penalty=patch_metric_duplicate_penalty,
            reg=reg,
        )
        basis = None if patch_bases is None else patch_bases.get(patch_map.atom_index)
        if basis is not None:
            patch_block = basis.T @ patch_block_values @ basis
            cross_local = (patch_map.eval_matrix.T * patch_map.fine_dv) @ basis
        else:
            patch_block = patch_block_values
            cross_local = patch_map.eval_matrix.T * patch_map.fine_dv
        size = patch_block.shape[0]
        total_metric = total_metric.at[offset: offset + size, offset: offset + size].set(patch_block)

        # Formal coarse-patch overlap block induced by coarse-to-patch
        # evaluation. This is the weighted adjoint pair behind
        # coarse_to_patch / patch_to_coarse_adjoint.
        for local_col, global_index in enumerate(jnp.asarray(patch_map.sample_indices).tolist()):
            total_metric = total_metric.at[global_index, offset: offset + size].set(
                cross_local[local_col]
            )
            total_metric = total_metric.at[offset: offset + size, global_index].set(
                cross_local[local_col]
            )
        offset += size
    return total_metric


def flatten_independent_patch_orbital(state, grid_shape, patch_maps, patch_bases=None):
    coarse_flat = jnp.asarray(state.coarse).reshape(-1)
    parts = [coarse_flat]
    dtype = coarse_flat.dtype
    for patch_map in patch_maps:
        length = (
            int(patch_bases[patch_map.atom_index].shape[1])
            if patch_bases is not None and patch_map.atom_index in patch_bases
            else int(patch_map.eval_matrix.shape[0])
        )
        correction = state.patch_corrections.get(
            patch_map.atom_index,
            jnp.zeros((length,), dtype=dtype),
        )
        parts.append(jnp.asarray(correction, dtype=dtype))
    return jnp.concatenate(parts, axis=0)


def unflatten_independent_patch_orbital(flat_state, grid_shape, patch_maps, patch_bases=None):
    flat_state = jnp.asarray(flat_state)
    coarse_size = int(jnp.prod(jnp.array(grid_shape)))
    coarse = flat_state[:coarse_size].reshape(grid_shape)
    patch_corrections = {}
    offset = coarse_size
    for patch_map in patch_maps:
        length = (
            int(patch_bases[patch_map.atom_index].shape[1])
            if patch_bases is not None and patch_map.atom_index in patch_bases
            else int(patch_map.eval_matrix.shape[0])
        )
        patch_corrections[patch_map.atom_index] = flat_state[offset: offset + length]
        offset += length
    return IndependentPatchOrbital(coarse=coarse, patch_corrections=patch_corrections)


def build_fixed_veff_mixed_apply_h(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_penalty=1.0,
    patch_stiffness=1.0,
    patch_metric_duplicate_penalty=10.0,
):
    v_eff = jnp.asarray(v_eff, dtype=jnp.float32)
    patch_specs = build_atom_patch_specs(
        grid,
        coords,
        pseudos,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
    )
    patch_maps = build_patch_maps(grid, patch_specs)
    patch_bases = {
        patch_map.atom_index: build_patch_complement_basis(patch_map)
        for patch_map in patch_maps
    }
    projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)
    v_eff_patch = {
        patch_map.atom_index: coarse_to_patch(v_eff, patch_map)
        for patch_map in patch_maps
    }
    patch_q = {
        patch_map.atom_index: build_patch_orthogonal_projector(patch_map)
        for patch_map in patch_maps
    }
    patch_kinetic_blocks = {
        patch_spec.atom_index: (
            patch_bases[patch_spec.atom_index].T
            @ build_patch_stiffness_matrix(
                patch_spec,
                patch_map,
                stiffness_scale=patch_stiffness,
            )
            @ patch_bases[patch_spec.atom_index]
        )
        for patch_spec, patch_map in zip(patch_specs, patch_maps)
    }
    patch_potential_blocks = {
        patch_map.atom_index: (
            patch_bases[patch_map.atom_index].T
            @ build_patch_potential_matrix(
                patch_map,
                v_eff_patch[patch_map.atom_index],
            )
            @ patch_bases[patch_map.atom_index]
        )
        for patch_map in patch_maps
    }
    patch_penalty_blocks = {
        patch_map.atom_index: (
            patch_bases[patch_map.atom_index].T
            @ build_patch_penalty_matrix(
                patch_map,
                patch_penalty,
            )
            @ patch_bases[patch_map.atom_index]
        )
        for patch_map in patch_maps
    }
    total_size = mixed_state_size(grid.shape, patch_maps, patch_bases=patch_bases)

    def apply_h(flat_state):
        state = unflatten_independent_patch_orbital(
            flat_state,
            grid.shape,
            patch_maps,
            patch_bases=patch_bases,
        )
        projected_state = state
        projected = apply_independent_patch_projector(
            projected_state,
            projector_data,
            patch_maps,
            grid.shape,
            patch_bases=patch_bases,
        )
        lap = laplacian_8th(state.coarse, grid.spacing, grid.mask)
        coarse_part = -0.5 * lap + v_eff * state.coarse + projected.coarse

        patch_part = {}
        for patch_map in patch_maps:
            atom_index = patch_map.atom_index
            basis = patch_bases[atom_index]
            orth_correction = projected_state.patch_corrections[atom_index]
            coarse_patch = coarse_to_patch(state.coarse, patch_map)
            projected_patch_response = projected.patch_corrections[atom_index]
            local_patch_from_coarse = (
                basis.T @ (
                    build_patch_potential_matrix(
                        patch_map,
                        v_eff_patch[atom_index],
                    ) @ coarse_patch
                    + build_patch_stiffness_matrix(
                        patch_specs[atom_index],
                        patch_map,
                        stiffness_scale=patch_stiffness,
                    ) @ coarse_patch
                )
            )
            local_patch_from_correction = (
                patch_potential_blocks[atom_index] @ orth_correction
                + patch_kinetic_blocks[atom_index] @ orth_correction
            )
            coarse_part = coarse_part + patch_to_coarse_adjoint(
                basis @ local_patch_from_correction,
                patch_map,
                grid.shape,
            )
            patch_part[atom_index] = (
                projected_patch_response
                + local_patch_from_coarse
                + local_patch_from_correction
                + patch_penalty_blocks[atom_index] @ orth_correction
            )

        return flatten_independent_patch_orbital(
            IndependentPatchOrbital(coarse=coarse_part, patch_corrections=patch_part),
            grid.shape,
            patch_maps,
            patch_bases=patch_bases,
        )

    return apply_h, total_size, patch_maps, patch_bases


def solve_fixed_veff_mixed_dense(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_penalty=1.0,
    patch_stiffness=1.0,
    patch_metric_duplicate_penalty=10.0,
):
    apply_h, total_size, patch_maps, patch_bases = build_fixed_veff_mixed_apply_h(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        patch_penalty=patch_penalty,
        patch_stiffness=patch_stiffness,
        patch_metric_duplicate_penalty=patch_metric_duplicate_penalty,
    )
    eye = jnp.eye(total_size, dtype=jnp.float32)
    h_dense = jax.vmap(apply_h, in_axes=1, out_axes=1)(eye)
    h_dense = 0.5 * (h_dense + h_dense.T)
    total_metric = build_total_mixed_metric(
        grid,
        patch_maps,
        patch_bases=patch_bases,
        patch_metric_duplicate_penalty=patch_metric_duplicate_penalty,
    )

    metric_evals, metric_evecs = jnp.linalg.eigh(0.5 * (total_metric + total_metric.T))
    metric_inv_sqrt = metric_evecs @ jnp.diag(1.0 / jnp.sqrt(metric_evals)) @ metric_evecs.T
    transformed = metric_inv_sqrt @ h_dense @ metric_inv_sqrt
    eigvals, eigvecs_y = jnp.linalg.eigh(0.5 * (transformed + transformed.T))
    eigvecs = metric_inv_sqrt @ eigvecs_y[:, :n_bands]
    return eigvals[:n_bands], eigvecs, total_size


def build_fixed_veff_mixed_matrices(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_penalty=1.0,
    patch_stiffness=1.0,
    patch_metric_duplicate_penalty=10.0,
):
    apply_h, total_size, patch_maps, patch_bases = build_fixed_veff_mixed_apply_h(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        patch_penalty=patch_penalty,
        patch_stiffness=patch_stiffness,
        patch_metric_duplicate_penalty=patch_metric_duplicate_penalty,
    )
    eye = jnp.eye(total_size, dtype=jnp.float32)
    h_dense = jax.vmap(apply_h, in_axes=1, out_axes=1)(eye)
    h_dense = 0.5 * (h_dense + h_dense.T)
    total_metric = build_total_mixed_metric(
        grid,
        patch_maps,
        patch_bases=patch_bases,
        patch_metric_duplicate_penalty=patch_metric_duplicate_penalty,
    )
    return h_dense, total_metric, total_size


def solve_fixed_veff_mixed_dense_host(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_penalty=1.0,
    patch_stiffness=1.0,
    patch_metric_duplicate_penalty=10.0,
):
    h_dense, total_metric, total_size = build_fixed_veff_mixed_matrices(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        patch_penalty=patch_penalty,
        patch_stiffness=patch_stiffness,
        patch_metric_duplicate_penalty=patch_metric_duplicate_penalty,
    )
    h_np = np.asarray(h_dense, dtype=np.float64)
    s_np = np.asarray(total_metric, dtype=np.float64)
    eigvals, eigvecs = scipy_linalg.eigh(h_np, s_np)
    return (
        jnp.asarray(eigvals[:n_bands], dtype=jnp.float32),
        jnp.asarray(eigvecs[:, :n_bands], dtype=jnp.float32),
        total_size,
    )
