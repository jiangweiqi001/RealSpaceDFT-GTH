from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg

from .hamiltonian import build_local_potential, laplacian_8th
from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch
from .patch_projector_operator import build_patch_projector_data

_PATCH_BASIS_RANK_TOL = 1e-6


@dataclass(frozen=True)
class MixedBasisGalerkinMetadata:
    patch_specs: list
    patch_maps: list
    patch_bases: dict
    patch_slices: dict


def _build_patch_overlap_value_matrix(patch_map):
    n_points = int(patch_map.eval_matrix.shape[0])
    return patch_map.fine_dv * jnp.eye(n_points, dtype=jnp.float32)


def _build_patch_potential_value_matrix(v_eff_patch, patch_map):
    values = jnp.asarray(v_eff_patch, dtype=jnp.float32)
    return patch_map.fine_dv * jnp.diag(values)


def _build_patch_dirichlet_kinetic_value_matrix(patch_spec, patch_map, kinetic_scale=1.0):
    _ = kinetic_scale
    positions = jnp.asarray(patch_spec.positions, dtype=jnp.float32)
    spacing = float(patch_spec.fine_spacing)
    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1)
    neighbor_mask = jnp.logical_and(dist2 > 0.0, jnp.abs(dist2 - spacing * spacing) < 1e-5)
    adjacency = neighbor_mask.astype(jnp.float32)
    lap = 6.0 * jnp.eye(adjacency.shape[0], dtype=jnp.float32) - adjacency
    matrix = 0.5 * patch_map.fine_dv * lap / (spacing * spacing)
    return 0.5 * (matrix + matrix.T)


def build_patch_kinetic_value_matrix_v3(patch_spec, patch_map, kinetic_scale=1.0):
    return _build_patch_dirichlet_kinetic_value_matrix(
        patch_spec,
        patch_map,
        kinetic_scale=kinetic_scale,
    )


def _build_projector_value_matrix(channel, patch_map):
    weight = jnp.asarray(channel.coeff, dtype=jnp.float32) * (patch_map.fine_dv ** 2)
    matrix = weight * jnp.outer(channel.p_i, channel.p_j)
    return 0.5 * (matrix + matrix.T)


def _insert_global_cross_block(global_block, local_block, sample_indices, patch_slice):
    rows = np.asarray(sample_indices, dtype=np.int32)
    cols = np.arange(patch_slice.start, patch_slice.stop, dtype=np.int32)
    global_block[np.ix_(rows, cols)] += np.asarray(local_block, dtype=np.float32)
    return global_block


def _insert_global_square_block(global_block, local_block, sample_indices):
    indices = np.asarray(sample_indices, dtype=np.int32)
    global_block[np.ix_(indices, indices)] += np.asarray(local_block, dtype=np.float32)
    return global_block


def build_patch_physical_basis_v3(patch_spec, patch_map, kinetic_scale=1.0, reg=1e-6):
    _ = kinetic_scale
    _ = reg
    e = np.asarray(patch_map.eval_matrix, dtype=np.float64)
    k = np.asarray(
        _build_patch_dirichlet_kinetic_value_matrix(
            patch_spec,
            patch_map,
        ),
        dtype=np.float64,
    )
    constraint = np.concatenate([e.T, e.T @ k], axis=0)
    _, s, vt = np.linalg.svd(constraint, full_matrices=True)
    rank = int(np.sum(s > _PATCH_BASIS_RANK_TOL))
    basis = vt[rank:].T
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return jnp.asarray(basis, dtype=jnp.float32)


def _build_coarse_kinetic_matrix(grid):
    coarse_size = int(jnp.prod(jnp.array(grid.shape)))
    eye = jnp.eye(coarse_size, dtype=jnp.float32)

    def apply_t(psi_flat):
        psi = psi_flat.reshape(grid.shape)
        return (-0.5 * laplacian_8th(psi, grid.spacing, grid.mask)).reshape(-1)

    t_operator = jax.vmap(apply_t, in_axes=1, out_axes=1)(eye)
    return grid.volume_element * 0.5 * (t_operator + t_operator.T)


def _build_coarse_local_matrix(grid, v_eff):
    return grid.volume_element * jnp.diag(jnp.asarray(v_eff, dtype=jnp.float32).reshape(-1))


def _build_coarse_local_baseline(grid, coords, pseudos):
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
    return build_local_potential(
        coords,
        grid.coords,
        zion,
        rloc,
        c,
        spacing=grid.spacing,
        local_subgrid=1,
        local_mode="cell_average",
    )


def build_fixed_veff_galerkin_components_v3(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_stiffness=1.0,
    vloc_cc_mode="patch",
):
    v_eff = jnp.asarray(v_eff, dtype=jnp.float32)
    coarse_size = int(jnp.prod(jnp.array(grid.shape)))

    patch_specs = build_atom_patch_specs(
        grid,
        coords,
        pseudos,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
    )
    patch_maps = build_patch_maps(grid, patch_specs)
    patch_bases = {
        patch_spec.atom_index: build_patch_physical_basis_v3(
            patch_spec,
            patch_map,
        )
        for patch_spec, patch_map in zip(patch_specs, patch_maps)
    }
    projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)

    patch_slices = {}
    offset = coarse_size
    for patch_map in patch_maps:
        atom_index = patch_map.atom_index
        size = int(patch_bases[atom_index].shape[1])
        patch_slices[atom_index] = slice(offset, offset + size)
        offset += size
    total_size = offset

    h_dense = np.zeros((total_size, total_size), dtype=np.float32)
    s_dense = np.zeros((total_size, total_size), dtype=np.float32)
    h_t = np.zeros((total_size, total_size), dtype=np.float32)
    h_v_loc = np.zeros((total_size, total_size), dtype=np.float32)
    h_v_nl = np.zeros((total_size, total_size), dtype=np.float32)

    if vloc_cc_mode not in ("patch", "coarse"):
        raise ValueError("vloc_cc_mode must be 'patch' or 'coarse'")

    h_cc_t = _build_coarse_kinetic_matrix(grid)
    v_eff_coarse = _build_coarse_local_baseline(grid, coords, pseudos)
    h_cc_v_loc = _build_coarse_local_matrix(grid, v_eff_coarse)
    delta_v_grid = v_eff - v_eff_coarse
    h_cc_v_nl = np.zeros((coarse_size, coarse_size), dtype=np.float32)
    s_cc = grid.volume_element * jnp.eye(coarse_size, dtype=jnp.float32)
    s_dense[:coarse_size, :coarse_size] = np.asarray(s_cc, dtype=np.float32)
    h_t[:coarse_size, :coarse_size] = np.asarray(h_cc_t, dtype=np.float32)
    h_v_loc[:coarse_size, :coarse_size] = np.asarray(h_cc_v_loc, dtype=np.float32)

    channels_by_atom = {}
    for channel in projector_data.channels:
        channels_by_atom.setdefault(channel.atom_index, []).append(channel)

    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        atom_index = patch_map.atom_index
        basis = patch_bases[atom_index]
        patch_slice = patch_slices[atom_index]
        eval_matrix = jnp.asarray(patch_map.eval_matrix, dtype=jnp.float32)
        overlap_value = _build_patch_overlap_value_matrix(patch_map)
        delta_v_patch = coarse_to_patch(delta_v_grid, patch_map)
        potential_value = _build_patch_potential_value_matrix(delta_v_patch, patch_map)
        kinetic_value = _build_patch_dirichlet_kinetic_value_matrix(
            patch_spec,
            patch_map,
        )

        s_pp = basis.T @ overlap_value @ basis
        s_cp_local = eval_matrix.T @ overlap_value @ basis
        h_pp_t = basis.T @ kinetic_value @ basis
        h_cp_t = eval_matrix.T @ kinetic_value @ basis
        h_pp_v_loc = basis.T @ potential_value @ basis
        h_cp_v_loc = eval_matrix.T @ potential_value @ basis
        h_pp_v_nl = jnp.zeros_like(h_pp_t)
        h_cp_v_nl = jnp.zeros_like(h_cp_t)
        h_cc_v_nl_local = jnp.zeros((eval_matrix.shape[1], eval_matrix.shape[1]), dtype=jnp.float32)

        for channel in channels_by_atom.get(atom_index, []):
            projector_value = _build_projector_value_matrix(channel, patch_map)
            h_cc_v_nl_local = h_cc_v_nl_local + eval_matrix.T @ projector_value @ eval_matrix
            h_pp_v_nl = h_pp_v_nl + basis.T @ projector_value @ basis
            h_cp_v_nl = h_cp_v_nl + eval_matrix.T @ projector_value @ basis

        h_cc_v_nl_local = 0.5 * (h_cc_v_nl_local + h_cc_v_nl_local.T)
        h_pp = h_pp_t + h_pp_v_loc + h_pp_v_nl
        h_cp_local = h_cp_t + h_cp_v_loc + h_cp_v_nl

        h_pp = 0.5 * (h_pp + h_pp.T)
        h_pp_t = 0.5 * (h_pp_t + h_pp_t.T)
        h_pp_v_loc = 0.5 * (h_pp_v_loc + h_pp_v_loc.T)
        h_pp_v_nl = 0.5 * (h_pp_v_nl + h_pp_v_nl.T)
        s_pp = 0.5 * (s_pp + s_pp.T)
        h_dense[patch_slice, patch_slice] = np.asarray(h_pp, dtype=np.float32)
        s_dense[patch_slice, patch_slice] = np.asarray(s_pp, dtype=np.float32)
        h_t[patch_slice, patch_slice] = np.asarray(h_pp_t, dtype=np.float32)
        h_v_loc[patch_slice, patch_slice] = np.asarray(h_pp_v_loc, dtype=np.float32)
        h_v_nl[patch_slice, patch_slice] = np.asarray(h_pp_v_nl, dtype=np.float32)
        h_cc_v_nl = _insert_global_square_block(h_cc_v_nl, h_cc_v_nl_local, patch_map.sample_indices)

        h_dense = _insert_global_cross_block(h_dense, h_cp_local, patch_map.sample_indices, patch_slice)
        s_dense = _insert_global_cross_block(s_dense, s_cp_local, patch_map.sample_indices, patch_slice)
        h_t = _insert_global_cross_block(h_t, h_cp_t, patch_map.sample_indices, patch_slice)
        h_v_loc = _insert_global_cross_block(h_v_loc, h_cp_v_loc, patch_map.sample_indices, patch_slice)
        h_v_nl = _insert_global_cross_block(h_v_nl, h_cp_v_nl, patch_map.sample_indices, patch_slice)
        h_dense[patch_slice, :coarse_size] = h_dense[:coarse_size, patch_slice].T
        s_dense[patch_slice, :coarse_size] = s_dense[:coarse_size, patch_slice].T
        h_t[patch_slice, :coarse_size] = h_t[:coarse_size, patch_slice].T
        h_v_loc[patch_slice, :coarse_size] = h_v_loc[:coarse_size, patch_slice].T
        h_v_nl[patch_slice, :coarse_size] = h_v_nl[:coarse_size, patch_slice].T

    h_cc = h_cc_t + h_cc_v_loc + h_cc_v_nl
    h_dense[:coarse_size, :coarse_size] = np.asarray(h_cc, dtype=np.float32)
    h_v_nl[:coarse_size, :coarse_size] = np.asarray(h_cc_v_nl, dtype=np.float32)

    metadata = MixedBasisGalerkinMetadata(
        patch_specs=patch_specs,
        patch_maps=patch_maps,
        patch_bases=patch_bases,
        patch_slices=patch_slices,
    )
    components = {
        "t": jnp.asarray(h_t, dtype=jnp.float32),
        "v_loc": jnp.asarray(h_v_loc, dtype=jnp.float32),
        "v_nl": jnp.asarray(h_v_nl, dtype=jnp.float32),
    }
    return (
        jnp.asarray(h_dense, dtype=jnp.float32),
        jnp.asarray(s_dense, dtype=jnp.float32),
        coarse_size,
        metadata,
        components,
    )


def build_fixed_veff_galerkin_matrices_v3(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_stiffness=1.0,
    vloc_cc_mode="patch",
):
    h_dense, s_dense, coarse_size, metadata, _ = build_fixed_veff_galerkin_components_v3(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        patch_stiffness=patch_stiffness,
        vloc_cc_mode=vloc_cc_mode,
    )
    return h_dense, s_dense, coarse_size, metadata


def compute_v3_generalized_fractions(eigvec, s_dense, coarse_size):
    vec = jnp.asarray(eigvec, dtype=jnp.float32)
    s = jnp.asarray(s_dense, dtype=jnp.float32)
    vc = vec[:coarse_size]
    vp = vec[coarse_size:]
    s_cc = s[:coarse_size, :coarse_size]
    s_cp = s[:coarse_size, coarse_size:]
    s_pp = s[coarse_size:, coarse_size:]
    total = jnp.dot(vec, s @ vec)
    coarse = jnp.dot(vc, s_cc @ vc) + jnp.dot(vc, s_cp @ vp)
    patch = jnp.dot(vp, s_pp @ vp) + jnp.dot(vp, s_cp.T @ vc)
    return {
        "coarse": coarse / total,
        "patch": patch / total,
        "norm": total,
    }


def compute_v3_energy_decomposition(eigvec, s_dense, components):
    vec = jnp.asarray(eigvec, dtype=jnp.float32)
    total = jnp.dot(vec, jnp.asarray(s_dense, dtype=jnp.float32) @ vec)
    return {
        name: jnp.dot(vec, matrix @ vec) / total
        for name, matrix in components.items()
    }


def build_v3_vloc_blocks(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_stiffness=1.0,
    vloc_cc_mode="patch",
):
    _, _, coarse_size, _, components = build_fixed_veff_galerkin_components_v3(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        patch_stiffness=patch_stiffness,
        vloc_cc_mode=vloc_cc_mode,
    )
    vloc = jnp.asarray(components["v_loc"], dtype=jnp.float32)
    total_size = vloc.shape[0]
    cc = jnp.zeros_like(vloc).at[:coarse_size, :coarse_size].set(vloc[:coarse_size, :coarse_size])
    cp = jnp.zeros_like(vloc)
    cp = cp.at[:coarse_size, coarse_size:].set(vloc[:coarse_size, coarse_size:])
    cp = cp.at[coarse_size:, :coarse_size].set(vloc[coarse_size:, :coarse_size])
    pp = jnp.zeros_like(vloc).at[coarse_size:total_size, coarse_size:total_size].set(
        vloc[coarse_size:total_size, coarse_size:total_size]
    )
    return {"cc": cc, "cp": cp, "pp": pp}


def compute_v3_vloc_block_expectations(eigvec, s_dense, vloc_blocks):
    vec = jnp.asarray(eigvec, dtype=jnp.float32)
    total = jnp.dot(vec, jnp.asarray(s_dense, dtype=jnp.float32) @ vec)
    return {
        name: jnp.dot(vec, block @ vec) / total
        for name, block in vloc_blocks.items()
    }


def solve_fixed_veff_galerkin_dense_host_v3(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    patch_stiffness=1.0,
    vloc_cc_mode="patch",
):
    h_dense, s_dense, coarse_size, metadata = build_fixed_veff_galerkin_matrices_v3(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        patch_stiffness=patch_stiffness,
        vloc_cc_mode=vloc_cc_mode,
    )
    h_np = np.asarray(h_dense, dtype=np.float64)
    s_np = np.asarray(s_dense, dtype=np.float64)
    eigvals, eigvecs = scipy_linalg.eigh(h_np, s_np)
    return (
        jnp.asarray(eigvals[:n_bands], dtype=jnp.float32),
        jnp.asarray(eigvecs[:, :n_bands], dtype=jnp.float32),
        coarse_size,
        metadata,
    )
