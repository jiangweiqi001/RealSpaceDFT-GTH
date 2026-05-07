"""V5b fixed-potential mixed-basis Galerkin prototype.

This module starts the V5 rewrite from the operator semantics instead of SCF:
the coarse block stays the stable baseline, while patch coordinates add
incremental S/T/Vloc/Vnl blocks built on the same atom-centered local basis.
"""

import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg

from .hamiltonian import get_gth_projector
from .mixed_basis_enriched_v5 import (
    EnrichedBasisMetadataV5,
    _raw_atom_centered_columns,
    _s_orthonormalize,
    _project_out_coarse_overlap,
)
from .mixed_basis_galerkin_v3 import (
    _build_coarse_kinetic_matrix,
    _build_coarse_local_matrix,
    _build_patch_dirichlet_kinetic_value_matrix,
    _build_patch_overlap_value_matrix,
    _build_patch_potential_value_matrix,
    _insert_global_cross_block,
)
from .mixed_basis_local_modes_v4 import _build_coarse_nonlocal_matrix_cell_average
from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch


def build_atom_centered_enriched_basis_v5b(patch_spec, patch_map, pseudo, max_modes=8):
    """Build a coarse-overlap-orthogonal atom-centered patch basis."""
    overlap = np.asarray(_build_patch_overlap_value_matrix(patch_map), dtype=np.float64)
    trace = np.asarray(patch_map.eval_matrix, dtype=np.float64)
    raw = _raw_atom_centered_columns(patch_spec, pseudo)
    projected = _project_out_coarse_overlap(raw, trace, overlap)
    basis = _s_orthonormalize(projected, overlap)
    if basis.shape[1] > max_modes:
        kinetic = np.asarray(
            _build_patch_dirichlet_kinetic_value_matrix(patch_spec, patch_map),
            dtype=np.float64,
        )
        reduced_t = 0.5 * (basis.T @ kinetic @ basis + (basis.T @ kinetic @ basis).T)
        eigvals, eigvecs = np.linalg.eigh(reduced_t)
        order = np.argsort(eigvals)[:max_modes]
        basis = _s_orthonormalize(basis @ eigvecs[:, order], overlap)
    return jnp.asarray(basis, dtype=jnp.float32)


def _projector_channels_on_patch(patch_spec, pseudo):
    offsets = jnp.asarray(patch_spec.offsets, dtype=jnp.float32)
    r = jnp.linalg.norm(offsets, axis=-1)
    channels = []
    for projector in pseudo.get("projectors", []):
        l = int(projector["l"])
        rp = float(projector["r"])
        h_mat = jnp.asarray(projector["h"], dtype=jnp.float32)
        if h_mat.ndim == 1:
            h_mat = jnp.diag(h_mat)
        n_proj = int(h_mat.shape[0])
        cache = {}

        def values(proj_idx, axis_idx=None):
            key = (proj_idx, axis_idx)
            if key in cache:
                return cache[key]
            radial = get_gth_projector(r, l, proj_idx, rp)
            if axis_idx is None:
                result = radial
            else:
                result = radial * (offsets[:, axis_idx] / (r + 1e-12))
            cache[key] = result
            return result

        for i_proj in range(1, n_proj + 1):
            for j_proj in range(1, n_proj + 1):
                h_ij = h_mat[i_proj - 1, j_proj - 1]
                if float(jnp.abs(h_ij)) < 1e-10:
                    continue
                if l == 0:
                    channels.append((
                        np.asarray(values(i_proj), dtype=np.float64),
                        np.asarray(values(j_proj), dtype=np.float64),
                        float(h_ij / (4.0 * jnp.pi)),
                    ))
                elif l == 1:
                    for axis in range(3):
                        channels.append((
                            np.asarray(values(i_proj, axis), dtype=np.float64),
                            np.asarray(values(j_proj, axis), dtype=np.float64),
                            float(3.0 * h_ij / (4.0 * jnp.pi)),
                        ))
    return channels


def _add_patch_nonlocal_blocks(h_v_nl, patch_spec, patch_map, basis, patch_slice, pseudo):
    channels = _projector_channels_on_patch(patch_spec, pseudo)
    if not channels or basis.shape[1] == 0:
        return h_v_nl
    trace = np.asarray(patch_map.eval_matrix, dtype=np.float64)
    basis_np = np.asarray(basis, dtype=np.float64)
    fine_dv = float(patch_map.fine_dv)
    rows = np.asarray(patch_map.sample_indices, dtype=np.int32)
    h_cp = np.zeros((trace.shape[1], basis_np.shape[1]), dtype=np.float64)
    h_pc = np.zeros((basis_np.shape[1], trace.shape[1]), dtype=np.float64)
    h_pp = np.zeros((basis_np.shape[1], basis_np.shape[1]), dtype=np.float64)
    for p_i, p_j, coeff in channels:
        q_i_c = fine_dv * (p_i @ trace)
        q_j_c = fine_dv * (p_j @ trace)
        q_i_p = fine_dv * (p_i @ basis_np)
        q_j_p = fine_dv * (p_j @ basis_np)
        h_cp += float(coeff) * np.outer(q_i_c, q_j_p)
        h_pc += float(coeff) * np.outer(q_i_p, q_j_c)
        h_pp += float(coeff) * np.outer(q_i_p, q_j_p)
    h_v_nl[np.ix_(rows, np.arange(patch_slice.start, patch_slice.stop))] += h_cp
    h_v_nl[np.ix_(np.arange(patch_slice.start, patch_slice.stop), rows)] += h_pc
    h_v_nl[patch_slice, patch_slice] += h_pp
    return h_v_nl


def build_fixed_veff_enriched_components_v5b(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
):
    """Build dense fixed-Veff V5b matrices with patch projector increments."""
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
    patch_bases = {}
    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        patch_bases[patch_spec.atom_index] = build_atom_centered_enriched_basis_v5b(
            patch_spec,
            patch_map,
            pseudos[patch_spec.atom_index],
            max_modes=max_modes_per_atom,
        )

    patch_slices = {}
    offset = coarse_size
    for patch_map in patch_maps:
        size = int(patch_bases[patch_map.atom_index].shape[1])
        patch_slices[patch_map.atom_index] = slice(offset, offset + size)
        offset += size
    total_size = offset

    h_t = np.zeros((total_size, total_size), dtype=np.float64)
    h_v_loc = np.zeros_like(h_t)
    h_v_nl = np.zeros_like(h_t)
    s_dense = np.zeros_like(h_t)

    h_t[:coarse_size, :coarse_size] = np.asarray(_build_coarse_kinetic_matrix(grid), dtype=np.float64)
    h_v_loc[:coarse_size, :coarse_size] = np.asarray(_build_coarse_local_matrix(grid, v_eff), dtype=np.float64)
    h_v_nl[:coarse_size, :coarse_size] = np.asarray(
        _build_coarse_nonlocal_matrix_cell_average(grid, coords, pseudos, projector_subgrid=1)[0],
        dtype=np.float64,
    )
    s_dense[:coarse_size, :coarse_size] = (
        float(grid.volume_element) * np.eye(coarse_size, dtype=np.float64)
    )

    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        atom_index = patch_spec.atom_index
        basis = patch_bases[atom_index]
        patch_slice = patch_slices[atom_index]
        if basis.shape[1] == 0:
            continue
        eval_matrix = jnp.asarray(patch_map.eval_matrix, dtype=jnp.float32)
        overlap = _build_patch_overlap_value_matrix(patch_map)
        kinetic = _build_patch_dirichlet_kinetic_value_matrix(patch_spec, patch_map)
        v_patch = coarse_to_patch(v_eff, patch_map)
        potential = _build_patch_potential_value_matrix(v_patch, patch_map)

        s_pp = basis.T @ overlap @ basis
        s_cp = eval_matrix.T @ overlap @ basis
        h_pp_t = basis.T @ kinetic @ basis
        h_cp_t = eval_matrix.T @ kinetic @ basis
        h_pp_v = basis.T @ potential @ basis
        h_cp_v = eval_matrix.T @ potential @ basis

        h_t[patch_slice, patch_slice] = np.asarray(0.5 * (h_pp_t + h_pp_t.T), dtype=np.float64)
        h_v_loc[patch_slice, patch_slice] = np.asarray(0.5 * (h_pp_v + h_pp_v.T), dtype=np.float64)
        s_dense[patch_slice, patch_slice] = np.asarray(0.5 * (s_pp + s_pp.T), dtype=np.float64)
        h_t = _insert_global_cross_block(h_t, h_cp_t, patch_map.sample_indices, patch_slice)
        h_v_loc = _insert_global_cross_block(h_v_loc, h_cp_v, patch_map.sample_indices, patch_slice)
        s_dense = _insert_global_cross_block(s_dense, s_cp, patch_map.sample_indices, patch_slice)
        h_v_nl = _add_patch_nonlocal_blocks(
            h_v_nl,
            patch_spec,
            patch_map,
            basis,
            patch_slice,
            pseudos[atom_index],
        )

    h_t = 0.5 * (h_t + h_t.T)
    h_v_loc = 0.5 * (h_v_loc + h_v_loc.T)
    h_v_nl = 0.5 * (h_v_nl + h_v_nl.T)
    s_dense = 0.5 * (s_dense + s_dense.T)
    h_dense = h_t + h_v_loc + h_v_nl
    metadata = EnrichedBasisMetadataV5(
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


def solve_fixed_veff_enriched_dense_host_v5b(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
):
    h_dense, s_dense, coarse_size, metadata, components = build_fixed_veff_enriched_components_v5b(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
    )
    eigvals, eigvecs = scipy_linalg.eigh(
        np.asarray(h_dense, dtype=np.float64),
        np.asarray(s_dense, dtype=np.float64),
        subset_by_index=[0, n_bands - 1],
    )
    eigvecs = eigvecs[:, :n_bands]
    s_np = np.asarray(s_dense, dtype=np.float64)
    norms = np.einsum("ik,ij,jk->k", eigvecs, s_np, eigvecs)
    s_cc = s_np[:coarse_size, :coarse_size]
    s_pp = s_np[coarse_size:, coarse_size:]
    coarse_norm = np.einsum("ik,ij,jk->k", eigvecs[:coarse_size], s_cc, eigvecs[:coarse_size])
    if s_pp.size == 0:
        patch_norm = np.zeros((n_bands,), dtype=np.float64)
    else:
        patch_norm = np.einsum("ik,ij,jk->k", eigvecs[coarse_size:], s_pp, eigvecs[coarse_size:])
    comp_np = {name: np.asarray(mat, dtype=np.float64) for name, mat in components.items()}
    decomposition = {
        name: jnp.asarray(
            np.einsum("ik,ij,jk->k", eigvecs, mat, eigvecs) / norms,
            dtype=jnp.float32,
        )
        for name, mat in comp_np.items()
    }
    return {
        "eigvals": jnp.asarray(eigvals[:n_bands], dtype=jnp.float32),
        "eigvecs": jnp.asarray(eigvecs, dtype=jnp.float32),
        "coarse_size": coarse_size,
        "metadata": metadata,
        "band_decomposition": decomposition,
        "coarse_metric_fraction_diag": jnp.asarray(coarse_norm / norms, dtype=jnp.float32),
        "patch_metric_fraction_diag": jnp.asarray(patch_norm / norms, dtype=jnp.float32),
    }
