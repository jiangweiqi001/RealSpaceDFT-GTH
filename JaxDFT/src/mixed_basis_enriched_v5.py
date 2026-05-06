from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg

from .functional import lda_xc
from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch, patch_to_coarse_adjoint
from .solver import precompute_poisson_kernel, solve_poisson
from .mixed_basis_galerkin_v3 import (
    _build_coarse_kinetic_matrix,
    _build_coarse_local_matrix,
    _build_patch_dirichlet_kinetic_value_matrix,
    _build_patch_overlap_value_matrix,
    _build_patch_potential_value_matrix,
    _insert_global_cross_block,
)
from .mixed_basis_local_modes_v4 import _build_coarse_nonlocal_matrix_cell_average


_V5_RANK_TOL = 1e-8


@dataclass(frozen=True)
class EnrichedBasisMetadataV5:
    patch_specs: list
    patch_maps: list
    patch_bases: dict
    patch_slices: dict


def _s_orthonormalize(columns, metric):
    if columns.size == 0:
        return columns
    gram = 0.5 * (columns.T @ metric @ columns + (columns.T @ metric @ columns).T)
    eigvals, eigvecs = np.linalg.eigh(gram)
    keep = eigvals > _V5_RANK_TOL
    if not np.any(keep):
        return np.zeros((columns.shape[0], 0), dtype=np.float64)
    return columns @ (eigvecs[:, keep] / np.sqrt(eigvals[keep])[None, :])


def _project_out_coarse_overlap(columns, trace, metric):
    if columns.size == 0:
        return columns
    gram = trace.T @ metric @ trace
    rhs = trace.T @ metric @ columns
    coeff = np.linalg.pinv(gram, rcond=_V5_RANK_TOL) @ rhs
    return columns - trace @ coeff


def _raw_atom_centered_columns(patch_spec, pseudo):
    offsets = np.asarray(patch_spec.offsets, dtype=np.float64)
    r = np.linalg.norm(offsets, axis=1)
    rloc = float(pseudo.get("rloc", 0.5))
    projector_r = float(pseudo.get("projectors", [{"r": rloc}])[0].get("r", rloc))
    widths = [max(0.25 * rloc, 1e-3), max(projector_r, 1e-3), max(2.0 * projector_r, 1e-3)]
    cols = []
    for width in widths:
        gaussian = np.exp(-0.5 * (r / width) ** 2)
        cols.append(gaussian)
        for axis in range(3):
            cols.append((offsets[:, axis] / width) * gaussian)
    raw = np.stack(cols, axis=1) if cols else np.zeros((offsets.shape[0], 0), dtype=np.float64)
    return raw


def build_atom_centered_enriched_basis_v5(
    patch_spec,
    patch_map,
    pseudo,
    max_modes=8,
):
    """Build S-orthogonal atom-centered local basis functions for V5 audits."""
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
        basis = basis @ eigvecs[:, order]
        basis = _s_orthonormalize(basis, overlap)
    return jnp.asarray(basis, dtype=jnp.float32)


def _build_coarse_nonlocal_apply_matrix(grid, coords, pseudos):
    return _build_coarse_nonlocal_matrix_cell_average(
        grid,
        coords,
        pseudos,
        projector_subgrid=1,
    )[0]


def build_fixed_veff_enriched_components_v5(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
):
    """Build dense fixed-Veff V5 audit matrices.

    This first V5 slice keeps the coarse block explicit and adds S-orthogonal
    atom-centered basis functions.  Nonlocal projector enrichment is intentionally
    not added yet; the returned ``v_nl`` component contains the coarse projector
    block only.
    """
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
        patch_spec.atom_index: build_atom_centered_enriched_basis_v5(
            patch_spec,
            patch_map,
            pseudos[patch_spec.atom_index],
            max_modes=max_modes_per_atom,
        )
        for patch_spec, patch_map in zip(patch_specs, patch_maps)
    }
    patch_slices = {}
    offset = coarse_size
    for patch_map in patch_maps:
        size = int(patch_bases[patch_map.atom_index].shape[1])
        patch_slices[patch_map.atom_index] = slice(offset, offset + size)
        offset += size
    total_size = offset

    h_dense = np.zeros((total_size, total_size), dtype=np.float64)
    s_dense = np.zeros((total_size, total_size), dtype=np.float64)
    h_t = np.zeros_like(h_dense)
    h_v_loc = np.zeros_like(h_dense)
    h_v_nl = np.zeros_like(h_dense)

    h_cc_t = _build_coarse_kinetic_matrix(grid)
    h_cc_v_loc = _build_coarse_local_matrix(grid, v_eff)
    h_cc_v_nl = _build_coarse_nonlocal_apply_matrix(grid, coords, pseudos)
    s_cc = grid.volume_element * jnp.eye(coarse_size, dtype=jnp.float32)
    h_t[:coarse_size, :coarse_size] = np.asarray(h_cc_t, dtype=np.float64)
    h_v_loc[:coarse_size, :coarse_size] = np.asarray(h_cc_v_loc, dtype=np.float64)
    h_v_nl[:coarse_size, :coarse_size] = np.asarray(h_cc_v_nl, dtype=np.float64)
    s_dense[:coarse_size, :coarse_size] = np.asarray(s_cc, dtype=np.float64)

    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        basis = patch_bases[patch_spec.atom_index]
        patch_slice = patch_slices[patch_spec.atom_index]
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
        h_t[patch_slice, :coarse_size] = h_t[:coarse_size, patch_slice].T
        h_v_loc[patch_slice, :coarse_size] = h_v_loc[:coarse_size, patch_slice].T
        s_dense[patch_slice, :coarse_size] = s_dense[:coarse_size, patch_slice].T

    h_dense = h_t + h_v_loc + h_v_nl
    h_dense = 0.5 * (h_dense + h_dense.T)
    s_dense = 0.5 * (s_dense + s_dense.T)
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


def solve_fixed_veff_enriched_dense_host_v5(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
):
    h_dense, s_dense, coarse_size, metadata, components = build_fixed_veff_enriched_components_v5(
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
    s_np = np.asarray(s_dense, dtype=np.float64)
    eigvecs = eigvecs[:, :n_bands]
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


def reconstruct_patch_values_v5(vec, coarse_size, metadata):
    """Evaluate the V5 mixed orbital on each atom-centered patch."""
    vec = jnp.asarray(vec, dtype=jnp.float32)
    coarse_size = int(coarse_size)
    coarse_flat = vec[:coarse_size]
    patch_values = {}
    for patch_map in metadata.patch_maps:
        atom_index = patch_map.atom_index
        patch_slice = metadata.patch_slices[atom_index]
        basis = metadata.patch_bases[atom_index]
        coarse_trace = patch_map.eval_matrix @ coarse_flat[patch_map.sample_indices]
        correction = basis @ vec[patch_slice]
        patch_values[atom_index] = coarse_trace + correction
    return patch_values


def compute_v5_density_on_coarse_grid(eigvecs, occ, grid, coarse_size, metadata):
    """Reconstruct a conservative coarse-grid density from V5 eigenvectors."""
    eigvecs = jnp.asarray(eigvecs, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    n_bands = eigvecs.shape[1]
    coarse_size = int(coarse_size)
    rho = jnp.sum((eigvecs[:coarse_size, :] ** 2) * occ[None, :], axis=1).reshape(grid.shape)

    for band in range(n_bands):
        occ_band = float(occ[band])
        if occ_band == 0.0:
            continue
        coarse_flat = eigvecs[:coarse_size, band]
        patch_values = reconstruct_patch_values_v5(eigvecs[:, band], coarse_size, metadata)
        for patch_map in metadata.patch_maps:
            atom_index = patch_map.atom_index
            trace = patch_map.eval_matrix @ coarse_flat[patch_map.sample_indices]
            total = patch_values[atom_index]
            delta_rho_patch = occ_band * (total ** 2 - trace ** 2)
            rho = rho + patch_to_coarse_adjoint(delta_rho_patch, patch_map, grid.shape)

    return jnp.clip(rho, 1e-12, None)


def _compute_v5_patch_density_delta(eigvecs, occ, grid, coarse_size, metadata):
    eigvecs = jnp.asarray(eigvecs, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    n_bands = eigvecs.shape[1]
    coarse_size = int(coarse_size)
    delta = jnp.zeros(grid.shape, dtype=jnp.float32)

    for band in range(n_bands):
        occ_band = float(occ[band])
        if occ_band == 0.0:
            continue
        coarse_flat = eigvecs[:coarse_size, band]
        patch_values = reconstruct_patch_values_v5(eigvecs[:, band], coarse_size, metadata)
        for patch_map in metadata.patch_maps:
            atom_index = patch_map.atom_index
            trace = patch_map.eval_matrix @ coarse_flat[patch_map.sample_indices]
            total = patch_values[atom_index]
            delta_rho_patch = occ_band * (total ** 2 - trace ** 2)
            delta = delta + patch_to_coarse_adjoint(delta_rho_patch, patch_map, grid.shape)
    return delta


def _compute_nuclear_density_weights_v5(grid, coords, rho, radius=0.75):
    rho = jnp.asarray(rho, dtype=jnp.float32)
    weights = []
    for atom_index in range(int(jnp.asarray(coords).shape[0])):
        r = jnp.linalg.norm(grid.coords - coords[atom_index], axis=-1)
        mask = r <= float(radius)
        weights.append(jnp.sum(jnp.where(mask, rho, 0.0)) * grid.volume_element)
    if not weights:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.asarray(weights, dtype=jnp.float32)


def _update_hxc_from_density_v5(grid, rho):
    kernel_k = precompute_poisson_kernel(grid.shape, grid.spacing)
    rho = jnp.clip(jnp.asarray(rho, dtype=jnp.float32), 1e-12, None)
    v_h = solve_poisson(rho, kernel_k, grid.spacing)
    _, v_xc = lda_xc(rho)
    return v_h, v_xc


def audit_fixed_veff_enriched_v5(
    grid,
    coords,
    pseudos,
    v_eff,
    baseline_eigvals,
    baseline_eigvecs,
    n_bands=None,
    occ=None,
    baseline_rho=None,
    nuclear_radius=0.75,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
):
    """Run a fixed-Veff V5 audit against an existing coarse eigensystem.

    This helper intentionally keeps the coarse reference explicit.  The
    embedded ``[c; 0]`` Rayleigh values must match the supplied coarse
    eigenvalues when the caller uses the same coarse Hamiltonian block.
    """
    baseline_eigvals = jnp.asarray(baseline_eigvals, dtype=jnp.float32)
    baseline_eigvecs = jnp.asarray(baseline_eigvecs, dtype=jnp.float32)
    if n_bands is None:
        n_bands = int(baseline_eigvals.shape[0])
    if occ is None:
        occ = jnp.ones((n_bands,), dtype=jnp.float32)
    else:
        occ = jnp.asarray(occ, dtype=jnp.float32)

    h_dense, s_dense, coarse_size, metadata, components = build_fixed_veff_enriched_components_v5(
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
    eigvals = eigvals[:n_bands]

    h_np = np.asarray(h_dense, dtype=np.float64)
    s_np = np.asarray(s_dense, dtype=np.float64)
    norms = np.einsum("ik,ij,jk->k", eigvecs, s_np, eigvecs)

    comp_np = {name: np.asarray(mat, dtype=np.float64) for name, mat in components.items()}
    band_decomposition = {
        name: jnp.asarray(np.einsum("ik,ij,jk->k", eigvecs, mat, eigvecs) / norms, dtype=jnp.float32)
        for name, mat in comp_np.items()
    }
    occupied_decomposition = {
        name: jnp.sum(values[:n_bands] * occ[:n_bands])
        for name, values in band_decomposition.items()
    }

    baseline_np = np.asarray(baseline_eigvecs[:, :n_bands], dtype=np.float64)
    s_cc = s_np[:coarse_size, :coarse_size]
    h_cc = h_np[:coarse_size, :coarse_size]
    embedded_norm = np.einsum("ik,ij,jk->k", baseline_np, s_cc, baseline_np)
    embedded_rayleigh = np.einsum("ik,ij,jk->k", baseline_np, h_cc, baseline_np) / embedded_norm
    embedded_decomposition = {
        name: jnp.asarray(
            np.einsum("ik,ij,jk->k", baseline_np, mat[:coarse_size, :coarse_size], baseline_np)
            / embedded_norm,
            dtype=jnp.float32,
        )
        for name, mat in comp_np.items()
    }

    s_pp = s_np[coarse_size:, coarse_size:]
    coarse_norm = np.einsum("ik,ij,jk->k", eigvecs[:coarse_size, :], s_cc, eigvecs[:coarse_size, :])
    if s_pp.size == 0:
        patch_norm = np.zeros((n_bands,), dtype=np.float64)
    else:
        patch_norm = np.einsum("ik,ij,jk->k", eigvecs[coarse_size:, :], s_pp, eigvecs[coarse_size:, :])

    corrected_eigvals = jnp.asarray(eigvals, dtype=jnp.float32)
    baseline_slice = baseline_eigvals[:n_bands]
    band_delta = corrected_eigvals - baseline_slice
    embedded_cc_rayleigh = jnp.asarray(embedded_rayleigh, dtype=jnp.float32)
    embedded_band_delta = embedded_cc_rayleigh - baseline_slice
    density = compute_v5_density_on_coarse_grid(
        jnp.asarray(eigvecs, dtype=jnp.float32),
        occ[:n_bands],
        grid,
        coarse_size,
        metadata,
    )
    patch_density_delta = _compute_v5_patch_density_delta(
        jnp.asarray(eigvecs, dtype=jnp.float32),
        occ[:n_bands],
        grid,
        coarse_size,
        metadata,
    )
    if baseline_rho is None:
        baseline_rho = jnp.sum(
            (baseline_eigvecs[:, :n_bands] ** 2) * occ[None, :n_bands],
            axis=1,
        ).reshape(grid.shape)
    else:
        baseline_rho = jnp.asarray(baseline_rho, dtype=jnp.float32)
    density_delta = density - baseline_rho
    nuclear_weights = _compute_nuclear_density_weights_v5(grid, coords, density, radius=nuclear_radius)
    baseline_nuclear_weights = _compute_nuclear_density_weights_v5(
        grid,
        coords,
        baseline_rho,
        radius=nuclear_radius,
    )
    return {
        "corrected_eigvals": corrected_eigvals,
        "corrected_eigvecs": jnp.asarray(eigvecs, dtype=jnp.float32),
        "baseline_eigvals": baseline_slice,
        "band_delta": band_delta,
        "band_sum_delta": jnp.sum(band_delta * occ[:n_bands]),
        "band_decomposition": band_decomposition,
        "occupied_decomposition": occupied_decomposition,
        "embedded_cc_rayleigh": embedded_cc_rayleigh,
        "embedded_band_delta": embedded_band_delta,
        "embedded_s_norm": jnp.asarray(embedded_norm, dtype=jnp.float32),
        "embedded_decomposition": embedded_decomposition,
        "density": density,
        "density_delta": density_delta,
        "density_change_l1": jnp.sum(jnp.abs(density_delta)) * grid.volume_element,
        "density_change_linf": jnp.max(jnp.abs(density_delta)),
        "patch_density_delta": patch_density_delta,
        "patch_density_delta_l1": jnp.sum(jnp.abs(patch_density_delta)) * grid.volume_element,
        "electron_count_baseline": jnp.sum(baseline_rho) * grid.volume_element,
        "electron_count_corrected": jnp.sum(density) * grid.volume_element,
        "nuclear_weight_baseline": baseline_nuclear_weights,
        "nuclear_weight_corrected": nuclear_weights,
        "nuclear_weight_delta": nuclear_weights - baseline_nuclear_weights,
        "coarse_metric_fraction_diag": jnp.asarray(coarse_norm / norms, dtype=jnp.float32),
        "patch_metric_fraction_diag": jnp.asarray(patch_norm / norms, dtype=jnp.float32),
        "coarse_size": coarse_size,
        "metadata": metadata,
    }


def post_scf_v5_one_shot_hxc_feedback(
    grid,
    coords,
    pseudos,
    baseline_rho,
    v_h,
    v_xc,
    baseline_eigvals,
    baseline_eigvecs,
    occ,
    v_loc_baseline,
    nuclear_radius=0.75,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
):
    """Run one fixed-Veff V5 density-to-Hxc feedback audit step."""
    baseline_rho = jnp.asarray(baseline_rho, dtype=jnp.float32)
    v_h = jnp.asarray(v_h, dtype=jnp.float32)
    v_xc = jnp.asarray(v_xc, dtype=jnp.float32)
    v_loc_baseline = jnp.asarray(v_loc_baseline, dtype=jnp.float32)
    baseline_eigvals = jnp.asarray(baseline_eigvals, dtype=jnp.float32)
    baseline_eigvecs = jnp.asarray(baseline_eigvecs, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)

    initial_v_eff = v_loc_baseline + v_h + v_xc
    baseline_v_h_check, baseline_v_xc_check = _update_hxc_from_density_v5(grid, baseline_rho)
    baseline_hxc_residual = (baseline_v_h_check + baseline_v_xc_check) - (v_h + v_xc)
    initial = audit_fixed_veff_enriched_v5(
        grid,
        coords,
        pseudos,
        initial_v_eff,
        baseline_eigvals,
        baseline_eigvecs,
        n_bands=int(baseline_eigvals.shape[0]),
        occ=occ,
        baseline_rho=baseline_rho,
        nuclear_radius=nuclear_radius,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
    )
    updated_v_h, updated_v_xc = _update_hxc_from_density_v5(grid, initial["density"])
    updated_v_eff = v_loc_baseline + updated_v_h + updated_v_xc
    updated = audit_fixed_veff_enriched_v5(
        grid,
        coords,
        pseudos,
        updated_v_eff,
        baseline_eigvals,
        baseline_eigvecs,
        n_bands=int(baseline_eigvals.shape[0]),
        occ=occ,
        baseline_rho=baseline_rho,
        nuclear_radius=nuclear_radius,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
    )
    hxc_delta = (updated_v_h + updated_v_xc) - (v_h + v_xc)
    return {
        "initial_audit": initial,
        "updated_audit": updated,
        "updated_v_h": updated_v_h,
        "updated_v_xc": updated_v_xc,
        "hxc_delta": hxc_delta,
        "hxc_delta_linf": jnp.max(jnp.abs(hxc_delta)),
        "hxc_delta_l1": jnp.sum(jnp.abs(hxc_delta)) * grid.volume_element,
        "baseline_hxc_residual": baseline_hxc_residual,
        "baseline_hxc_residual_linf": jnp.max(jnp.abs(baseline_hxc_residual)),
        "baseline_hxc_residual_l1": jnp.sum(jnp.abs(baseline_hxc_residual)) * grid.volume_element,
        "band_sum_feedback_delta": updated["band_sum_delta"] - initial["band_sum_delta"],
        "density_feedback_l1": updated["density_change_l1"] - initial["density_change_l1"],
        "nuclear_weight_before": _compute_nuclear_density_weights_v5(
            grid,
            coords,
            baseline_rho,
            radius=nuclear_radius,
        ),
        "nuclear_weight_after": initial["nuclear_weight_corrected"],
        "nuclear_weight_delta": initial["nuclear_weight_delta"],
    }
