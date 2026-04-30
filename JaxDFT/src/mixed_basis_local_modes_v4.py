from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as sparse_linalg

from .functional import lda_xc
from .patch_maps import patch_to_coarse_adjoint
from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch
from .patch_projector_operator import build_patch_projector_data
from .hamiltonian import build_local_potential, laplacian_8th, precompute_projectors
from .solver import anderson_mixing, ion_ion_energy, precompute_poisson_kernel, solve_poisson, total_energy
from .mixed_basis_galerkin_v3 import (
    _build_coarse_kinetic_matrix,
    _build_coarse_local_baseline,
    _build_coarse_local_matrix,
    _build_patch_dirichlet_kinetic_value_matrix,
    _build_patch_overlap_value_matrix,
    _build_patch_potential_value_matrix,
    _build_projector_value_matrix,
    _insert_global_cross_block,
    _insert_global_square_block,
)

_LOCAL_MODE_ORTH_TOL = 1e-8
_LOBPCG_INIT_SEED = 0


@dataclass(frozen=True)
class MixedBasisLocalModesMetadata:
    patch_specs: list
    patch_maps: list
    patch_bases: dict
    patch_slices: dict


@dataclass(frozen=True)
class LocalModePatchBlockV4:
    atom_index: int
    sample_indices: np.ndarray
    patch_slice: slice
    s_pp: np.ndarray
    s_cp: np.ndarray
    h_pp: np.ndarray
    h_cp: np.ndarray


@dataclass(frozen=True)
class MixedBasisLocalModesOperatorV4:
    grid: object
    coarse_size: int
    total_size: int
    coarse_local_baseline_flat: np.ndarray
    coarse_projector_channels: tuple
    patch_blocks: tuple
    metadata: MixedBasisLocalModesMetadata

    @property
    def coarse_kinetic_diagonal(self):
        spacing = float(self.grid.spacing)
        inv_h2 = 1.0 / (spacing * spacing)
        return np.full(
            (self.coarse_size,),
            float(self.grid.volume_element) * (205.0 / 48.0) * inv_h2,
            dtype=np.float64,
        )

    @property
    def coarse_nonlocal_diagonal(self):
        diag = np.zeros((self.coarse_size,), dtype=np.float64)
        dv2 = float(self.grid.volume_element) ** 2
        for p_i, p_j, coeff in self.coarse_projector_channels:
            pi = np.asarray(p_i, dtype=np.float64).reshape(-1)
            pj = np.asarray(p_j, dtype=np.float64).reshape(-1)
            diag += dv2 * float(coeff) * pi * pj
        return diag

    def _coerce_columns(self, vecs):
        arr = np.asarray(vecs, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[:, None]
        return arr, squeeze

    def _restore_shape(self, arr, squeeze):
        if squeeze:
            return arr[:, 0]
        return arr

    def _apply_coarse_kinetic(self, coarse_vecs):
        result = np.zeros_like(coarse_vecs, dtype=np.float64)
        for col in range(coarse_vecs.shape[1]):
            psi = jnp.asarray(coarse_vecs[:, col], dtype=jnp.float32).reshape(self.grid.shape)
            applied = (-0.5 * laplacian_8th(psi, self.grid.spacing, self.grid.mask)).reshape(-1)
            result[:, col] = np.asarray(self.grid.volume_element * applied, dtype=np.float64)
        return result

    def _apply_coarse_local(self, coarse_vecs):
        return float(self.grid.volume_element) * self.coarse_local_baseline_flat[:, None] * coarse_vecs

    def _apply_coarse_nonlocal(self, coarse_vecs):
        result = np.zeros_like(coarse_vecs, dtype=np.float64)
        dv2 = float(self.grid.volume_element) ** 2
        for p_i, p_j, coeff in self.coarse_projector_channels:
            pi = np.asarray(p_i, dtype=np.float64).reshape(-1)
            pj = np.asarray(p_j, dtype=np.float64).reshape(-1)
            overlaps = pj @ coarse_vecs
            result += dv2 * float(coeff) * pi[:, None] * overlaps[None, :]
        return result

    def apply_h(self, vecs):
        x, squeeze = self._coerce_columns(vecs)
        coarse = x[: self.coarse_size, :]
        result = np.zeros((self.total_size, x.shape[1]), dtype=np.float64)
        result[: self.coarse_size, :] = (
            self._apply_coarse_kinetic(coarse)
            + self._apply_coarse_local(coarse)
            + self._apply_coarse_nonlocal(coarse)
        )

        for block in self.patch_blocks:
            rows = block.sample_indices
            patch_slice = block.patch_slice
            coarse_local = coarse[rows, :]
            patch_local = x[patch_slice, :]
            result[rows, :] += block.h_cp @ patch_local
            result[patch_slice, :] += block.h_cp.T @ coarse_local + block.h_pp @ patch_local

        return self._restore_shape(result, squeeze)

    def apply_s(self, vecs):
        x, squeeze = self._coerce_columns(vecs)
        coarse = x[: self.coarse_size, :]
        result = np.zeros((self.total_size, x.shape[1]), dtype=np.float64)
        result[: self.coarse_size, :] = float(self.grid.volume_element) * coarse

        for block in self.patch_blocks:
            rows = block.sample_indices
            patch_slice = block.patch_slice
            coarse_local = coarse[rows, :]
            patch_local = x[patch_slice, :]
            result[rows, :] += block.s_cp @ patch_local
            result[patch_slice, :] += block.s_cp.T @ coarse_local + block.s_pp @ patch_local

        return self._restore_shape(result, squeeze)


def _s_orthonormalize_columns(columns, metric):
    if columns.size == 0:
        return columns
    gram = 0.5 * (columns.T @ metric @ columns + (columns.T @ metric @ columns).T)
    eigvals, eigvecs = np.linalg.eigh(gram)
    keep = eigvals > _LOCAL_MODE_ORTH_TOL
    if not np.any(keep):
        return np.zeros((columns.shape[0], 0), dtype=np.float64)
    return columns @ (eigvecs[:, keep] / np.sqrt(eigvals[keep])[None, :])


def _project_out_coarse_trace(columns, trace_matrix, metric):
    if columns.size == 0:
        return columns
    gram = trace_matrix.T @ metric @ trace_matrix
    rhs = trace_matrix.T @ metric @ columns
    coeff = np.linalg.pinv(gram, rcond=_LOCAL_MODE_ORTH_TOL) @ rhs
    return columns - trace_matrix @ coeff


def _nullspace_basis(rows):
    if rows.size == 0:
        return np.eye(rows.shape[1], dtype=np.float64)
    _, singular_vals, vt = np.linalg.svd(rows, full_matrices=True)
    rank = int(np.sum(singular_vals > _LOCAL_MODE_ORTH_TOL))
    return vt[rank:].T


def _seeded_random_subspace(size, n_bands):
    rng = np.random.default_rng(_LOBPCG_INIT_SEED)
    init = rng.standard_normal((size, n_bands)).astype(np.float64)
    init, _ = np.linalg.qr(init, mode="reduced")
    return init


def _complete_initial_subspace(previous_vecs, size, n_bands):
    random_block = _seeded_random_subspace(size, n_bands)
    if previous_vecs is None:
        return random_block
    prev = np.asarray(previous_vecs, dtype=np.float64)
    if prev.ndim == 1:
        prev = prev[:, None]
    if prev.shape[0] != size:
        return random_block
    if prev.shape[1] >= n_bands:
        return prev[:, :n_bands]
    needed = n_bands - prev.shape[1]
    return np.concatenate([prev, random_block[:, :needed]], axis=1)


def _build_lobpcg_diagonal_preconditioner(operator):
    diag = np.zeros((operator.total_size,), dtype=np.float64)
    diag[: operator.coarse_size] = (
        operator.coarse_kinetic_diagonal
        + float(operator.grid.volume_element) * operator.coarse_local_baseline_flat
        + operator.coarse_nonlocal_diagonal
    )
    for block in operator.patch_blocks:
        patch_diag = np.diag(block.h_pp)
        diag[block.patch_slice] = patch_diag
    floor = max(1e-6, 1e-3 * np.max(np.abs(diag)) if diag.size else 1e-6)
    safe_diag = np.maximum(np.abs(diag), floor)

    def apply(vecs):
        arr = np.asarray(vecs, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[:, None]
        out = arr / safe_diag[:, None]
        if squeeze:
            return out[:, 0]
        return out

    return sparse_linalg.LinearOperator(
        (operator.total_size, operator.total_size),
        matvec=lambda x: apply(x),
        matmat=lambda x: apply(x),
        dtype=np.float64,
    )


def _schedule_lobpcg_controls(iteration, max_iter, eig_maxiter, eig_tol):
    if iteration == 0:
        return max(20, eig_maxiter // 3), max(eig_tol * 1e3, 1e-3)
    if iteration < max(1, max_iter - 2):
        return max(30, eig_maxiter // 2), max(eig_tol * 1e2, 1e-4)
    return eig_maxiter, eig_tol


def build_patch_local_mode_basis_v4(patch_spec, patch_map, num_modes=8):
    overlap = np.asarray(_build_patch_overlap_value_matrix(patch_map), dtype=np.float64)
    kinetic = np.asarray(
        _build_patch_dirichlet_kinetic_value_matrix(patch_spec, patch_map),
        dtype=np.float64,
    )
    trace = np.asarray(patch_map.eval_matrix, dtype=np.float64)
    constraint_rows = np.concatenate([trace.T @ overlap, trace.T @ kinetic], axis=0)
    nullspace = _nullspace_basis(constraint_rows)
    if nullspace.shape[1] == 0:
        return jnp.zeros((overlap.shape[0], 0), dtype=jnp.float32)
    reduced_overlap = 0.5 * (nullspace.T @ overlap @ nullspace + (nullspace.T @ overlap @ nullspace).T)
    reduced_kinetic = 0.5 * (nullspace.T @ kinetic @ nullspace + (nullspace.T @ kinetic @ nullspace).T)
    reduced_size = reduced_overlap.shape[0]
    n_keep = min(num_modes, reduced_size)
    eigvals, eigvecs = scipy_linalg.eigh(
        reduced_kinetic,
        reduced_overlap,
        subset_by_index=[0, n_keep - 1],
    )
    _ = eigvals
    basis = nullspace @ eigvecs[:, :n_keep]
    basis = _s_orthonormalize_columns(basis, overlap)
    return jnp.asarray(basis, dtype=jnp.float32)


def _build_coarse_nonlocal_matrix_cell_average(grid, coords, pseudos, projector_subgrid=1):
    coarse_proj = precompute_projectors(
        grid,
        coords,
        pseudos,
        projector_subgrid=projector_subgrid,
        projector_mode="cell_average",
    )
    coarse_size = int(jnp.prod(jnp.array(grid.shape)))
    if coarse_proj is None:
        return jnp.zeros((coarse_size, coarse_size), dtype=jnp.float32), ()

    p_i, p_j, coeffs = coarse_proj
    h_cc_v_nl = jnp.zeros((coarse_size, coarse_size), dtype=jnp.float32)
    coarse_channels = []
    for idx in range(p_i.shape[0]):
        pi = jnp.asarray(p_i[idx], dtype=jnp.float32).reshape(-1)
        pj = jnp.asarray(p_j[idx], dtype=jnp.float32).reshape(-1)
        coeff = jnp.asarray(coeffs[idx], dtype=jnp.float32)
        h_cc_v_nl = h_cc_v_nl + (grid.volume_element ** 2) * coeff * jnp.outer(pi, pj)
        coarse_channels.append((p_i[idx], p_j[idx], coeff))
    h_cc_v_nl = 0.5 * (h_cc_v_nl + h_cc_v_nl.T)
    return h_cc_v_nl, tuple(coarse_channels)


def build_fixed_veff_local_modes_operator_v4(
    grid,
    coords,
    pseudos,
    v_eff,
    coarse_local_baseline=None,
    delta_v_grid=None,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
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
    if coarse_local_baseline is None:
        coarse_local_baseline = _build_coarse_local_baseline(grid, coords, pseudos)
    else:
        coarse_local_baseline = jnp.asarray(coarse_local_baseline, dtype=jnp.float32)
    if delta_v_grid is None:
        delta_v_grid = v_eff - coarse_local_baseline
    else:
        delta_v_grid = jnp.asarray(delta_v_grid, dtype=jnp.float32)
    patch_bases = {
        patch_spec.atom_index: build_patch_local_mode_basis_v4(
            patch_spec,
            patch_map,
            num_modes=num_local_modes,
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

    _, coarse_projector_channels = _build_coarse_nonlocal_matrix_cell_average(
        grid,
        coords,
        pseudos,
        projector_subgrid=1,
    )

    channels_by_atom = {}
    for channel in projector_data.channels:
        channels_by_atom.setdefault(channel.atom_index, []).append(channel)
    channel_idx = 0
    patch_blocks = []

    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        atom_index = patch_map.atom_index
        basis = patch_bases[atom_index]
        patch_slice = patch_slices[atom_index]
        eval_matrix = jnp.asarray(patch_map.eval_matrix, dtype=jnp.float32)
        overlap_value = _build_patch_overlap_value_matrix(patch_map)
        kinetic_value = _build_patch_dirichlet_kinetic_value_matrix(patch_spec, patch_map)
        delta_v_patch = coarse_to_patch(delta_v_grid, patch_map)
        potential_value = _build_patch_potential_value_matrix(delta_v_patch, patch_map)

        s_pp = basis.T @ overlap_value @ basis
        s_cp_local = eval_matrix.T @ overlap_value @ basis
        h_pp = basis.T @ kinetic_value @ basis + basis.T @ potential_value @ basis
        h_cp_local = eval_matrix.T @ kinetic_value @ basis + eval_matrix.T @ potential_value @ basis

        for channel in channels_by_atom.get(atom_index, []):
            projector_value_fine = _build_projector_value_matrix(channel, patch_map)
            coarse_p_i_patch = coarse_to_patch(coarse_projector_channels[channel_idx][0], patch_map)
            coarse_p_j_patch = coarse_to_patch(coarse_projector_channels[channel_idx][1], patch_map)
            projector_value_coarse = channel.coeff * (patch_map.fine_dv ** 2) * jnp.outer(
                coarse_p_i_patch,
                coarse_p_j_patch,
            )
            delta_projector_value = projector_value_fine - projector_value_coarse
            delta_projector_value = 0.5 * (delta_projector_value + delta_projector_value.T)
            h_pp = h_pp + basis.T @ delta_projector_value @ basis
            h_cp_local = h_cp_local + eval_matrix.T @ delta_projector_value @ basis
            channel_idx += 1

        patch_blocks.append(
            LocalModePatchBlockV4(
                atom_index=atom_index,
                sample_indices=np.asarray(patch_map.sample_indices, dtype=np.int32),
                patch_slice=patch_slice,
                s_pp=np.asarray(0.5 * (s_pp + s_pp.T), dtype=np.float64),
                s_cp=np.asarray(s_cp_local, dtype=np.float64),
                h_pp=np.asarray(0.5 * (h_pp + h_pp.T), dtype=np.float64),
                h_cp=np.asarray(h_cp_local, dtype=np.float64),
            )
        )

    metadata = MixedBasisLocalModesMetadata(
        patch_specs=patch_specs,
        patch_maps=patch_maps,
        patch_bases=patch_bases,
        patch_slices=patch_slices,
    )
    return MixedBasisLocalModesOperatorV4(
        grid=grid,
        coarse_size=coarse_size,
        total_size=total_size,
        coarse_local_baseline_flat=np.asarray(coarse_local_baseline, dtype=np.float64).reshape(-1),
        coarse_projector_channels=tuple(
            (
                np.asarray(p_i, dtype=np.float64),
                np.asarray(p_j, dtype=np.float64),
                float(coeff),
            )
            for p_i, p_j, coeff in coarse_projector_channels
        ),
        patch_blocks=tuple(patch_blocks),
        metadata=metadata,
    )


def build_fixed_veff_local_modes_components_v4(
    grid,
    coords,
    pseudos,
    v_eff,
    coarse_local_baseline=None,
    delta_v_grid=None,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
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

    h_cc_t = _build_coarse_kinetic_matrix(grid)
    if coarse_local_baseline is None:
        coarse_local_baseline = _build_coarse_local_baseline(grid, coords, pseudos)
    else:
        coarse_local_baseline = jnp.asarray(coarse_local_baseline, dtype=jnp.float32)
    if delta_v_grid is None:
        delta_v_grid = v_eff - coarse_local_baseline
    else:
        delta_v_grid = jnp.asarray(delta_v_grid, dtype=jnp.float32)
    patch_bases = {
        patch_spec.atom_index: build_patch_local_mode_basis_v4(
            patch_spec,
            patch_map,
            num_modes=num_local_modes,
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
    h_cc_v_loc = _build_coarse_local_matrix(grid, coarse_local_baseline)
    h_cc_v_nl, coarse_projector_channels = _build_coarse_nonlocal_matrix_cell_average(
        grid,
        coords,
        pseudos,
        projector_subgrid=1,
    )
    s_cc = grid.volume_element * jnp.eye(coarse_size, dtype=jnp.float32)
    s_dense[:coarse_size, :coarse_size] = np.asarray(s_cc, dtype=np.float32)
    h_t[:coarse_size, :coarse_size] = np.asarray(h_cc_t, dtype=np.float32)
    h_v_loc[:coarse_size, :coarse_size] = np.asarray(h_cc_v_loc, dtype=np.float32)
    h_v_nl[:coarse_size, :coarse_size] = np.asarray(h_cc_v_nl, dtype=np.float32)

    channels_by_atom = {}
    for channel in projector_data.channels:
        channels_by_atom.setdefault(channel.atom_index, []).append(channel)
    channel_idx = 0

    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        atom_index = patch_map.atom_index
        basis = patch_bases[atom_index]
        patch_slice = patch_slices[atom_index]
        eval_matrix = jnp.asarray(patch_map.eval_matrix, dtype=jnp.float32)
        overlap_value = _build_patch_overlap_value_matrix(patch_map)
        kinetic_value = _build_patch_dirichlet_kinetic_value_matrix(patch_spec, patch_map)
        delta_v_patch = coarse_to_patch(delta_v_grid, patch_map)
        potential_value = _build_patch_potential_value_matrix(delta_v_patch, patch_map)

        s_pp = basis.T @ overlap_value @ basis
        s_cp_local = eval_matrix.T @ overlap_value @ basis
        h_pp_t = basis.T @ kinetic_value @ basis
        h_cp_t = eval_matrix.T @ kinetic_value @ basis
        h_pp_v_loc = basis.T @ potential_value @ basis
        h_cp_v_loc = eval_matrix.T @ potential_value @ basis
        h_pp_v_nl = jnp.zeros_like(h_pp_t)
        h_cp_v_nl = jnp.zeros_like(h_cp_t)

        for channel in channels_by_atom.get(atom_index, []):
            projector_value_fine = _build_projector_value_matrix(channel, patch_map)
            coarse_p_i_patch = coarse_to_patch(coarse_projector_channels[channel_idx][0], patch_map)
            coarse_p_j_patch = coarse_to_patch(coarse_projector_channels[channel_idx][1], patch_map)
            projector_value_coarse = channel.coeff * (patch_map.fine_dv ** 2) * jnp.outer(
                coarse_p_i_patch,
                coarse_p_j_patch,
            )
            delta_projector_value = projector_value_fine - projector_value_coarse
            delta_projector_value = 0.5 * (delta_projector_value + delta_projector_value.T)
            h_pp_v_nl = h_pp_v_nl + basis.T @ delta_projector_value @ basis
            h_cp_v_nl = h_cp_v_nl + eval_matrix.T @ delta_projector_value @ basis
            channel_idx += 1

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

    metadata = MixedBasisLocalModesMetadata(
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


def build_fixed_veff_local_modes_matrices_v4(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
):
    h_dense, s_dense, coarse_size, metadata, _ = build_fixed_veff_local_modes_components_v4(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        num_local_modes=num_local_modes,
    )
    return h_dense, s_dense, coarse_size, metadata


def solve_fixed_veff_local_modes_dense_host_v4(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
):
    h_dense, s_dense, coarse_size, metadata = build_fixed_veff_local_modes_matrices_v4(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        num_local_modes=num_local_modes,
    )
    eigvals, eigvecs = scipy_linalg.eigh(np.asarray(h_dense, dtype=np.float64), np.asarray(s_dense, dtype=np.float64))
    return (
        jnp.asarray(eigvals[:n_bands], dtype=jnp.float32),
        jnp.asarray(eigvecs[:, :n_bands], dtype=jnp.float32),
        coarse_size,
        metadata,
    )


def solve_fixed_veff_local_modes_iterative_host_v4(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
    maxiter=100,
    tol=1e-6,
):
    operator = build_fixed_veff_local_modes_operator_v4(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        num_local_modes=num_local_modes,
    )
    size = operator.total_size

    a_op = sparse_linalg.LinearOperator(
        (size, size),
        matvec=lambda x: operator.apply_h(x),
        matmat=lambda x: operator.apply_h(x),
        dtype=np.float64,
    )
    b_op = sparse_linalg.LinearOperator(
        (size, size),
        matvec=lambda x: operator.apply_s(x),
        matmat=lambda x: operator.apply_s(x),
        dtype=np.float64,
    )
    m_op = _build_lobpcg_diagonal_preconditioner(operator)
    init = _seeded_random_subspace(size, n_bands)
    eigvals, eigvecs = sparse_linalg.lobpcg(
        a_op,
        init,
        B=b_op,
        M=m_op,
        largest=False,
        maxiter=maxiter,
        tol=tol,
    )
    order = np.argsort(eigvals)
    eigvals = eigvals[order][:n_bands]
    eigvecs = eigvecs[:, order][:, :n_bands]
    return (
        jnp.asarray(eigvals, dtype=jnp.float32),
        jnp.asarray(eigvecs, dtype=jnp.float32),
        operator.coarse_size,
        operator.metadata,
    )


def reconstruct_patch_values_v4(eigvec, coarse_size, metadata):
    vec = jnp.asarray(eigvec, dtype=jnp.float32)
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


def compute_v4_density_on_coarse_grid(eigvecs, occ, grid, coarse_size, metadata):
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
        patch_values = reconstruct_patch_values_v4(eigvecs[:, band], coarse_size, metadata)
        for patch_map in metadata.patch_maps:
            atom_index = patch_map.atom_index
            trace = patch_map.eval_matrix @ coarse_flat[patch_map.sample_indices]
            total = patch_values[atom_index]
            delta_rho_patch = occ_band * (total ** 2 - trace ** 2)
            rho = rho + patch_to_coarse_adjoint(delta_rho_patch, patch_map, grid.shape)

    return jnp.clip(rho, 1e-12, None)


def build_selfconsistent_local_modes_matrices_v4(
    grid,
    coords,
    pseudos,
    v_h,
    v_xc,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
):
    v_h = jnp.asarray(v_h, dtype=jnp.float32)
    v_xc = jnp.asarray(v_xc, dtype=jnp.float32)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
    v_loc_patch = build_local_potential(
        coords,
        grid.coords,
        zion,
        rloc,
        c,
        spacing=grid.spacing,
        local_subgrid=patch_subgrid,
        local_mode="patch",
        local_patch_radius_factor=patch_radius_factor,
    )
    v_loc_coarse = _build_coarse_local_baseline(grid, coords, pseudos)
    v_eff_coarse = v_loc_coarse + v_h + v_xc
    delta_v_grid = v_loc_patch - v_loc_coarse
    v_eff = v_eff_coarse + delta_v_grid
    return build_fixed_veff_local_modes_components_v4(
        grid,
        coords,
        pseudos,
        v_eff,
        coarse_local_baseline=v_eff_coarse,
        delta_v_grid=delta_v_grid,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        num_local_modes=num_local_modes,
    )


def build_selfconsistent_local_modes_operator_v4(
    grid,
    coords,
    pseudos,
    v_h,
    v_xc,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
):
    v_h = jnp.asarray(v_h, dtype=jnp.float32)
    v_xc = jnp.asarray(v_xc, dtype=jnp.float32)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
    v_loc_patch = build_local_potential(
        coords,
        grid.coords,
        zion,
        rloc,
        c,
        spacing=grid.spacing,
        local_subgrid=patch_subgrid,
        local_mode="patch",
        local_patch_radius_factor=patch_radius_factor,
    )
    v_loc_coarse = _build_coarse_local_baseline(grid, coords, pseudos)
    v_eff_coarse = v_loc_coarse + v_h + v_xc
    delta_v_grid = v_loc_patch - v_loc_coarse
    v_eff = v_eff_coarse + delta_v_grid
    return build_fixed_veff_local_modes_operator_v4(
        grid,
        coords,
        pseudos,
        v_eff,
        coarse_local_baseline=v_eff_coarse,
        delta_v_grid=delta_v_grid,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        num_local_modes=num_local_modes,
    )


def solve_selfconsistent_local_modes_dense_reference_v4(
    grid,
    coords,
    pseudos,
    n_bands,
    occ,
    max_iter,
    mix_alpha,
    tolerance,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
):
    coords = jnp.asarray(coords, dtype=jnp.float32)
    volume_element = grid.volume_element
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for atom_index in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[atom_index], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = rho / (jnp.sum(rho) * volume_element) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    kernel_k = precompute_poisson_kernel(grid.shape, grid.spacing)
    eigvals = jnp.zeros((n_bands,), dtype=jnp.float32)
    eigvecs = jnp.zeros((rho.size, n_bands), dtype=jnp.float32)
    coarse_size = int(rho.size)
    metadata = None
    v_h = jnp.zeros(grid.shape, dtype=jnp.float32)
    eps_xc = jnp.zeros(grid.shape, dtype=jnp.float32)
    v_xc = jnp.zeros(grid.shape, dtype=jnp.float32)

    for iteration in range(max_iter):
        rho = jnp.clip(rho, 1e-12, None)
        v_h = solve_poisson(rho, kernel_k, grid.spacing)
        eps_xc, v_xc = lda_xc(rho)
        h_dense, s_dense, coarse_size, metadata, _ = build_selfconsistent_local_modes_matrices_v4(
            grid,
            coords,
            pseudos,
            v_h,
            v_xc,
            patch_subgrid=patch_subgrid,
            patch_radius_factor=patch_radius_factor,
            num_local_modes=num_local_modes,
        )
        eigvals_np, eigvecs_np = scipy_linalg.eigh(
            np.asarray(h_dense, dtype=np.float64),
            np.asarray(s_dense, dtype=np.float64),
            subset_by_index=[0, n_bands - 1],
        )
        eigvals = jnp.asarray(eigvals_np, dtype=jnp.float32)
        eigvecs = jnp.asarray(eigvecs_np, dtype=jnp.float32)
        rho_new = compute_v4_density_on_coarse_grid(eigvecs, occ, grid, coarse_size, metadata)
        diff = jnp.max(jnp.abs(rho_new - rho))
        rho_flat, f_hist = anderson_mixing(
            rho.reshape(-1),
            rho_new.reshape(-1),
            f_hist,
            mix_alpha,
            iteration,
        )
        rho = rho_flat.reshape(grid.shape)
        if float(diff) < tolerance:
            break

    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    ion_ion = ion_ion_energy(coords, zion)
    v_loc_patch = build_local_potential(
        coords,
        grid.coords,
        zion,
        jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32),
        jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32),
        spacing=grid.spacing,
        local_subgrid=patch_subgrid,
        local_mode="patch",
        local_patch_radius_factor=patch_radius_factor,
    )
    energy = total_energy(
        rho,
        eigvals,
        occ,
        v_loc_patch,
        v_h,
        eps_xc,
        v_xc,
        volume_element,
        ion_ion,
    )
    return rho, eigvals, eigvecs, v_h, eps_xc, v_xc, energy, coarse_size, metadata


def solve_selfconsistent_local_modes_dense_host_v4(
    grid,
    coords,
    pseudos,
    n_bands,
    occ,
    max_iter,
    mix_alpha,
    tolerance,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    num_local_modes=8,
    eig_maxiter=100,
    eig_tol=1e-6,
):
    coords = jnp.asarray(coords, dtype=jnp.float32)
    volume_element = grid.volume_element
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for atom_index in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[atom_index], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = rho / (jnp.sum(rho) * volume_element) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    kernel_k = precompute_poisson_kernel(grid.shape, grid.spacing)
    eigvals = jnp.zeros((n_bands,), dtype=jnp.float32)
    eigvecs = jnp.zeros((rho.size, n_bands), dtype=jnp.float32)
    coarse_size = int(rho.size)
    metadata = None
    v_h = jnp.zeros(grid.shape, dtype=jnp.float32)
    eps_xc = jnp.zeros(grid.shape, dtype=jnp.float32)
    v_xc = jnp.zeros(grid.shape, dtype=jnp.float32)
    previous_eigvecs = None

    for iteration in range(max_iter):
        rho = jnp.clip(rho, 1e-12, None)
        v_h = solve_poisson(rho, kernel_k, grid.spacing)
        eps_xc, v_xc = lda_xc(rho)
        operator = build_selfconsistent_local_modes_operator_v4(
            grid,
            coords,
            pseudos,
            v_h,
            v_xc,
            patch_subgrid=patch_subgrid,
            patch_radius_factor=patch_radius_factor,
            num_local_modes=num_local_modes,
        )
        size = operator.total_size
        a_op = sparse_linalg.LinearOperator(
            (size, size),
            matvec=lambda x: operator.apply_h(x),
            matmat=lambda x: operator.apply_h(x),
            dtype=np.float64,
        )
        b_op = sparse_linalg.LinearOperator(
            (size, size),
            matvec=lambda x: operator.apply_s(x),
            matmat=lambda x: operator.apply_s(x),
            dtype=np.float64,
        )
        m_op = _build_lobpcg_diagonal_preconditioner(operator)
        init = _complete_initial_subspace(previous_eigvecs, size, n_bands)
        local_maxiter, local_tol = _schedule_lobpcg_controls(
            iteration,
            max_iter,
            eig_maxiter,
            eig_tol,
        )
        eigvals_np, eigvecs_np = sparse_linalg.lobpcg(
            a_op,
            init,
            B=b_op,
            M=m_op,
            largest=False,
            maxiter=local_maxiter,
            tol=local_tol,
        )
        order = np.argsort(eigvals_np)
        eigvals = jnp.asarray(eigvals_np[order][:n_bands], dtype=jnp.float32)
        eigvecs = jnp.asarray(eigvecs_np[:, order][:, :n_bands], dtype=jnp.float32)
        previous_eigvecs = np.asarray(eigvecs_np[:, order][:, :n_bands], dtype=np.float64)
        coarse_size = operator.coarse_size
        metadata = operator.metadata
        rho_new = compute_v4_density_on_coarse_grid(eigvecs, occ, grid, coarse_size, metadata)
        diff = jnp.max(jnp.abs(rho_new - rho))
        rho_flat, f_hist = anderson_mixing(
            rho.reshape(-1),
            rho_new.reshape(-1),
            f_hist,
            mix_alpha,
            iteration,
        )
        rho = rho_flat.reshape(grid.shape)
        if float(diff) < tolerance:
            break

    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    ion_ion = ion_ion_energy(coords, zion)
    v_loc_patch = build_local_potential(
        coords,
        grid.coords,
        zion,
        jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32),
        jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32),
        spacing=grid.spacing,
        local_subgrid=patch_subgrid,
        local_mode="patch",
        local_patch_radius_factor=patch_radius_factor,
    )
    energy = total_energy(
        rho,
        eigvals,
        occ,
        v_loc_patch,
        v_h,
        eps_xc,
        v_xc,
        volume_element,
        ion_ion,
    )
    return rho, eigvals, eigvecs, v_h, eps_xc, v_xc, energy, coarse_size, metadata
