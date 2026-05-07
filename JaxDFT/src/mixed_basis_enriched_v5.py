from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as sparse_linalg

from .functional import lda_xc
from .hamiltonian import laplacian_8th
from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch, patch_to_coarse_adjoint
from .solver import ion_ion_energy, precompute_poisson_kernel, solve_poisson
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
_V5_LOBPCG_INIT_SEED = 0


@dataclass(frozen=True)
class EnrichedBasisMetadataV5:
    patch_specs: list
    patch_maps: list
    patch_bases: dict
    patch_slices: dict


@dataclass(frozen=True)
class EnrichedPatchBlockV5:
    atom_index: int
    sample_indices: np.ndarray
    patch_slice: slice
    s_pp: np.ndarray
    s_cp: np.ndarray
    h_pp: np.ndarray
    h_cp: np.ndarray
    h_t_pp: np.ndarray
    h_t_cp: np.ndarray
    h_v_loc_pp: np.ndarray
    h_v_loc_cp: np.ndarray


@dataclass(frozen=True)
class EnrichedOperatorV5:
    grid: object
    coarse_size: int
    total_size: int
    v_eff_flat: np.ndarray
    coarse_projector_channels: tuple
    patch_blocks: tuple
    metadata: EnrichedBasisMetadataV5

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
        return float(self.grid.volume_element) * self.v_eff_flat[:, None] * coarse_vecs

    def _apply_coarse_nonlocal(self, coarse_vecs):
        result = np.zeros_like(coarse_vecs, dtype=np.float64)
        dv2 = float(self.grid.volume_element) ** 2
        for p_i, p_j, coeff in self.coarse_projector_channels:
            pi = np.asarray(p_i, dtype=np.float64).reshape(-1)
            pj = np.asarray(p_j, dtype=np.float64).reshape(-1)
            overlaps = pj @ coarse_vecs
            result += dv2 * float(coeff) * pi[:, None] * overlaps[None, :]
        return result

    def _apply_patch_blocks(self, x, component):
        coarse = x[: self.coarse_size, :]
        result = np.zeros((self.total_size, x.shape[1]), dtype=np.float64)
        for block in self.patch_blocks:
            rows = block.sample_indices
            patch_slice = block.patch_slice
            coarse_local = coarse[rows, :]
            patch_local = x[patch_slice, :]
            if component == "t":
                h_pp = block.h_t_pp
                h_cp = block.h_t_cp
            elif component == "v_loc":
                h_pp = block.h_v_loc_pp
                h_cp = block.h_v_loc_cp
            elif component == "h":
                h_pp = block.h_pp
                h_cp = block.h_cp
            else:
                raise ValueError(f"unknown V5 component: {component}")
            result[rows, :] += h_cp @ patch_local
            result[patch_slice, :] += h_cp.T @ coarse_local + h_pp @ patch_local
        return result

    def apply_t(self, vecs):
        x, squeeze = self._coerce_columns(vecs)
        coarse = x[: self.coarse_size, :]
        result = np.zeros((self.total_size, x.shape[1]), dtype=np.float64)
        result[: self.coarse_size, :] = self._apply_coarse_kinetic(coarse)
        result += self._apply_patch_blocks(x, "t")
        return self._restore_shape(result, squeeze)

    def apply_vloc(self, vecs):
        x, squeeze = self._coerce_columns(vecs)
        coarse = x[: self.coarse_size, :]
        result = np.zeros((self.total_size, x.shape[1]), dtype=np.float64)
        result[: self.coarse_size, :] = self._apply_coarse_local(coarse)
        result += self._apply_patch_blocks(x, "v_loc")
        return self._restore_shape(result, squeeze)

    def apply_vnl(self, vecs):
        x, squeeze = self._coerce_columns(vecs)
        coarse = x[: self.coarse_size, :]
        result = np.zeros((self.total_size, x.shape[1]), dtype=np.float64)
        result[: self.coarse_size, :] = self._apply_coarse_nonlocal(coarse)
        return self._restore_shape(result, squeeze)

    def apply_h(self, vecs):
        x, squeeze = self._coerce_columns(vecs)
        result = (
            np.asarray(self.apply_t(x), dtype=np.float64)
            + np.asarray(self.apply_vloc(x), dtype=np.float64)
            + np.asarray(self.apply_vnl(x), dtype=np.float64)
        )
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


def _project_out_vloc_weighted_coarse(columns, trace, patch_map, v_eff_patch):
    if columns.size == 0 or v_eff_patch is None:
        return columns
    v_eff_patch = np.asarray(v_eff_patch, dtype=np.float64).reshape(-1)
    weights = np.maximum(-v_eff_patch, 0.0)
    if not np.any(weights > 0.0):
        return columns
    weighted_metric = np.diag(weights * float(patch_map.fine_dv))
    return _project_out_coarse_overlap(columns, trace, weighted_metric)


def _project_out_combined_coarse_constraints(columns, trace, overlap, patch_map, v_eff_patch):
    if columns.size == 0 or v_eff_patch is None:
        return _project_out_coarse_overlap(columns, trace, overlap)
    v_eff_patch = np.asarray(v_eff_patch, dtype=np.float64).reshape(-1)
    weights = np.maximum(-v_eff_patch, 0.0)
    if not np.any(weights > 0.0):
        return _project_out_coarse_overlap(columns, trace, overlap)
    weighted_metric = np.diag(weights * float(patch_map.fine_dv))
    constraints = np.concatenate([trace.T @ overlap, trace.T @ weighted_metric], axis=0)
    gram = constraints @ constraints.T
    coeff = np.linalg.pinv(gram, rcond=_V5_RANK_TOL) @ (constraints @ columns)
    return columns - constraints.T @ coeff


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
    vloc_aware_constraint=False,
    v_eff_patch=None,
):
    """Build S-orthogonal atom-centered local basis functions for V5 audits."""
    overlap = np.asarray(_build_patch_overlap_value_matrix(patch_map), dtype=np.float64)
    trace = np.asarray(patch_map.eval_matrix, dtype=np.float64)
    raw = _raw_atom_centered_columns(patch_spec, pseudo)
    if vloc_aware_constraint:
        projected = _project_out_combined_coarse_constraints(raw, trace, overlap, patch_map, v_eff_patch)
    else:
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


def _representative_projector_radius_v5(pseudo, fallback):
    radii = [
        float(projector.get("r", fallback))
        for projector in pseudo.get("projectors", [])
        if float(projector.get("r", fallback)) > 0.0
    ]
    if not radii:
        return float(fallback)
    return float(min(radii))


def _classify_v5_radial_channel(r_rms, rloc, projector_r):
    core_cut = max(float(rloc), 0.5 * float(projector_r))
    projector_cut = max(1.5 * float(projector_r), core_cut)
    if float(r_rms) <= core_cut:
        return "core"
    if float(r_rms) <= projector_cut:
        return "projector"
    return "tail"


def _classify_v5_angular_channel(mode, offsets, overlap, r_scale):
    weighted = overlap @ mode
    monopole = abs(float(np.sum(weighted)))
    dipole = np.linalg.norm(offsets.T @ weighted) / max(float(r_scale), 1e-8)
    return "p" if dipole > monopole else "s"


def _build_v5_mode_diagnostics(metadata, pseudos, components, s_dense, coarse_size, eigvecs=None, occ=None):
    diagnostics = []
    s_np = np.asarray(s_dense, dtype=np.float64)
    comp_np = {name: np.asarray(mat, dtype=np.float64) for name, mat in components.items()}
    eigvecs_np = None if eigvecs is None else np.asarray(eigvecs, dtype=np.float64)
    occ_np = None if occ is None else np.asarray(occ, dtype=np.float64)
    applied_components = {}
    if eigvecs_np is not None and occ_np is not None:
        applied_components = {
            name: mat @ eigvecs_np
            for name, mat in comp_np.items()
        }
    for patch_map in metadata.patch_maps:
        atom_index = int(patch_map.atom_index)
        pseudo = pseudos[atom_index]
        basis = np.asarray(metadata.patch_bases[atom_index], dtype=np.float64)
        patch_slice = metadata.patch_slices[atom_index]
        if basis.shape[1] == 0:
            continue
        offsets = np.asarray(metadata.patch_specs[atom_index].offsets, dtype=np.float64)
        r = np.linalg.norm(offsets, axis=1)
        rloc = float(pseudo.get("rloc", 0.5))
        projector_r = _representative_projector_radius_v5(pseudo, rloc)
        overlap = np.asarray(_build_patch_overlap_value_matrix(patch_map), dtype=np.float64)
        local_s = s_np[patch_slice, patch_slice]
        local_t = comp_np["t"][patch_slice, patch_slice]
        local_v = comp_np["v_loc"][patch_slice, patch_slice]
        local_vnl = comp_np["v_nl"][patch_slice, patch_slice]
        for mode_index in range(basis.shape[1]):
            mode = basis[:, mode_index]
            coeff = np.zeros((basis.shape[1],), dtype=np.float64)
            coeff[mode_index] = 1.0
            s_norm = float(coeff @ local_s @ coeff)
            safe_s = max(abs(s_norm), 1e-14)
            mode_weight = mode * (overlap @ mode)
            r_rms = float(np.sqrt(max(np.sum((r ** 2) * mode_weight) / safe_s, 0.0)))
            near_core_weight = float(np.sum(np.where(r <= rloc, mode_weight, 0.0)) / safe_s)
            v_loc_expectation = float(coeff @ local_v @ coeff / safe_s)
            patch_index = int(patch_slice.start + mode_index)
            occupied_contrib = {
                name: 0.0
                for name in ("t", "v_loc", "v_nl")
            }
            if eigvecs_np is not None and occ_np is not None:
                for name, applied in applied_components.items():
                    occupied_contrib[name] = float(
                        np.sum(occ_np[: eigvecs_np.shape[1]] * eigvecs_np[patch_index, :] * applied[patch_index, :])
                    )
            diagnostics.append({
                "atom_index": atom_index,
                "symbol": str(pseudo.get("symbol", "")),
                "mode_index": int(mode_index),
                "patch_index": patch_index,
                "rloc": float(rloc),
                "projector_radius": float(projector_r),
                "r_rms": float(r_rms),
                "radial_channel": _classify_v5_radial_channel(r_rms, rloc, projector_r),
                "angular_channel": _classify_v5_angular_channel(mode, offsets, overlap, projector_r),
                "s_norm": float(s_norm),
                "t_expectation": float(coeff @ local_t @ coeff / safe_s),
                "v_loc_expectation": v_loc_expectation,
                "v_nl_expectation": float(coeff @ local_vnl @ coeff / safe_s),
                "vloc_sensitivity": float(abs(v_loc_expectation)),
                "occupied_t_contribution": occupied_contrib["t"],
                "occupied_vloc_contribution": occupied_contrib["v_loc"],
                "occupied_vnl_contribution": occupied_contrib["v_nl"],
                "near_core_weight": float(max(near_core_weight, 0.0)),
            })
    return diagnostics


def _build_v5_operator_mode_diagnostics(operator, pseudos, eigvecs=None, occ=None):
    diagnostics = []
    block_by_atom = {int(block.atom_index): block for block in operator.patch_blocks}
    eigvecs_np = None if eigvecs is None else np.asarray(eigvecs, dtype=np.float64)
    occ_np = None if occ is None else np.asarray(occ, dtype=np.float64)
    applied_components = {}
    if eigvecs_np is not None and occ_np is not None:
        applied_components = {
            "t": np.asarray(operator.apply_t(eigvecs_np), dtype=np.float64),
            "v_loc": np.asarray(operator.apply_vloc(eigvecs_np), dtype=np.float64),
            "v_nl": np.asarray(operator.apply_vnl(eigvecs_np), dtype=np.float64),
        }
    for patch_map in operator.metadata.patch_maps:
        atom_index = int(patch_map.atom_index)
        if atom_index not in block_by_atom:
            continue
        block = block_by_atom[atom_index]
        pseudo = pseudos[atom_index]
        basis = np.asarray(operator.metadata.patch_bases[atom_index], dtype=np.float64)
        if basis.shape[1] == 0:
            continue
        offsets = np.asarray(operator.metadata.patch_specs[atom_index].offsets, dtype=np.float64)
        r = np.linalg.norm(offsets, axis=1)
        rloc = float(pseudo.get("rloc", 0.5))
        projector_r = _representative_projector_radius_v5(pseudo, rloc)
        overlap = np.asarray(_build_patch_overlap_value_matrix(patch_map), dtype=np.float64)
        zero_vnl = np.zeros_like(block.s_pp, dtype=np.float64)
        for mode_index in range(basis.shape[1]):
            mode = basis[:, mode_index]
            coeff = np.zeros((basis.shape[1],), dtype=np.float64)
            coeff[mode_index] = 1.0
            s_norm = float(coeff @ block.s_pp @ coeff)
            safe_s = max(abs(s_norm), 1e-14)
            mode_weight = mode * (overlap @ mode)
            r_rms = float(np.sqrt(max(np.sum((r ** 2) * mode_weight) / safe_s, 0.0)))
            near_core_weight = float(np.sum(np.where(r <= rloc, mode_weight, 0.0)) / safe_s)
            v_loc_expectation = float(coeff @ block.h_v_loc_pp @ coeff / safe_s)
            patch_index = int(block.patch_slice.start + mode_index)
            occupied_contrib = {
                name: 0.0
                for name in ("t", "v_loc", "v_nl")
            }
            if eigvecs_np is not None and occ_np is not None:
                for name, applied in applied_components.items():
                    occupied_contrib[name] = float(
                        np.sum(occ_np[: eigvecs_np.shape[1]] * eigvecs_np[patch_index, :] * applied[patch_index, :])
                    )
            diagnostics.append({
                "atom_index": atom_index,
                "symbol": str(pseudo.get("symbol", "")),
                "mode_index": int(mode_index),
                "patch_index": patch_index,
                "rloc": float(rloc),
                "projector_radius": float(projector_r),
                "r_rms": float(r_rms),
                "radial_channel": _classify_v5_radial_channel(r_rms, rloc, projector_r),
                "angular_channel": _classify_v5_angular_channel(mode, offsets, overlap, projector_r),
                "s_norm": float(s_norm),
                "t_expectation": float(coeff @ block.h_t_pp @ coeff / safe_s),
                "v_loc_expectation": v_loc_expectation,
                "v_nl_expectation": float(coeff @ zero_vnl @ coeff / safe_s),
                "vloc_sensitivity": float(abs(v_loc_expectation)),
                "occupied_t_contribution": occupied_contrib["t"],
                "occupied_vloc_contribution": occupied_contrib["v_loc"],
                "occupied_vnl_contribution": occupied_contrib["v_nl"],
                "near_core_weight": float(max(near_core_weight, 0.0)),
            })
    return diagnostics


def _build_coarse_nonlocal_apply_matrix(grid, coords, pseudos):
    return _build_coarse_nonlocal_matrix_cell_average(
        grid,
        coords,
        pseudos,
        projector_subgrid=1,
    )[0]


def _seeded_random_subspace_v5(size, n_bands):
    rng = np.random.default_rng(_V5_LOBPCG_INIT_SEED)
    init = rng.standard_normal((size, n_bands)).astype(np.float64)
    init, _ = np.linalg.qr(init, mode="reduced")
    return init


def _complete_initial_subspace_v5(previous_vecs, size, n_bands):
    random_block = _seeded_random_subspace_v5(size, n_bands)
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


def _build_v5_diagonal_preconditioner(operator):
    diag = np.zeros((operator.total_size,), dtype=np.float64)
    diag[: operator.coarse_size] = (
        operator.coarse_kinetic_diagonal
        + float(operator.grid.volume_element) * operator.v_eff_flat
        + operator.coarse_nonlocal_diagonal
    )
    for block in operator.patch_blocks:
        diag[block.patch_slice] = np.diag(block.h_pp)
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


def build_fixed_veff_enriched_operator_v5(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
):
    """Build a matrix-free fixed-Veff V5 operator."""
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
        v_patch = coarse_to_patch(v_eff, patch_map)
        patch_bases[patch_spec.atom_index] = build_atom_centered_enriched_basis_v5(
            patch_spec,
            patch_map,
            pseudos[patch_spec.atom_index],
            max_modes=max_modes_per_atom,
            vloc_aware_constraint=vloc_aware_constraint,
            v_eff_patch=v_patch,
        )
    patch_slices = {}
    offset = coarse_size
    for patch_map in patch_maps:
        size = int(patch_bases[patch_map.atom_index].shape[1])
        patch_slices[patch_map.atom_index] = slice(offset, offset + size)
        offset += size
    total_size = offset
    _, coarse_projector_channels = _build_coarse_nonlocal_matrix_cell_average(
        grid,
        coords,
        pseudos,
        projector_subgrid=1,
    )
    patch_blocks = []
    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        atom_index = patch_map.atom_index
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
        h_t_pp = basis.T @ kinetic @ basis
        h_t_cp = eval_matrix.T @ kinetic @ basis
        h_v_loc_pp = basis.T @ potential @ basis
        h_v_loc_cp = eval_matrix.T @ potential @ basis
        h_pp = h_t_pp + h_v_loc_pp
        h_cp = h_t_cp + h_v_loc_cp
        patch_blocks.append(
            EnrichedPatchBlockV5(
                atom_index=atom_index,
                sample_indices=np.asarray(patch_map.sample_indices, dtype=np.int32),
                patch_slice=patch_slice,
                s_pp=np.asarray(0.5 * (s_pp + s_pp.T), dtype=np.float64),
                s_cp=np.asarray(s_cp, dtype=np.float64),
                h_pp=np.asarray(0.5 * (h_pp + h_pp.T), dtype=np.float64),
                h_cp=np.asarray(h_cp, dtype=np.float64),
                h_t_pp=np.asarray(0.5 * (h_t_pp + h_t_pp.T), dtype=np.float64),
                h_t_cp=np.asarray(h_t_cp, dtype=np.float64),
                h_v_loc_pp=np.asarray(0.5 * (h_v_loc_pp + h_v_loc_pp.T), dtype=np.float64),
                h_v_loc_cp=np.asarray(h_v_loc_cp, dtype=np.float64),
            )
        )
    metadata = EnrichedBasisMetadataV5(
        patch_specs=patch_specs,
        patch_maps=patch_maps,
        patch_bases=patch_bases,
        patch_slices=patch_slices,
    )
    return EnrichedOperatorV5(
        grid=grid,
        coarse_size=coarse_size,
        total_size=total_size,
        v_eff_flat=np.asarray(v_eff, dtype=np.float64).reshape(-1),
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


def build_fixed_veff_enriched_components_v5(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
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
    patch_bases = {}
    for patch_spec, patch_map in zip(patch_specs, patch_maps):
        v_patch = coarse_to_patch(v_eff, patch_map)
        patch_bases[patch_spec.atom_index] = build_atom_centered_enriched_basis_v5(
            patch_spec,
            patch_map,
            pseudos[patch_spec.atom_index],
            max_modes=max_modes_per_atom,
            vloc_aware_constraint=vloc_aware_constraint,
            v_eff_patch=v_patch,
        )
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
    vloc_aware_constraint=False,
):
    h_dense, s_dense, coarse_size, metadata, components = build_fixed_veff_enriched_components_v5(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
        vloc_aware_constraint=vloc_aware_constraint,
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


def solve_fixed_veff_enriched_iterative_host_v5(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
    maxiter=160,
    tol=1e-6,
    x_init=None,
):
    operator = build_fixed_veff_enriched_operator_v5(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
        vloc_aware_constraint=vloc_aware_constraint,
    )
    a_op = sparse_linalg.LinearOperator(
        (operator.total_size, operator.total_size),
        matvec=lambda x: operator.apply_h(x),
        matmat=lambda x: operator.apply_h(x),
        dtype=np.float64,
    )
    b_op = sparse_linalg.LinearOperator(
        (operator.total_size, operator.total_size),
        matvec=lambda x: operator.apply_s(x),
        matmat=lambda x: operator.apply_s(x),
        dtype=np.float64,
    )
    m_op = _build_v5_diagonal_preconditioner(operator)
    init = _complete_initial_subspace_v5(x_init, operator.total_size, n_bands)
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
    vloc_aware_constraint=False,
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
        vloc_aware_constraint=vloc_aware_constraint,
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
        "mode_diagnostics": _build_v5_mode_diagnostics(
            metadata,
            pseudos,
            components,
            s_dense,
            coarse_size,
            eigvecs=eigvecs,
            occ=occ[:n_bands],
        ),
        "coarse_size": coarse_size,
        "metadata": metadata,
    }


def _operator_component_expectations_v5(operator, eigvecs):
    eigvecs_np = np.asarray(eigvecs, dtype=np.float64)
    s_applied = np.asarray(operator.apply_s(eigvecs_np), dtype=np.float64)
    norms = np.einsum("ik,ik->k", eigvecs_np, s_applied)
    expectations = {}
    for name, apply_fn in (
        ("t", operator.apply_t),
        ("v_loc", operator.apply_vloc),
        ("v_nl", operator.apply_vnl),
    ):
        applied = np.asarray(apply_fn(eigvecs_np), dtype=np.float64)
        expectations[name] = jnp.asarray(
            np.einsum("ik,ik->k", eigvecs_np, applied) / norms,
            dtype=jnp.float32,
        )
    return expectations, norms


def audit_fixed_veff_enriched_iterative_v5(
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
    vloc_aware_constraint=False,
    maxiter=160,
    tol=1e-6,
    x_init=None,
):
    baseline_eigvals = jnp.asarray(baseline_eigvals, dtype=jnp.float32)
    baseline_eigvecs = jnp.asarray(baseline_eigvecs, dtype=jnp.float32)
    if n_bands is None:
        n_bands = int(baseline_eigvals.shape[0])
    if occ is None:
        occ = jnp.ones((n_bands,), dtype=jnp.float32)
    else:
        occ = jnp.asarray(occ, dtype=jnp.float32)
    corrected_eigvals, corrected_eigvecs, coarse_size, metadata = solve_fixed_veff_enriched_iterative_host_v5(
        grid,
        coords,
        pseudos,
        v_eff,
        n_bands=n_bands,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
        vloc_aware_constraint=vloc_aware_constraint,
        maxiter=maxiter,
        tol=tol,
        x_init=x_init,
    )
    density = compute_v5_density_on_coarse_grid(
        corrected_eigvecs,
        occ[:n_bands],
        grid,
        coarse_size,
        metadata,
    )
    patch_density_delta = _compute_v5_patch_density_delta(
        corrected_eigvecs,
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
    operator = build_fixed_veff_enriched_operator_v5(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
        vloc_aware_constraint=vloc_aware_constraint,
    )
    band_decomposition, s_norm_np = _operator_component_expectations_v5(
        operator,
        corrected_eigvecs,
    )
    embedded_baseline = jnp.vstack([
        baseline_eigvecs[:, :n_bands],
        jnp.zeros((operator.total_size - coarse_size, n_bands), dtype=jnp.float32),
    ])
    embedded_baseline_decomposition, _ = _operator_component_expectations_v5(
        operator,
        embedded_baseline,
    )
    s_norm = jnp.asarray(s_norm_np, dtype=jnp.float32)
    coarse_norm = grid.volume_element * jnp.sum(corrected_eigvecs[:coarse_size, :] ** 2, axis=0)
    patch_norm = jnp.maximum(s_norm - coarse_norm, 0.0)
    nuclear_weights = _compute_nuclear_density_weights_v5(grid, coords, density, radius=nuclear_radius)
    baseline_nuclear_weights = _compute_nuclear_density_weights_v5(grid, coords, baseline_rho, radius=nuclear_radius)
    band_delta = corrected_eigvals - baseline_eigvals[:n_bands]
    occupied_decomposition = {
        name: jnp.sum(values * occ[:n_bands])
        for name, values in band_decomposition.items()
    }
    embedded_occupied_decomposition = {
        name: jnp.sum(values * occ[:n_bands])
        for name, values in embedded_baseline_decomposition.items()
    }
    return {
        "corrected_eigvals": corrected_eigvals,
        "corrected_eigvecs": corrected_eigvecs,
        "baseline_eigvals": baseline_eigvals[:n_bands],
        "band_delta": band_delta,
        "band_sum_delta": jnp.sum(band_delta * occ[:n_bands]),
        "band_decomposition": band_decomposition,
        "occupied_decomposition": occupied_decomposition,
        "embedded_baseline_decomposition": embedded_baseline_decomposition,
        "embedded_baseline_occupied_decomposition": embedded_occupied_decomposition,
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
        "coarse_metric_fraction_diag": coarse_norm / s_norm,
        "patch_metric_fraction_diag": patch_norm / s_norm,
        "mode_diagnostics": _build_v5_operator_mode_diagnostics(
            operator,
            pseudos,
            eigvecs=corrected_eigvecs,
            occ=occ[:n_bands],
        ),
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
    vloc_aware_constraint=False,
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
        vloc_aware_constraint=vloc_aware_constraint,
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
        vloc_aware_constraint=vloc_aware_constraint,
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


def _solve_v5_coarse_block_dense(grid, coords, pseudos, v_eff, n_bands, **basis_kwargs):
    h_dense, s_dense, coarse_size, _, _ = build_fixed_veff_enriched_components_v5(
        grid,
        coords,
        pseudos,
        v_eff,
        **basis_kwargs,
    )
    eigvals, eigvecs = scipy_linalg.eigh(
        np.asarray(h_dense[:coarse_size, :coarse_size], dtype=np.float64),
        np.asarray(s_dense[:coarse_size, :coarse_size], dtype=np.float64),
        subset_by_index=[0, n_bands - 1],
    )
    return (
        jnp.asarray(eigvals, dtype=jnp.float32),
        jnp.asarray(eigvecs, dtype=jnp.float32),
        coarse_size,
    )


def dense_baseline_v5_one_shot_hxc_diagnostic(
    grid,
    coords,
    pseudos,
    occ,
    v_loc_baseline,
    n_steps=60,
    mix_alpha=0.12,
    initial_rho=None,
    nuclear_radius=0.75,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
):
    """Build a dense, internally consistent coarse baseline for V5 Hxc audits."""
    coords = jnp.asarray(coords, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    v_loc_baseline = jnp.asarray(v_loc_baseline, dtype=jnp.float32)
    n_bands = int(occ.shape[0])
    basis_kwargs = {
        "patch_subgrid": patch_subgrid,
        "patch_radius_factor": patch_radius_factor,
        "max_modes_per_atom": max_modes_per_atom,
        "vloc_aware_constraint": vloc_aware_constraint,
    }
    if initial_rho is None:
        rho = jnp.zeros(grid.shape, dtype=jnp.float32)
        for atom_index in range(int(coords.shape[0])):
            r = jnp.linalg.norm(grid.coords - coords[atom_index], axis=-1)
            rho = rho + jnp.exp(-2.0 * r * r)
        rho = rho / (jnp.sum(rho) * grid.volume_element) * jnp.sum(occ)
    else:
        rho = jnp.asarray(initial_rho, dtype=jnp.float32)

    trace = []
    eigvals = jnp.zeros((n_bands,), dtype=jnp.float32)
    eigvecs = jnp.zeros((int(jnp.prod(jnp.asarray(grid.shape))), n_bands), dtype=jnp.float32)
    v_h = jnp.zeros(grid.shape, dtype=jnp.float32)
    v_xc = jnp.zeros(grid.shape, dtype=jnp.float32)
    for step in range(int(n_steps)):
        v_h, v_xc = _update_hxc_from_density_v5(grid, rho)
        eigvals, eigvecs, _ = _solve_v5_coarse_block_dense(
            grid,
            coords,
            pseudos,
            v_loc_baseline + v_h + v_xc,
            n_bands,
            **basis_kwargs,
        )
        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        diff_linf = jnp.max(jnp.abs(rho_new - rho))
        diff_l1 = jnp.sum(jnp.abs(rho_new - rho)) * grid.volume_element
        trace.append({
            "step": step,
            "rho_linf": diff_linf,
            "rho_l1": diff_l1,
            "band_sum": jnp.sum(eigvals * occ),
            "electron_count": jnp.sum(rho_new) * grid.volume_element,
        })
        rho = (1.0 - float(mix_alpha)) * rho + float(mix_alpha) * rho_new

    # Resync once so the returned rho, Hxc, and eigensystem share one fixed operator.
    v_h, v_xc = _update_hxc_from_density_v5(grid, rho)
    eigvals, eigvecs, _ = _solve_v5_coarse_block_dense(
        grid,
        coords,
        pseudos,
        v_loc_baseline + v_h + v_xc,
        n_bands,
        **basis_kwargs,
    )
    rho = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
    v_h, v_xc = _update_hxc_from_density_v5(grid, rho)
    eigvals, eigvecs, _ = _solve_v5_coarse_block_dense(
        grid,
        coords,
        pseudos,
        v_loc_baseline + v_h + v_xc,
        n_bands,
        **basis_kwargs,
    )
    one_shot = post_scf_v5_one_shot_hxc_feedback(
        grid,
        coords,
        pseudos,
        rho,
        v_h,
        v_xc,
        eigvals,
        eigvecs,
        occ,
        v_loc_baseline,
        nuclear_radius=nuclear_radius,
        **basis_kwargs,
    )
    return {
        "baseline": {
            "rho": rho,
            "v_h": v_h,
            "v_xc": v_xc,
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "electron_count": jnp.sum(rho) * grid.volume_element,
        },
        "one_shot": one_shot,
        "trace": trace,
    }


def post_scf_v5_damped_feedback_trace(
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
    n_steps=3,
    alpha=0.05,
    nuclear_radius=0.75,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
):
    """Trace damped V5 density feedback without entering a full SCF loop."""
    baseline_rho = jnp.asarray(baseline_rho, dtype=jnp.float32)
    v_h_initial = jnp.asarray(v_h, dtype=jnp.float32)
    v_xc_initial = jnp.asarray(v_xc, dtype=jnp.float32)
    baseline_eigvals = jnp.asarray(baseline_eigvals, dtype=jnp.float32)
    baseline_eigvecs = jnp.asarray(baseline_eigvecs, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    v_loc_baseline = jnp.asarray(v_loc_baseline, dtype=jnp.float32)
    rho_current = baseline_rho
    trace = []
    basis_kwargs = {
        "patch_subgrid": patch_subgrid,
        "patch_radius_factor": patch_radius_factor,
        "max_modes_per_atom": max_modes_per_atom,
        "vloc_aware_constraint": vloc_aware_constraint,
    }

    for step in range(int(n_steps)):
        v_h_current, v_xc_current = _update_hxc_from_density_v5(grid, rho_current)
        audit = audit_fixed_veff_enriched_v5(
            grid,
            coords,
            pseudos,
            v_loc_baseline + v_h_current + v_xc_current,
            baseline_eigvals,
            baseline_eigvecs,
            n_bands=int(baseline_eigvals.shape[0]),
            occ=occ,
            baseline_rho=rho_current,
            nuclear_radius=nuclear_radius,
            **basis_kwargs,
        )
        raw_rho = audit["density"]
        raw_delta = raw_rho - rho_current
        damped_rho = rho_current + float(alpha) * raw_delta
        hxc_delta = (v_h_current + v_xc_current) - (v_h_initial + v_xc_initial)
        max_patch_fraction = jnp.max(audit["patch_metric_fraction_diag"])
        trace.append({
            "step": step,
            "alpha": jnp.asarray(alpha, dtype=jnp.float32),
            "corrected_eigvals": audit["corrected_eigvals"],
            "band_sum_delta": audit["band_sum_delta"],
            "band_decomposition": audit["band_decomposition"],
            "raw_rho_l1": jnp.sum(jnp.abs(raw_delta)) * grid.volume_element,
            "raw_rho_linf": jnp.max(jnp.abs(raw_delta)),
            "damped_rho_l1": jnp.sum(jnp.abs(damped_rho - rho_current)) * grid.volume_element,
            "damped_rho_linf": jnp.max(jnp.abs(damped_rho - rho_current)),
            "hxc_delta_linf": jnp.max(jnp.abs(hxc_delta)),
            "hxc_delta_l1": jnp.sum(jnp.abs(hxc_delta)) * grid.volume_element,
            "electron_count_raw": audit["electron_count_corrected"],
            "electron_count_damped": jnp.sum(damped_rho) * grid.volume_element,
            "max_patch_fraction": max_patch_fraction,
            "patch_density_delta_l1": audit["patch_density_delta_l1"],
            "nuclear_weight_delta": audit["nuclear_weight_delta"],
        })
        rho_current = jnp.clip(damped_rho, 1e-12, None)

    return trace


def solve_baseline_initialized_damped_v5_scf(
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
    n_steps=10,
    alpha=0.05,
    rho_l1_tolerance=1e-4,
    solver_mode="dense",
    eig_maxiter=160,
    eig_tol=1e-6,
    nuclear_radius=0.75,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
):
    """Experimental baseline-initialized damped V5 SCF bridge.

    This is a diagnostic bridge only: it deliberately uses fixed damping and
    does not report a mixed-basis total energy.
    """
    baseline_rho = jnp.asarray(baseline_rho, dtype=jnp.float32)
    v_h_initial = jnp.asarray(v_h, dtype=jnp.float32)
    v_xc_initial = jnp.asarray(v_xc, dtype=jnp.float32)
    baseline_eigvals = jnp.asarray(baseline_eigvals, dtype=jnp.float32)
    baseline_eigvecs = jnp.asarray(baseline_eigvecs, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    v_loc_baseline = jnp.asarray(v_loc_baseline, dtype=jnp.float32)
    rho_current = baseline_rho
    trace = []
    last_audit = None
    previous_vecs = None
    converged = False
    final_raw_rho_l1 = jnp.asarray(jnp.inf, dtype=jnp.float32)
    final_raw_rho_linf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    basis_kwargs = {
        "patch_subgrid": patch_subgrid,
        "patch_radius_factor": patch_radius_factor,
        "max_modes_per_atom": max_modes_per_atom,
        "vloc_aware_constraint": vloc_aware_constraint,
    }
    if solver_mode not in ("dense", "iterative"):
        raise ValueError("solver_mode must be 'dense' or 'iterative'")

    for step in range(int(n_steps)):
        v_h_current, v_xc_current = _update_hxc_from_density_v5(grid, rho_current)
        if solver_mode == "dense":
            last_audit = audit_fixed_veff_enriched_v5(
                grid,
                coords,
                pseudos,
                v_loc_baseline + v_h_current + v_xc_current,
                baseline_eigvals,
                baseline_eigvecs,
                n_bands=int(baseline_eigvals.shape[0]),
                occ=occ,
                baseline_rho=rho_current,
                nuclear_radius=nuclear_radius,
                **basis_kwargs,
            )
        else:
            last_audit = audit_fixed_veff_enriched_iterative_v5(
                grid,
                coords,
                pseudos,
                v_loc_baseline + v_h_current + v_xc_current,
                baseline_eigvals,
                baseline_eigvecs,
                n_bands=int(baseline_eigvals.shape[0]),
                occ=occ,
                baseline_rho=rho_current,
                nuclear_radius=nuclear_radius,
                maxiter=eig_maxiter,
                tol=eig_tol,
                x_init=previous_vecs,
                **basis_kwargs,
            )
            previous_vecs = last_audit["corrected_eigvecs"]
        raw_rho = last_audit["density"]
        raw_delta = raw_rho - rho_current
        raw_rho_l1 = jnp.sum(jnp.abs(raw_delta)) * grid.volume_element
        raw_rho_linf = jnp.max(jnp.abs(raw_delta))
        damped_rho = rho_current + float(alpha) * raw_delta
        hxc_delta = (v_h_current + v_xc_current) - (v_h_initial + v_xc_initial)
        trace.append({
            "step": step,
            "alpha": jnp.asarray(alpha, dtype=jnp.float32),
            "eigvals": last_audit["corrected_eigvals"],
            "band_sum_delta": last_audit["band_sum_delta"],
            "band_decomposition": last_audit["band_decomposition"],
            "raw_rho_l1": raw_rho_l1,
            "raw_rho_linf": raw_rho_linf,
            "damped_rho_l1": jnp.sum(jnp.abs(damped_rho - rho_current)) * grid.volume_element,
            "damped_rho_linf": jnp.max(jnp.abs(damped_rho - rho_current)),
            "hxc_delta_linf": jnp.max(jnp.abs(hxc_delta)),
            "hxc_delta_l1": jnp.sum(jnp.abs(hxc_delta)) * grid.volume_element,
            "electron_count_raw": last_audit["electron_count_corrected"],
            "electron_count_damped": jnp.sum(damped_rho) * grid.volume_element,
            "max_patch_fraction": jnp.max(last_audit["patch_metric_fraction_diag"]),
            "patch_density_delta_l1": last_audit["patch_density_delta_l1"],
            "nuclear_weight_delta": last_audit["nuclear_weight_delta"],
        })
        final_raw_rho_l1 = raw_rho_l1
        final_raw_rho_linf = raw_rho_linf
        if float(raw_rho_l1) <= float(rho_l1_tolerance):
            converged = True
            rho_current = jnp.clip(raw_rho, 1e-12, None)
            break
        rho_current = jnp.clip(damped_rho, 1e-12, None)

    v_h_final, v_xc_final = _update_hxc_from_density_v5(grid, rho_current)
    if solver_mode == "dense":
        final_audit = audit_fixed_veff_enriched_v5(
            grid,
            coords,
            pseudos,
            v_loc_baseline + v_h_final + v_xc_final,
            baseline_eigvals,
            baseline_eigvecs,
            n_bands=int(baseline_eigvals.shape[0]),
            occ=occ,
            baseline_rho=rho_current,
            nuclear_radius=nuclear_radius,
            **basis_kwargs,
        )
    else:
        final_audit = audit_fixed_veff_enriched_iterative_v5(
            grid,
            coords,
            pseudos,
            v_loc_baseline + v_h_final + v_xc_final,
            baseline_eigvals,
            baseline_eigvecs,
            n_bands=int(baseline_eigvals.shape[0]),
            occ=occ,
            baseline_rho=rho_current,
            nuclear_radius=nuclear_radius,
            maxiter=eig_maxiter,
            tol=eig_tol,
            x_init=previous_vecs,
            **basis_kwargs,
        )
    return {
        "rho": rho_current,
        "v_h": v_h_final,
        "v_xc": v_xc_final,
        "eigvals": final_audit["corrected_eigvals"],
        "eigvecs": final_audit["corrected_eigvecs"],
        "electron_count": jnp.sum(rho_current) * grid.volume_element,
        "converged": converged,
        "n_iter": len(trace),
        "final_raw_rho_l1": final_raw_rho_l1,
        "final_raw_rho_linf": final_raw_rho_linf,
        "trace": trace,
        "final_audit": final_audit,
    }


def mixed_basis_v5_total_energy_audit(
    grid,
    coords,
    pseudos,
    rho,
    v_h,
    v_xc,
    eigvals,
    occ,
    v_loc_baseline,
    baseline_eigvals,
    baseline_eigvecs,
    nuclear_radius=0.75,
    patch_subgrid=2,
    patch_radius_factor=2.0,
    max_modes_per_atom=8,
    vloc_aware_constraint=False,
):
    """Audit V5 total-energy components for one fixed mixed-basis state.

    This diagnostic uses the standard Kohn-Sham double-counting correction with
    the supplied density and eigenvalues, then reports a same-Hxc re-solve delta.
    It is not promoted as a production total-energy path until the mixed density
    and feedback loop are fully validated on real systems.
    """
    coords = jnp.asarray(coords, dtype=jnp.float32)
    rho = jnp.asarray(rho, dtype=jnp.float32)
    v_h = jnp.asarray(v_h, dtype=jnp.float32)
    v_xc = jnp.asarray(v_xc, dtype=jnp.float32)
    eigvals = jnp.asarray(eigvals, dtype=jnp.float32)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    v_loc_baseline = jnp.asarray(v_loc_baseline, dtype=jnp.float32)
    baseline_eigvals = jnp.asarray(baseline_eigvals, dtype=jnp.float32)
    baseline_eigvecs = jnp.asarray(baseline_eigvecs, dtype=jnp.float32)
    eps_xc, v_xc_from_rho = lda_xc(jnp.clip(rho, 1e-12, None))
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)

    e_band = jnp.sum(eigvals * occ)
    e_hartree = 0.5 * grid.volume_element * jnp.sum(rho * v_h)
    e_xc = grid.volume_element * jnp.sum(eps_xc)
    e_rho_vxc = grid.volume_element * jnp.sum(rho * v_xc)
    e_ion = ion_ion_energy(coords, zion)
    e_total = e_band - e_hartree + e_xc - e_rho_vxc + e_ion

    resolved = audit_fixed_veff_enriched_v5(
        grid,
        coords,
        pseudos,
        v_loc_baseline + v_h + v_xc,
        baseline_eigvals,
        baseline_eigvecs,
        n_bands=int(eigvals.shape[0]),
        occ=occ,
        baseline_rho=rho,
        nuclear_radius=nuclear_radius,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
        max_modes_per_atom=max_modes_per_atom,
        vloc_aware_constraint=vloc_aware_constraint,
    )
    resolved_e_band = jnp.sum(resolved["corrected_eigvals"] * occ)
    resolved_e_total = resolved_e_band - e_hartree + e_xc - e_rho_vxc + e_ion
    hxc_residual = (v_xc_from_rho - v_xc)
    return {
        "e_total": e_total,
        "e_band": e_band,
        "e_hartree": e_hartree,
        "e_xc": e_xc,
        "e_rho_vxc": e_rho_vxc,
        "e_ion": e_ion,
        "electron_count": jnp.sum(rho) * grid.volume_element,
        "same_hxc_resolve_e_total": resolved_e_total,
        "same_hxc_resolve_band": resolved_e_band,
        "same_hxc_resolve_energy_delta": resolved_e_total - e_total,
        "resolved_density_l1": resolved["density_change_l1"],
        "resolved_density_linf": resolved["density_change_linf"],
        "resolved_patch_fraction_max": jnp.max(resolved["patch_metric_fraction_diag"]),
        "rho_vxc_consistency_linf": jnp.max(jnp.abs(hxc_residual)),
        "resolved_audit": resolved,
    }
