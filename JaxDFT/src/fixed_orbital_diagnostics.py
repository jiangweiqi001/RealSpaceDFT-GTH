"""Fixed-orbital operator diagnostics for local and nonlocal GTH terms."""

import jax.numpy as jnp
import numpy as np
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as sparse_linalg

from .hamiltonian import (
    _make_atom_patch_points,
    apply_nonlocal_fine_integral,
    apply_nonlocal_precomputed,
    build_fine_interpolation_data,
    gather_fine_values,
    get_gth_projector,
    gth_local_potential_value,
    laplacian_8th,
    precompute_projectors,
    scatter_fine_values_adjoint,
)
from .mixed_basis_galerkin_v3 import _build_coarse_kinetic_matrix, _build_coarse_local_matrix


def _coerce_orbital_columns(eigvecs):
    eigvecs = jnp.asarray(eigvecs, dtype=jnp.float32)
    if eigvecs.ndim == 1:
        eigvecs = eigvecs[:, None]
    return eigvecs


def _fine_local_points(grid, coords, pseudos, local_fine_subgrid, local_patch_radius_factor):
    fine_spacing = jnp.asarray(grid.spacing, dtype=jnp.float32) / int(local_fine_subgrid)
    positions = []
    for atom_index, pseudo in enumerate(pseudos):
        radius = float(local_patch_radius_factor) * float(pseudo["rloc"]) + 0.8660254 * float(grid.spacing)
        _, atom_positions, _ = _make_atom_patch_points(coords[atom_index], radius, fine_spacing)
        positions.append(atom_positions)
    if not positions:
        return jnp.zeros((0, 3), dtype=jnp.float32), fine_spacing ** 3
    return jnp.concatenate(positions, axis=0), fine_spacing ** 3


def _cell_fine_points(grid, fine_subgrid):
    fine_spacing = jnp.asarray(grid.spacing, dtype=jnp.float32) / int(fine_subgrid)
    axis = (jnp.arange(int(fine_subgrid), dtype=jnp.float32) + 0.5) / int(fine_subgrid) - 0.5
    axis = axis * jnp.asarray(grid.spacing, dtype=jnp.float32)
    ox, oy, oz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    offsets = jnp.stack([ox, oy, oz], axis=-1).reshape(-1, 3)
    n_grid = grid.coords.reshape(-1, 3).shape[0]
    n_offsets = offsets.shape[0]
    positions = (grid.coords.reshape(-1, 3)[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    cell_ids = jnp.repeat(jnp.arange(n_grid, dtype=jnp.int32), n_offsets)
    return positions, fine_spacing ** 3, cell_ids


def _local_potential_values_at_positions(positions, coords, pseudos):
    values = jnp.zeros((positions.shape[0],), dtype=jnp.float32)
    for atom_index, pseudo in enumerate(pseudos):
        r = jnp.linalg.norm(positions - coords[atom_index], axis=-1)
        values = values + gth_local_potential_value(
            r,
            float(pseudo["zion"]),
            float(pseudo["rloc"]),
            jnp.asarray(pseudo["c"], dtype=jnp.float32),
        )
    return values


def _projector_channels_at_positions(positions, coords, pseudos):
    p_i_list = []
    p_j_list = []
    coeff_list = []
    for atom_index, pseudo in enumerate(pseudos):
        if not pseudo["projectors"]:
            continue
        diff = positions - coords[atom_index]
        r = jnp.linalg.norm(diff, axis=-1)
        for channel in pseudo["projectors"]:
            l = channel["l"]
            rp = channel["r"]
            h_mat = jnp.asarray(channel["h"], dtype=jnp.float32)
            if h_mat.ndim == 1:
                h_mat = jnp.diag(h_mat)
            n_proj = h_mat.shape[0]
            for i in range(1, n_proj + 1):
                for j in range(1, n_proj + 1):
                    h_ij = h_mat[i - 1, j - 1]
                    if float(jnp.abs(h_ij)) < 1e-10:
                        continue
                    if l == 0:
                        p_i_list.append(get_gth_projector(r, l, i, rp))
                        p_j_list.append(get_gth_projector(r, l, j, rp))
                        coeff_list.append(h_ij / (4.0 * jnp.pi))
                    elif l == 1:
                        p_i_rad = get_gth_projector(r, l, i, rp)
                        p_j_rad = get_gth_projector(r, l, j, rp)
                        for axis in range(3):
                            angular = diff[:, axis] / (r + 1e-12)
                            p_i_list.append(p_i_rad * angular)
                            p_j_list.append(p_j_rad * angular)
                            coeff_list.append(3.0 * h_ij / (4.0 * jnp.pi))
    if not p_i_list:
        return None
    return (
        jnp.stack(p_i_list, axis=0),
        jnp.stack(p_j_list, axis=0),
        jnp.asarray(coeff_list, dtype=jnp.float32),
    )


def _coarse_vloc_expectations(grid, eigvecs, occ, v_loc):
    v_loc = jnp.asarray(v_loc, dtype=jnp.float32).reshape(-1)
    out = []
    for band in range(eigvecs.shape[1]):
        psi = eigvecs[:, band]
        out.append(float(occ[band]) * grid.volume_element * jnp.sum(v_loc * psi * psi))
    return jnp.asarray(out, dtype=jnp.float32)


def _fine_vloc_expectations(
    grid,
    coords,
    pseudos,
    eigvecs,
    occ,
    local_fine_subgrid,
    local_patch_radius_factor,
    fine_mode,
):
    if fine_mode == "cell":
        positions, fine_dv, _ = _cell_fine_points(grid, local_fine_subgrid)
    elif fine_mode == "patch":
        positions, fine_dv = _fine_local_points(
            grid,
            coords,
            pseudos,
            local_fine_subgrid,
            local_patch_radius_factor,
        )
    else:
        raise ValueError("fine_mode must be 'cell' or 'patch'")
    if positions.shape[0] == 0:
        return jnp.zeros((eigvecs.shape[1],), dtype=jnp.float32)
    flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
    v_fine = _local_potential_values_at_positions(positions, coords, pseudos)
    out = []
    for band in range(eigvecs.shape[1]):
        psi = eigvecs[:, band].reshape(grid.shape)
        psi_fine = gather_fine_values(psi, flat_indices, weights)
        out.append(float(occ[band]) * fine_dv * jnp.sum(v_fine * psi_fine * psi_fine))
    return jnp.asarray(out, dtype=jnp.float32)


def _cell_supported_vloc_expectations(grid, coords, pseudos, eigvecs, occ, local_fine_subgrid):
    positions, fine_dv, cell_ids = _cell_fine_points(grid, local_fine_subgrid)
    flat_indices, weights, valid = build_fine_interpolation_data(grid, positions)
    del flat_indices, weights
    v_fine = _local_potential_values_at_positions(positions, coords, pseudos)
    valid = valid.astype(jnp.float32)
    out = []
    for band in range(eigvecs.shape[1]):
        psi_center = eigvecs[:, band][cell_ids]
        out.append(float(occ[band]) * fine_dv * jnp.sum(valid * v_fine * psi_center * psi_center))
    return jnp.asarray(out, dtype=jnp.float32)


def _cell_fine_mass(grid, eigvecs, occ, fine_subgrid):
    positions, fine_dv, _ = _cell_fine_points(grid, fine_subgrid)
    flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
    out = []
    for band in range(eigvecs.shape[1]):
        psi = eigvecs[:, band].reshape(grid.shape)
        psi_fine = gather_fine_values(psi, flat_indices, weights)
        out.append(float(occ[band]) * fine_dv * jnp.sum(psi_fine * psi_fine))
    return jnp.asarray(out, dtype=jnp.float32)


def _nonlocal_expectation_from_applied(grid, eigvecs, occ, applied_by_band):
    out = []
    for band, applied in enumerate(applied_by_band):
        psi = eigvecs[:, band].reshape(grid.shape)
        out.append(float(occ[band]) * grid.volume_element * jnp.sum(psi * applied))
    return jnp.asarray(out, dtype=jnp.float32)


def _coarse_vnl_expectations(grid, coords, pseudos, eigvecs, occ, coarse_projector_subgrid):
    data = precompute_projectors(
        grid,
        coords,
        pseudos,
        projector_subgrid=coarse_projector_subgrid,
        projector_mode="cell_average",
    )
    if data is None:
        return jnp.zeros((eigvecs.shape[1],), dtype=jnp.float32)
    p_i, p_j, coeffs = data
    applied = []
    for band in range(eigvecs.shape[1]):
        psi = eigvecs[:, band].reshape(grid.shape)
        applied.append(apply_nonlocal_precomputed(psi, p_i, p_j, coeffs, grid.volume_element))
    return _nonlocal_expectation_from_applied(grid, eigvecs, occ, applied)


def _fine_vnl_expectations(
    grid,
    coords,
    pseudos,
    eigvecs,
    occ,
    fine_projector_subgrid,
    projector_patch_radius_factor,
    fine_mode,
):
    if fine_mode == "cell":
        positions, fine_dv, _ = _cell_fine_points(grid, fine_projector_subgrid)
        flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
        channels = _projector_channels_at_positions(positions, coords, pseudos)
        if channels is None:
            return jnp.zeros((eigvecs.shape[1],), dtype=jnp.float32)
        p_i, p_j, coeffs = channels
        out = []
        for band in range(eigvecs.shape[1]):
            psi = eigvecs[:, band].reshape(grid.shape)
            psi_fine = gather_fine_values(psi, flat_indices, weights)
            overlaps_i = jnp.sum(p_i * psi_fine[None, :], axis=1) * fine_dv
            overlaps_j = jnp.sum(p_j * psi_fine[None, :], axis=1) * fine_dv
            out.append(float(occ[band]) * jnp.sum(coeffs * overlaps_i * overlaps_j))
        return jnp.asarray(out, dtype=jnp.float32)
    if fine_mode != "patch":
        raise ValueError("fine_mode must be 'cell' or 'patch'")
    data = precompute_projectors(
        grid,
        coords,
        pseudos,
        projector_subgrid=fine_projector_subgrid,
        projector_mode="patch",
        projector_patch_radius_factor=projector_patch_radius_factor,
    )
    if data is None:
        return jnp.zeros((eigvecs.shape[1],), dtype=jnp.float32)
    applied = []
    for band in range(eigvecs.shape[1]):
        psi = eigvecs[:, band].reshape(grid.shape)
        applied.append(apply_nonlocal_fine_integral(psi, *data[1:]))
    return _nonlocal_expectation_from_applied(grid, eigvecs, occ, applied)


def fixed_orbital_operator_audit(
    grid,
    coords,
    pseudos,
    eigvecs,
    occ,
    v_loc,
    local_fine_subgrid=5,
    local_patch_radius_factor=4.0,
    coarse_projector_subgrid=1,
    fine_projector_subgrid=3,
    projector_patch_radius_factor=6.0,
    fine_mode="patch",
):
    """Compare current coarse operator matrix elements with fixed-orbital fine integrals."""
    coords = jnp.asarray(coords, dtype=jnp.float32)
    eigvecs = _coerce_orbital_columns(eigvecs)
    occ = jnp.asarray(occ, dtype=jnp.float32)
    vloc_coarse = _coarse_vloc_expectations(grid, eigvecs, occ, v_loc)
    vloc_fine = _fine_vloc_expectations(
        grid,
        coords,
        pseudos,
        eigvecs,
        occ,
        local_fine_subgrid,
        local_patch_radius_factor,
        fine_mode,
    )
    vnl_coarse = _coarse_vnl_expectations(
        grid,
        coords,
        pseudos,
        eigvecs,
        occ,
        coarse_projector_subgrid,
    )
    vnl_fine = _fine_vnl_expectations(
        grid,
        coords,
        pseudos,
        eigvecs,
        occ,
        fine_projector_subgrid,
        projector_patch_radius_factor,
        fine_mode,
    )
    vloc_delta = vloc_fine - vloc_coarse
    vnl_delta = vnl_fine - vnl_coarse
    if fine_mode == "cell":
        vloc_supported = _cell_supported_vloc_expectations(
            grid,
            coords,
            pseudos,
            eigvecs,
            occ,
            local_fine_subgrid,
        )
        fine_mass = _cell_fine_mass(grid, eigvecs, occ, local_fine_subgrid)
    else:
        vloc_supported = None
        fine_mass = None
    return {
        "v_loc": {
            "coarse_by_band": vloc_coarse,
            "coarse_on_fine_support_by_band": vloc_supported,
            "fine_by_band": vloc_fine,
            "delta_by_band": vloc_delta,
            "coarse_total": jnp.sum(vloc_coarse),
            "fine_total": jnp.sum(vloc_fine),
            "delta_total": jnp.sum(vloc_delta),
        },
        "v_nl": {
            "coarse_by_band": vnl_coarse,
            "fine_by_band": vnl_fine,
            "delta_by_band": vnl_delta,
            "coarse_total": jnp.sum(vnl_coarse),
            "fine_total": jnp.sum(vnl_fine),
            "delta_total": jnp.sum(vnl_delta),
        },
        "summary": {
            "combined_delta_by_band": vloc_delta + vnl_delta,
            "combined_delta_total": jnp.sum(vloc_delta + vnl_delta),
            "fine_mode": fine_mode,
            "fine_mass_by_band": fine_mass,
        },
    }


def _dense_cell_interpolation(grid, fine_subgrid):
    positions, fine_dv, _ = _cell_fine_points(grid, fine_subgrid)
    flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
    flat_indices_np = np.asarray(flat_indices, dtype=np.int64)
    weights_np = np.asarray(weights, dtype=np.float64)
    n_fine = int(flat_indices_np.shape[0])
    n_grid = int(np.prod(np.asarray(grid.shape)))
    interp = np.zeros((n_fine, n_grid), dtype=np.float64)
    rows = np.repeat(np.arange(n_fine), flat_indices_np.shape[1])
    np.add.at(interp, (rows, flat_indices_np.reshape(-1)), weights_np.reshape(-1))
    return positions, float(fine_dv), interp, flat_indices, weights


def _coarse_nonlocal_matrix(grid, coords, pseudos, coarse_projector_subgrid):
    n_grid = int(np.prod(np.asarray(grid.shape)))
    data = precompute_projectors(
        grid,
        coords,
        pseudos,
        projector_subgrid=coarse_projector_subgrid,
        projector_mode="cell_average",
    )
    if data is None:
        return np.zeros((n_grid, n_grid), dtype=np.float64)
    p_i, p_j, coeffs = data
    p_i = np.asarray(p_i, dtype=np.float64).reshape((p_i.shape[0], n_grid))
    p_j = np.asarray(p_j, dtype=np.float64).reshape((p_j.shape[0], n_grid))
    coeffs = np.asarray(coeffs, dtype=np.float64)
    mat = (float(grid.volume_element) ** 2) * (p_i.T @ (coeffs[:, None] * p_j))
    return 0.5 * (mat + mat.T)


def _fine_cell_local_matrix_from_grid_potential(grid, v_eff, fine_subgrid):
    positions, fine_dv, interp, flat_indices, weights = _dense_cell_interpolation(grid, fine_subgrid)
    del positions
    v_eff = jnp.asarray(v_eff, dtype=jnp.float32).reshape(grid.shape)
    v_fine = np.asarray(gather_fine_values(v_eff, flat_indices, weights), dtype=np.float64)
    mat = interp.T @ ((fine_dv * v_fine)[:, None] * interp)
    return 0.5 * (mat + mat.T)


def _fine_cell_nonlocal_matrix(grid, coords, pseudos, fine_subgrid):
    positions, fine_dv, interp, _, _ = _dense_cell_interpolation(grid, fine_subgrid)
    channels = _projector_channels_at_positions(positions, coords, pseudos)
    n_grid = int(np.prod(np.asarray(grid.shape)))
    if channels is None:
        return np.zeros((n_grid, n_grid), dtype=np.float64)
    p_i, p_j, coeffs = channels
    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    overlaps_i = (fine_dv * p_i) @ interp
    overlaps_j = (fine_dv * p_j) @ interp
    mat = overlaps_i.T @ (coeffs[:, None] * overlaps_j)
    return 0.5 * (mat + mat.T)


def _solve_dense_variant(h_matrix, s_matrix, components, n_bands):
    eigvals, eigvecs = scipy_linalg.eigh(h_matrix, s_matrix)
    eigvals = eigvals[:n_bands]
    eigvecs = eigvecs[:, :n_bands]
    decomposition = {}
    for name, matrix in components.items():
        vals = []
        for band in range(n_bands):
            vec = eigvecs[:, band]
            norm = float(vec.T @ s_matrix @ vec)
            vals.append(float(vec.T @ matrix @ vec) / norm)
        decomposition[name] = np.asarray(vals, dtype=np.float64)
    decomposition["total"] = sum(decomposition.values())
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "band_decomposition": decomposition,
    }


def fixed_veff_dense_operator_ab(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    fine_subgrid=2,
    coarse_projector_subgrid=1,
):
    """Dense fixed-Veff A/B solve for coarse vs cell-fine local/projector operators.

    This is a small-system diagnostic. The supplied ``v_eff`` is treated as the
    fixed local effective potential and is interpolated to the same cell-fine
    support used by the fine projector matrix.
    """
    coords = jnp.asarray(coords, dtype=jnp.float32)
    n_grid = int(np.prod(np.asarray(grid.shape)))
    n_bands = int(n_bands)
    if n_bands < 1 or n_bands > n_grid:
        raise ValueError("n_bands must be between 1 and the coarse grid size")

    h_t = np.asarray(_build_coarse_kinetic_matrix(grid), dtype=np.float64)
    h_local_coarse = np.asarray(_build_coarse_local_matrix(grid, v_eff), dtype=np.float64)
    h_vnl_coarse = _coarse_nonlocal_matrix(grid, coords, pseudos, coarse_projector_subgrid)
    h_local_fine = _fine_cell_local_matrix_from_grid_potential(grid, v_eff, fine_subgrid)
    h_vnl_fine = _fine_cell_nonlocal_matrix(grid, coords, pseudos, fine_subgrid)
    s_matrix = float(grid.volume_element) * np.eye(n_grid, dtype=np.float64)

    variants = {
        "coarse": (h_local_coarse, h_vnl_coarse),
        "fine_local": (h_local_fine, h_vnl_coarse),
        "fine_projector": (h_local_coarse, h_vnl_fine),
        "fine_both": (h_local_fine, h_vnl_fine),
    }
    solved = {}
    for name, (h_local, h_vnl) in variants.items():
        components = {
            "t": h_t,
            "local": h_local,
            "v_nl": h_vnl,
        }
        solved[name] = _solve_dense_variant(h_t + h_local + h_vnl, s_matrix, components, n_bands)

    return {
        "variants": solved,
        "matrix_deltas": {
            "fine_local_minus_coarse_local_norm": float(np.linalg.norm(h_local_fine - h_local_coarse)),
            "fine_projector_minus_coarse_projector_norm": float(np.linalg.norm(h_vnl_fine - h_vnl_coarse)),
        },
        "summary": {
            "fine_mode": "cell",
            "fine_subgrid": int(fine_subgrid),
            "coarse_projector_subgrid": int(coarse_projector_subgrid),
            "coarse_size": n_grid,
        },
    }


def _cell_fine_data_for_operator(grid, v_eff, fine_subgrid):
    positions, fine_dv, _ = _cell_fine_points(grid, fine_subgrid)
    flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
    v_eff = jnp.asarray(v_eff, dtype=jnp.float32).reshape(grid.shape)
    v_fine = gather_fine_values(v_eff, flat_indices, weights)
    return positions, fine_dv, flat_indices, weights, v_fine


def _apply_matrix_free_component(grid, vector, component):
    arr = jnp.asarray(vector, dtype=jnp.float32).reshape(grid.shape)
    return np.asarray(component(arr).reshape(-1), dtype=np.float64)


def _solve_matrix_free_variant(grid, apply_components, n_bands, maxiter, tol, seed):
    n_grid = int(np.prod(np.asarray(grid.shape)))
    dv = float(grid.volume_element)

    def apply_h_over_s(vec):
        out = np.zeros((n_grid,), dtype=np.float64)
        for apply_component in apply_components.values():
            out += _apply_matrix_free_component(grid, vec, apply_component)
        return out / dv

    op = sparse_linalg.LinearOperator(
        (n_grid, n_grid),
        matvec=apply_h_over_s,
        dtype=np.float64,
    )
    eigvals, eigvecs = sparse_linalg.eigsh(
        op,
        k=int(n_bands),
        which="SA",
        maxiter=int(maxiter),
        tol=float(tol),
        v0=np.random.default_rng(seed).standard_normal(n_grid),
    )
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    decomposition = {}
    for name, apply_component in apply_components.items():
        vals = []
        for band in range(int(n_bands)):
            vec = eigvecs[:, band]
            h_vec = _apply_matrix_free_component(grid, vec, apply_component)
            vals.append(float(vec @ h_vec) / (dv * float(vec @ vec)))
        decomposition[name] = np.asarray(vals, dtype=np.float64)
    decomposition["total"] = sum(decomposition.values())
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "band_decomposition": decomposition,
    }


def fixed_veff_matrix_free_operator_ab(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    fine_subgrid=2,
    coarse_projector_subgrid=1,
    maxiter=200,
    tol=1e-6,
    seed=0,
):
    """Matrix-free fixed-Veff A/B solve using the same cell-fine operator semantics as dense A/B."""
    coords = jnp.asarray(coords, dtype=jnp.float32)
    n_grid = int(np.prod(np.asarray(grid.shape)))
    n_bands = int(n_bands)
    if n_bands < 1 or n_bands >= n_grid:
        raise ValueError("n_bands must be between 1 and coarse_size - 1 for eigsh")

    v_eff = jnp.asarray(v_eff, dtype=jnp.float32).reshape(grid.shape)
    coarse_proj_data = precompute_projectors(
        grid,
        coords,
        pseudos,
        projector_subgrid=coarse_projector_subgrid,
        projector_mode="cell_average",
    )
    positions, fine_dv, flat_indices, weights, v_fine = _cell_fine_data_for_operator(grid, v_eff, fine_subgrid)
    fine_channels = _projector_channels_at_positions(positions, coords, pseudos)

    def apply_t(psi):
        return grid.volume_element * (-0.5 * laplacian_8th(psi, grid.spacing, grid.mask))

    def apply_local_coarse(psi):
        return grid.volume_element * v_eff * psi

    def apply_local_fine(psi):
        psi_fine = gather_fine_values(psi, flat_indices, weights)
        physical = scatter_fine_values_adjoint(
            v_fine * psi_fine,
            grid.shape,
            flat_indices,
            weights,
            fine_dv,
            grid.volume_element,
        )
        return grid.volume_element * physical

    def apply_vnl_coarse(psi):
        if coarse_proj_data is None:
            return jnp.zeros_like(psi)
        p_i, p_j, coeffs = coarse_proj_data
        return grid.volume_element * apply_nonlocal_precomputed(psi, p_i, p_j, coeffs, grid.volume_element)

    def apply_vnl_fine(psi):
        if fine_channels is None:
            return jnp.zeros_like(psi)
        p_i, p_j, coeffs = fine_channels
        psi_fine = gather_fine_values(psi, flat_indices, weights)
        overlap = jnp.sum(p_j * psi_fine[None, :], axis=1) * fine_dv
        values_fine = jnp.sum(p_i * (coeffs * overlap)[:, None], axis=0)
        physical = scatter_fine_values_adjoint(
            values_fine,
            grid.shape,
            flat_indices,
            weights,
            fine_dv,
            grid.volume_element,
        )
        return grid.volume_element * physical

    variants = {
        "coarse": {"t": apply_t, "local": apply_local_coarse, "v_nl": apply_vnl_coarse},
        "fine_local": {"t": apply_t, "local": apply_local_fine, "v_nl": apply_vnl_coarse},
        "fine_projector": {"t": apply_t, "local": apply_local_coarse, "v_nl": apply_vnl_fine},
        "fine_both": {"t": apply_t, "local": apply_local_fine, "v_nl": apply_vnl_fine},
    }
    solved = {}
    for idx, (name, apply_components) in enumerate(variants.items()):
        solved[name] = _solve_matrix_free_variant(
            grid,
            apply_components,
            n_bands,
            maxiter,
            tol,
            int(seed) + idx,
        )

    return {
        "variants": solved,
        "summary": {
            "fine_mode": "cell",
            "fine_subgrid": int(fine_subgrid),
            "coarse_projector_subgrid": int(coarse_projector_subgrid),
            "coarse_size": n_grid,
            "solver": "eigsh",
        },
    }
