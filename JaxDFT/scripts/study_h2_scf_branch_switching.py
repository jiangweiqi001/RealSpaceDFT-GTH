"""H2 adaptive SCF / eigensolver coupling branch-switching study.

This is not a final benchmark.
The goal is to explain why the symmetric_fv kinetic prototype showed positive
signals under frozen V_eff but fell onto different low-energy branches once we
entered real adaptive SCF.

We replay a small adaptive SCF trace in script space, with old/new kinetic
operators starting from exactly the same:
  - initial rho
  - initial subspace X0
  - mixing parameters

For each box we record the first 8 SCF iterations and compare:
  - eigvals[:4]
  - occupied-like state index matched to the box-30 reference family
  - occupied-like overlap to previous iteration
  - matched overlap to reference state 0
  - rho update / mixing step norms
  - same-iteration old-vs-new occupied-like overlap

We also add a frozen same-X0 control under a common frozen V_eff, to see
whether branching is already present before self-consistent feedback.
"""

from __future__ import annotations

import itertools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends import AdaptiveBackend
    from JaxDFT.src.functional import lda_xc
    from JaxDFT.src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import anderson_mixing, scf, solve_orbitals_subspace
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.functional import lda_xc
    from src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from src.io import load_pseudopotentials
    from src.solver import anderson_mixing, scf, solve_orbitals_subspace


DISTANCE = 1.4
BOX_TRACE = [14.0, 22.0]
BOX_REF = 30.0
H_MIN = 0.25
H_MAX = 0.80
R_CORE = 1.0
STRETCH_BETA = 5.0
SCF_KWARGS = {
    "max_iter": 4,
    "mix_alpha": 0.30,
    "tolerance": 5.0e-4,
}
TRACE_ITERS = 8
TRACE_SOLVE_MAX_ITER = 8
TRACE_SOLVE_TOL = 1.0e-4
K_EIG = 4
SUBSPACE_M = 2
MAIN_PROBE_BOUND = 4.0
PROBE_SPACING = 0.3
OVERLAP_THRESH = 0.7
RHO_SEP_THRESH = 1.0e-4
SMALL = 1.0e-12
MODES = {
    "old": "prototype_fd2",
    "new": "symmetric_fv",
}


def fmt_float(value, width=11, precision=6):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=12, precision=3):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def build_occ(pseudos):
    n_electrons = float(jnp.sum(jnp.asarray([p["q"] for p in pseudos], dtype=jnp.float32)))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float32)
    rem = n_electrons
    for i in range(n_bands):
        val = min(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val
    return n_electrons, n_bands, occ


def build_probe_axis(bound: float, spacing: float) -> jnp.ndarray:
    n = int(np.floor(bound / spacing))
    return spacing * jnp.arange(-n, n + 1, dtype=jnp.float32)


def build_probe_points(bound: float, spacing: float):
    axis = build_probe_axis(bound, spacing)
    xx, yy, zz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    points = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return axis, points


def _sample_axis_trilinear(axis_values, coord):
    axis_np = np.asarray(axis_values, dtype=np.float64)
    coord_np = np.asarray(coord, dtype=np.float64)
    upper = np.searchsorted(axis_np, coord_np, side="right")
    lower = upper - 1
    lower = np.clip(lower, 0, axis_np.size - 2)
    upper = np.clip(upper, 1, axis_np.size - 1)
    x0 = axis_np[lower]
    x1 = axis_np[upper]
    denom = np.maximum(x1 - x0, SMALL)
    frac = np.clip((coord_np - x0) / denom, 0.0, 1.0)
    return lower, upper, frac


def sample_adaptive_trilinear(grid, field, target_points):
    field_np = np.asarray(jnp.asarray(field), dtype=np.float64)
    points_np = np.asarray(jnp.asarray(target_points), dtype=np.float64)
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    z = np.asarray(grid.z, dtype=np.float64)

    outside = (
        (points_np[:, 0] < x[0]) | (points_np[:, 0] > x[-1]) |
        (points_np[:, 1] < y[0]) | (points_np[:, 1] > y[-1]) |
        (points_np[:, 2] < z[0]) | (points_np[:, 2] > z[-1])
    )
    if np.any(outside):
        bad_idx = int(np.argmax(outside))
        raise ValueError(f"target point {points_np[bad_idx].tolist()} lies outside source adaptive box")

    ix0, ix1, tx = _sample_axis_trilinear(x, points_np[:, 0])
    iy0, iy1, ty = _sample_axis_trilinear(y, points_np[:, 1])
    iz0, iz1, tz = _sample_axis_trilinear(z, points_np[:, 2])

    c000 = field_np[ix0, iy0, iz0]
    c001 = field_np[ix0, iy0, iz1]
    c010 = field_np[ix0, iy1, iz0]
    c011 = field_np[ix0, iy1, iz1]
    c100 = field_np[ix1, iy0, iz0]
    c101 = field_np[ix1, iy0, iz1]
    c110 = field_np[ix1, iy1, iz0]
    c111 = field_np[ix1, iy1, iz1]

    wx0 = 1.0 - tx
    wy0 = 1.0 - ty
    wz0 = 1.0 - tz
    wx1 = tx
    wy1 = ty
    wz1 = tz

    sampled = (
        c000 * wx0 * wy0 * wz0
        + c001 * wx0 * wy0 * wz1
        + c010 * wx0 * wy1 * wz0
        + c011 * wx0 * wy1 * wz1
        + c100 * wx1 * wy0 * wz0
        + c101 * wx1 * wy0 * wz1
        + c110 * wx1 * wy1 * wz0
        + c111 * wx1 * wy1 * wz1
    )
    return jnp.asarray(sampled, dtype=jnp.float32)


def sample_field_to_grid(source_grid, field, target_grid):
    sampled = sample_adaptive_trilinear(source_grid, field, target_grid.coords.reshape(-1, 3))
    return sampled.reshape(target_grid.shape)


def rms_diff(a, b):
    diff = jnp.asarray(a, dtype=jnp.float32) - jnp.asarray(b, dtype=jnp.float32)
    return float(jnp.sqrt(jnp.mean(diff * diff)))


def project_field(grid, field):
    return jnp.asarray(field, dtype=jnp.float32) * grid.mask


def project_block(grid, block_flat):
    mask_flat = jnp.asarray(grid.mask, dtype=jnp.float32).reshape(-1, 1)
    return jnp.asarray(block_flat, dtype=jnp.float32) * mask_flat


def normalize_field(backend, grid, psi):
    psi = project_field(grid, psi)
    norm = jnp.sqrt(jnp.maximum(backend.inner_product(grid, psi, psi), SMALL))
    psi = psi / norm
    psi = project_field(grid, psi)
    norm2 = jnp.sqrt(jnp.maximum(backend.inner_product(grid, psi, psi), SMALL))
    psi = psi / norm2
    return psi


def metric_gram(backend, grid, block_a, block_b):
    m = block_a.shape[0]
    n = block_b.shape[0]
    gram = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            gram[i, j] = float(backend.inner_product(grid, block_a[i], block_b[j]))
    return gram


def metric_orthonormalize_block(backend, grid, block):
    block = jnp.asarray(block, dtype=jnp.float32)
    flat = block.reshape(block.shape[0], -1)
    gram = metric_gram(backend, grid, block, block)
    gram = 0.5 * (gram + gram.T)
    evals, evecs = np.linalg.eigh(gram)
    evals = np.clip(evals, 1.0e-8, None)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    ortho_flat = inv_sqrt @ np.asarray(flat, dtype=np.float64)
    ortho = jnp.asarray(ortho_flat, dtype=jnp.float32).reshape(block.shape)
    ortho = jnp.stack([normalize_field(backend, grid, psi) for psi in ortho], axis=0)
    return ortho


def build_initial_rho(grid, coords, occ, backend):
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-2.0 * r ** 2)
    rho = project_field(grid, rho)
    rho = rho / backend.integrate(grid, rho) * jnp.sum(occ)
    return rho


def build_initial_X0(grid, n_states, seed, backend):
    key = jax.random.PRNGKey(seed)
    block = jax.random.normal(key, (n_states,) + tuple(int(n) for n in grid.shape), dtype=jnp.float32)
    block = jnp.stack([normalize_field(backend, grid, psi) for psi in block], axis=0)
    block = metric_orthonormalize_block(backend, grid, block)
    return block


def sample_state_block_to_grid(source_grid, state_block, target_grid, backend):
    sampled_states = []
    target_points = target_grid.coords.reshape(-1, 3)
    for state in state_block:
        sampled = sample_adaptive_trilinear(source_grid, state, target_points).reshape(target_grid.shape)
        sampled_states.append(normalize_field(backend, target_grid, sampled))
    return jnp.stack(sampled_states, axis=0)


def compute_overlap_matrix(backend, grid, states_a, states_b):
    k_a = states_a.shape[0]
    k_b = states_b.shape[0]
    matrix = np.zeros((k_a, k_b), dtype=np.float64)
    for i in range(k_a):
        for j in range(k_b):
            matrix[i, j] = float(jnp.abs(backend.inner_product(grid, states_a[i], states_b[j])))
    return matrix


def best_assignment(overlap_matrix):
    k = overlap_matrix.shape[0]
    best_perm = None
    best_score = -1.0
    best_overlaps = None
    for perm in itertools.permutations(range(k)):
        overlaps = [float(overlap_matrix[i, perm[i]]) for i in range(k)]
        score = float(sum(overlaps))
        if score > best_score:
            best_score = score
            best_perm = perm
            best_overlaps = overlaps
    inverse = [0] * k
    for i, j in enumerate(best_perm):
        inverse[j] = i
    return {
        "perm": tuple(best_perm),
        "inverse_perm": tuple(inverse),
        "matched_overlaps": best_overlaps,
        "total_assignment_overlap": best_score / max(k, 1),
        "min_matched_overlap": min(best_overlaps) if best_overlaps else 0.0,
        "current_index_for_ref0": int(inverse[0]),
        "overlap_for_ref0": float(overlap_matrix[inverse[0], 0]),
    }


def subspace_metrics(backend, grid, current_states, ref_states, m):
    cur = metric_orthonormalize_block(backend, grid, current_states[:m])
    ref = metric_orthonormalize_block(backend, grid, ref_states[:m])
    s = metric_gram(backend, grid, cur, ref)
    sigma = np.linalg.svd(s, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    min_sigma = float(np.min(sigma))
    projector_dist = float(np.sqrt(max(2.0 * m - 2.0 * float(np.sum(sigma * sigma)), 0.0)) / np.sqrt(2.0 * m))
    return {
        "min_sigma": min_sigma,
        "projector_dist": projector_dist,
    }


def solve_spectrum_from_potential(backend, grid, pseudos, coords, V_eff, x_init_flat, key_seed):
    n_states = x_init_flat.shape[1]
    proj_data = backend.precompute_nonlocal(grid, coords, pseudos)

    def apply_h(psi_flat):
        psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
        psi = project_field(grid, psi)
        kinetic_psi = backend.apply_kinetic(grid, psi)
        v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
        hpsi = kinetic_psi + V_eff * psi + v_nonlocal
        hpsi = project_field(grid, hpsi)
        return hpsi.reshape(-1)

    eigvals, eigvecs = solve_orbitals_subspace(
        apply_h,
        int(np.prod(grid.shape)),
        n_states,
        x_init=x_init_flat,
        max_iter=TRACE_SOLVE_MAX_ITER,
        tol=TRACE_SOLVE_TOL,
        key=jax.random.PRNGKey(key_seed),
        grid=grid,
        backend=backend,
    )
    eigvecs = project_block(grid, eigvecs)
    eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_states,)), -1, 0)
    norms = jnp.sqrt(jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields))
    eigvecs = eigvecs / norms[None, :]
    eigvecs = project_block(grid, eigvecs)
    states = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_states,)), -1, 0)
    return {
        "eigvals": jnp.asarray(eigvals, dtype=jnp.float32),
        "eigvecs_flat": eigvecs,
        "states": states,
    }


def run_reference(coords, pseudos):
    rows = {}
    box = jnp.array([BOX_REF, BOX_REF, BOX_REF], dtype=jnp.float32)
    builder = AdaptiveBackend()
    grid = builder.create_grid(
        spacing=H_MIN,
        box_size=box,
        atom_coords=coords,
        h_min=H_MIN,
        h_max=H_MAX,
        r_core=R_CORE,
        stretch_beta=STRETCH_BETA,
    )
    _, n_bands, occ = build_occ(pseudos)
    for offset, (label, kinetic_mode) in enumerate(MODES.items()):
        backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior", kinetic_mode=kinetic_mode)
        V_loc = backend.build_local_potential(grid, coords, pseudos)
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
            grid,
            coords,
            n_bands,
            occ,
            V_loc,
            pseudos,
            key=jax.random.PRNGKey(6100 + offset),
            backend=backend,
            **SCF_KWARGS,
        )
        V_eff = V_loc + V_H + v_xc
        X0_ref = build_initial_X0(grid, K_EIG, 7100 + offset, backend)
        x_init_ref = jnp.moveaxis(X0_ref, 0, -1).reshape(-1, K_EIG)
        spec = solve_spectrum_from_potential(
            backend,
            grid,
            pseudos,
            coords,
            V_eff,
            x_init_ref,
            key_seed=7200 + offset,
        )
        rows[label] = {
            "grid": grid,
            "backend": backend,
            "rho": rho,
            "V_H": V_H,
            "v_xc": v_xc,
            "V_eff": V_eff,
            "ref_spec": spec,
        }
    return grid, occ, rows


def run_trace_for_mode(grid, coords, pseudos, occ, kinetic_mode, rho0, X0_fields, ref_states_on_current, key_base):
    backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior", kinetic_mode=kinetic_mode)
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    proj_data = backend.precompute_nonlocal(grid, coords, pseudos)
    rho_cur = jnp.asarray(rho0, dtype=jnp.float32)
    x_prev = jnp.moveaxis(jnp.asarray(X0_fields, dtype=jnp.float32), 0, -1).reshape(-1, K_EIG)
    f_hist = jnp.zeros((5, rho_cur.size), dtype=jnp.float32)
    prev_occ_idx = None
    prev_occ_state = None
    trace = []

    def apply_h_factory(V_eff):
        def apply_h(psi_flat):
            psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
            psi = project_field(grid, psi)
            kinetic_psi = backend.apply_kinetic(grid, psi)
            v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
            hpsi = kinetic_psi + V_eff * psi + v_nonlocal
            hpsi = project_field(grid, hpsi)
            return hpsi.reshape(-1)
        return apply_h

    for iter_idx in range(TRACE_ITERS):
        rho_in = project_field(grid, jnp.clip(rho_cur, 1.0e-12, None))
        V_H = backend.solve_hartree(grid, rho_in)
        eps_xc, v_xc = lda_xc(rho_in)
        V_eff = V_loc + V_H + v_xc
        apply_h = apply_h_factory(V_eff)
        eigvals, eigvecs = solve_orbitals_subspace(
            apply_h,
            int(np.prod(grid.shape)),
            K_EIG,
            x_init=x_prev,
            max_iter=TRACE_SOLVE_MAX_ITER,
            tol=TRACE_SOLVE_TOL,
            key=jax.random.PRNGKey(key_base + iter_idx),
            grid=grid,
            backend=backend,
        )
        eigvecs = project_block(grid, eigvecs)
        eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (K_EIG,)), -1, 0)
        norms = jnp.sqrt(jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields))
        eigvecs = eigvecs / norms[None, :]
        eigvecs = project_block(grid, eigvecs)
        states = jnp.moveaxis(eigvecs.reshape(grid.shape + (K_EIG,)), -1, 0)

        overlap_ref = compute_overlap_matrix(backend, grid, states, ref_states_on_current)
        match_ref = best_assignment(overlap_ref)
        occ_idx = match_ref["current_index_for_ref0"]
        occ_state = states[occ_idx]
        if prev_occ_state is None:
            occ_overlap_prev = None
            branch_jump = False
        else:
            occ_overlap_prev = float(jnp.abs(backend.inner_product(grid, occ_state, prev_occ_state)))
            branch_jump = (occ_idx != prev_occ_idx) or (occ_overlap_prev < OVERLAP_THRESH)

        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        rho_new = project_field(grid, rho_new)
        rho_update_norm = rms_diff(rho_new, rho_in)
        rho_mixed_flat, f_hist = anderson_mixing(rho_in.reshape(-1), rho_new.reshape(-1), f_hist, SCF_KWARGS["mix_alpha"], iter_idx)
        rho_mixed = project_field(grid, rho_mixed_flat.reshape(grid.shape))
        rho_mix_step_norm = rms_diff(rho_mixed, rho_in)

        trace.append({
            "iter": iter_idx,
            "eigvals": np.asarray(jnp.asarray(eigvals), dtype=float)[:4],
            "states": states,
            "rho_mixed": rho_mixed,
            "occ_idx": occ_idx,
            "occ_overlap_prev": occ_overlap_prev,
            "matched_overlap_ref0": match_ref["overlap_for_ref0"],
            "total_assignment_overlap_ref": match_ref["total_assignment_overlap"],
            "branch_jump": branch_jump,
            "rho_update_norm": rho_update_norm,
            "rho_mix_step_norm": rho_mix_step_norm,
        })

        prev_occ_idx = occ_idx
        prev_occ_state = occ_state
        x_prev = eigvecs
        rho_cur = rho_mixed
    return trace


def first_iter_where(values, predicate):
    for value in values:
        if predicate(value):
            return value["iter"]
    return None


def run_frozen_same_x0_compare(grid, coords, pseudos, X0_fields, ref_old, key_seed_base):
    old_backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior", kinetic_mode=MODES["old"])
    new_backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior", kinetic_mode=MODES["new"])
    V_loc_current = old_backend.build_local_potential(grid, coords, pseudos)
    V_H_ref_on_current = sample_field_to_grid(ref_old["grid"], ref_old["V_H"], grid)
    v_xc_ref_on_current = sample_field_to_grid(ref_old["grid"], ref_old["v_xc"], grid)
    V_eff_frozen = V_loc_current + V_H_ref_on_current + v_xc_ref_on_current
    x_init = jnp.moveaxis(jnp.asarray(X0_fields, dtype=jnp.float32), 0, -1).reshape(-1, K_EIG)

    old_spec = solve_spectrum_from_potential(old_backend, grid, pseudos, coords, V_eff_frozen, x_init, key_seed_base)
    new_spec = solve_spectrum_from_potential(new_backend, grid, pseudos, coords, V_eff_frozen, x_init, key_seed_base + 1)

    old_ref_states_on_current = sample_state_block_to_grid(ref_old["grid"], ref_old["ref_spec"]["states"], grid, old_backend)
    old_match = best_assignment(compute_overlap_matrix(old_backend, grid, old_spec["states"], old_ref_states_on_current))
    new_match = best_assignment(compute_overlap_matrix(old_backend, grid, new_spec["states"], old_ref_states_on_current))
    same_overlap = compute_overlap_matrix(old_backend, grid, old_spec["states"], new_spec["states"])
    same_match = best_assignment(same_overlap)
    old_idx = old_match["current_index_for_ref0"]
    new_idx = new_match["current_index_for_ref0"]
    occ_overlap = float(same_overlap[old_idx, new_idx])
    m2 = subspace_metrics(old_backend, grid, new_spec["states"], old_spec["states"], SUBSPACE_M)
    return {
        "occ_overlap": occ_overlap,
        "sigma_m2": m2["min_sigma"],
        "projector_dist_m2": m2["projector_dist"],
        "assignment_overlap": same_match["total_assignment_overlap"],
    }


def analyze_box(box_length, coords, pseudos, occ, reference):
    builder = AdaptiveBackend()
    box = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    grid = builder.create_grid(
        spacing=H_MIN,
        box_size=box,
        atom_coords=coords,
        h_min=H_MIN,
        h_max=H_MAX,
        r_core=R_CORE,
        stretch_beta=STRETCH_BETA,
    )
    init_backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior", kinetic_mode=MODES["old"])
    rho0 = build_initial_rho(grid, coords, occ, init_backend)
    X0 = build_initial_X0(grid, K_EIG, 8100 + int(round(box_length)), init_backend)

    traces = {}
    for label, kinetic_mode in MODES.items():
        ref_states_on_current = sample_state_block_to_grid(reference[label]["grid"], reference[label]["ref_spec"]["states"], grid, init_backend)
        traces[label] = run_trace_for_mode(
            grid,
            coords,
            pseudos,
            occ,
            kinetic_mode,
            rho0,
            X0,
            ref_states_on_current,
            key_base=9100 + (0 if label == "old" else 100),
        )

    iter_rows = []
    for i in range(TRACE_ITERS):
        row_old = traces["old"][i]
        row_new = traces["new"][i]
        same_overlap = float(jnp.abs(init_backend.inner_product(grid, row_old["states"][row_old["occ_idx"]], row_new["states"][row_new["occ_idx"]])))
        rho_sep = rms_diff(row_old["rho_mixed"], row_new["rho_mixed"])
        iter_rows.append({
            "iter": i,
            "old": row_old,
            "new": row_new,
            "old_vs_new_occ_like_overlap_same_iter": same_overlap,
            "old_vs_new_rho_sep_norm": rho_sep,
        })

    frozen = run_frozen_same_x0_compare(grid, coords, pseudos, X0, reference["old"], key_seed_base=10100 + int(round(box_length)))

    summary = {
        "box": box_length,
        "first_iter_old_branch_jump": first_iter_where(traces["old"], lambda row: row["branch_jump"]),
        "first_iter_new_branch_jump": first_iter_where(traces["new"], lambda row: row["branch_jump"]),
        "num_branch_jumps_old": sum(int(row["branch_jump"]) for row in traces["old"]),
        "num_branch_jumps_new": sum(int(row["branch_jump"]) for row in traces["new"]),
        "first_iter_old_new_occ_overlap_below_thresh": first_iter_where(iter_rows, lambda row: row["old_vs_new_occ_like_overlap_same_iter"] < OVERLAP_THRESH),
        "first_iter_old_new_rho_sep": first_iter_where(iter_rows, lambda row: row["old_vs_new_rho_sep_norm"] > RHO_SEP_THRESH),
        "frozen_same_X0_occ_overlap": frozen["occ_overlap"],
        "frozen_same_X0_sigma_m2": frozen["sigma_m2"],
        "frozen_same_X0_projector_dist_m2": frozen["projector_dist_m2"],
        "frozen_same_X0_assignment_overlap": frozen["assignment_overlap"],
        "min_old_new_occ_like_overlap_same_iter": min(row["old_vs_new_occ_like_overlap_same_iter"] for row in iter_rows),
        "max_old_new_rho_sep": max(row["old_vs_new_rho_sep_norm"] for row in iter_rows),
    }
    return iter_rows, summary


def fmt_eigs(arr):
    vals = [f"{float(v):.4f}" for v in arr]
    return "[" + ", ".join(vals) + "]"


def print_iteration_trace(box_length, iter_rows):
    print(f"=== Iteration Trace Table (box={box_length:.1f}) ===")
    header = (
        f"{'it':>3} {'eig_old[:4]':<34} {'idx_o':>5} {'ovprev_o':>9} {'match_o':>9} {'rho_up_o':>10} {'rho_mix_o':>10} "
        f"{'eig_new[:4]':<34} {'idx_n':>5} {'ovprev_n':>9} {'match_n':>9} {'rho_up_n':>10} {'rho_mix_n':>10} {'ovlp_o_n':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in iter_rows:
        old = row["old"]
        new = row["new"]
        print(
            f"{row['iter']:>3d} {fmt_eigs(old['eigvals']):<34} {old['occ_idx']:>5d} {fmt_float(old['occ_overlap_prev'], 9, 6)} "
            f"{fmt_float(old['matched_overlap_ref0'], 9, 6)} {fmt_sci(old['rho_update_norm'], 10, 2)} {fmt_sci(old['rho_mix_step_norm'], 10, 2)} "
            f"{fmt_eigs(new['eigvals']):<34} {new['occ_idx']:>5d} {fmt_float(new['occ_overlap_prev'], 9, 6)} "
            f"{fmt_float(new['matched_overlap_ref0'], 9, 6)} {fmt_sci(new['rho_update_norm'], 10, 2)} {fmt_sci(new['rho_mix_step_norm'], 10, 2)} "
            f"{fmt_float(row['old_vs_new_occ_like_overlap_same_iter'], 10, 6)}"
        )
    print()


def print_branch_summary(summaries):
    print("=== Branch Switching Summary ===")
    header = (
        f"{'Box':>5} {'jump_o':>8} {'jump_n':>8} {'njump_o':>8} {'njump_n':>8} {'ov<0.7':>8} {'rho_sep':>8} "
        f"{'froz_ov':>10} {'froz_sig2':>10} {'froz_proj2':>11} {'min_ov_o_n':>11} {'max_rho_sep':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        print(
            f"{fmt_float(row['box'], 5, 1)} {str(row['first_iter_old_branch_jump']):>8} {str(row['first_iter_new_branch_jump']):>8} "
            f"{row['num_branch_jumps_old']:>8d} {row['num_branch_jumps_new']:>8d} {str(row['first_iter_old_new_occ_overlap_below_thresh']):>8} "
            f"{str(row['first_iter_old_new_rho_sep']):>8} {fmt_float(row['frozen_same_X0_occ_overlap'], 10, 6)} "
            f"{fmt_float(row['frozen_same_X0_sigma_m2'], 10, 6)} {fmt_float(row['frozen_same_X0_projector_dist_m2'], 11, 6)} "
            f"{fmt_float(row['min_old_new_occ_like_overlap_same_iter'], 11, 6)} {fmt_sci(row['max_old_new_rho_sep'], 11, 2)}"
        )
    print()


def diagnose(summaries):
    eig_first = any(
        (row["frozen_same_X0_occ_overlap"] < OVERLAP_THRESH or row["frozen_same_X0_sigma_m2"] < OVERLAP_THRESH)
        and row["first_iter_old_new_occ_overlap_below_thresh"] is not None
        and (
            row["first_iter_old_new_rho_sep"] is None
            or row["first_iter_old_new_occ_overlap_below_thresh"] <= row["first_iter_old_new_rho_sep"]
        )
        for row in summaries
    )
    mixing_first = all(
        row["frozen_same_X0_occ_overlap"] >= OVERLAP_THRESH and row["frozen_same_X0_sigma_m2"] >= OVERLAP_THRESH
        for row in summaries
    ) and any(
        row["first_iter_old_new_rho_sep"] is not None
        and row["first_iter_old_new_occ_overlap_below_thresh"] is not None
        and row["first_iter_old_new_rho_sep"] <= row["first_iter_old_new_occ_overlap_below_thresh"]
        for row in summaries
    )

    if eig_first:
        label = "eigensolver-branching-first"
    elif mixing_first:
        label = "mixing-amplified-branching"
    else:
        label = "mixed"

    print("=== Diagnosis Summary ===")
    print(f"label: {label}")
    for row in summaries:
        print(
            f"box={row['box']:.1f}: first_old_jump={row['first_iter_old_branch_jump']}, "
            f"first_new_jump={row['first_iter_new_branch_jump']}, "
            f"first_old_new_occ_overlap_below_thresh={row['first_iter_old_new_occ_overlap_below_thresh']}, "
            f"first_old_new_rho_sep={row['first_iter_old_new_rho_sep']}, "
            f"frozen_occ_overlap={row['frozen_same_X0_occ_overlap']:.6f}, "
            f"frozen_sigma_m2={row['frozen_same_X0_sigma_m2']:.6f}"
        )
    return {"label": label}


def main():
    print("H2 adaptive SCF branch-switching study")
    print("This is not a final benchmark.")
    print("The goal is to locate whether old/new branch separation starts in eigensolve or is amplified by mixing.")
    print()

    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    _, occ, reference = run_reference(coords, pseudos)

    summaries = []
    for box_length in BOX_TRACE:
        iter_rows, summary = analyze_box(box_length, coords, pseudos, occ, reference)
        print_iteration_trace(box_length, iter_rows)
        summaries.append(summary)

    print_branch_summary(summaries)
    diagnosis = diagnose(summaries)

    checks = []
    checks.append(check(
        "trace_length",
        len(summaries) == len(BOX_TRACE),
        f"n_boxes={len(summaries)}",
    ))
    checks.append(check(
        "frozen_metrics_finite",
        all(np.isfinite(row["frozen_same_X0_occ_overlap"]) and np.isfinite(row["frozen_same_X0_sigma_m2"]) for row in summaries),
        f"min_frozen_overlap={min(row['frozen_same_X0_occ_overlap'] for row in summaries):.6f}",
    ))
    checks.append(check(
        "diagnosis_label_valid",
        diagnosis["label"] in {"eigensolver-branching-first", "mixing-amplified-branching", "mixed"},
        f"label={diagnosis['label']}",
    ))

    overall = all(checks)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
