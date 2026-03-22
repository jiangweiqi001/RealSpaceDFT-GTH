"""H2 state-matched frozen response study.

This is not a final benchmark.
The goal is to reassess the frozen-V_eff response after correcting for the
state reordering identified in the frozen eigenspectrum sanity check.

For each smaller box we:
  1. solve the lowest k states under the current SCF V_eff,
  2. solve the lowest k states under the frozen V_eff built from the reference
     Hartree/XC fields,
  3. match both spectra to the reference spectrum with weighted-overlap
     assignment,
  4. compare the state matched to ref-state-0 on a common probe,
  5. compare the low-energy m=2 and m=3 subspaces.

If the state-matched frozen response remains as large as the SCF response while
m=2 remains stable, the fixed-potential eigenproblem / finite-box direction is
more strongly supported. If the matched frozen response collapses, the earlier
frozen conclusion was mainly contaminated by state reordering.
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
    from JaxDFT.src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import scf, solve_orbitals_subspace
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from src.io import load_pseudopotentials
    from src.solver import scf, solve_orbitals_subspace


DISTANCE = 1.4
BOXES = [14.0, 18.0, 22.0, 26.0, 30.0]
H_MIN = 0.25
H_MAX = 0.80
R_CORE = 1.0
STRETCH_BETA = 5.0
SCF_KWARGS = {
    "max_iter": 4,
    "mix_alpha": 0.30,
    "tolerance": 5.0e-4,
}
K_EIG = 4
SUBSPACE_M_PRIMARY = 2
SUBSPACE_M_AUX = 3
FIXED_SOLVE_MAX_ITER = 16
FIXED_SOLVE_TOL = 1.0e-5
MAIN_PROBE_BOUND = 4.0
PROBE_SPACING = 0.3
NORM_TOL = 2.0e-2
SMALL = 1.0e-12


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
    if bound <= 0.0 or spacing <= 0.0:
        raise ValueError("probe bound and spacing must be positive")
    n = int(np.floor(bound / spacing))
    return spacing * jnp.arange(-n, n + 1, dtype=jnp.float32)


def build_probe_points(bound: float, spacing: float):
    axis = build_probe_axis(bound, spacing)
    xx, yy, zz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    points = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    shape = tuple(int(n) for n in xx.shape)
    return axis, points, shape


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


def field_metrics(current, reference):
    cur = jnp.asarray(current, dtype=jnp.float32).reshape(-1)
    ref = jnp.asarray(reference, dtype=jnp.float32).reshape(-1)
    diff = cur - ref
    cur_c = cur - jnp.mean(cur)
    ref_c = ref - jnp.mean(ref)
    diff_c = cur_c - ref_c
    return {
        "rms": float(jnp.sqrt(jnp.mean(diff * diff))),
        "rms_demeaned": float(jnp.sqrt(jnp.mean(diff_c * diff_c))),
        "mean_delta": float(jnp.mean(cur) - jnp.mean(ref)),
    }


def normalize_field(backend, grid, psi):
    psi = jnp.asarray(psi, dtype=jnp.float32) * grid.mask
    norm = jnp.sqrt(jnp.maximum(backend.inner_product(grid, psi, psi), SMALL))
    psi = psi / norm
    psi = psi * grid.mask
    norm2 = jnp.sqrt(jnp.maximum(backend.inner_product(grid, psi, psi), SMALL))
    psi = psi / norm2
    return psi


def sample_state_block_to_grid(source_grid, state_block, target_grid, backend):
    sampled_states = []
    target_points = target_grid.coords.reshape(-1, 3)
    for state in state_block:
        sampled = sample_adaptive_trilinear(source_grid, state, target_points).reshape(target_grid.shape)
        sampled_states.append(normalize_field(backend, target_grid, sampled))
    return jnp.stack(sampled_states, axis=0)


def solve_fixed_spectrum(backend, grid, pseudos, coords, V_eff, n_states, key_seed, x_init_states=None):
    proj_data = backend.precompute_nonlocal(grid, coords, pseudos)

    def apply_h(psi_flat):
        psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
        psi = psi * grid.mask
        kinetic_psi = backend.apply_kinetic(grid, psi)
        v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
        hpsi = kinetic_psi + V_eff * psi + v_nonlocal
        hpsi = hpsi * grid.mask
        return hpsi.reshape(-1)

    if x_init_states is None:
        x_init = None
    else:
        x_init = jnp.moveaxis(jnp.asarray(x_init_states, dtype=jnp.float32), 0, -1).reshape(-1, n_states)

    eigvals, eigvecs = solve_orbitals_subspace(
        apply_h,
        int(np.prod(grid.shape)),
        n_states,
        x_init=x_init,
        max_iter=FIXED_SOLVE_MAX_ITER,
        tol=FIXED_SOLVE_TOL,
        key=jax.random.PRNGKey(key_seed),
        grid=grid,
        backend=backend,
    )
    states = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_states,)), -1, 0)
    states = jnp.stack([normalize_field(backend, grid, psi) for psi in states], axis=0)
    norm_dev = float(max(abs(float(backend.inner_product(grid, psi, psi)) - 1.0) for psi in states))
    return {
        "eigvals": jnp.asarray(eigvals, dtype=jnp.float32),
        "states": states,
        "norm_dev": norm_dev,
    }


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
        "perm": best_perm,
        "inverse_perm": tuple(inverse),
        "matched_overlaps": best_overlaps,
        "total_assignment_overlap": best_score / max(k, 1),
        "min_matched_overlap": min(best_overlaps) if best_overlaps else 0.0,
        "current_index_for_ref0": int(inverse[0]),
        "overlap_for_ref0": float(overlap_matrix[inverse[0], 0]),
    }


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


def subspace_metrics(backend, grid, current_states, ref_states, m):
    cur = metric_orthonormalize_block(backend, grid, current_states[:m])
    ref = metric_orthonormalize_block(backend, grid, ref_states[:m])
    s = metric_gram(backend, grid, cur, ref)
    sigma = np.linalg.svd(s, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    min_sigma = float(np.min(sigma))
    max_angle_deg = float(np.degrees(np.arccos(np.clip(min_sigma, -1.0, 1.0))))
    projector_dist = float(np.sqrt(max(2.0 * m - 2.0 * float(np.sum(sigma * sigma)), 0.0)) / np.sqrt(2.0 * m))
    return {
        "min_sigma": min_sigma,
        "max_angle_deg": max_angle_deg,
        "projector_dist": projector_dist,
    }


def run_scf_case(grid, coords, pseudos, occ, n_occ_bands, key_seed):
    backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior")
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid,
        coords,
        n_occ_bands,
        occ,
        V_loc,
        pseudos,
        key=jax.random.PRNGKey(key_seed),
        backend=backend,
        **SCF_KWARGS,
    )
    return {
        "box": float(grid.box_size[0]),
        "grid": grid,
        "backend": backend,
        "V_loc": V_loc,
        "V_H": V_H,
        "v_xc": v_xc,
        "V_eff": V_loc + V_H + v_xc,
        "rho": rho,
        "eigvals_occ": jnp.asarray(eigvals, dtype=jnp.float32),
    }


def build_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    _, n_occ_bands, occ = build_occ(pseudos)
    grid_builder = AdaptiveBackend()
    base_key = jax.random.PRNGKey(456)
    rows = []
    for box_idx, box_length in enumerate(BOXES):
        box = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
        grid = grid_builder.create_grid(
            spacing=H_MIN,
            box_size=box,
            atom_coords=coords,
            h_min=H_MIN,
            h_max=H_MAX,
            r_core=R_CORE,
            stretch_beta=STRETCH_BETA,
        )
        key_seed = int(jax.random.fold_in(base_key, box_idx)[0])
        rows.append(run_scf_case(grid, coords, pseudos, occ, n_occ_bands, key_seed))
    return coords, pseudos, rows


def analyze(rows, coords, pseudos):
    _, probe_points, _ = build_probe_points(MAIN_PROBE_BOUND, PROBE_SPACING)
    probe_points = jnp.asarray(probe_points, dtype=jnp.float32)
    ref = rows[-1]
    ref_backend = ref["backend"]

    ref_spec = solve_fixed_spectrum(
        ref_backend,
        ref["grid"],
        pseudos,
        coords,
        ref["V_eff"],
        K_EIG,
        key_seed=1200,
        x_init_states=None,
    )
    ref_state0_probe = sample_adaptive_trilinear(ref["grid"], jnp.abs(ref_spec["states"][0]), probe_points)
    ref_rho0_probe = 2.0 * (ref_state0_probe ** 2)

    analyses = []
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    coeff = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)

    for idx, row in enumerate(rows[:-1]):
        backend = row["backend"]
        grid = row["grid"]
        ref_states_on_current = sample_state_block_to_grid(ref["grid"], ref_spec["states"], grid, backend)

        V_loc_current = build_local_potential_pointwise(coords, grid.coords, zion, rloc, coeff)
        V_H_ref_on_current = sample_field_to_grid(ref["grid"], ref["V_H"], grid)
        v_xc_ref_on_current = sample_field_to_grid(ref["grid"], ref["v_xc"], grid)
        V_eff_frozen = V_loc_current + V_H_ref_on_current + v_xc_ref_on_current

        init_states = ref_states_on_current
        scf_spec = solve_fixed_spectrum(
            backend,
            grid,
            pseudos,
            coords,
            row["V_eff"],
            K_EIG,
            key_seed=1300 + idx,
            x_init_states=init_states,
        )
        frozen_spec = solve_fixed_spectrum(
            backend,
            grid,
            pseudos,
            coords,
            V_eff_frozen,
            K_EIG,
            key_seed=1400 + idx,
            x_init_states=init_states,
        )

        scf_ref_matrix = compute_overlap_matrix(backend, grid, scf_spec["states"], ref_states_on_current)
        frozen_ref_matrix = compute_overlap_matrix(backend, grid, frozen_spec["states"], ref_states_on_current)
        scf_frozen_matrix = compute_overlap_matrix(backend, grid, scf_spec["states"], frozen_spec["states"])

        scf_ref_match = best_assignment(scf_ref_matrix)
        frozen_ref_match = best_assignment(frozen_ref_matrix)
        scf_frozen_match = best_assignment(scf_frozen_matrix)

        scf_match_idx = scf_ref_match["current_index_for_ref0"]
        frozen_match_idx = frozen_ref_match["current_index_for_ref0"]

        scf_match_probe = sample_adaptive_trilinear(grid, jnp.abs(scf_spec["states"][scf_match_idx]), probe_points)
        frozen_match_probe = sample_adaptive_trilinear(grid, jnp.abs(frozen_spec["states"][frozen_match_idx]), probe_points)
        scf_match_rho_probe = 2.0 * (scf_match_probe ** 2)
        frozen_match_rho_probe = 2.0 * (frozen_match_probe ** 2)

        scf_match_psi_m = field_metrics(scf_match_probe, ref_state0_probe)
        frozen_match_psi_m = field_metrics(frozen_match_probe, ref_state0_probe)
        scf_match_rho_m = field_metrics(scf_match_rho_probe, ref_rho0_probe)
        frozen_match_rho_m = field_metrics(frozen_match_rho_probe, ref_rho0_probe)

        m2_scf = subspace_metrics(backend, grid, scf_spec["states"], ref_states_on_current, SUBSPACE_M_PRIMARY)
        m2_frozen = subspace_metrics(backend, grid, frozen_spec["states"], ref_states_on_current, SUBSPACE_M_PRIMARY)
        m3_scf = subspace_metrics(backend, grid, scf_spec["states"], ref_states_on_current, SUBSPACE_M_AUX)
        m3_frozen = subspace_metrics(backend, grid, frozen_spec["states"], ref_states_on_current, SUBSPACE_M_AUX)

        analyses.append({
            "box": row["box"],
            "eigvals_scf": np.asarray(scf_spec["eigvals"], dtype=float),
            "eigvals_frozen": np.asarray(frozen_spec["eigvals"], dtype=float),
            "gap01_scf": float(np.asarray(scf_spec["eigvals"], dtype=float)[1] - np.asarray(scf_spec["eigvals"], dtype=float)[0]),
            "gap01_frozen": float(np.asarray(frozen_spec["eigvals"], dtype=float)[1] - np.asarray(frozen_spec["eigvals"], dtype=float)[0]),
            "norm_dev_scf": scf_spec["norm_dev"],
            "norm_dev_frozen": frozen_spec["norm_dev"],
            "scf_ref_match": scf_ref_match,
            "frozen_ref_match": frozen_ref_match,
            "scf_frozen_match": scf_frozen_match,
            "matched_state_index_scf": scf_match_idx,
            "matched_state_index_frozen": frozen_match_idx,
            "matched_overlap_scf": scf_ref_match["overlap_for_ref0"],
            "matched_overlap_frozen": frozen_ref_match["overlap_for_ref0"],
            "matched_psi_probe_rms_scf": scf_match_psi_m["rms"],
            "matched_psi_probe_rms_demeaned_scf": scf_match_psi_m["rms_demeaned"],
            "matched_psi_probe_rms_frozen": frozen_match_psi_m["rms"],
            "matched_psi_probe_rms_demeaned_frozen": frozen_match_psi_m["rms_demeaned"],
            "matched_rho_probe_rms_scf": scf_match_rho_m["rms"],
            "matched_rho_probe_rms_demeaned_scf": scf_match_rho_m["rms_demeaned"],
            "matched_rho_probe_rms_frozen": frozen_match_rho_m["rms"],
            "matched_rho_probe_rms_demeaned_frozen": frozen_match_rho_m["rms_demeaned"],
            "psi_matched_reduction": scf_match_psi_m["rms"] / max(frozen_match_psi_m["rms"], SMALL),
            "rho_matched_reduction": scf_match_rho_m["rms"] / max(frozen_match_rho_m["rms"], SMALL),
            "m2_scf": m2_scf,
            "m2_frozen": m2_frozen,
            "m3_scf": m3_scf,
            "m3_frozen": m3_frozen,
        })
    return analyses


def fmt_perm(perm):
    return "(" + ",".join(str(int(x)) for x in perm) + ")"


def fmt_eigs(arr):
    vals = [f"{float(v):.4f}" for v in arr]
    return "[" + ", ".join(vals) + "]"


def print_spectrum_table(rows):
    print("=== Spectrum Table ===")
    header = f"{'Box':>5} {'eigvals_scf[:4]':<38} {'gap01_scf':>11} {'eigvals_frozen[:4]':<38} {'gap01_frozen':>13}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_eigs(row['eigvals_scf']):<38} {fmt_float(row['gap01_scf'], 11, 6)} "
            f"{fmt_eigs(row['eigvals_frozen']):<38} {fmt_float(row['gap01_frozen'], 13, 6)}"
        )


def print_matched_response_table(rows):
    print("=== Matched Single-State Response Table ===")
    header = (
        f"{'Box':>5} {'idx_scf':>8} {'idx_froz':>9} {'ovlp_scf':>10} {'ovlp_froz':>10} "
        f"{'psi_scf':>10} {'psi_froz':>10} {'psi_red':>9} {'rho_scf':>10} {'rho_froz':>10} {'rho_red':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} {row['matched_state_index_scf']:>8d} {row['matched_state_index_frozen']:>9d} "
            f"{fmt_float(row['matched_overlap_scf'], 10, 6)} {fmt_float(row['matched_overlap_frozen'], 10, 6)} "
            f"{fmt_sci(row['matched_psi_probe_rms_scf'], 10, 2)} {fmt_sci(row['matched_psi_probe_rms_frozen'], 10, 2)} "
            f"{fmt_float(row['psi_matched_reduction'], 9, 3)} {fmt_sci(row['matched_rho_probe_rms_scf'], 10, 2)} "
            f"{fmt_sci(row['matched_rho_probe_rms_frozen'], 10, 2)} {fmt_float(row['rho_matched_reduction'], 9, 3)}"
        )


def print_subspace_table(rows):
    print("=== Low-Energy Subspace Response Table ===")
    header = (
        f"{'Box':>5} {'min_sigma_m2_scf':>16} {'min_sigma_m2_froz':>17} {'proj_m2_scf':>12} {'proj_m2_froz':>13} "
        f"{'min_sigma_m3_scf':>16} {'min_sigma_m3_froz':>17} {'proj_m3_scf':>12} {'proj_m3_froz':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_float(row['m2_scf']['min_sigma'], 16, 6)} {fmt_float(row['m2_frozen']['min_sigma'], 17, 6)} "
            f"{fmt_float(row['m2_scf']['projector_dist'], 12, 6)} {fmt_float(row['m2_frozen']['projector_dist'], 13, 6)} "
            f"{fmt_float(row['m3_scf']['min_sigma'], 16, 6)} {fmt_float(row['m3_frozen']['min_sigma'], 17, 6)} "
            f"{fmt_float(row['m3_scf']['projector_dist'], 12, 6)} {fmt_float(row['m3_frozen']['projector_dist'], 13, 6)}"
        )


def diagnose(rows):
    psi_scf_max = max(row['matched_psi_probe_rms_scf'] for row in rows)
    psi_frozen_max = max(row['matched_psi_probe_rms_frozen'] for row in rows)
    rho_scf_max = max(row['matched_rho_probe_rms_scf'] for row in rows)
    rho_frozen_max = max(row['matched_rho_probe_rms_frozen'] for row in rows)
    min_sigma_m2_scf = min(row['m2_scf']['min_sigma'] for row in rows)
    min_sigma_m2_frozen = min(row['m2_frozen']['min_sigma'] for row in rows)
    max_proj_m2_scf = max(row['m2_scf']['projector_dist'] for row in rows)
    max_proj_m2_frozen = max(row['m2_frozen']['projector_dist'] for row in rows)
    psi_matched_reduction = psi_scf_max / max(psi_frozen_max, SMALL)
    rho_matched_reduction = rho_scf_max / max(rho_frozen_max, SMALL)

    if (
        psi_frozen_max >= 0.8 * psi_scf_max
        and rho_frozen_max >= 0.8 * rho_scf_max
        and min_sigma_m2_frozen >= 0.7
    ):
        label = 'eigenproblem-suspected-next'
    elif (
        (psi_frozen_max <= 0.5 * psi_scf_max or rho_frozen_max <= 0.5 * rho_scf_max)
        and min_sigma_m2_frozen >= 0.7
    ):
        label = 'reordering-contaminated'
    else:
        label = 'mixed'

    print("=== Diagnosis Summary ===")
    print(f"label: {label}")
    print(f"max_matched_psi_scf = {psi_scf_max:.6e}")
    print(f"max_matched_psi_frozen = {psi_frozen_max:.6e}")
    print(f"max_matched_rho_scf = {rho_scf_max:.6e}")
    print(f"max_matched_rho_frozen = {rho_frozen_max:.6e}")
    print(f"min_sigma_m2_scf = {min_sigma_m2_scf:.6f}")
    print(f"min_sigma_m2_frozen = {min_sigma_m2_frozen:.6f}")
    print(f"max_projector_dist_m2_scf = {max_proj_m2_scf:.6f}")
    print(f"max_projector_dist_m2_frozen = {max_proj_m2_frozen:.6f}")
    print(f"psi_matched_reduction_maxboxwise = {psi_matched_reduction:.6f}")
    print(f"rho_matched_reduction_maxboxwise = {rho_matched_reduction:.6f}")
    print()
    print("Interpretation:")
    print("  - m=2 is the primary subspace sanity check; m=3 is an auxiliary reference.")
    print("  - If matched frozen response stays close to matched SCF response while m=2")
    print("    remains stable, the fixed-potential eigenproblem direction is better supported.")
    print("  - If matched frozen response collapses under stable m=2 matching, the earlier")
    print("    frozen conclusion was mainly contaminated by state reordering.")
    return label


def main():
    print("H2 state-matched frozen response study")
    print("This is not a final benchmark.")
    print("The goal is to reassess the frozen-V_eff response after correcting for")
    print("tracked state reordering via weighted-overlap matching and low-energy subspace checks.")
    print()

    coords, pseudos, rows = build_results()
    analyses = analyze(rows, coords, pseudos)
    print_spectrum_table(analyses)
    print()
    print_matched_response_table(analyses)
    print()
    print_subspace_table(analyses)
    print()
    label = diagnose(analyses)

    checks = []
    checks.append(check(
        "spectrum_shapes_ok",
        all(len(row['eigvals_scf']) == K_EIG and len(row['eigvals_frozen']) == K_EIG for row in analyses),
        f"k={K_EIG}",
    ))
    checks.append(check(
        "norms_reasonable",
        all(row['norm_dev_scf'] <= NORM_TOL and row['norm_dev_frozen'] <= NORM_TOL for row in analyses),
        f"max_norm_dev={max(max(row['norm_dev_scf'], row['norm_dev_frozen']) for row in analyses):.3e}",
    ))
    checks.append(check(
        "overlaps_reasonable",
        all(row['matched_overlap_scf'] > 0.0 and row['matched_overlap_frozen'] > 0.0 for row in analyses),
        f"min_matched_overlap={min(min(row['matched_overlap_scf'], row['matched_overlap_frozen']) for row in analyses):.6f}",
    ))
    checks.append(check(
        "diagnosis_label_valid",
        label in {'eigenproblem-suspected-next', 'reordering-contaminated', 'mixed'},
        f"label={label}",
    ))

    overall = all(checks)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
