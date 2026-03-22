"""H2 frozen eigenspectrum sanity check.

This is not a final benchmark.
The goal is to verify that the frozen-V_eff comparison is tracking the same
physical low-energy state across box sizes, rather than being contaminated by
state reordering or overlap collapse.

For each smaller box we compare three spectra on the same adaptive grid:
  - fixed-current SCF V_eff spectrum vs reference spectrum
  - frozen-reference V_eff spectrum vs reference spectrum
  - fixed-current SCF V_eff spectrum vs frozen-reference spectrum

We solve the lowest k states and use weighted-overlap matching to determine
whether the lowest state remains identifiable.
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
FIXED_SOLVE_MAX_ITER = 16
FIXED_SOLVE_TOL = 1.0e-5
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
    total_assignment_overlap = best_score / max(k, 1)
    min_matched_overlap = min(best_overlaps) if best_overlaps else 0.0
    return {
        "perm": best_perm,
        "matched_overlaps": best_overlaps,
        "total_assignment_overlap": total_assignment_overlap,
        "min_matched_overlap": min_matched_overlap,
        "state0_best_match_index": int(best_perm[0]),
        "state0_best_match_overlap": float(best_overlaps[0]),
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
    psi_occ = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_occ_bands,)), -1, 0)
    psi0 = normalize_field(backend, grid, psi_occ[0])
    return {
        "box": float(grid.box_size[0]),
        "grid": grid,
        "backend": backend,
        "V_loc": V_loc,
        "V_H": V_H,
        "v_xc": v_xc,
        "V_eff": V_loc + V_H + v_xc,
        "rho": rho,
        "psi0": psi0,
        "eig0_occ": float(jnp.asarray(eigvals).reshape(-1)[0]),
    }


def build_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    _, n_occ_bands, occ = build_occ(pseudos)
    grid_builder = AdaptiveBackend()
    base_key = jax.random.PRNGKey(123)
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
    ref = rows[-1]
    ref_backend = ref["backend"]
    ref_states = solve_fixed_spectrum(
        ref_backend,
        ref["grid"],
        pseudos,
        coords,
        ref["V_eff"],
        K_EIG,
        key_seed=900,
        x_init_states=None,
    )

    analyses = []
    for idx, row in enumerate(rows[:-1]):
        backend = row["backend"]
        grid = row["grid"]
        ref_states_on_current = sample_state_block_to_grid(ref["grid"], ref_states["states"], grid, backend)

        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
        coeff = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
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
            key_seed=1000 + idx,
            x_init_states=init_states,
        )
        frozen_spec = solve_fixed_spectrum(
            backend,
            grid,
            pseudos,
            coords,
            V_eff_frozen,
            K_EIG,
            key_seed=1100 + idx,
            x_init_states=init_states,
        )

        scf_ref_matrix = compute_overlap_matrix(backend, grid, scf_spec["states"], ref_states_on_current)
        frozen_ref_matrix = compute_overlap_matrix(backend, grid, frozen_spec["states"], ref_states_on_current)
        scf_frozen_matrix = compute_overlap_matrix(backend, grid, scf_spec["states"], frozen_spec["states"])

        scf_ref_match = best_assignment(scf_ref_matrix)
        frozen_ref_match = best_assignment(frozen_ref_matrix)
        scf_frozen_match = best_assignment(scf_frozen_matrix)

        analyses.append({
            "box": row["box"],
            "eigvals_scf": np.asarray(scf_spec["eigvals"], dtype=float),
            "eigvals_frozen": np.asarray(frozen_spec["eigvals"], dtype=float),
            "gap01_scf": float(np.asarray(scf_spec["eigvals"], dtype=float)[1] - np.asarray(scf_spec["eigvals"], dtype=float)[0]),
            "gap01_frozen": float(np.asarray(frozen_spec["eigvals"], dtype=float)[1] - np.asarray(frozen_spec["eigvals"], dtype=float)[0]),
            "norm_dev_scf": scf_spec["norm_dev"],
            "norm_dev_frozen": frozen_spec["norm_dev"],
            "scf_ref_matrix": scf_ref_matrix,
            "frozen_ref_matrix": frozen_ref_matrix,
            "scf_frozen_matrix": scf_frozen_matrix,
            "scf_ref_match": scf_ref_match,
            "frozen_ref_match": frozen_ref_match,
            "scf_frozen_match": scf_frozen_match,
            "state0_scf_vs_frozen_overlap": float(scf_frozen_matrix[0, 0]),
        })
    return ref_states, analyses


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


def print_match_summary(title, rows, key):
    print(f"=== {title} ===")
    header = (
        f"{'Box':>5} {'best_perm':<12} {'state0_idx':>10} {'state0_ovlp':>12} "
        f"{'total_assign':>13} {'min_match':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        match = row[key]
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_perm(match['perm']):<12} "
            f"{match['state0_best_match_index']:>10d} {fmt_float(match['state0_best_match_overlap'], 12, 6)} "
            f"{fmt_float(match['total_assignment_overlap'], 13, 6)} {fmt_float(match['min_matched_overlap'], 11, 6)}"
        )


def print_scf_frozen_summary(rows):
    print("=== SCF vs Frozen Matching Summary ===")
    header = (
        f"{'Box':>5} {'state0_scf_vs_frozen':>21} {'best_perm_scf_frozen':<22} {'total_assign':>13} {'min_match':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        match = row['scf_frozen_match']
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_float(row['state0_scf_vs_frozen_overlap'], 21, 6)} "
            f"{fmt_perm(match['perm']):<22} {fmt_float(match['total_assignment_overlap'], 13, 6)} "
            f"{fmt_float(match['min_matched_overlap'], 11, 6)}"
        )


def print_overlap_matrices(rows):
    for row in rows:
        print(f"=== Overlap Matrices for box={row['box']:.1f} ===")
        print("SCF vs Reference:")
        print(np.array2string(row['scf_ref_matrix'], precision=4, suppress_small=False))
        print("Frozen vs Reference:")
        print(np.array2string(row['frozen_ref_matrix'], precision=4, suppress_small=False))
        print("SCF vs Frozen:")
        print(np.array2string(row['scf_frozen_matrix'], precision=4, suppress_small=False))
        print()


def diagnose(rows):
    min_state0_frozen_overlap = min(row['frozen_ref_match']['state0_best_match_overlap'] for row in rows)
    min_state0_scf_overlap = min(row['scf_ref_match']['state0_best_match_overlap'] for row in rows)
    min_total_assignment_overlap_frozen = min(row['frozen_ref_match']['total_assignment_overlap'] for row in rows)
    min_total_assignment_overlap_scf = min(row['scf_ref_match']['total_assignment_overlap'] for row in rows)
    min_total_assignment_overlap_scf_frozen = min(row['scf_frozen_match']['total_assignment_overlap'] for row in rows)
    min_state0_scf_vs_frozen_overlap = min(row['state0_scf_vs_frozen_overlap'] for row in rows)
    num_frozen_state0_match_to_ref0 = sum(int(row['frozen_ref_match']['state0_best_match_index'] == 0) for row in rows)
    num_scf_state0_match_to_ref0 = sum(int(row['scf_ref_match']['state0_best_match_index'] == 0) for row in rows)

    if (
        num_frozen_state0_match_to_ref0 == len(rows)
        and min_state0_frozen_overlap >= 0.8
        and min_total_assignment_overlap_frozen >= 0.7
        and min_state0_scf_vs_frozen_overlap >= 0.7
    ):
        label = 'matching-stable'
    elif min_state0_frozen_overlap >= 0.5 and min_total_assignment_overlap_frozen >= 0.5:
        label = 'reordered-but-trackable'
    else:
        label = 'overlap-collapse'

    print("=== Lowest-State Overlap Consistency Summary ===")
    print(f"label: {label}")
    print(f"min_state0_frozen_overlap = {min_state0_frozen_overlap:.6f}")
    print(f"min_state0_scf_overlap = {min_state0_scf_overlap:.6f}")
    print(f"num_frozen_state0_match_to_ref0 = {num_frozen_state0_match_to_ref0}")
    print(f"num_scf_state0_match_to_ref0 = {num_scf_state0_match_to_ref0}")
    print(f"min_total_assignment_overlap_frozen = {min_total_assignment_overlap_frozen:.6f}")
    print(f"min_total_assignment_overlap_scf = {min_total_assignment_overlap_scf:.6f}")
    print(f"min_total_assignment_overlap_scf_frozen = {min_total_assignment_overlap_scf_frozen:.6f}")
    print(f"min_state0_scf_vs_frozen_overlap = {min_state0_scf_vs_frozen_overlap:.6f}")
    return label


def main():
    print("H2 frozen eigenspectrum sanity check")
    print("This is not a final benchmark.")
    print("The goal is to verify that the frozen-V_eff path is still tracking the same")
    print("physical low-energy state, rather than being distorted by state reordering.")
    print()

    coords, pseudos, rows = build_results()
    ref_states, analyses = analyze(rows, coords, pseudos)
    _ = ref_states
    print_spectrum_table(analyses)
    print()
    print_match_summary("SCF vs Reference Matching Summary", analyses, 'scf_ref_match')
    print()
    print_match_summary("Frozen vs Reference Matching Summary", analyses, 'frozen_ref_match')
    print()
    print_scf_frozen_summary(analyses)
    print()
    print_overlap_matrices(analyses)
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
        "overlaps_finite",
        all(
            np.all(np.isfinite(row['scf_ref_matrix']))
            and np.all(np.isfinite(row['frozen_ref_matrix']))
            and np.all(np.isfinite(row['scf_frozen_matrix']))
            for row in analyses
        ),
        f"min_overlap={min(float(np.min(row['frozen_ref_matrix'])) for row in analyses):.6f}",
    ))
    checks.append(check(
        "diagnosis_label_valid",
        label in {'matching-stable', 'reordered-but-trackable', 'overlap-collapse'},
        f"label={label}",
    ))

    overall = all(checks)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
