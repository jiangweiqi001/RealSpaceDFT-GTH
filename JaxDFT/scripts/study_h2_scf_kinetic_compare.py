"""H2 adaptive SCF old/new kinetic comparison.

This is not a final benchmark.
The goal is to compare the current adaptive SCF path with:
  - kinetic_mode='prototype_fd2'
  - kinetic_mode='symmetric_fv'

under the same H2 / box / Hartree settings and decide whether the symmetric
finite-volume-style kinetic prototype carries its frozen-Veff gains into real
adaptive SCF.

We compare:
  - SCF total-energy box drift
  - matched single-state response against each kinetic family's own largest-box
    reference
  - m=2 low-energy subspace stability
  - common-interior rho / |psi0| / corrected V_eff drift
  - same-box old vs new direct differences

If the new kinetic shows clear improvements here, the next step should be a
small stricter-SCF recheck before considering broader rollout.
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
    from JaxDFT.src.solver import scf, solve_orbitals_subspace, total_energy, ion_ion_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from src.io import load_pseudopotentials
    from src.solver import scf, solve_orbitals_subspace, total_energy, ion_ion_energy


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
SUBSPACE_M = 2
FIXED_SOLVE_MAX_ITER = 16
FIXED_SOLVE_TOL = 1.0e-5
MAIN_PROBE_BOUND = 4.0
PROBE_SPACING = 0.3
NORM_TOL = 2.0e-2
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
        "perm": tuple(best_perm),
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
    projector_dist = float(np.sqrt(max(2.0 * m - 2.0 * float(np.sum(sigma * sigma)), 0.0)) / np.sqrt(2.0 * m))
    return {
        "min_sigma": min_sigma,
        "projector_dist": projector_dist,
    }


def eval_vloc_on_points(coords, pseudos, points):
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    coeff = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
    return build_local_potential_pointwise(coords, points, zion, rloc, coeff)


def run_scf_case(grid, coords, pseudos, occ, n_occ_bands, key_seed, kinetic_mode):
    backend = AdaptiveBackend(
        hartree_boundary_mode="uniform_exterior",
        kinetic_mode=kinetic_mode,
    )
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
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
    psi0 = normalize_field(
        backend,
        grid,
        jnp.asarray(eigvecs[:, 0], dtype=jnp.float32).reshape(grid.shape),
    )
    ion_e = ion_ion_energy(coords, zion)
    total_e = float(total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, ion_e, backend=backend))
    hartree_proxy = float(0.5 * backend.integrate(grid, rho * V_H))
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
        "eigvals_occ": np.asarray(jnp.asarray(eigvals), dtype=float),
        "total_energy": total_e,
        "orbital_norm_maxdev": abs(float(backend.inner_product(grid, psi0, psi0)) - 1.0),
        "electron_count": float(backend.integrate(grid, rho)),
        "hartree_proxy": hartree_proxy,
        "grid_shape": tuple(int(n) for n in grid.shape),
    }


def build_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    _, n_occ_bands, occ = build_occ(pseudos)
    grid_builder = AdaptiveBackend()
    rows = {label: [] for label in MODES}
    for mode_offset, (label, kinetic_mode) in enumerate(MODES.items()):
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
            key_seed = 4100 + 100 * mode_offset + box_idx
            rows[label].append(run_scf_case(grid, coords, pseudos, occ, n_occ_bands, key_seed, kinetic_mode))
    return coords, pseudos, rows


def analyze(rows_by_mode, coords, pseudos):
    _, probe_points, _ = build_probe_points(MAIN_PROBE_BOUND, PROBE_SPACING)
    probe_points = jnp.asarray(probe_points, dtype=jnp.float32)
    analyses = []
    refs = {}

    for label, rows in rows_by_mode.items():
        ref = rows[-1]
        ref_spec = solve_fixed_spectrum(
            ref["backend"],
            ref["grid"],
            pseudos,
            coords,
            ref["V_eff"],
            K_EIG,
            key_seed=5100 + (0 if label == "old" else 100),
            x_init_states=None,
        )
        ref_vloc_probe = eval_vloc_on_points(coords, pseudos, probe_points)
        ref_vh_probe = sample_adaptive_trilinear(ref["grid"], ref["V_H"], probe_points)
        ref_vxc_probe = sample_adaptive_trilinear(ref["grid"], ref["v_xc"], probe_points)
        ref_veff_probe = ref_vloc_probe + ref_vh_probe + ref_vxc_probe
        ref_psi0_probe = sample_adaptive_trilinear(ref["grid"], jnp.abs(ref["psi0"]), probe_points)
        ref_rho_probe = sample_adaptive_trilinear(ref["grid"], ref["rho"], probe_points)
        refs[label] = {
            "row": ref,
            "spec": ref_spec,
            "V_eff_probe": ref_veff_probe,
            "psi0_probe": ref_psi0_probe,
            "rho_probe": ref_rho_probe,
        }

    for box_idx, box_length in enumerate(BOXES[:-1]):
        row = {"box": box_length}
        same_box_store = {}
        for label in ("old", "new"):
            scf_row = rows_by_mode[label][box_idx]
            ref = refs[label]
            backend = scf_row["backend"]
            grid = scf_row["grid"]
            ref_states_on_current = sample_state_block_to_grid(ref["row"]["grid"], ref["spec"]["states"], grid, backend)
            spec = solve_fixed_spectrum(
                backend,
                grid,
                pseudos,
                coords,
                scf_row["V_eff"],
                K_EIG,
                key_seed=5300 + 100 * box_idx + (0 if label == "old" else 10),
                x_init_states=ref_states_on_current,
            )
            overlap = compute_overlap_matrix(backend, grid, spec["states"], ref_states_on_current)
            match = best_assignment(overlap)
            matched_idx = match["current_index_for_ref0"]

            matched_probe = sample_adaptive_trilinear(grid, jnp.abs(spec["states"][matched_idx]), probe_points)
            ref_state0_probe = sample_adaptive_trilinear(ref["row"]["grid"], jnp.abs(ref["spec"]["states"][0]), probe_points)
            matched_rho_probe = 2.0 * (matched_probe ** 2)
            ref_matched_rho_probe = 2.0 * (ref_state0_probe ** 2)
            psi_m = field_metrics(matched_probe, ref_state0_probe)
            rho_m = field_metrics(matched_rho_probe, ref_matched_rho_probe)

            rho_probe = sample_adaptive_trilinear(grid, scf_row["rho"], probe_points)
            psi0_probe = sample_adaptive_trilinear(grid, jnp.abs(scf_row["psi0"]), probe_points)
            vh_probe = sample_adaptive_trilinear(grid, scf_row["V_H"], probe_points)
            vxc_probe = sample_adaptive_trilinear(grid, scf_row["v_xc"], probe_points)
            vloc_direct_probe = eval_vloc_on_points(coords, pseudos, probe_points)
            veff_probe = vloc_direct_probe + vh_probe + vxc_probe

            rho_ctx = field_metrics(rho_probe, ref["rho_probe"])
            psi_ctx = field_metrics(psi0_probe, ref["psi0_probe"])
            veff_ctx = field_metrics(veff_probe, ref["V_eff_probe"])
            m2 = subspace_metrics(backend, grid, spec["states"], ref_states_on_current, SUBSPACE_M)

            row.update({
                f"total_energy_{label}": scf_row["total_energy"],
                f"eig0_{label}": float(scf_row["eigvals_occ"][0]),
                f"gap01_{label}": float(np.asarray(spec["eigvals"], dtype=float)[1] - np.asarray(spec["eigvals"], dtype=float)[0]),
                f"matched_overlap_ref0_{label}": match["overlap_for_ref0"],
                f"matched_psi_probe_rms_{label}": psi_m["rms"],
                f"matched_rho_probe_rms_{label}": rho_m["rms"],
                f"min_sigma_m2_{label}": m2["min_sigma"],
                f"projector_dist_m2_{label}": m2["projector_dist"],
                f"rho_probe_rms_{label}": rho_ctx["rms"],
                f"psi0_probe_rms_{label}": psi_ctx["rms"],
                f"V_eff_probe_rms_demeaned_{label}": veff_ctx["rms_demeaned"],
            })
            same_box_store[label] = {
                "backend": backend,
                "grid": grid,
                "spec": spec,
                "match": match,
                "scf_row": scf_row,
            }

        old_store = same_box_store["old"]
        new_store = same_box_store["new"]
        same_overlap_matrix = compute_overlap_matrix(
            old_store["backend"],
            old_store["grid"],
            old_store["spec"]["states"],
            new_store["spec"]["states"],
        )
        same_match = best_assignment(same_overlap_matrix)
        old_occ_idx = old_store["match"]["current_index_for_ref0"]
        new_occ_idx = new_store["match"]["current_index_for_ref0"]
        same_occ_overlap = float(same_overlap_matrix[old_occ_idx, new_occ_idx])
        row.update({
            "delta_energy_same_box": float(new_store["scf_row"]["total_energy"] - old_store["scf_row"]["total_energy"]),
            "same_box_matched_overlap_old_new": same_occ_overlap,
            "best_match_perm_old_new": same_match["perm"],
            "min_total_assignment_overlap_old_new": same_match["total_assignment_overlap"],
        })
        analyses.append(row)

    energy_drifts = {}
    scf_rows = []
    for label, rows in rows_by_mode.items():
        ref_energy = rows[-1]["total_energy"]
        energy_drifts[label] = [abs(row["total_energy"] - ref_energy) for row in rows[:-1]]
        for row in rows:
            scf_rows.append({
                "mode": label,
                "box": row["box"],
                "energy": row["total_energy"],
                "electron_count": row["electron_count"],
                "orbital_norm_maxdev": row["orbital_norm_maxdev"],
                "eig0": float(row["eigvals_occ"][0]),
                "hartree_proxy": row["hartree_proxy"],
                "grid_shape": row["grid_shape"],
            })
    return analyses, scf_rows, energy_drifts


def print_scf_result_table(rows):
    print("=== SCF Result Table ===")
    header = (
        f"{'Mode':>5} {'Box':>5} {'Energy':>12} {'Elec':>10} {'NormDev':>10} "
        f"{'eig0':>10} {'Hproxy':>10} {'grid.shape':>16}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:>5} {fmt_float(row['box'], 5, 1)} {fmt_float(row['energy'], 12, 6)} "
            f"{fmt_float(row['electron_count'], 10, 6)} {fmt_sci(row['orbital_norm_maxdev'], 10, 2)} "
            f"{fmt_float(row['eig0'], 10, 6)} {fmt_float(row['hartree_proxy'], 10, 6)} "
            f"{str(row['grid_shape']):>16}"
        )


def print_energy_drift_summary(energy_drifts):
    print("=== Energy Drift Summary ===")
    print(f"{'Box':>5} {'drift_old':>12} {'drift_new':>12}")
    print("-" * 31)
    for idx, box in enumerate(BOXES[:-1]):
        print(
            f"{fmt_float(box, 5, 1)} "
            f"{fmt_sci(energy_drifts['old'][idx], 12, 3)} "
            f"{fmt_sci(energy_drifts['new'][idx], 12, 3)}"
        )


def print_matched_table(rows):
    print("=== Matched Single-State Response Table ===")
    header = (
        f"{'Box':>5} {'ovlp_old':>10} {'ovlp_new':>10} {'psi_old':>10} {'psi_new':>10} "
        f"{'rho_old':>10} {'rho_new':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_float(row['matched_overlap_ref0_old'], 10, 6)} {fmt_float(row['matched_overlap_ref0_new'], 10, 6)} "
            f"{fmt_sci(row['matched_psi_probe_rms_old'], 10, 2)} {fmt_sci(row['matched_psi_probe_rms_new'], 10, 2)} "
            f"{fmt_sci(row['matched_rho_probe_rms_old'], 10, 2)} {fmt_sci(row['matched_rho_probe_rms_new'], 10, 2)}"
        )


def print_subspace_table(rows):
    print("=== m=2 Subspace Table ===")
    header = (
        f"{'Box':>5} {'sigma_old':>10} {'sigma_new':>10} {'proj_old':>10} {'proj_new':>10} "
        f"{'gap01_old':>10} {'gap01_new':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_float(row['min_sigma_m2_old'], 10, 6)} {fmt_float(row['min_sigma_m2_new'], 10, 6)} "
            f"{fmt_float(row['projector_dist_m2_old'], 10, 6)} {fmt_float(row['projector_dist_m2_new'], 10, 6)} "
            f"{fmt_float(row['gap01_old'], 10, 6)} {fmt_float(row['gap01_new'], 10, 6)}"
        )


def print_common_drift_table(rows):
    print("=== Common-Interior Drift Table ===")
    header = (
        f"{'Box':>5} {'rho_old':>10} {'rho_new':>10} {'psi_old':>10} {'psi_new':>10} "
        f"{'Veffdm_old':>12} {'Veffdm_new':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['rho_probe_rms_old'], 10, 2)} {fmt_sci(row['rho_probe_rms_new'], 10, 2)} "
            f"{fmt_sci(row['psi0_probe_rms_old'], 10, 2)} {fmt_sci(row['psi0_probe_rms_new'], 10, 2)} "
            f"{fmt_sci(row['V_eff_probe_rms_demeaned_old'], 12, 2)} {fmt_sci(row['V_eff_probe_rms_demeaned_new'], 12, 2)}"
        )


def print_same_box_table(rows):
    print("=== Same-Box Old vs New Table ===")
    header = (
        f"{'Box':>5} {'dE_new-old':>12} {'occ_overlap':>12} {'assign_overlap':>14} {'perm_old_new':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['delta_energy_same_box'], 12, 3)} "
            f"{fmt_float(row['same_box_matched_overlap_old_new'], 12, 6)} "
            f"{fmt_float(row['min_total_assignment_overlap_old_new'], 14, 6)} "
            f"{str(row['best_match_perm_old_new']):>14}"
        )


def summarize(rows, energy_drifts):
    max_energy_drift_old = max(energy_drifts["old"])
    max_energy_drift_new = max(energy_drifts["new"])
    min_sigma_old = min(row["min_sigma_m2_old"] for row in rows)
    min_sigma_new = min(row["min_sigma_m2_new"] for row in rows)
    max_proj_old = max(row["projector_dist_m2_old"] for row in rows)
    max_proj_new = max(row["projector_dist_m2_new"] for row in rows)
    worst_overlap_old = min(row["matched_overlap_ref0_old"] for row in rows)
    worst_overlap_new = min(row["matched_overlap_ref0_new"] for row in rows)
    max_rho_old = max(row["rho_probe_rms_old"] for row in rows)
    max_rho_new = max(row["rho_probe_rms_new"] for row in rows)
    max_psi_old = max(row["psi0_probe_rms_old"] for row in rows)
    max_psi_new = max(row["psi0_probe_rms_new"] for row in rows)
    max_veff_old = max(row["V_eff_probe_rms_demeaned_old"] for row in rows)
    max_veff_new = max(row["V_eff_probe_rms_demeaned_new"] for row in rows)

    energy_improved = max_energy_drift_new < max_energy_drift_old - 1.0e-4
    sigma_improved = min_sigma_new > min_sigma_old + 0.02
    proj_improved = max_proj_new < max_proj_old - 0.01
    overlap_improved = worst_overlap_new > worst_overlap_old + 0.01
    common_improvements = sum([
        max_rho_new < max_rho_old - 1.0e-5,
        max_psi_new < max_psi_old - 1.0e-4,
        max_veff_new < max_veff_old - 1.0e-4,
    ])
    spectral_ok = sigma_improved or proj_improved
    scf_ok = energy_improved or common_improvements >= 1
    improvement_groups = sum([
        energy_improved,
        sigma_improved,
        proj_improved,
        overlap_improved,
        common_improvements >= 1,
    ])

    if improvement_groups >= 3 and spectral_ok and scf_ok:
        label = "promising"
    elif improvement_groups >= 2 and spectral_ok:
        label = "mixed-but-positive"
    else:
        label = "no-clear-scf-gain"

    print("=== Diagnosis Summary ===")
    print(f"label: {label}")
    print(f"max_energy_drift_old = {max_energy_drift_old:.6e}")
    print(f"max_energy_drift_new = {max_energy_drift_new:.6e}")
    print(f"min_sigma_m2_old = {min_sigma_old:.6f}")
    print(f"min_sigma_m2_new = {min_sigma_new:.6f}")
    print(f"max_projector_dist_m2_old = {max_proj_old:.6f}")
    print(f"max_projector_dist_m2_new = {max_proj_new:.6f}")
    print(f"worst_matched_overlap_ref0_old = {worst_overlap_old:.6f}")
    print(f"worst_matched_overlap_ref0_new = {worst_overlap_new:.6f}")
    print(f"max_rho_probe_rms_old = {max_rho_old:.6e}")
    print(f"max_rho_probe_rms_new = {max_rho_new:.6e}")
    print(f"max_psi0_probe_rms_old = {max_psi_old:.6e}")
    print(f"max_psi0_probe_rms_new = {max_psi_new:.6e}")
    print(f"max_V_eff_probe_rms_demeaned_old = {max_veff_old:.6e}")
    print(f"max_V_eff_probe_rms_demeaned_new = {max_veff_new:.6e}")
    print(f"energy_improved = {energy_improved}")
    print(f"sigma_improved = {sigma_improved}")
    print(f"projector_improved = {proj_improved}")
    print(f"matched_overlap_improved = {overlap_improved}")
    print(f"common_drift_improvement_count = {common_improvements}")
    if label in {"promising", "mixed-but-positive"}:
        print("follow_up_suggestion: rerun a small-box subset with slightly stricter SCF settings before broader rollout.")
    return {
        "label": label,
        "energy_improved": energy_improved,
        "sigma_improved": sigma_improved,
        "proj_improved": proj_improved,
        "overlap_improved": overlap_improved,
        "common_improvements": common_improvements,
        "improvement_groups": improvement_groups,
    }


def main():
    print("H2 adaptive SCF kinetic comparison")
    print("This is not a final benchmark.")
    print("The goal is to compare old vs new adaptive kinetic operators under real adaptive SCF.")
    print()

    coords, pseudos, rows_by_mode = build_results()
    rows, scf_rows, energy_drifts = analyze(rows_by_mode, coords, pseudos)

    print_scf_result_table(scf_rows)
    print()
    print_energy_drift_summary(energy_drifts)
    print()
    print_matched_table(rows)
    print()
    print_subspace_table(rows)
    print()
    print_common_drift_table(rows)
    print()
    print_same_box_table(rows)
    print()
    summary = summarize(rows, energy_drifts)

    checks = []
    checks.append(check(
        "scf_norms_reasonable",
        all(row["orbital_norm_maxdev"] <= NORM_TOL for mode_rows in rows_by_mode.values() for row in mode_rows),
        f"max_norm_dev={max(row['orbital_norm_maxdev'] for mode_rows in rows_by_mode.values() for row in mode_rows):.3e}",
    ))
    checks.append(check(
        "spectral_metrics_finite",
        all(np.isfinite(row["matched_overlap_ref0_old"]) and np.isfinite(row["matched_overlap_ref0_new"]) for row in rows),
        f"worst_overlap={min(min(row['matched_overlap_ref0_old'], row['matched_overlap_ref0_new']) for row in rows):.6f}",
    ))
    checks.append(check(
        "summary_label_valid",
        summary["label"] in {"promising", "mixed-but-positive", "no-clear-scf-gain"},
        f"label={summary['label']}",
    ))

    overall = all(checks)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
