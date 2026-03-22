"""H2 main-solver vs replay parity audit for adaptive SCF.

This is not a final benchmark.
The goal is to locate the first stage where the real adaptive main solver path
diverges from a script-side replay, and to determine whether old/new kinetic
branch separation starts:
  - before orbital solve (x_init mismatch)
  - inside orbital solve
  - after rho_new before mixing
  - after mixing

This audit keeps the main adaptive H2 SCF setup as close as possible to the
current production path. In particular, it uses the same occupied-band count as
real H2 SCF, so the strict parity trace only has as many bands as the real
solver tracks.
"""

from __future__ import annotations

import os
import sys
from typing import Any

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
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import (
        _dirichlet_project_block,
        _dirichlet_project_field,
        anderson_mixing,
        scf,
        solve_orbitals_subspace,
    )
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.functional import lda_xc
    from src.io import load_pseudopotentials
    from src.solver import (
        _dirichlet_project_block,
        _dirichlet_project_field,
        anderson_mixing,
        scf,
        solve_orbitals_subspace,
    )


DISTANCE = 1.4
BOXES = [14.0, 22.0, 30.0]
H_MIN = 0.25
H_MAX = 0.80
R_CORE = 1.0
STRETCH_BETA = 5.0
SCF_KWARGS = {
    "max_iter": 8,
    "mix_alpha": 0.30,
    "tolerance": 5.0e-4,
}
MAIN_TRACE_MODE = "adaptive_debug"
KINETIC_MODES = {
    "old": "prototype_fd2",
    "new": "symmetric_fv",
}

PARITY_XINIT_TOL = 1.0e-5
PARITY_EIG_TOL = 1.0e-5
PARITY_RITZ_TOL = 1.0e-5
PARITY_RES_TOL = 1.0e-5
PARITY_STATE_OVERLAP_TOL = 0.9999
PARITY_RHO_TOL = 1.0e-5

OLD_NEW_XINIT_TOL = 1.0e-5
OLD_NEW_RITZ_TOL = 1.0e-5
OLD_NEW_RES_TOL = 1.0e-5
OLD_NEW_STATE_OVERLAP_THRESH = 0.7
OLD_NEW_RHO_TOL = 1.0e-4

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


def rms_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def overlap_state0(grid, backend, eigvecs_a, eigvecs_b):
    arr_a = jnp.asarray(eigvecs_a, dtype=jnp.float32)
    arr_b = jnp.asarray(eigvecs_b, dtype=jnp.float32)
    if arr_a.ndim != 2 or arr_b.ndim != 2 or arr_a.shape[1] == 0 or arr_b.shape[1] == 0:
        return None
    psi_a = arr_a[:, 0].reshape(grid.shape)
    psi_b = arr_b[:, 0].reshape(grid.shape)
    return float(jnp.abs(backend.inner_product(grid, psi_a, psi_b)))


def overlap_occupied_subspace(grid, backend, eigvecs_a, eigvecs_b, max_m=2):
    arr_a = jnp.asarray(eigvecs_a, dtype=jnp.float32)
    arr_b = jnp.asarray(eigvecs_b, dtype=jnp.float32)
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        return None
    m = min(int(arr_a.shape[1]), int(arr_b.shape[1]), int(max_m))
    if m <= 0:
        return None
    fields_a = jnp.moveaxis(arr_a[:, :m].reshape(grid.shape + (m,)), -1, 0)
    fields_b = jnp.moveaxis(arr_b[:, :m].reshape(grid.shape + (m,)), -1, 0)
    gram = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            gram[i, j] = float(backend.inner_product(grid, fields_a[i], fields_b[j]))
    sigma = np.linalg.svd(gram, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    return float(np.min(sigma))


def build_initial_rho(grid, coords, occ, backend):
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-2.0 * r ** 2)
    rho = _dirichlet_project_field(grid, rho)
    rho = rho / backend.integrate(grid, rho) * jnp.sum(occ)
    return rho


def normalize_eigvecs(grid, backend, eigvecs_flat):
    eigvecs = _dirichlet_project_block(grid, eigvecs_flat)
    n_bands = int(eigvecs.shape[1])
    eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_bands,)), -1, 0)
    norms = jnp.sqrt(jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields))
    eigvecs = eigvecs / norms[None, :]
    eigvecs = _dirichlet_project_block(grid, eigvecs)
    return eigvecs


def run_replay_trace(grid, coords, n_bands, occ, V_loc, projectors, key, backend):
    coords = jnp.asarray(coords, dtype=jnp.float32)
    rho = build_initial_rho(grid, coords, occ, backend)
    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    n_grid = rho.size
    proj_data = backend.precompute_nonlocal(grid, coords, projectors)
    eigvecs0 = jnp.zeros((n_grid, n_bands), dtype=jnp.float32)
    diff = jnp.array(jnp.inf, dtype=jnp.float32)
    trace = []

    for i in range(int(SCF_KWARGS["max_iter"])):
        if not bool(diff > SCF_KWARGS["tolerance"]):
            break

        rho_cur = jnp.clip(rho, 1e-12, None)
        rho_cur = _dirichlet_project_field(grid, rho_cur)
        V_H = backend.solve_hartree(grid, rho_cur)
        eps_xc, v_xc = lda_xc(rho_cur)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
            psi = _dirichlet_project_field(grid, psi)
            kinetic_psi = backend.apply_kinetic(grid, psi)
            v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
            hpsi = kinetic_psi + V_eff * psi + v_nonlocal
            hpsi = _dirichlet_project_field(grid, hpsi)
            return hpsi.reshape(-1)

        iter_key = jax.random.fold_in(key, jnp.asarray(i, dtype=jnp.int32))
        metric_backend = backend if getattr(backend, "name", None) != "uniform" else None
        metric_grid = grid if metric_backend is not None else None
        orbital_max_iter = 30 if metric_backend is None else 8
        orbital_tol = 1e-5 if metric_backend is None else 1e-4
        orbital_trace = []
        eigvals, eigvecs = solve_orbitals_subspace(
            apply_h,
            n_grid,
            n_bands,
            x_init=eigvecs0,
            max_iter=orbital_max_iter,
            tol=orbital_tol,
            key=iter_key,
            grid=metric_grid,
            backend=metric_backend,
            trace_sink=orbital_trace,
            trace_context={"scf_iter": int(i)},
        )
        eigvecs = normalize_eigvecs(grid, backend, eigvecs)
        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        rho_new = _dirichlet_project_field(grid, rho_new)
        diff = jnp.max(jnp.abs(rho_new - rho_cur))
        rho_flat, f_hist = anderson_mixing(
            rho_cur.reshape(-1), rho_new.reshape(-1), f_hist, SCF_KWARGS["mix_alpha"], i
        )
        rho_mixed = _dirichlet_project_field(grid, rho_flat.reshape(grid.shape))
        trace.append(
            {
                "scf_iter": int(i),
                "x_init": np.asarray(jnp.asarray(eigvecs0), dtype=np.float32),
                "x_init_norm": float(jnp.linalg.norm(eigvecs0)),
                "eigvals": np.asarray(jnp.asarray(eigvals[: min(4, eigvals.shape[0])]), dtype=np.float32),
                "eigvecs": np.asarray(jnp.asarray(eigvecs), dtype=np.float32),
                "state0_overlap_prev": overlap_state0(grid, backend, eigvecs0, eigvecs),
                "occupied_subspace_overlap_prev": overlap_occupied_subspace(grid, backend, eigvecs0, eigvecs, max_m=2),
                "rho_in": np.asarray(jnp.asarray(rho_cur), dtype=np.float32),
                "rho_new": np.asarray(jnp.asarray(rho_new), dtype=np.float32),
                "rho_mixed": np.asarray(jnp.asarray(rho_mixed), dtype=np.float32),
                "rho_update_norm": float(jnp.sqrt(jnp.mean((rho_new - rho_cur) ** 2))),
                "rho_mix_step_norm": float(jnp.sqrt(jnp.mean((rho_mixed - rho_cur) ** 2))),
                "rho_new_mixed_diff_norm": float(jnp.sqrt(jnp.mean((rho_mixed - rho_new) ** 2))),
                "orbital_trace": orbital_trace,
            }
        )
        rho = rho_mixed
        eigvecs0 = eigvecs
    return trace


def compare_orbital_traces(trace_a, trace_b, eig_tol, res_tol):
    n = min(len(trace_a), len(trace_b))
    for idx in range(n):
        row_a = trace_a[idx]
        row_b = trace_b[idx]
        if row_a.get("stage") != row_b.get("stage") or row_a.get("sub_iter") != row_b.get("sub_iter"):
            return {
                "sub_iter": idx,
                "detail": "stage-mismatch",
            }
        eig_rms = rms_diff(row_a.get("ritz_eigvals", []), row_b.get("ritz_eigvals", []))
        res_diff = abs(float(row_a.get("max_residual_norm", 0.0)) - float(row_b.get("max_residual_norm", 0.0)))
        if eig_rms > eig_tol or res_diff > res_tol:
            return {
                "sub_iter": int(row_a.get("sub_iter", idx)),
                "detail": f"eig_rms={eig_rms:.3e}, res_diff={res_diff:.3e}",
            }
    if len(trace_a) != len(trace_b):
        return {
            "sub_iter": n,
            "detail": "orbital-trace-length-mismatch",
        }
    return None


def classify_main_vs_replay(main_trace, replay_trace, grid, backend):
    n = min(len(main_trace), len(replay_trace))
    if len(main_trace) != len(replay_trace):
        return {
            "stage": "trace_length_mismatch",
            "iter": n,
            "detail": f"main={len(main_trace)}, replay={len(replay_trace)}",
        }
    for i in range(n):
        main_row = main_trace[i]
        replay_row = replay_trace[i]
        xinit_rms = rms_diff(main_row["x_init"], replay_row["x_init"])
        if xinit_rms > PARITY_XINIT_TOL:
            return {
                "stage": "before_orbital_solve",
                "iter": i,
                "detail": f"x_init_rms={xinit_rms:.3e}",
            }
        orbital_div = compare_orbital_traces(main_row["orbital_trace"], replay_row["orbital_trace"], PARITY_RITZ_TOL, PARITY_RES_TOL)
        if orbital_div is not None:
            return {
                "stage": "inside_orbital_solve",
                "iter": i,
                "detail": orbital_div["detail"],
            }
        eig_rms = rms_diff(main_row["eigvals"], replay_row["eigvals"])
        state_overlap = overlap_state0(grid, backend, main_row["eigvecs"], replay_row["eigvecs"])
        occ_sub_overlap = overlap_occupied_subspace(grid, backend, main_row["eigvecs"], replay_row["eigvecs"], max_m=2)
        if eig_rms > PARITY_EIG_TOL or (state_overlap is not None and state_overlap < PARITY_STATE_OVERLAP_TOL) or (occ_sub_overlap is not None and occ_sub_overlap < PARITY_STATE_OVERLAP_TOL):
            return {
                "stage": "inside_orbital_solve",
                "iter": i,
                "detail": f"eig_rms={eig_rms:.3e}, state_overlap={state_overlap}, occ_sub_overlap={occ_sub_overlap}",
            }
        rho_new_rms = rms_diff(main_row["rho_new"], replay_row["rho_new"])
        if rho_new_rms > PARITY_RHO_TOL:
            return {
                "stage": "after_rho_new_before_mixing",
                "iter": i,
                "detail": f"rho_new_rms={rho_new_rms:.3e}",
            }
        rho_mixed_rms = rms_diff(main_row["rho_mixed"], replay_row["rho_mixed"])
        if rho_mixed_rms > PARITY_RHO_TOL:
            return {
                "stage": "after_mixing",
                "iter": i,
                "detail": f"rho_mixed_rms={rho_mixed_rms:.3e}",
            }
    return {"stage": "no_divergence", "iter": None, "detail": "traces match within parity tolerances"}


def first_iter_where(rows, predicate):
    for row in rows:
        if predicate(row):
            return row["iter"]
    return None


def classify_old_new_main(main_old, main_new, grid, backend):
    iter_rows = []
    for i in range(min(len(main_old), len(main_new))):
        row_old = main_old[i]
        row_new = main_new[i]
        same_xinit_rms = rms_diff(row_old["x_init"], row_new["x_init"])
        same_occ_overlap = overlap_state0(grid, backend, row_old["eigvecs"], row_new["eigvecs"])
        same_occ_sub_overlap = overlap_occupied_subspace(grid, backend, row_old["eigvecs"], row_new["eigvecs"], max_m=2)
        same_rho_new_rms = rms_diff(row_old["rho_new"], row_new["rho_new"])
        same_rho_mixed_rms = rms_diff(row_old["rho_mixed"], row_new["rho_mixed"])
        orbital_div = compare_orbital_traces(row_old["orbital_trace"], row_new["orbital_trace"], OLD_NEW_RITZ_TOL, OLD_NEW_RES_TOL)
        iter_rows.append({
            "iter": i,
            "same_xinit_rms": same_xinit_rms,
            "same_occ_overlap": same_occ_overlap,
            "same_occ_sub_overlap": same_occ_sub_overlap,
            "same_rho_new_rms": same_rho_new_rms,
            "same_rho_mixed_rms": same_rho_mixed_rms,
            "old_state0_prev": row_old.get("state0_overlap_prev"),
            "new_state0_prev": row_new.get("state0_overlap_prev"),
            "old_occ_sub_prev": row_old.get("occupied_subspace_overlap_prev"),
            "new_occ_sub_prev": row_new.get("occupied_subspace_overlap_prev"),
            "old_eigvals": row_old.get("eigvals"),
            "new_eigvals": row_new.get("eigvals"),
            "orbital_div": orbital_div,
        })

    first_stage = {"stage": "no_divergence", "iter": None, "detail": "old/new traces remain aligned within thresholds"}
    for row in iter_rows:
        if row["same_xinit_rms"] > OLD_NEW_XINIT_TOL:
            first_stage = {
                "stage": "before_orbital_solve",
                "iter": row["iter"],
                "detail": f"x_init_rms={row['same_xinit_rms']:.3e}",
            }
            break
        if row["orbital_div"] is not None:
            first_stage = {
                "stage": "inside_orbital_solve",
                "iter": row["iter"],
                "detail": row["orbital_div"]["detail"],
            }
            break
        if row["same_occ_overlap"] is not None and row["same_occ_overlap"] < OLD_NEW_STATE_OVERLAP_THRESH:
            first_stage = {
                "stage": "inside_orbital_solve",
                "iter": row["iter"],
                "detail": f"occ_overlap={row['same_occ_overlap']:.6f}",
            }
            break
        if row["same_rho_new_rms"] > OLD_NEW_RHO_TOL:
            first_stage = {
                "stage": "after_rho_new_before_mixing",
                "iter": row["iter"],
                "detail": f"rho_new_rms={row['same_rho_new_rms']:.3e}",
            }
            break
        if row["same_rho_mixed_rms"] > OLD_NEW_RHO_TOL:
            first_stage = {
                "stage": "after_mixing",
                "iter": row["iter"],
                "detail": f"rho_mixed_rms={row['same_rho_mixed_rms']:.3e}",
            }
            break

    summary = {
        "first_stage": first_stage,
        "first_iter_old_new_occ_overlap_below_thresh": first_iter_where(iter_rows, lambda row: row["same_occ_overlap"] is not None and row["same_occ_overlap"] < OLD_NEW_STATE_OVERLAP_THRESH),
        "first_iter_old_new_occ_subspace_below_thresh": first_iter_where(iter_rows, lambda row: row["same_occ_sub_overlap"] is not None and row["same_occ_sub_overlap"] < OLD_NEW_STATE_OVERLAP_THRESH),
        "first_iter_old_new_xinit_divergence": first_iter_where(iter_rows, lambda row: row["same_xinit_rms"] > OLD_NEW_XINIT_TOL),
        "first_iter_old_new_rho_new_divergence": first_iter_where(iter_rows, lambda row: row["same_rho_new_rms"] > OLD_NEW_RHO_TOL),
        "first_iter_old_new_rho_mixed_divergence": first_iter_where(iter_rows, lambda row: row["same_rho_mixed_rms"] > OLD_NEW_RHO_TOL),
        "same_iter_old_new_occ_overlap_min": min((row["same_occ_overlap"] for row in iter_rows if row["same_occ_overlap"] is not None), default=None),
        "same_iter_old_new_occ_subspace_overlap_min": min((row["same_occ_sub_overlap"] for row in iter_rows if row["same_occ_sub_overlap"] is not None), default=None),
        "iter_rows": iter_rows,
    }
    return summary


def analyze_box(box_length, coords, pseudos, n_bands, occ):
    box = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    grid_builder = AdaptiveBackend(hartree_boundary_mode="uniform_exterior")
    grid = grid_builder.create_grid(
        spacing=H_MIN,
        box_size=box,
        atom_coords=coords,
        h_min=H_MIN,
        h_max=H_MAX,
        r_core=R_CORE,
        stretch_beta=STRETCH_BETA,
    )
    results = {}
    key = jax.random.PRNGKey(20260323 + int(round(box_length)))
    for label, kinetic_mode in KINETIC_MODES.items():
        backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior", kinetic_mode=kinetic_mode)
        V_loc = backend.build_local_potential(grid, coords, pseudos)
        main_trace = []
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
            grid,
            coords,
            n_bands,
            occ,
            V_loc,
            pseudos,
            key=key,
            backend=backend,
            trace_sink=main_trace,
            trace_mode=MAIN_TRACE_MODE,
            **SCF_KWARGS,
        )
        replay_trace = run_replay_trace(grid, coords, n_bands, occ, V_loc, pseudos, key, backend)
        parity = classify_main_vs_replay(main_trace, replay_trace, grid, backend)
        results[label] = {
            "backend": backend,
            "main_trace": main_trace,
            "replay_trace": replay_trace,
            "parity": parity,
            "final": {
                "rho": rho,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "V_H": V_H,
                "eps_xc": eps_xc,
                "v_xc": v_xc,
            },
        }

    old_new_main = classify_old_new_main(results["old"]["main_trace"], results["new"]["main_trace"], grid, results["old"]["backend"])
    return {
        "grid": grid,
        "results": results,
        "old_new_main": old_new_main,
    }


def print_trace_table(box_length, analysis):
    print(f"=== Iteration Trace Table (box={box_length:.1f}) ===")
    header = (
        f"{'it':>3} {'old_m eig':<12} {'old_r eig':<12} {'x_rms_o':>10} {'ov_mr_o':>9} {'subprev_o':>9} "
        f"{'new_m eig':<12} {'new_r eig':<12} {'x_rms_n':>10} {'ov_mr_n':>9} {'subprev_n':>9} {'ov_o_n':>9}"
    )
    print(header)
    print("-" * len(header))
    old_trace = analysis["results"]["old"]["main_trace"]
    old_replay = analysis["results"]["old"]["replay_trace"]
    new_trace = analysis["results"]["new"]["main_trace"]
    new_replay = analysis["results"]["new"]["replay_trace"]
    rows = analysis["old_new_main"]["iter_rows"]
    n = min(len(rows), len(old_trace), len(old_replay), len(new_trace), len(new_replay))
    for i in range(n):
        old_m = old_trace[i]
        old_r = old_replay[i]
        new_m = new_trace[i]
        new_r = new_replay[i]
        row = rows[i]
        old_m_e = float(np.asarray(old_m["eigvals"])[0]) if len(old_m["eigvals"]) else float("nan")
        old_r_e = float(np.asarray(old_r["eigvals"])[0]) if len(old_r["eigvals"]) else float("nan")
        new_m_e = float(np.asarray(new_m["eigvals"])[0]) if len(new_m["eigvals"]) else float("nan")
        new_r_e = float(np.asarray(new_r["eigvals"])[0]) if len(new_r["eigvals"]) else float("nan")
        print(
            f"{i:>3d} {old_m_e:<12.6f} {old_r_e:<12.6f} {fmt_sci(rms_diff(old_m['x_init'], old_r['x_init']), 10, 2)} "
            f"{fmt_float(overlap_state0(analysis['grid'], analysis['results']['old']['backend'], old_m['eigvecs'], old_r['eigvecs']), 9, 6)} "
            f"{fmt_float(old_m.get('occupied_subspace_overlap_prev'), 9, 6)} "
            f"{new_m_e:<12.6f} {new_r_e:<12.6f} {fmt_sci(rms_diff(new_m['x_init'], new_r['x_init']), 10, 2)} "
            f"{fmt_float(overlap_state0(analysis['grid'], analysis['results']['new']['backend'], new_m['eigvecs'], new_r['eigvecs']), 9, 6)} "
            f"{fmt_float(new_m.get('occupied_subspace_overlap_prev'), 9, 6)} "
            f"{fmt_float(row['same_occ_overlap'], 9, 6)}"
        )
    print()


def print_summary(box_length, analysis):
    print(f"=== Branch Switching Summary (box={box_length:.1f}) ===")
    old_parity = analysis["results"]["old"]["parity"]
    new_parity = analysis["results"]["new"]["parity"]
    old_new = analysis["old_new_main"]
    print(f"main_vs_replay_old: stage={old_parity['stage']}, iter={old_parity['iter']}, detail={old_parity['detail']}")
    print(f"main_vs_replay_new: stage={new_parity['stage']}, iter={new_parity['iter']}, detail={new_parity['detail']}")
    print(
        "main_old_vs_new: "
        f"stage={old_new['first_stage']['stage']}, iter={old_new['first_stage']['iter']}, detail={old_new['first_stage']['detail']}"
    )
    print(f"first_iter_old_new_occ_overlap_below_thresh={old_new['first_iter_old_new_occ_overlap_below_thresh']}")
    print(f"first_iter_old_new_occ_subspace_below_thresh={old_new['first_iter_old_new_occ_subspace_below_thresh']}")
    print(f"first_iter_old_new_xinit_divergence={old_new['first_iter_old_new_xinit_divergence']}")
    print(f"first_iter_old_new_rho_new_divergence={old_new['first_iter_old_new_rho_new_divergence']}")
    print(f"first_iter_old_new_rho_mixed_divergence={old_new['first_iter_old_new_rho_mixed_divergence']}")
    print(f"same_iter_old_new_occ_overlap_min={old_new['same_iter_old_new_occ_overlap_min']}")
    print(f"same_iter_old_new_occ_subspace_overlap_min={old_new['same_iter_old_new_occ_subspace_overlap_min']}")
    print()


def diagnose(box_results):
    parity_ok = True
    old_new_stages = []
    for analysis in box_results.values():
        old_stage = analysis["results"]["old"]["parity"]["stage"]
        new_stage = analysis["results"]["new"]["parity"]["stage"]
        parity_ok &= old_stage == "no_divergence"
        parity_ok &= new_stage == "no_divergence"
        old_new_stages.append(analysis["old_new_main"]["first_stage"]["stage"])

    if parity_ok and any(stage == "inside_orbital_solve" for stage in old_new_stages):
        label = "inside_orbital_solve"
        next_step = "prioritize solver.py internal orbital-subspace / root-selection investigation"
    elif parity_ok and any(stage == "after_mixing" for stage in old_new_stages):
        label = "after_mixing"
        next_step = "prioritize density mixing / fixed-point selection investigation"
    elif not parity_ok:
        label = "main_vs_replay_mismatch"
        next_step = "fix replay parity first before changing solver root-selection logic"
    else:
        label = "mixed"
        next_step = "inspect solver orbital-subspace and density-feedback coupling together"

    print("=== Diagnosis Summary ===")
    print(f"label: {label}")
    print(f"next_step: {next_step}")
    print()
    return label


def main():
    print("=== H2 Solver Parity Audit ===")
    print("This is not a final benchmark.")
    print("Goal: compare the real adaptive main solver path against a script-side replay and locate the first divergence stage.")
    print("Strict parity uses the actual H2 occupied-band count from the main solver path.")
    print()

    pseudos = load_pseudopotentials(["H", "H"], "pbe")
    _, n_bands, occ = build_occ(pseudos)
    coords = jnp.asarray([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)

    overall_ok = True
    box_results = {}
    for box in BOXES:
        analysis = analyze_box(box, coords, pseudos, n_bands, occ)
        box_results[box] = analysis
        print_trace_table(box, analysis)
        print_summary(box, analysis)

    label = diagnose(box_results)

    for box, analysis in box_results.items():
        old_stage = analysis["results"]["old"]["parity"]["stage"]
        new_stage = analysis["results"]["new"]["parity"]["stage"]
        overall_ok &= check(
            f"main_replay_parity_box_{int(box)}",
            old_stage == "no_divergence" and new_stage == "no_divergence",
            f"old={old_stage}, new={new_stage}",
        )

    overall_ok &= check(
        "diagnosis_produced",
        label in {"inside_orbital_solve", "after_mixing", "main_vs_replay_mismatch", "mixed"},
        f"label={label}",
    )

    print("=== Overall Summary ===")
    print(f"n_bands_strict_parity={n_bands}")
    print(f"label={label}")
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
