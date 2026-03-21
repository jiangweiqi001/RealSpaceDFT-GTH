"""Fixed-density H2 Hartree oracle study for adaptive boundary/exterior attribution.

This is not a final benchmark.
The goal is to separate:
  1. Hartree boundary/exterior provider error, from
  2. the remaining SCF / kinetic / finite-box error.

Workflow:
  - run adaptive SCF with the current true-Dirichlet solver and
    ``multipole_dirichlet`` Hartree to obtain a converged density
  - freeze that density on the same adaptive grid
  - compare several Hartree boundary providers against a direct-Coulomb
    free-space oracle evaluated on the adaptive box faces

The oracle is intentionally kept out of the main SCF path. It is a diagnostic
tool for deciding whether the next milestone should stay on boundary/exterior
work or move to kinetic/Laplacian work.
"""

from __future__ import annotations

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
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
    from JaxDFT.src.grids.adaptive_poisson import (
        build_monopole_dirichlet_faces,
        build_multipole_dirichlet_faces,
        build_uniform_exterior_dirichlet_faces,
        solve_hartree_dirichlet_3d,
        solve_hartree_monopole_dirichlet_3d,
        solve_hartree_multipole_dirichlet_3d,
        solve_hartree_uniform_exterior_dirichlet_3d,
        solve_poisson_dirichlet_3d,
    )
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy
    from src.grids.adaptive_poisson import (
        build_monopole_dirichlet_faces,
        build_multipole_dirichlet_faces,
        build_uniform_exterior_dirichlet_faces,
        solve_hartree_dirichlet_3d,
        solve_hartree_monopole_dirichlet_3d,
        solve_hartree_multipole_dirichlet_3d,
        solve_hartree_uniform_exterior_dirichlet_3d,
        solve_poisson_dirichlet_3d,
    )


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
ELECTRON_TOL = 5.0e-3
NORM_TOL = 2.0e-2
ORACLE_CHARGE_RETAIN = 0.999999
ORACLE_BATCH_SIZE = 128


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


def build_zero_dirichlet_faces(grid):
    dtype = jnp.float32
    return {
        "x_lo": jnp.zeros((grid.shape[1] - 2, grid.shape[2] - 2), dtype=dtype),
        "x_hi": jnp.zeros((grid.shape[1] - 2, grid.shape[2] - 2), dtype=dtype),
        "y_lo": jnp.zeros((grid.shape[0] - 2, grid.shape[2] - 2), dtype=dtype),
        "y_hi": jnp.zeros((grid.shape[0] - 2, grid.shape[2] - 2), dtype=dtype),
        "z_lo": jnp.zeros((grid.shape[0] - 2, grid.shape[1] - 2), dtype=dtype),
        "z_hi": jnp.zeros((grid.shape[0] - 2, grid.shape[1] - 2), dtype=dtype),
    }


def build_face_area_weights(grid):
    wx = jnp.asarray(grid.wx)
    wy = jnp.asarray(grid.wy)
    wz = jnp.asarray(grid.wz)
    return {
        "x_lo": wy[1:-1, None] * wz[1:-1][None, :],
        "x_hi": wy[1:-1, None] * wz[1:-1][None, :],
        "y_lo": wx[1:-1, None] * wz[1:-1][None, :],
        "y_hi": wx[1:-1, None] * wz[1:-1][None, :],
        "z_lo": wx[1:-1, None] * wy[1:-1][None, :],
        "z_hi": wx[1:-1, None] * wy[1:-1][None, :],
    }


def flatten_face_dict(face_dict):
    keys = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    return jnp.concatenate([jnp.ravel(jnp.asarray(face_dict[key])) for key in keys], axis=0)


def flatten_face_weights(face_weights):
    keys = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    return jnp.concatenate([jnp.ravel(jnp.asarray(face_weights[key])) for key in keys], axis=0)


def face_error_metrics(mode_faces, oracle_faces, face_weights):
    mode_flat = flatten_face_dict(mode_faces)
    oracle_flat = flatten_face_dict(oracle_faces)
    weight_flat = flatten_face_weights(face_weights)
    diff = mode_flat - oracle_flat
    linf_abs = float(jnp.max(jnp.abs(diff)))
    linf_rel = linf_abs / max(float(jnp.max(jnp.abs(oracle_flat))), 1.0e-12)
    num = float(jnp.sum(weight_flat * diff * diff))
    den = float(jnp.sum(weight_flat * oracle_flat * oracle_flat))
    l2_rel = float(np.sqrt(num / max(den, 1.0e-30)))
    return {
        "face_linf_abs": linf_abs,
        "face_linf_rel": linf_rel,
        "face_l2_rel": l2_rel,
    }


def min_positive_spacing(grid):
    h_all = jnp.concatenate([jnp.ravel(grid.hx), jnp.ravel(grid.hy), jnp.ravel(grid.hz)])
    h_pos = h_all[h_all > 0.0]
    return float(jnp.min(h_pos))


def compress_oracle_sources(points, charges, retain_fraction=ORACLE_CHARGE_RETAIN):
    abs_q = np.abs(charges)
    total_abs = float(np.sum(abs_q))
    if total_abs <= 0.0:
        return points[:1], charges[:1], {
            "n_sources_total": int(points.shape[0]),
            "n_sources_retained": 1,
            "charge_total_abs": total_abs,
            "charge_retained_abs": 0.0,
            "charge_retained_fraction": 1.0,
        }

    order = np.argsort(abs_q)[::-1]
    sorted_abs = abs_q[order]
    cumulative = np.cumsum(sorted_abs)
    cutoff = retain_fraction * total_abs
    n_keep = int(np.searchsorted(cumulative, cutoff, side="left")) + 1
    keep = order[:n_keep]
    retained_abs = float(np.sum(abs_q[keep]))
    diagnostics = {
        "n_sources_total": int(points.shape[0]),
        "n_sources_retained": int(n_keep),
        "charge_total_abs": total_abs,
        "charge_retained_abs": retained_abs,
        "charge_retained_fraction": retained_abs / max(total_abs, 1.0e-30),
    }
    return points[keep], charges[keep], diagnostics


def evaluate_coulomb_points(source_points, source_charges, target_points, r_cut, batch_size=ORACLE_BATCH_SIZE):
    values = np.zeros(target_points.shape[0], dtype=np.float64)
    for start in range(0, target_points.shape[0], batch_size):
        stop = min(start + batch_size, target_points.shape[0])
        targets = target_points[start:stop]
        diff = source_points[:, None, :] - targets[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        values[start:stop] = np.sum(source_charges[:, None] / np.maximum(dist, r_cut), axis=0)
    return values


def build_oracle_dirichlet_faces(grid, rho, *, retain_fraction=ORACLE_CHARGE_RETAIN, batch_size=ORACLE_BATCH_SIZE):
    coords = np.asarray(grid.coords, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(jnp.asarray(grid.volume_weights) * jnp.asarray(rho), dtype=np.float64).reshape(-1)
    src_points, src_charges, src_diag = compress_oracle_sources(coords, charges, retain_fraction=retain_fraction)
    r_cut = 0.5 * min_positive_spacing(grid)

    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    z = np.asarray(grid.z, dtype=np.float64)
    x_int = x[1:-1]
    y_int = y[1:-1]
    z_int = z[1:-1]

    y_grid_x, z_grid_x = np.meshgrid(y_int, z_int, indexing="ij")
    x_grid_y, z_grid_y = np.meshgrid(x_int, z_int, indexing="ij")
    x_grid_z, y_grid_z = np.meshgrid(x_int, y_int, indexing="ij")

    face_points = {
        "x_lo": np.stack([np.full_like(y_grid_x, x[0]), y_grid_x, z_grid_x], axis=-1),
        "x_hi": np.stack([np.full_like(y_grid_x, x[-1]), y_grid_x, z_grid_x], axis=-1),
        "y_lo": np.stack([x_grid_y, np.full_like(x_grid_y, y[0]), z_grid_y], axis=-1),
        "y_hi": np.stack([x_grid_y, np.full_like(x_grid_y, y[-1]), z_grid_y], axis=-1),
        "z_lo": np.stack([x_grid_z, y_grid_z, np.full_like(x_grid_z, z[0])], axis=-1),
        "z_hi": np.stack([x_grid_z, y_grid_z, np.full_like(x_grid_z, z[-1])], axis=-1),
    }

    faces = {}
    for key, pts in face_points.items():
        flat_pts = pts.reshape(-1, 3)
        vals = evaluate_coulomb_points(src_points, src_charges, flat_pts, r_cut=r_cut, batch_size=batch_size)
        faces[key] = jnp.asarray(vals.reshape(pts.shape[:-1]), dtype=jnp.float32)

    diagnostics = {
        "boundary_model": "oracle_dirichlet_faces",
        "r_cut": float(r_cut),
        "retain_fraction_target": float(retain_fraction),
        "batch_size": int(batch_size),
        "charge_total": float(np.sum(charges)),
    }
    diagnostics.update(src_diag)
    return faces, diagnostics


def run_scf_box(grid, coords, pseudos, zion, occ, n_bands, key_seed):
    backend = AdaptiveBackend(hartree_boundary_mode="multipole_dirichlet")
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid,
        coords,
        n_bands,
        occ,
        V_loc,
        pseudos,
        key=jax.random.PRNGKey(key_seed),
        backend=backend,
        **SCF_KWARGS,
    )
    ion_e = ion_ion_energy(coords, zion)
    energy = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, ion_e, backend=backend)
    eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_bands,)), -1, 0)
    norms = jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields)
    result = {
        "box": float(grid.box_size[0]),
        "energy": float(energy),
        "electron_count": float(backend.integrate(grid, rho)),
        "electron_error": abs(float(backend.integrate(grid, rho)) - float(jnp.sum(occ))),
        "orbital_norm_maxdev": float(jnp.max(jnp.abs(norms - 1.0))),
        "eig0": float(jnp.asarray(eigvals).reshape(-1)[0]),
        "shape": tuple(int(n) for n in grid.shape),
        "rho": rho,
        "V_H_scf": V_H,
    }
    return result


def run_provider_study(grid, rho):
    rhs = (4.0 * jnp.pi) * jnp.asarray(rho)
    face_weights = build_face_area_weights(grid)

    zero_faces = build_zero_dirichlet_faces(grid)
    zero_V, zero_diag = solve_hartree_dirichlet_3d(grid, rho)

    mono_faces, mono_face_diag = build_monopole_dirichlet_faces(grid, rho, center_mode="charge_center")
    mono_V, mono_diag = solve_hartree_monopole_dirichlet_3d(grid, rho, center_mode="charge_center")

    multi_faces, multi_face_diag = build_multipole_dirichlet_faces(grid, rho)
    multi_V, multi_diag = solve_hartree_multipole_dirichlet_3d(grid, rho)

    uniform_faces, uniform_face_diag = build_uniform_exterior_dirichlet_faces(grid, rho)
    uniform_V, uniform_diag = solve_hartree_uniform_exterior_dirichlet_3d(grid, rho)

    oracle_faces, oracle_face_diag = build_oracle_dirichlet_faces(grid, rho)
    oracle_V, oracle_diag = solve_poisson_dirichlet_3d(grid, rhs, boundary_faces=oracle_faces)

    rows = []
    provider_specs = [
        ("zero_dirichlet", zero_faces, zero_V, {**zero_diag}),
        ("monopole_charge_center", mono_faces, mono_V, {**mono_diag, **mono_face_diag}),
        ("multipole_dirichlet", multi_faces, multi_V, {**multi_diag, **multi_face_diag}),
        ("uniform_exterior", uniform_faces, uniform_V, {**uniform_diag, **uniform_face_diag}),
        ("oracle", oracle_faces, oracle_V, {**oracle_diag, **oracle_face_diag}),
    ]

    oracle_proxy = float(grid.integrate(jnp.asarray(rho) * oracle_V) * 0.5)
    for mode, faces, V_H, diagnostics in provider_specs:
        metrics = face_error_metrics(faces, oracle_faces, face_weights)
        eh_proxy = float(grid.integrate(jnp.asarray(rho) * V_H) * 0.5)
        rows.append({
            "mode": mode,
            "box": float(grid.box_size[0]),
            "all_finite": bool(jnp.all(jnp.isfinite(V_H))),
            "hartree_proxy": eh_proxy,
            "hartree_proxy_delta": eh_proxy - oracle_proxy,
            "face_linf_abs": metrics["face_linf_abs"],
            "face_linf_rel": metrics["face_linf_rel"],
            "face_l2_rel": metrics["face_l2_rel"],
            "charge_total": float(oracle_face_diag["charge_total"]),
            "r_cut": float(oracle_face_diag["r_cut"]),
            "oracle_sources_retained": int(oracle_face_diag["n_sources_retained"]),
            "oracle_sources_total": int(oracle_face_diag["n_sources_total"]),
            "oracle_charge_retained_fraction": float(oracle_face_diag["charge_retained_fraction"]),
        })
    return rows


def build_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    _, n_bands, occ = build_occ(pseudos)
    grid_builder = AdaptiveBackend()
    base_key = jax.random.PRNGKey(42)

    scf_rows = []
    provider_rows = []
    for box_idx, box_length in enumerate(BOXES):
        print(f"--- SCF on box {box_length:.1f} Bohr (multipole_dirichlet) ---")
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
        scf_row = run_scf_box(grid, coords, pseudos, zion, occ, n_bands, key_seed)
        scf_rows.append(scf_row)

        print(f"--- Frozen-density Hartree providers on box {box_length:.1f} Bohr ---")
        provider_rows.extend(run_provider_study(grid, scf_row["rho"]))

    return scf_rows, provider_rows


def print_scf_table(rows):
    print("=== SCF Result Table ===")
    header = (
        f"{'Box':>5} {'Energy':>12} {'N':>8} {'dN':>10} {'NormDev':>10} "
        f"{'eig0':>11} {'Shape':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_float(row['energy'], 12, 6)} "
            f"{fmt_float(row['electron_count'], 8, 4)} {fmt_sci(row['electron_error'], 10, 2)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 10, 2)} {fmt_float(row['eig0'], 11, 6)} {str(row['shape']):<14}"
        )


def print_provider_table(rows):
    print("=== Hartree Provider vs Oracle Table ===")
    header = (
        f"{'Mode':<24} {'Box':>5} {'Finite':<6} {'E_H_proxy':>12} {'dEH_oracle':>12} "
        f"{'LinfAbs':>12} {'LinfRel':>12} {'L2Rel':>12} {'r_cut':>10} {'SrcKeep':>8} {'KeepFrac':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<24} {fmt_float(row['box'], 5, 1)} "
            f"{('PASS' if row['all_finite'] else 'FAIL'):<6} "
            f"{fmt_float(row['hartree_proxy'], 12, 6)} {fmt_sci(row['hartree_proxy_delta'], 12, 3)} "
            f"{fmt_sci(row['face_linf_abs'], 12, 3)} {fmt_sci(row['face_linf_rel'], 12, 3)} "
            f"{fmt_sci(row['face_l2_rel'], 12, 3)} {fmt_sci(row['r_cut'], 10, 3)} "
            f"{str(row['oracle_sources_retained']).rjust(8)} {fmt_sci(row['oracle_charge_retained_fraction'], 10, 3)}"
        )


def build_box_drift_summary(scf_rows, provider_rows):
    summary = []
    scf_ref = scf_rows[-1]["energy"]
    scf_max_drift = max(abs(row["energy"] - scf_ref) for row in scf_rows)

    by_mode = {}
    for row in provider_rows:
        by_mode.setdefault(row["mode"], []).append(row)
    for rows in by_mode.values():
        rows.sort(key=lambda r: r["box"])

    for mode, rows in by_mode.items():
        ref = rows[-1]["hartree_proxy"]
        max_drift = max(abs(row["hartree_proxy"] - ref) for row in rows)
        max_delta_oracle = max(abs(row["hartree_proxy_delta"]) for row in rows)
        max_l2_rel = max(row["face_l2_rel"] for row in rows)
        summary.append({
            "mode": mode,
            "max_proxy_box_drift": max_drift,
            "max_proxy_vs_oracle": max_delta_oracle,
            "max_face_l2_rel": max_l2_rel,
            "scf_max_energy_drift": scf_max_drift,
        })
    return summary


def print_box_drift_summary(summary):
    print("=== Box Drift vs Oracle Summary ===")
    header = (
        f"{'Mode':<24} {'Max|dEH_box|':>14} {'Max|dEH_oracle|':>16} "
        f"{'MaxFaceL2Rel':>14} {'SCF_Max|dE|':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['mode']:<24} {fmt_sci(row['max_proxy_box_drift'], 14, 3)} "
            f"{fmt_sci(row['max_proxy_vs_oracle'], 16, 3)} {fmt_sci(row['max_face_l2_rel'], 14, 3)} "
            f"{fmt_sci(row['scf_max_energy_drift'], 12, 3)}"
        )


def diagnose(scf_rows, provider_rows):
    summary = build_box_drift_summary(scf_rows, provider_rows)
    by_mode = {row["mode"]: row for row in summary}
    scf_max_drift = summary[0]["scf_max_energy_drift"] if summary else None

    multi = by_mode["multipole_dirichlet"]
    uniform = by_mode["uniform_exterior"]
    best_provider_gap = min(multi["max_proxy_vs_oracle"], uniform["max_proxy_vs_oracle"])
    best_face_gap = min(multi["max_face_l2_rel"], uniform["max_face_l2_rel"])
    uniform_beats_multi = uniform["max_proxy_vs_oracle"] < multi["max_proxy_vs_oracle"]

    if (
        best_provider_gap <= 0.10 * max(scf_max_drift, 1.0e-8)
        and best_face_gap <= 5.0e-2
    ):
        label = "boundary-provider-saturated"
    elif (
        best_provider_gap >= 0.40 * max(scf_max_drift, 1.0e-8)
        or best_face_gap >= 1.5e-1
    ):
        label = "boundary/exterior-dominated"
    else:
        label = "mixed"

    return {
        "label": label,
        "scf_max_energy_drift": scf_max_drift,
        "multi_max_proxy_vs_oracle": multi["max_proxy_vs_oracle"],
        "uniform_max_proxy_vs_oracle": uniform["max_proxy_vs_oracle"],
        "multi_max_face_l2_rel": multi["max_face_l2_rel"],
        "uniform_max_face_l2_rel": uniform["max_face_l2_rel"],
        "uniform_beats_multi": uniform_beats_multi,
    }


def print_diagnosis(diag):
    print("=== Diagnosis Summary ===")
    print(f"label                        = {diag['label']}")
    print(f"scf_max_energy_drift         = {diag['scf_max_energy_drift']:.6f}")
    print(f"multi_max_proxy_vs_oracle    = {diag['multi_max_proxy_vs_oracle']:.6f}")
    print(f"uniform_max_proxy_vs_oracle  = {diag['uniform_max_proxy_vs_oracle']:.6f}")
    print(f"multi_max_face_l2_rel        = {diag['multi_max_face_l2_rel']:.6e}")
    print(f"uniform_max_face_l2_rel      = {diag['uniform_max_face_l2_rel']:.6e}")
    print(f"uniform_beats_multi          = {diag['uniform_beats_multi']}")
    print()
    print("Interpret conservatively:")
    print("- If multipole/uniform_exterior are both already close to the oracle but total-energy box drift is still large, shift attention to kinetic/Laplacian or the remaining finite-box eigenproblem.")
    print("- If the best nonzero provider is still far from the oracle, keep improving boundary/exterior treatment before touching Laplacian.")
    print("- If the evidence is mixed, avoid over-claiming a single dominant source.")


def main():
    print("=== H2 Fixed-Density Hartree Oracle Study ===")
    print("Note: this is not a final benchmark.")
    print("Note: the current goal is to separate Hartree boundary/exterior provider error from the remaining SCF / kinetic / finite-box error.")
    print("Note: the oracle is a diagnostic free-space-like direct-Coulomb boundary evaluator on the adaptive box faces; it is not part of the main SCF path.")
    print(f"Setup: d={DISTANCE} Bohr, boxes={BOXES}")
    print(
        "Adaptive params: "
        f"h_min={H_MIN}, h_max={H_MAX}, r_core={R_CORE}, stretch_beta={STRETCH_BETA}"
    )
    print(
        "SCF: "
        f"max_iter={SCF_KWARGS['max_iter']}, mix_alpha={SCF_KWARGS['mix_alpha']}, "
        f"tolerance={SCF_KWARGS['tolerance']}"
    )
    print(
        "Oracle controls: "
        f"retain_fraction={ORACLE_CHARGE_RETAIN}, batch_size={ORACLE_BATCH_SIZE}"
    )
    print()

    scf_rows, provider_rows = build_results()
    print()
    print_scf_table(scf_rows)
    print()
    print_provider_table(provider_rows)
    print()
    summary = build_box_drift_summary(scf_rows, provider_rows)
    print_box_drift_summary(summary)
    print()
    diag = diagnose(scf_rows, provider_rows)
    print_diagnosis(diag)

    ok = True
    ok &= check(
        "scf_runs_ok",
        all(
            jnp.isfinite(row["energy"])
            and abs(row["electron_error"]) <= ELECTRON_TOL
            and row["orbital_norm_maxdev"] <= NORM_TOL
            for row in scf_rows
        ),
        "all SCF boxes completed with finite energies, correct electron counts, and reasonable orbital norms",
    )
    ok &= check(
        "provider_runs_ok",
        all(row["all_finite"] for row in provider_rows),
        "all frozen-density Hartree providers produced finite potentials",
    )
    ok &= check(
        "oracle_charge_retention_ok",
        all(row["oracle_charge_retained_fraction"] >= ORACLE_CHARGE_RETAIN for row in provider_rows if row["mode"] == "oracle"),
        "oracle retained the requested fraction of absolute source charge",
    )
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
