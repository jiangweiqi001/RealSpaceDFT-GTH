"""H2 common-interior V_eff decomposition study on the adaptive backend.

This is not a final benchmark.
The goal is to decompose the remaining common-interior drift of V_eff into:
  1. V_H-driven drift,
  2. v_xc-driven drift,
  3. and a possibly non-negligible V_loc / grid-sampling contribution.

The study keeps the current best Hartree path fixed (uniform_exterior) and
compares all boxes on the same interior probe regions.
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
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


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
PROBE_SPACING = 0.3
MAIN_PROBE_BOUND = 4.0
EXT_PROBE_BOUND = 6.0


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
    denom = np.maximum(x1 - x0, 1.0e-12)
    frac = np.clip((coord_np - x0) / denom, 0.0, 1.0)
    return lower, upper, frac


def sample_adaptive_trilinear(grid, field, probe_points):
    field_np = np.asarray(jnp.asarray(field), dtype=np.float64)
    points_np = np.asarray(jnp.asarray(probe_points), dtype=np.float64)
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
        raise ValueError(f"probe point {points_np[bad_idx].tolist()} lies outside adaptive box")

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


def probe_metrics(current, reference):
    cur = jnp.asarray(current, dtype=jnp.float32).reshape(-1)
    ref = jnp.asarray(reference, dtype=jnp.float32).reshape(-1)
    diff = cur - ref
    cur_c = cur - jnp.mean(cur)
    ref_c = ref - jnp.mean(ref)
    diff_c = cur_c - ref_c
    return {
        "rms": float(jnp.sqrt(jnp.mean(diff * diff))),
        "linf": float(jnp.max(jnp.abs(diff))),
        "rms_demeaned": float(jnp.sqrt(jnp.mean(diff_c * diff_c))),
        "mean_delta": float(jnp.mean(cur) - jnp.mean(ref)),
    }


def run_case(grid, coords, pseudos, zion, occ, n_bands, key_seed):
    backend = AdaptiveBackend(hartree_boundary_mode="uniform_exterior")
    result = {
        "box": float(grid.box_size[0]),
        "completed": False,
        "all_finite": False,
        "energy": None,
        "electron_count": None,
        "electron_error": None,
        "orbital_norm_maxdev": None,
        "eig0": None,
        "hartree_proxy": None,
        "shape": tuple(int(n) for n in grid.shape),
        "grid": grid,
        "V_loc": None,
        "V_H": None,
        "v_xc": None,
        "V_eff": None,
        "rho": None,
        "psi0_abs": None,
        "error": None,
    }
    try:
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
        psi0_abs = jnp.abs(eigvec_fields[0])
        electron_count = float(backend.integrate(grid, rho))
        norm_maxdev = float(jnp.max(jnp.abs(norms - 1.0)))
        V_eff = V_loc + V_H + v_xc
        hartree_proxy = float(0.5 * backend.integrate(grid, rho * V_H))
        all_finite = bool(
            jnp.all(jnp.isfinite(V_loc))
            and jnp.all(jnp.isfinite(V_H))
            and jnp.all(jnp.isfinite(v_xc))
            and jnp.all(jnp.isfinite(V_eff))
            and jnp.all(jnp.isfinite(rho))
            and jnp.all(jnp.isfinite(psi0_abs))
            and jnp.all(jnp.isfinite(eigvals))
            and jnp.isfinite(energy)
        )
        result.update({
            "completed": True,
            "all_finite": all_finite,
            "energy": float(energy),
            "electron_count": electron_count,
            "electron_error": abs(electron_count - float(jnp.sum(occ))),
            "orbital_norm_maxdev": norm_maxdev,
            "eig0": float(jnp.asarray(eigvals).reshape(-1)[0]),
            "hartree_proxy": hartree_proxy,
            "V_loc": V_loc,
            "V_H": V_H,
            "v_xc": v_xc,
            "V_eff": V_eff,
            "rho": rho,
            "psi0_abs": psi0_abs,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    _, n_bands, occ = build_occ(pseudos)
    grid_builder = AdaptiveBackend()
    results = []
    base_key = jax.random.PRNGKey(7)
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
        results.append(run_case(grid, coords, pseudos, zion, occ, n_bands, key_seed))
    return results


def build_probe_decomposition(rows, bound, spacing):
    _, probe_points, _ = build_probe_points(bound, spacing)
    reference = rows[-1]
    ref_samples = {
        "V_loc": sample_adaptive_trilinear(reference["grid"], reference["V_loc"], probe_points),
        "V_H": sample_adaptive_trilinear(reference["grid"], reference["V_H"], probe_points),
        "v_xc": sample_adaptive_trilinear(reference["grid"], reference["v_xc"], probe_points),
        "V_eff": sample_adaptive_trilinear(reference["grid"], reference["V_eff"], probe_points),
        "rho": sample_adaptive_trilinear(reference["grid"], reference["rho"], probe_points),
        "psi0_abs": sample_adaptive_trilinear(reference["grid"], reference["psi0_abs"], probe_points),
    }
    table = []
    for row in rows[:-1]:
        loc_m = probe_metrics(sample_adaptive_trilinear(row["grid"], row["V_loc"], probe_points), ref_samples["V_loc"])
        vh_m = probe_metrics(sample_adaptive_trilinear(row["grid"], row["V_H"], probe_points), ref_samples["V_H"])
        vxc_m = probe_metrics(sample_adaptive_trilinear(row["grid"], row["v_xc"], probe_points), ref_samples["v_xc"])
        veff_m = probe_metrics(sample_adaptive_trilinear(row["grid"], row["V_eff"], probe_points), ref_samples["V_eff"])
        rho_m = probe_metrics(sample_adaptive_trilinear(row["grid"], row["rho"], probe_points), ref_samples["rho"])
        psi_m = probe_metrics(sample_adaptive_trilinear(row["grid"], row["psi0_abs"], probe_points), ref_samples["psi0_abs"])
        table.append({
            "box": row["box"],
            "V_loc_probe_rms": loc_m["rms"],
            "V_loc_probe_rms_demeaned": loc_m["rms_demeaned"],
            "V_loc_probe_mean_delta": loc_m["mean_delta"],
            "V_H_probe_rms": vh_m["rms"],
            "V_H_probe_rms_demeaned": vh_m["rms_demeaned"],
            "V_H_probe_mean_delta": vh_m["mean_delta"],
            "v_xc_probe_rms": vxc_m["rms"],
            "v_xc_probe_rms_demeaned": vxc_m["rms_demeaned"],
            "v_xc_probe_mean_delta": vxc_m["mean_delta"],
            "V_eff_probe_rms": veff_m["rms"],
            "V_eff_probe_rms_demeaned": veff_m["rms_demeaned"],
            "V_eff_probe_mean_delta": veff_m["mean_delta"],
            "rho_probe_rms": rho_m["rms"],
            "rho_probe_rms_demeaned": rho_m["rms_demeaned"],
            "rho_probe_mean_delta": rho_m["mean_delta"],
            "psi0_probe_rms": psi_m["rms"],
        })
    return table


def print_scf_table(results):
    print("=== SCF Result Table ===")
    header = (
        f"{'Box':>5} {'Done':<5} {'Finite':<6} {'Energy':>12} {'N':>8} {'dN':>10} "
        f"{'NormDev':>10} {'eig0':>11} {'E_H_proxy':>12} {'Shape':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{fmt_float(row['box'], 5, 1)} {('PASS' if row['completed'] else 'FAIL'):<5} "
            f"{('PASS' if row['all_finite'] else 'FAIL'):<6} {fmt_float(row['energy'], 12, 6)} "
            f"{fmt_float(row['electron_count'], 8, 4)} {fmt_sci(row['electron_error'], 10, 2)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 10, 2)} {fmt_float(row['eig0'], 11, 6)} "
            f"{fmt_float(row['hartree_proxy'], 12, 6)} {str(row['shape']):<14}"
        )
        if row["error"] is not None:
            print(f"  error: {row['error']}")


def print_decomposition_table(title, rows):
    print(f"=== {title} ===")
    header = (
        f"{'Box':>5} {'Vloc_dm':>10} {'VH_dm':>10} {'Vxc_dm':>10} {'Veff_dm':>10} "
        f"{'rho_rms':>10} {'rho_dm':>10} {'rho_mean':>10} {'|psi0|_rms':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['V_loc_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['V_H_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['v_xc_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['V_eff_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['rho_probe_rms'], 10, 2)} "
            f"{fmt_sci(row['rho_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['rho_probe_mean_delta'], 10, 2)} "
            f"{fmt_sci(row['psi0_probe_rms'], 11, 2)}"
        )


def diagnose(main_rows):
    loc_dm = max(row["V_loc_probe_rms_demeaned"] for row in main_rows)
    vh_dm = max(row["V_H_probe_rms_demeaned"] for row in main_rows)
    vxc_dm = max(row["v_xc_probe_rms_demeaned"] for row in main_rows)
    veff_dm = max(row["V_eff_probe_rms_demeaned"] for row in main_rows)
    rho_dm = max(row["rho_probe_rms_demeaned"] for row in main_rows)
    psi_rms = max(row["psi0_probe_rms"] for row in main_rows)
    loc_mean = max(abs(row["V_loc_probe_mean_delta"]) for row in main_rows)
    veff_mean = max(abs(row["V_eff_probe_mean_delta"]) for row in main_rows)

    if vh_dm >= 1.5 * max(vxc_dm, loc_dm) and vh_dm >= 0.25 * veff_dm:
        label = "vh-dominated"
    elif vxc_dm >= 1.5 * max(vh_dm, loc_dm) and vxc_dm >= 0.25 * veff_dm:
        label = "vxc-dominated"
    else:
        label = "mixed"

    vloc_not_negligible = (loc_dm >= 0.3 * max(vh_dm, vxc_dm)) or (loc_mean >= 0.3 * max(veff_mean, 1.0e-12))
    return {
        "label": label,
        "vloc_not_negligible": vloc_not_negligible,
        "max_V_loc_probe_rms_demeaned": loc_dm,
        "max_V_H_probe_rms_demeaned": vh_dm,
        "max_v_xc_probe_rms_demeaned": vxc_dm,
        "max_V_eff_probe_rms_demeaned": veff_dm,
        "max_rho_probe_rms_demeaned": rho_dm,
        "max_|psi0|_probe_rms": psi_rms,
        "max_abs_V_loc_probe_mean_delta": loc_mean,
        "max_abs_V_eff_probe_mean_delta": veff_mean,
        "main_probe_box": "[-4,4]^3 requested; exact centered 0.3 lattice is [-3.9, 3.9]^3",
        "extended_probe_box": "[-6,6]^3",
    }


def print_diagnosis(summary):
    print("=== Diagnosis Summary ===")
    print(f"label                         : {summary['label']}")
    print(f"vloc-not-negligible           : {summary['vloc_not_negligible']}")
    print(f"max_V_loc_probe_rms_demeaned  : {summary['max_V_loc_probe_rms_demeaned']:.6e}")
    print(f"max_V_H_probe_rms_demeaned    : {summary['max_V_H_probe_rms_demeaned']:.6e}")
    print(f"max_v_xc_probe_rms_demeaned   : {summary['max_v_xc_probe_rms_demeaned']:.6e}")
    print(f"max_V_eff_probe_rms_demeaned  : {summary['max_V_eff_probe_rms_demeaned']:.6e}")
    print(f"max_rho_probe_rms_demeaned    : {summary['max_rho_probe_rms_demeaned']:.6e}")
    print(f"max_|psi0|_probe_rms          : {summary['max_|psi0|_probe_rms']:.6e}")
    print(f"max_abs_V_loc_mean_delta      : {summary['max_abs_V_loc_probe_mean_delta']:.6e}")
    print(f"max_abs_V_eff_mean_delta      : {summary['max_abs_V_eff_probe_mean_delta']:.6e}")
    print(f"main_probe_box                : {summary['main_probe_box']}")
    print(f"extended_probe_box            : {summary['extended_probe_box']}")


def main():
    print("H2 common-interior V_eff decomposition study")
    print("This is not a final benchmark.")
    print("The goal is to identify whether the remaining common-interior V_eff drift")
    print("looks more like V_H-driven, v_xc-driven, or mixed, while also checking")
    print("whether V_loc / grid sampling is already non-negligible.")
    print()
    print(f"Fixed setup: d={DISTANCE} Bohr, boxes={BOXES}")
    print(
        f"Adaptive params: h_min={H_MIN}, h_max={H_MAX}, "
        f"r_core={R_CORE}, stretch_beta={STRETCH_BETA}"
    )
    print("Hartree path: AdaptiveBackend(hartree_boundary_mode='uniform_exterior')")
    print()

    results = build_results()
    print_scf_table(results)
    print()

    main_rows = build_probe_decomposition(results, MAIN_PROBE_BOUND, PROBE_SPACING)
    ext_rows = build_probe_decomposition(results, EXT_PROBE_BOUND, PROBE_SPACING)
    print_decomposition_table("Main-Probe V_eff Decomposition Table", main_rows)
    print()
    print_decomposition_table("Extended-Probe V_eff Decomposition Table", ext_rows)
    print()

    summary = diagnose(main_rows)
    print_diagnosis(summary)
    print()

    all_completed = all(row["completed"] for row in results)
    all_finite = all(row["all_finite"] for row in results)
    electron_ok = all(
        row["electron_error"] is not None and row["electron_error"] <= ELECTRON_TOL
        for row in results if row["completed"]
    )
    norm_ok = all(
        row["orbital_norm_maxdev"] is not None and row["orbital_norm_maxdev"] <= NORM_TOL
        for row in results if row["completed"]
    )

    ok = True
    ok &= check("scf_completed", all_completed, f"all boxes completed={all_completed}")
    ok &= check("scf_finite", all_finite, f"all boxes finite={all_finite}")
    ok &= check("electron_count", electron_ok, f"electron error <= {ELECTRON_TOL}")
    ok &= check("orbital_norms", norm_ok, f"orbital norm maxdev <= {NORM_TOL}")
    ok &= check(
        "probe_rows",
        len(main_rows) == len(BOXES) - 1 and len(ext_rows) == len(BOXES) - 1,
        f"main_rows={len(main_rows)}, ext_rows={len(ext_rows)}",
    )

    print()
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
