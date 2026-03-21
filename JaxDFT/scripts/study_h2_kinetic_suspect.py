"""H2 common-interior consistency study under the best current Hartree path.

This is not a final benchmark.
The current goal is not to prove that the Laplacian is definitely the dominant
remaining error source. The goal is narrower:

  1. decide whether Hartree/effective-potential drift has already become small
     inside a common interior probe region, or
  2. decide that Hartree/effective-potential is still not sufficiently cleared.

If the interior drift of ``V_H`` and ``V_eff`` is already much smaller than the
interior drift of ``|psi0|`` and ``rho``, the next milestone should move toward
kinetic/Laplacian or the remaining finite-box eigenproblem rather than toward
more Hartree boundary tuning.
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
    axis = spacing * jnp.arange(-n, n + 1, dtype=jnp.float32)
    return axis


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

    x_min, x_max = float(x[0]), float(x[-1])
    y_min, y_max = float(y[0]), float(y[-1])
    z_min, z_max = float(z[0]), float(z[-1])
    outside = (
        (points_np[:, 0] < x_min)
        | (points_np[:, 0] > x_max)
        | (points_np[:, 1] < y_min)
        | (points_np[:, 1] > y_max)
        | (points_np[:, 2] < z_min)
        | (points_np[:, 2] > z_max)
    )
    if np.any(outside):
        bad_idx = int(np.argmax(outside))
        raise ValueError(
            f"probe point {points_np[bad_idx].tolist()} lies outside adaptive box "
            f"[{x_min}, {x_max}] x [{y_min}, {y_max}] x [{z_min}, {z_max}]"
        )

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
    rms = float(jnp.sqrt(jnp.mean(diff * diff)))
    linf = float(jnp.max(jnp.abs(diff)))
    cur_c = cur - jnp.mean(cur)
    ref_c = ref - jnp.mean(ref)
    diff_c = cur_c - ref_c
    rms_demeaned = float(jnp.sqrt(jnp.mean(diff_c * diff_c)))
    return {
        "rms": rms,
        "linf": linf,
        "rms_demeaned": rms_demeaned,
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
        "rho": None,
        "psi0_abs": None,
        "V_H": None,
        "V_eff": None,
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
        hartree_proxy = float(0.5 * backend.integrate(grid, rho * V_H))
        V_eff = V_loc + V_H + v_xc

        all_finite = bool(
            jnp.all(jnp.isfinite(rho))
            and jnp.all(jnp.isfinite(eigvals))
            and jnp.all(jnp.isfinite(eigvecs))
            and jnp.all(jnp.isfinite(V_H))
            and jnp.all(jnp.isfinite(V_eff))
            and jnp.isfinite(energy)
            and jnp.all(jnp.isfinite(norms))
        )
        result.update(
            {
                "completed": True,
                "all_finite": all_finite,
                "energy": float(energy),
                "electron_count": electron_count,
                "electron_error": abs(electron_count - float(jnp.sum(occ))),
                "orbital_norm_maxdev": norm_maxdev,
                "eig0": float(jnp.asarray(eigvals).reshape(-1)[0]),
                "hartree_proxy": hartree_proxy,
                "rho": rho,
                "psi0_abs": psi0_abs,
                "V_H": V_H,
                "V_eff": V_eff,
            }
        )
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
    base_key = jax.random.PRNGKey(42)

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


def build_probe_drift(rows, bound, spacing):
    _, probe_points, _ = build_probe_points(bound, spacing)
    reference = rows[-1]
    ref_samples = {
        "rho": sample_adaptive_trilinear(reference["grid"], reference["rho"], probe_points),
        "psi0_abs": sample_adaptive_trilinear(reference["grid"], reference["psi0_abs"], probe_points),
        "V_H": sample_adaptive_trilinear(reference["grid"], reference["V_H"], probe_points),
        "V_eff": sample_adaptive_trilinear(reference["grid"], reference["V_eff"], probe_points),
    }

    table = []
    for row in rows[:-1]:
        rho_metrics = probe_metrics(
            sample_adaptive_trilinear(row["grid"], row["rho"], probe_points),
            ref_samples["rho"],
        )
        psi_metrics = probe_metrics(
            sample_adaptive_trilinear(row["grid"], row["psi0_abs"], probe_points),
            ref_samples["psi0_abs"],
        )
        vh_metrics = probe_metrics(
            sample_adaptive_trilinear(row["grid"], row["V_H"], probe_points),
            ref_samples["V_H"],
        )
        veff_metrics = probe_metrics(
            sample_adaptive_trilinear(row["grid"], row["V_eff"], probe_points),
            ref_samples["V_eff"],
        )
        table.append(
            {
                "box": row["box"],
                "rho_probe_rms": rho_metrics["rms"],
                "rho_probe_linf": rho_metrics["linf"],
                "psi0_probe_rms": psi_metrics["rms"],
                "psi0_probe_linf": psi_metrics["linf"],
                "V_H_probe_rms": vh_metrics["rms"],
                "V_H_probe_rms_demeaned": vh_metrics["rms_demeaned"],
                "V_eff_probe_rms": veff_metrics["rms"],
                "V_eff_probe_rms_demeaned": veff_metrics["rms_demeaned"],
            }
        )
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


def print_probe_table(title, rows):
    print(f"=== {title} ===")
    header = (
        f"{'Box':>5} {'rho_rms':>12} {'rho_linf':>12} {'|psi0|_rms':>12} {'|psi0|_linf':>12} "
        f"{'V_H_rms':>12} {'V_H_dm_rms':>12} {'V_eff_rms':>12} {'V_eff_dm_rms':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_sci(row['rho_probe_rms'])} {fmt_sci(row['rho_probe_linf'])} "
            f"{fmt_sci(row['psi0_probe_rms'])} {fmt_sci(row['psi0_probe_linf'])} "
            f"{fmt_sci(row['V_H_probe_rms'])} {fmt_sci(row['V_H_probe_rms_demeaned'])} "
            f"{fmt_sci(row['V_eff_probe_rms'])} {fmt_sci(row['V_eff_probe_rms_demeaned'], 14, 3)}"
        )


def diagnose(main_rows):
    psi_max = max(row["psi0_probe_rms"] for row in main_rows)
    rho_max = max(row["rho_probe_rms"] for row in main_rows)
    vh_dm_max = max(row["V_H_probe_rms_demeaned"] for row in main_rows)
    veff_dm_max = max(row["V_eff_probe_rms_demeaned"] for row in main_rows)

    if (
        vh_dm_max >= 0.3 * psi_max
        and veff_dm_max >= 0.3 * psi_max
        and max(vh_dm_max, veff_dm_max) >= 1.0e-3
    ):
        label = "hartree-not-cleared"
    elif (
        vh_dm_max <= 0.1 * psi_max
        and veff_dm_max <= 0.25 * psi_max
        and max(psi_max, rho_max) >= 1.0e-3
    ):
        label = "kinetic-suspected-next"
    else:
        label = "mixed"

    return {
        "label": label,
        "max_rho_probe_rms": rho_max,
        "max_psi0_probe_rms": psi_max,
        "max_V_H_probe_rms_demeaned": vh_dm_max,
        "max_V_eff_probe_rms_demeaned": veff_dm_max,
        "main_probe_box": "[-4,4]^3 requested; exact centered 0.3 lattice is [-3.9, 3.9]^3",
        "extended_probe_box": "[-6,6]^3",
    }


def print_diagnosis(summary):
    print("=== Diagnosis Summary ===")
    print(f"label                      : {summary['label']}")
    print(f"max_rho_probe_rms          : {summary['max_rho_probe_rms']:.6e}")
    print(f"max_psi0_probe_rms         : {summary['max_psi0_probe_rms']:.6e}")
    print(f"max_V_H_probe_rms_demeaned : {summary['max_V_H_probe_rms_demeaned']:.6e}")
    print(f"max_V_eff_probe_rms_demean : {summary['max_V_eff_probe_rms_demeaned']:.6e}")
    print(f"main_probe_box             : {summary['main_probe_box']}")
    print(f"extended_probe_box         : {summary['extended_probe_box']}")


def main():
    print("H2 common-interior consistency study")
    print("This is not a final benchmark.")
    print("The current goal is to decide whether Hartree/effective-potential drift")
    print("has already become much smaller than |psi0|/rho drift inside a shared")
    print("interior probe region under the current best Hartree path.")
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

    all_completed = all(row["completed"] for row in results)
    all_finite = all(row["all_finite"] for row in results)
    electron_ok = all(
        row["electron_error"] is not None and row["electron_error"] <= ELECTRON_TOL
        for row in results
        if row["completed"]
    )
    norm_ok = all(
        row["orbital_norm_maxdev"] is not None and row["orbital_norm_maxdev"] <= NORM_TOL
        for row in results
        if row["completed"]
    )

    main_probe_rows = build_probe_drift(results, MAIN_PROBE_BOUND, PROBE_SPACING)
    ext_probe_rows = build_probe_drift(results, EXT_PROBE_BOUND, PROBE_SPACING)
    print()
    print_probe_table("Common-Interior Drift Table", main_probe_rows)
    print()
    print_probe_table("Extended-Probe Drift Table", ext_probe_rows)
    print()

    summary = diagnose(main_probe_rows)
    print_diagnosis(summary)
    print()

    ok = True
    ok &= check("scf_completed", all_completed, f"all boxes completed={all_completed}")
    ok &= check("scf_finite", all_finite, f"all boxes finite={all_finite}")
    ok &= check("electron_count", electron_ok, f"electron error <= {ELECTRON_TOL}")
    ok &= check("orbital_norms", norm_ok, f"orbital norm maxdev <= {NORM_TOL}")
    ok &= check(
        "probe_rows",
        len(main_probe_rows) == len(BOXES) - 1 and len(ext_probe_rows) == len(BOXES) - 1,
        f"main_rows={len(main_probe_rows)}, ext_rows={len(ext_probe_rows)}",
    )

    status = "PASS" if ok else "FAIL"
    print()
    print(f"OVERALL: {status}")


if __name__ == "__main__":
    main()
