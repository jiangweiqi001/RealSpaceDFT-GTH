"""H2 V_loc grid-vs-sampling diagnosis on the adaptive backend.

This is not a final benchmark.
The goal is to determine whether the remaining V_loc-related common-interior
issues are primarily caused by:
  1. grid-level V_loc tabulation on the adaptive tensor grid, or
  2. sampling / interpolation from the adaptive grid onto a common probe.

The study keeps the current best Hartree path fixed (uniform_exterior) and
uses the same H2/box settings as the earlier local-field studies.
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
    from JaxDFT.src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.hamiltonian import build_local_potential as build_local_potential_pointwise
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
CORE_RADIUS = 1.0
SMALL = 1.0e-12
FIELD_FLOOR = 1.0e-3


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
        "linf": float(jnp.max(jnp.abs(diff))),
        "mean_delta": float(jnp.mean(cur) - jnp.mean(ref)),
    }


def nearest_atom_distance(points, atom_coords):
    pts = jnp.asarray(points, dtype=jnp.float32).reshape(-1, 3)
    atoms = jnp.asarray(atom_coords, dtype=jnp.float32)
    diff = pts[:, None, :] - atoms[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    return jnp.min(dist, axis=1)


def masked_rms(diff, mask):
    diff = jnp.asarray(diff, dtype=jnp.float32).reshape(-1)
    mask = jnp.asarray(mask)
    if int(jnp.sum(mask)) == 0:
        return 0.0
    vals = diff[mask]
    return float(jnp.sqrt(jnp.mean(vals * vals)))


def eval_vloc_on_points(atom_coords, pseudos, points, shape):
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    coeff = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
    coords_grid = jnp.asarray(points, dtype=jnp.float32).reshape(shape + (3,))
    field = build_local_potential_pointwise(
        jnp.asarray(atom_coords, dtype=jnp.float32),
        coords_grid,
        zion,
        rloc,
        coeff,
    )
    return field.reshape(-1)


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
        electron_count = float(backend.integrate(grid, rho))
        norm_maxdev = float(jnp.max(jnp.abs(norms - 1.0)))
        hartree_proxy = float(0.5 * backend.integrate(grid, rho * V_H))
        all_finite = bool(
            jnp.all(jnp.isfinite(V_loc))
            and jnp.all(jnp.isfinite(rho))
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
    base_key = jax.random.PRNGKey(27)
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
    return coords, pseudos, results


def build_grid_level_table(coords, pseudos, rows):
    table = []
    for row in rows:
        grid = row["grid"]
        vloc_grid = jnp.asarray(row["V_loc"]).reshape(-1)
        vloc_direct = eval_vloc_on_points(coords, pseudos, grid.coords.reshape(-1, 3), grid.shape)
        metrics = field_metrics(vloc_grid, vloc_direct)
        nearest = nearest_atom_distance(grid.coords.reshape(-1, 3), coords)
        core_mask = nearest <= CORE_RADIUS
        outer_mask = nearest > CORE_RADIUS
        diff = jnp.asarray(vloc_grid - vloc_direct)
        table.append({
            "box": row["box"],
            "grid_vs_direct_rms": metrics["rms"],
            "grid_vs_direct_rms_demeaned": metrics["rms_demeaned"],
            "grid_vs_direct_linf": metrics["linf"],
            "grid_vs_direct_mean_delta": metrics["mean_delta"],
            "grid_vs_direct_core_rms": masked_rms(diff, core_mask),
            "grid_vs_direct_outer_rms": masked_rms(diff, outer_mask),
        })
    return table


def build_probe_level_table(coords, pseudos, rows, bound, spacing):
    _, probe_points, probe_shape = build_probe_points(bound, spacing)
    probe_points = jnp.asarray(probe_points, dtype=jnp.float32)
    vloc_direct = eval_vloc_on_points(coords, pseudos, probe_points, probe_shape)
    nearest = nearest_atom_distance(probe_points, coords)
    core_mask = nearest <= CORE_RADIUS
    outer_mask = nearest > CORE_RADIUS

    ref_sampled = sample_adaptive_trilinear(rows[-1]["grid"], rows[-1]["V_loc"], probe_points)
    ref_direct = vloc_direct

    table = []
    for row in rows[:-1]:
        sampled = sample_adaptive_trilinear(row["grid"], row["V_loc"], probe_points)
        sampled_ref = field_metrics(sampled, ref_sampled)
        direct_ref = field_metrics(vloc_direct, ref_direct)
        sampled_vs_direct = field_metrics(sampled, vloc_direct)
        diff = jnp.asarray(sampled - vloc_direct)
        table.append({
            "box": row["box"],
            "sampled_ref_rms": sampled_ref["rms"],
            "sampled_ref_rms_demeaned": sampled_ref["rms_demeaned"],
            "sampled_ref_linf": sampled_ref["linf"],
            "sampled_ref_mean_delta": sampled_ref["mean_delta"],
            "direct_ref_rms": direct_ref["rms"],
            "direct_ref_rms_demeaned": direct_ref["rms_demeaned"],
            "direct_ref_linf": direct_ref["linf"],
            "direct_ref_mean_delta": direct_ref["mean_delta"],
            "sampled_vs_direct_rms": sampled_vs_direct["rms"],
            "sampled_vs_direct_rms_demeaned": sampled_vs_direct["rms_demeaned"],
            "sampled_vs_direct_linf": sampled_vs_direct["linf"],
            "sampled_vs_direct_mean_delta": sampled_vs_direct["mean_delta"],
            "sampled_vs_direct_core_rms": masked_rms(diff, core_mask),
            "sampled_vs_direct_outer_rms": masked_rms(diff, outer_mask),
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


def print_grid_table(rows):
    print("=== Grid-Level V_loc Table ===")
    header = (
        f"{'Box':>5} {'rms':>10} {'dm_rms':>10} {'linf':>10} {'mean':>10} {'core_rms':>10} {'outer_rms':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['grid_vs_direct_rms'], 10, 2)} "
            f"{fmt_sci(row['grid_vs_direct_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['grid_vs_direct_linf'], 10, 2)} "
            f"{fmt_sci(row['grid_vs_direct_mean_delta'], 10, 2)} "
            f"{fmt_sci(row['grid_vs_direct_core_rms'], 10, 2)} "
            f"{fmt_sci(row['grid_vs_direct_outer_rms'], 11, 2)}"
        )


def print_probe_table(title, rows):
    print(f"=== {title} ===")
    header = (
        f"{'Box':>5} {'s_ref_dm':>10} {'d_ref_dm':>10} {'s-d_dm':>10} {'s-d_linf':>10} {'s-d_mean':>10} {'core_rms':>10} {'outer_rms':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['sampled_ref_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['direct_ref_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['sampled_vs_direct_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['sampled_vs_direct_linf'], 10, 2)} "
            f"{fmt_sci(row['sampled_vs_direct_mean_delta'], 10, 2)} "
            f"{fmt_sci(row['sampled_vs_direct_core_rms'], 10, 2)} "
            f"{fmt_sci(row['sampled_vs_direct_outer_rms'], 11, 2)}"
        )


def diagnose(grid_rows, probe_rows):
    grid_dm_max = max(row["grid_vs_direct_rms_demeaned"] for row in grid_rows)
    probe_dm_max = max(row["sampled_vs_direct_rms_demeaned"] for row in probe_rows)
    sampled_ref_dm_max = max(row["sampled_ref_rms_demeaned"] for row in probe_rows)
    direct_ref_dm_max = max(row["direct_ref_rms_demeaned"] for row in probe_rows)
    core_probe_max = max(row["sampled_vs_direct_core_rms"] for row in probe_rows)
    outer_probe_max = max(row["sampled_vs_direct_outer_rms"] for row in probe_rows)

    if grid_dm_max >= FIELD_FLOOR and grid_dm_max >= 1.5 * max(probe_dm_max, SMALL):
        label = "grid-level-tabulation"
    elif probe_dm_max >= FIELD_FLOOR and probe_dm_max >= 1.5 * max(grid_dm_max, SMALL) and direct_ref_dm_max < 1.0e-6:
        label = "probe-sampling"
    elif grid_dm_max >= FIELD_FLOOR and probe_dm_max >= FIELD_FLOOR and max(grid_dm_max, probe_dm_max) / max(min(grid_dm_max, probe_dm_max), SMALL) <= 1.5:
        label = "both"
    else:
        label = "weak"

    grid_level_suspected = label in {"grid-level-tabulation", "both"}
    probe_sampling_suspected = label in {"probe-sampling", "both"}
    return {
        "main_label": label,
        "grid_level_suspected": grid_level_suspected,
        "probe_sampling_suspected": probe_sampling_suspected,
        "grid_dm_max": grid_dm_max,
        "probe_dm_max": probe_dm_max,
        "sampled_ref_dm_max": sampled_ref_dm_max,
        "direct_ref_dm_max": direct_ref_dm_max,
        "core_probe_max": core_probe_max,
        "outer_probe_max": outer_probe_max,
    }


def print_summary(summary):
    print("=== Summary ===")
    print(f"main_label               : {summary['main_label']}")
    print(f"grid_level_suspected     : {summary['grid_level_suspected']}")
    print(f"probe_sampling_suspected : {summary['probe_sampling_suspected']}")
    print(f"grid_dm_max              : {summary['grid_dm_max']:.6e}")
    print(f"probe_dm_max             : {summary['probe_dm_max']:.6e}")
    print(f"sampled_ref_dm_max       : {summary['sampled_ref_dm_max']:.6e}")
    print(f"direct_ref_dm_max        : {summary['direct_ref_dm_max']:.6e}")
    print(f"core_probe_max           : {summary['core_probe_max']:.6e}")
    print(f"outer_probe_max          : {summary['outer_probe_max']:.6e}")


def main():
    print("H2 V_loc grid-vs-sampling diagnosis")
    print("This is not a final benchmark.")
    print("The goal is to decide whether the remaining V_loc issue is already present")
    print("on adaptive grid nodes, or mainly appears when V_loc is sampled to a common probe.")
    print()
    print(f"Fixed setup: d={DISTANCE} Bohr, boxes={BOXES}")
    print(
        f"Adaptive params: h_min={H_MIN}, h_max={H_MAX}, "
        f"r_core={R_CORE}, stretch_beta={STRETCH_BETA}"
    )
    print("Hartree path: AdaptiveBackend(hartree_boundary_mode='uniform_exterior')")
    print()

    coords, pseudos, results = build_results()
    print_scf_table(results)
    print()

    grid_rows = build_grid_level_table(coords, pseudos, results)
    main_probe_rows = build_probe_level_table(coords, pseudos, results, MAIN_PROBE_BOUND, PROBE_SPACING)
    ext_probe_rows = build_probe_level_table(coords, pseudos, results, EXT_PROBE_BOUND, PROBE_SPACING)
    print_grid_table(grid_rows)
    print()
    print_probe_table("Probe-Level V_loc Table", main_probe_rows)
    print()
    print_probe_table("Extended-Probe V_loc Table", ext_probe_rows)
    print()

    summary = diagnose(grid_rows, main_probe_rows)
    print_summary(summary)
    print()

    all_completed = all(row["completed"] for row in results)
    all_finite = all(row["all_finite"] for row in results)
    electron_ok = all(row["electron_error"] is not None and row["electron_error"] <= ELECTRON_TOL for row in results if row["completed"])
    norm_ok = all(row["orbital_norm_maxdev"] is not None and row["orbital_norm_maxdev"] <= NORM_TOL for row in results if row["completed"])
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

    print()
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
