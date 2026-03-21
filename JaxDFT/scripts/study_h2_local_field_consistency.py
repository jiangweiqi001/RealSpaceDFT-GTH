"""H2 local-field consistency study on the adaptive backend.

This is not a final benchmark.
The goal is to attribute the remaining common-interior V_eff drift more
specifically between:
  1. V_loc adaptive tabulation / sampling drift,
  2. v_xc grid-representation / sampling-chain drift,
  3. or drift that remains after both are accounted for.

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
    from JaxDFT.src.functional import lda_xc
    from JaxDFT.src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.functional import lda_xc
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
SMALL = 1.0e-12
FIELD_SUSPECT_FLOOR = 1.0e-3


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


def build_local_potential_on_probe(probe_shape, probe_points, atom_coords, pseudos):
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    coeff = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
    coords_grid = jnp.asarray(probe_points, dtype=jnp.float32).reshape(probe_shape + (3,))
    field = build_local_potential_pointwise(
        jnp.asarray(atom_coords, dtype=jnp.float32),
        coords_grid,
        zion,
        rloc,
        coeff,
    )
    return field.reshape(-1)


def recompute_vxc_from_probe_rho(rho_probe):
    _, v_xc = lda_xc(jnp.asarray(rho_probe, dtype=jnp.float32))
    return v_xc.reshape(-1)


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
    base_key = jax.random.PRNGKey(17)
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


def build_probe_consistency(coords, pseudos, rows, bound, spacing):
    _, probe_points, probe_shape = build_probe_points(bound, spacing)
    probe_points = jnp.asarray(probe_points, dtype=jnp.float32)
    vloc_direct = build_local_potential_on_probe(probe_shape, probe_points, coords, pseudos)

    sampled_ref = rows[-1]
    ref_data = {
        "V_loc_sampled": sample_adaptive_trilinear(sampled_ref["grid"], sampled_ref["V_loc"], probe_points),
        "V_loc_direct": vloc_direct,
        "rho": sample_adaptive_trilinear(sampled_ref["grid"], sampled_ref["rho"], probe_points),
        "v_xc_sampled": sample_adaptive_trilinear(sampled_ref["grid"], sampled_ref["v_xc"], probe_points),
        "V_eff": sample_adaptive_trilinear(sampled_ref["grid"], sampled_ref["V_eff"], probe_points),
        "psi0_abs": sample_adaptive_trilinear(sampled_ref["grid"], sampled_ref["psi0_abs"], probe_points),
    }
    ref_data["v_xc_recomputed"] = recompute_vxc_from_probe_rho(ref_data["rho"])

    rows_out = []
    for row in rows[:-1]:
        vloc_sampled = sample_adaptive_trilinear(row["grid"], row["V_loc"], probe_points)
        rho_probe = sample_adaptive_trilinear(row["grid"], row["rho"], probe_points)
        vxc_sampled = sample_adaptive_trilinear(row["grid"], row["v_xc"], probe_points)
        vxc_recomputed = recompute_vxc_from_probe_rho(rho_probe)
        veff_probe = sample_adaptive_trilinear(row["grid"], row["V_eff"], probe_points)
        psi_probe = sample_adaptive_trilinear(row["grid"], row["psi0_abs"], probe_points)

        loc_sampled_ref = probe_metrics(vloc_sampled, ref_data["V_loc_sampled"])
        loc_direct_ref = probe_metrics(vloc_direct, ref_data["V_loc_direct"])
        loc_same = probe_metrics(vloc_sampled, vloc_direct)
        vxc_sampled_ref = probe_metrics(vxc_sampled, ref_data["v_xc_sampled"])
        vxc_recomputed_ref = probe_metrics(vxc_recomputed, ref_data["v_xc_recomputed"])
        vxc_same = probe_metrics(vxc_sampled, vxc_recomputed)
        rho_ref = probe_metrics(rho_probe, ref_data["rho"])
        psi_ref = probe_metrics(psi_probe, ref_data["psi0_abs"])
        veff_ref = probe_metrics(veff_probe, ref_data["V_eff"])

        rows_out.append({
            "box": row["box"],
            "V_loc_sampled_ref_rms": loc_sampled_ref["rms"],
            "V_loc_sampled_ref_rms_demeaned": loc_sampled_ref["rms_demeaned"],
            "V_loc_direct_ref_rms": loc_direct_ref["rms"],
            "V_loc_direct_ref_rms_demeaned": loc_direct_ref["rms_demeaned"],
            "V_loc_sampled_vs_direct_rms": loc_same["rms"],
            "V_loc_sampled_vs_direct_rms_demeaned": loc_same["rms_demeaned"],
            "V_loc_sampled_vs_direct_mean_delta": loc_same["mean_delta"],
            "rho_probe_rms": rho_ref["rms"],
            "rho_probe_rms_demeaned": rho_ref["rms_demeaned"],
            "rho_probe_mean_delta": rho_ref["mean_delta"],
            "v_xc_sampled_ref_rms": vxc_sampled_ref["rms"],
            "v_xc_sampled_ref_rms_demeaned": vxc_sampled_ref["rms_demeaned"],
            "v_xc_recomputed_ref_rms": vxc_recomputed_ref["rms"],
            "v_xc_recomputed_ref_rms_demeaned": vxc_recomputed_ref["rms_demeaned"],
            "v_xc_sampled_vs_recomputed_rms": vxc_same["rms"],
            "v_xc_sampled_vs_recomputed_rms_demeaned": vxc_same["rms_demeaned"],
            "v_xc_sampled_vs_recomputed_mean_delta": vxc_same["mean_delta"],
            "V_eff_probe_rms": veff_ref["rms"],
            "V_eff_probe_rms_demeaned": veff_ref["rms_demeaned"],
            "psi0_probe_rms": psi_ref["rms"],
        })
    return rows_out


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


def print_vloc_table(title, rows):
    print(f"=== {title} ===")
    header = (
        f"{'Box':>5} {'s_ref_dm':>10} {'d_ref_dm':>10} {'s-vs-d_dm':>12} {'s-vs-d_mean':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['V_loc_sampled_ref_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['V_loc_direct_ref_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['V_loc_sampled_vs_direct_rms_demeaned'], 12, 2)} "
            f"{fmt_sci(row['V_loc_sampled_vs_direct_mean_delta'], 12, 2)}"
        )


def print_vxc_table(title, rows):
    print(f"=== {title} ===")
    header = (
        f"{'Box':>5} {'rho_dm':>10} {'rho_mean':>10} {'vxc_s_dm':>10} {'vxc_r_dm':>10} "
        f"{'s-vs-r_dm':>11} {'s-vs-r_mean':>12} {'Veff_dm':>10} {'|psi0|_rms':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['rho_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['rho_probe_mean_delta'], 10, 2)} "
            f"{fmt_sci(row['v_xc_sampled_ref_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['v_xc_recomputed_ref_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['v_xc_sampled_vs_recomputed_rms_demeaned'], 11, 2)} "
            f"{fmt_sci(row['v_xc_sampled_vs_recomputed_mean_delta'], 12, 2)} "
            f"{fmt_sci(row['V_eff_probe_rms_demeaned'], 10, 2)} "
            f"{fmt_sci(row['psi0_probe_rms'], 11, 2)}"
        )


def diagnose(main_rows):
    loc_ref_dm = max(row["V_loc_sampled_ref_rms_demeaned"] for row in main_rows)
    loc_direct_ref_dm = max(row["V_loc_direct_ref_rms_demeaned"] for row in main_rows)
    loc_same_dm = max(row["V_loc_sampled_vs_direct_rms_demeaned"] for row in main_rows)
    vxc_sampled_ref_dm = max(row["v_xc_sampled_ref_rms_demeaned"] for row in main_rows)
    vxc_recomputed_ref_dm = max(row["v_xc_recomputed_ref_rms_demeaned"] for row in main_rows)
    vxc_same_dm = max(row["v_xc_sampled_vs_recomputed_rms_demeaned"] for row in main_rows)
    rho_dm = max(row["rho_probe_rms_demeaned"] for row in main_rows)
    veff_dm = max(row["V_eff_probe_rms_demeaned"] for row in main_rows)
    psi_rms = max(row["psi0_probe_rms"] for row in main_rows)

    vloc_sampling_suspected = (
        loc_ref_dm >= FIELD_SUSPECT_FLOOR and
        loc_same_dm >= 0.5 * max(loc_ref_dm, SMALL)
    )

    vxc_sampling_chain_suspected = (
        vxc_sampled_ref_dm >= FIELD_SUSPECT_FLOOR and
        vxc_same_dm >= 0.5 * max(vxc_sampled_ref_dm, SMALL)
    )

    vxc_rho_driven = (
        vxc_recomputed_ref_dm >= 0.5 * max(vxc_sampled_ref_dm, SMALL) and
        vxc_same_dm < 0.3 * max(vxc_sampled_ref_dm, SMALL) and
        rho_dm >= 1.0e-4
    )

    if vloc_sampling_suspected or vxc_sampling_chain_suspected:
        next_step_label = "local-field-first"
    elif veff_dm <= 0.5 * max(psi_rms, SMALL):
        next_step_label = "kinetic-suspected-next"
    else:
        next_step_label = "mixed"

    return {
        "V_loc_sampling_suspected": vloc_sampling_suspected,
        "v_xc_sampling_chain_suspected": vxc_sampling_chain_suspected,
        "v_xc_rho_driven": vxc_rho_driven,
        "next_step_label": next_step_label,
        "max_V_loc_sampled_ref_rms_demeaned": loc_ref_dm,
        "max_V_loc_direct_ref_rms_demeaned": loc_direct_ref_dm,
        "max_V_loc_sampled_vs_direct_rms_demeaned": loc_same_dm,
        "max_v_xc_sampled_ref_rms_demeaned": vxc_sampled_ref_dm,
        "max_v_xc_recomputed_ref_rms_demeaned": vxc_recomputed_ref_dm,
        "max_v_xc_sampled_vs_recomputed_rms_demeaned": vxc_same_dm,
        "max_rho_probe_rms_demeaned": rho_dm,
        "max_V_eff_probe_rms_demeaned": veff_dm,
        "max_|psi0|_probe_rms": psi_rms,
        "main_probe_box": "[-4,4]^3 requested; exact centered 0.3 lattice is [-3.9, 3.9]^3",
        "extended_probe_box": "[-6,6]^3",
    }


def print_summary(summary):
    print("=== Summary ===")
    print(f"V_loc_sampling_suspected          : {summary['V_loc_sampling_suspected']}")
    print(f"v_xc_sampling_chain_suspected     : {summary['v_xc_sampling_chain_suspected']}")
    print(f"v_xc_rho_driven                   : {summary['v_xc_rho_driven']}")
    print(f"next_step_label                   : {summary['next_step_label']}")
    print(f"max_V_loc_sampled_ref_dm          : {summary['max_V_loc_sampled_ref_rms_demeaned']:.6e}")
    print(f"max_V_loc_direct_ref_dm           : {summary['max_V_loc_direct_ref_rms_demeaned']:.6e}")
    print(f"max_V_loc_sampled_vs_direct_dm    : {summary['max_V_loc_sampled_vs_direct_rms_demeaned']:.6e}")
    print(f"max_v_xc_sampled_ref_dm           : {summary['max_v_xc_sampled_ref_rms_demeaned']:.6e}")
    print(f"max_v_xc_recomputed_ref_dm        : {summary['max_v_xc_recomputed_ref_rms_demeaned']:.6e}")
    print(f"max_v_xc_sampled_vs_recomputed_dm : {summary['max_v_xc_sampled_vs_recomputed_rms_demeaned']:.6e}")
    print(f"max_rho_probe_rms_demeaned        : {summary['max_rho_probe_rms_demeaned']:.6e}")
    print(f"max_V_eff_probe_rms_demeaned      : {summary['max_V_eff_probe_rms_demeaned']:.6e}")
    print(f"max_|psi0|_probe_rms              : {summary['max_|psi0|_probe_rms']:.6e}")
    print(f"main_probe_box                    : {summary['main_probe_box']}")
    print(f"extended_probe_box                : {summary['extended_probe_box']}")


def main():
    print("H2 local-field consistency study")
    print("This is not a final benchmark.")
    print("The goal is to determine whether the remaining common-interior V_eff drift")
    print("is still explained by local-field representation / sampling, or whether it")
    print("is small enough that the next step should move toward kinetic/Laplacian.")
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

    main_rows = build_probe_consistency(coords, pseudos, results, MAIN_PROBE_BOUND, PROBE_SPACING)
    ext_rows = build_probe_consistency(coords, pseudos, results, EXT_PROBE_BOUND, PROBE_SPACING)
    print_vloc_table("V_loc Consistency Table", main_rows)
    print()
    print_vxc_table("v_xc Consistency Table", main_rows)
    print()
    print_vloc_table("Extended-Probe V_loc Table", ext_rows)
    print()
    print_vxc_table("Extended-Probe v_xc Table", ext_rows)
    print()

    summary = diagnose(main_rows)
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
        len(main_rows) == len(BOXES) - 1 and len(ext_rows) == len(BOXES) - 1,
        f"main_rows={len(main_rows)}, ext_rows={len(ext_rows)}",
    )

    print()
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
