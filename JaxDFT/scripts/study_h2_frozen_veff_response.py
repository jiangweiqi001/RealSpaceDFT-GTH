"""H2 frozen-V_eff response study.

This is not a final benchmark.
The goal is to separate self-consistent density-response feedback from the
fixed-potential eigenproblem / finite-box effect for adaptive H2.

Method:
  1. Run a normal SCF on the largest box to obtain reference rho, V_H, v_xc,
     V_eff, and psi0.
  2. For each smaller box, also run a normal SCF.
  3. Build a frozen effective potential on the smaller box using
       V_loc(current, direct-on-grid)
     + sample(V_H_ref -> current grid)
     + sample(v_xc_ref -> current grid)
  4. Under this frozen V_eff, solve only the lowest occupied state.
  5. Compare SCF and frozen psi0 / rho against the largest-box reference on a
     common interior probe.

If frozen psi0/rho drift remains close to the SCF drift, the remaining error is
more consistent with a fixed-potential eigenproblem / kinetic / finite-box box
issue. If frozen drift collapses, the remaining error is more consistent with
self-consistent density-response feedback.
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
    from JaxDFT.src.solver import (
        ion_ion_energy,
        scf,
        solve_orbitals_subspace,
        total_energy,
    )
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, solve_orbitals_subspace, total_energy


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
MAIN_PROBE_BOUND = 4.0
PROBE_SPACING = 0.3
ELECTRON_TOL = 5.0e-3
NORM_TOL = 2.0e-2
SMALL = 1.0e-12
FROZEN_SOLVE_MAX_ITER = 12
FROZEN_SOLVE_TOL = 1.0e-5


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


def solve_frozen_lowest_state(backend, grid, pseudos, coords, V_eff_frozen, psi_init, key_seed):
    proj_data = backend.precompute_nonlocal(grid, coords, pseudos)

    def apply_h(psi_flat):
        psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
        psi = psi * grid.mask
        kinetic_psi = backend.apply_kinetic(grid, psi)
        v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
        hpsi = kinetic_psi + V_eff_frozen * psi + v_nonlocal
        hpsi = hpsi * grid.mask
        return hpsi.reshape(-1)

    x_init = normalize_field(backend, grid, psi_init).reshape(-1, 1)
    eigvals, eigvecs = solve_orbitals_subspace(
        apply_h,
        int(np.prod(grid.shape)),
        1,
        x_init=x_init,
        max_iter=FROZEN_SOLVE_MAX_ITER,
        tol=FROZEN_SOLVE_TOL,
        key=jax.random.PRNGKey(key_seed),
        grid=grid,
        backend=backend,
    )
    psi0 = eigvecs.reshape(grid.shape)
    psi0 = normalize_field(backend, grid, psi0)
    norm_dev = float(abs(float(backend.inner_product(grid, psi0, psi0)) - 1.0))
    rho_frozen = 2.0 * (psi0 ** 2)
    rho_frozen = rho_frozen * grid.mask
    return {
        "eig0_frozen": float(jnp.asarray(eigvals).reshape(-1)[0]),
        "psi0_frozen": psi0,
        "rho_frozen": rho_frozen,
        "frozen_norm_dev": norm_dev,
    }


def run_scf_case(grid, coords, pseudos, zion, occ, n_bands, key_seed):
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
        "backend": backend,
        "V_loc": None,
        "V_H": None,
        "v_xc": None,
        "V_eff": None,
        "rho": None,
        "psi0": None,
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
        psi0 = normalize_field(backend, grid, eigvec_fields[0])
        psi0_abs = jnp.abs(psi0)
        electron_count = float(backend.integrate(grid, rho))
        norm_maxdev = float(jnp.max(jnp.abs(norms - 1.0)))
        hartree_proxy = float(0.5 * backend.integrate(grid, rho * V_H))
        all_finite = bool(
            jnp.all(jnp.isfinite(V_loc))
            and jnp.all(jnp.isfinite(V_H))
            and jnp.all(jnp.isfinite(v_xc))
            and jnp.all(jnp.isfinite(rho))
            and jnp.all(jnp.isfinite(psi0))
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
            "V_eff": V_loc + V_H + v_xc,
            "rho": rho,
            "psi0": psi0,
            "psi0_abs": psi0_abs,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_scf_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    _, n_bands, occ = build_occ(pseudos)
    grid_builder = AdaptiveBackend()
    results = []
    base_key = jax.random.PRNGKey(91)
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
        results.append(run_scf_case(grid, coords, pseudos, zion, occ, n_bands, key_seed))
    return coords, pseudos, results


def build_frozen_rows(coords, pseudos, scf_rows, bound, spacing):
    _, probe_points, _ = build_probe_points(bound, spacing)
    probe_points = jnp.asarray(probe_points, dtype=jnp.float32)
    ref = scf_rows[-1]
    ref_grid = ref["grid"]
    ref_psi_probe = sample_adaptive_trilinear(ref_grid, ref["psi0_abs"], probe_points)
    ref_rho_probe = sample_adaptive_trilinear(ref_grid, ref["rho"], probe_points)

    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
    coeff = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)

    rows = []
    for box_idx, row in enumerate(scf_rows[:-1]):
        grid = row["grid"]
        backend = row["backend"]
        V_loc_current = build_local_potential_pointwise(coords, grid.coords, zion, rloc, coeff)
        V_H_ref_on_current = sample_field_to_grid(ref_grid, ref["V_H"], grid)
        v_xc_ref_on_current = sample_field_to_grid(ref_grid, ref["v_xc"], grid)
        psi0_ref_on_current = sample_field_to_grid(ref_grid, ref["psi0"], grid)
        psi0_ref_on_current = normalize_field(backend, grid, psi0_ref_on_current)
        V_eff_frozen = V_loc_current + V_H_ref_on_current + v_xc_ref_on_current

        frozen = solve_frozen_lowest_state(
            backend,
            grid,
            pseudos,
            coords,
            V_eff_frozen,
            psi0_ref_on_current,
            key_seed=700 + box_idx,
        )

        psi0_scf_probe = sample_adaptive_trilinear(grid, row["psi0_abs"], probe_points)
        rho_scf_probe = sample_adaptive_trilinear(grid, row["rho"], probe_points)
        psi0_frozen_probe = sample_adaptive_trilinear(grid, jnp.abs(frozen["psi0_frozen"]), probe_points)
        rho_frozen_probe = sample_adaptive_trilinear(grid, frozen["rho_frozen"], probe_points)

        psi_scf_m = field_metrics(psi0_scf_probe, ref_psi_probe)
        psi_frozen_m = field_metrics(psi0_frozen_probe, ref_psi_probe)
        rho_scf_m = field_metrics(rho_scf_probe, ref_rho_probe)
        rho_frozen_m = field_metrics(rho_frozen_probe, ref_rho_probe)

        overlap_scf_ref = float(
            jnp.abs(backend.inner_product(grid, normalize_field(backend, grid, row["psi0"]), psi0_ref_on_current))
        )
        overlap_frozen_ref = float(
            jnp.abs(backend.inner_product(grid, normalize_field(backend, grid, frozen["psi0_frozen"]), psi0_ref_on_current))
        )

        rows.append({
            "box": row["box"],
            "eig0_frozen": frozen["eig0_frozen"],
            "frozen_norm_dev": frozen["frozen_norm_dev"],
            "psi0_frozen_probe_rms": psi_frozen_m["rms"],
            "psi0_frozen_probe_rms_demeaned": psi_frozen_m["rms_demeaned"],
            "rho_frozen_probe_rms": rho_frozen_m["rms"],
            "rho_frozen_probe_rms_demeaned": rho_frozen_m["rms_demeaned"],
            "psi0_scf_probe_rms": psi_scf_m["rms"],
            "psi0_scf_probe_rms_demeaned": psi_scf_m["rms_demeaned"],
            "rho_scf_probe_rms": rho_scf_m["rms"],
            "rho_scf_probe_rms_demeaned": rho_scf_m["rms_demeaned"],
            "psi_reduction_factor": psi_scf_m["rms"] / max(psi_frozen_m["rms"], SMALL),
            "rho_reduction_factor": rho_scf_m["rms"] / max(rho_frozen_m["rms"], SMALL),
            "overlap_scf_ref": overlap_scf_ref,
            "overlap_frozen_ref": overlap_frozen_ref,
        })
    return rows


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


def print_frozen_table(rows):
    print("=== Frozen Response Table ===")
    header = (
        f"{'Box':>5} {'eig0_frozen':>12} {'NormDev':>10} {'psi_f_rms':>11} {'psi_f_dm':>11} "
        f"{'rho_f_rms':>11} {'rho_f_dm':>11} {'ovlp_f_ref':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_float(row['eig0_frozen'], 12, 6)} "
            f"{fmt_sci(row['frozen_norm_dev'], 10, 2)} "
            f"{fmt_sci(row['psi0_frozen_probe_rms'], 11, 2)} "
            f"{fmt_sci(row['psi0_frozen_probe_rms_demeaned'], 11, 2)} "
            f"{fmt_sci(row['rho_frozen_probe_rms'], 11, 2)} "
            f"{fmt_sci(row['rho_frozen_probe_rms_demeaned'], 11, 2)} "
            f"{fmt_float(row['overlap_frozen_ref'], 11, 6)}"
        )


def print_compare_table(rows):
    print("=== SCF vs Frozen Comparison Table ===")
    header = (
        f"{'Box':>5} {'psi_scf':>10} {'psi_scf_dm':>11} {'psi_froz':>10} {'psi_f_dm':>11} "
        f"{'psi_red':>9} {'rho_scf':>10} {'rho_scf_dm':>11} {'rho_froz':>10} {'rho_f_dm':>11} "
        f"{'rho_red':>9} {'ovlp_scf':>10} {'ovlp_froz':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} "
            f"{fmt_sci(row['psi0_scf_probe_rms'], 10, 2)} "
            f"{fmt_sci(row['psi0_scf_probe_rms_demeaned'], 11, 2)} "
            f"{fmt_sci(row['psi0_frozen_probe_rms'], 10, 2)} "
            f"{fmt_sci(row['psi0_frozen_probe_rms_demeaned'], 11, 2)} "
            f"{fmt_float(row['psi_reduction_factor'], 9, 3)} "
            f"{fmt_sci(row['rho_scf_probe_rms'], 10, 2)} "
            f"{fmt_sci(row['rho_scf_probe_rms_demeaned'], 11, 2)} "
            f"{fmt_sci(row['rho_frozen_probe_rms'], 10, 2)} "
            f"{fmt_sci(row['rho_frozen_probe_rms_demeaned'], 11, 2)} "
            f"{fmt_float(row['rho_reduction_factor'], 9, 3)} "
            f"{fmt_float(row['overlap_scf_ref'], 10, 6)} "
            f"{fmt_float(row['overlap_frozen_ref'], 10, 6)}"
        )


def diagnose(rows):
    psi_scf_max = max(row['psi0_scf_probe_rms'] for row in rows)
    psi_frozen_max = max(row['psi0_frozen_probe_rms'] for row in rows)
    rho_scf_max = max(row['rho_scf_probe_rms'] for row in rows)
    rho_frozen_max = max(row['rho_frozen_probe_rms'] for row in rows)
    psi_reduction = psi_scf_max / max(psi_frozen_max, SMALL)
    rho_reduction = rho_scf_max / max(rho_frozen_max, SMALL)

    if psi_frozen_max <= 0.5 * psi_scf_max and rho_frozen_max <= 0.5 * rho_scf_max:
        label = 'self-consistency-dominated'
    elif psi_frozen_max >= 0.8 * psi_scf_max and rho_frozen_max >= 0.8 * rho_scf_max:
        label = 'eigenproblem-suspected-next'
    else:
        label = 'mixed'

    print("=== Diagnosis Summary ===")
    print(f"label: {label}")
    print(f"max_psi0_scf_probe_rms = {psi_scf_max:.6e}")
    print(f"max_psi0_frozen_probe_rms = {psi_frozen_max:.6e}")
    print(f"max_rho_scf_probe_rms = {rho_scf_max:.6e}")
    print(f"max_rho_frozen_probe_rms = {rho_frozen_max:.6e}")
    print(f"psi_reduction_factor_maxboxwise = {psi_reduction:.6f}")
    print(f"rho_reduction_factor_maxboxwise = {rho_reduction:.6f}")
    print()
    print("Interpretation:")
    print("  - If frozen drift stays close to SCF drift, the remaining box effect is more")
    print("    consistent with a fixed-potential eigenproblem / kinetic / finite-box issue.")
    print("  - If frozen drift collapses, the remaining box effect is more consistent with")
    print("    self-consistent density-response feedback.")
    return label


def main():
    print("H2 frozen-V_eff response study")
    print("This is not a final benchmark.")
    print("The goal is to test whether freezing V_eff removes most of the common-interior")
    print("psi0 / rho drift, or whether the drift survives even under a fixed effective potential.")
    print()

    coords, pseudos, scf_results = build_scf_results()
    print_scf_table(scf_results)
    print()

    frozen_rows = build_frozen_rows(coords, pseudos, scf_results, MAIN_PROBE_BOUND, PROBE_SPACING)
    print_frozen_table(frozen_rows)
    print()
    print_compare_table(frozen_rows)
    print()
    label = diagnose(frozen_rows)

    checks = []
    checks.append(check(
        "all_scf_completed",
        all(row['completed'] and row['all_finite'] for row in scf_results),
        f"completed={sum(int(row['completed'] and row['all_finite']) for row in scf_results)}/{len(scf_results)}",
    ))
    checks.append(check(
        "electron_counts_reasonable",
        all(row['electron_error'] is not None and row['electron_error'] <= ELECTRON_TOL for row in scf_results),
        f"max_electron_error={max(row['electron_error'] for row in scf_results):.3e}",
    ))
    checks.append(check(
        "orbital_norms_reasonable",
        all(row['orbital_norm_maxdev'] is not None and row['orbital_norm_maxdev'] <= NORM_TOL for row in scf_results),
        f"max_orbital_norm_dev={max(row['orbital_norm_maxdev'] for row in scf_results):.3e}",
    ))
    checks.append(check(
        "frozen_norms_reasonable",
        all(row['frozen_norm_dev'] <= NORM_TOL for row in frozen_rows),
        f"max_frozen_norm_dev={max(row['frozen_norm_dev'] for row in frozen_rows):.3e}",
    ))
    checks.append(check(
        "frozen_overlaps_finite",
        all(np.isfinite(row['overlap_scf_ref']) and np.isfinite(row['overlap_frozen_ref']) for row in frozen_rows),
        f"min_overlap={min(min(row['overlap_scf_ref'], row['overlap_frozen_ref']) for row in frozen_rows):.6f}",
    ))
    checks.append(check(
        "diagnosis_label_valid",
        label in {'self-consistency-dominated', 'eigenproblem-suspected-next', 'mixed'},
        f"label={label}",
    ))

    overall = all(checks)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
