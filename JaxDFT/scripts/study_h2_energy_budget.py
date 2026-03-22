"""H2 adaptive SCF energy-budget audit.

This is not a final benchmark.
The goal is to identify which energy component dominates the box drift of the
real adaptive SCF total energy for H2 at fixed bond length.

We decompose the total energy into:
  - Ts   = sum_i occ_i <psi_i | T | psi_i>
  - Eloc = integral rho * V_loc
  - Enl  = sum_i occ_i <psi_i | V_nonlocal | psi_i>
  - Eh   = 0.5 * integral rho * V_H
  - Exc  = integral eps_xc
  - Eion = ion-ion Coulomb energy

and compare drifts relative to the largest-box reference.
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
    from JaxDFT.src.solver import energy_and_forces, ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.io import load_pseudopotentials
    from src.solver import energy_and_forces, ion_ion_energy, scf, total_energy


DISTANCE = 1.4
BOXES = [14.0, 18.0, 22.0, 26.0, 30.0]
H_MIN = 0.25
H_MAX = 0.80
R_CORE = 1.0
STRETCH_BETA = 5.0
KINETIC_MODE = "prototype_fd2"
SCF_KWARGS = {
    "max_iter": 4,
    "mix_alpha": 0.30,
    "tolerance": 5.0e-4,
}
SMALL = 1.0e-12
CONSISTENCY_TOL = 5.0e-3


def fmt_float(value, width=13, precision=6):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=13, precision=3):
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


def orbital_fields_from_flat(grid, eigvecs_flat):
    n_bands = int(eigvecs_flat.shape[1])
    return jnp.moveaxis(eigvecs_flat.reshape(grid.shape + (n_bands,)), -1, 0)


def decompose_energy(grid, backend, coords, pseudos, occ, rho, eigvals, eigvecs, V_loc, V_H, eps_xc, v_xc):
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    eion = float(ion_ion_energy(coords, zion))
    e_total = float(total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, eion, backend=backend))

    states = orbital_fields_from_flat(grid, eigvecs)
    norms = []
    ts = 0.0
    enl = 0.0
    cache = backend.precompute_nonlocal(grid, coords, pseudos)
    occ_np = np.asarray(jnp.asarray(occ), dtype=np.float64)
    for i in range(states.shape[0]):
        psi = states[i]
        occ_i = float(occ_np[i])
        norm_i = float(backend.inner_product(grid, psi, psi))
        norms.append(norm_i)
        if occ_i <= 0.0:
            continue
        tpsi = backend.apply_kinetic(grid, psi)
        vnl_psi = backend.apply_nonlocal(grid, psi, cache)
        ts += occ_i * float(backend.inner_product(grid, psi, tpsi))
        enl += occ_i * float(backend.inner_product(grid, psi, vnl_psi))

    eloc = float(backend.integrate(grid, rho * V_loc))
    eh = float(0.5 * backend.integrate(grid, rho * V_H))
    exc = float(backend.integrate(grid, eps_xc))
    e_sum = ts + eloc + enl + eh + exc + eion
    consistency_error = e_total - e_sum
    rel_consistency_error = consistency_error / max(abs(e_total), SMALL)
    electron_count = float(backend.integrate(grid, rho))
    orbital_norm_maxdev = float(np.max(np.abs(np.asarray(norms, dtype=np.float64) - 1.0))) if norms else 0.0
    return {
        "total_energy": e_total,
        "Ts": ts,
        "Eloc": eloc,
        "Enl": enl,
        "Eh": eh,
        "Exc": exc,
        "Eion": eion,
        "E_sum": e_sum,
        "consistency_error": consistency_error,
        "relative_consistency_error": rel_consistency_error,
        "electron_count": electron_count,
        "orbital_norm_maxdev": orbital_norm_maxdev,
        "eigvals": np.asarray(jnp.asarray(eigvals), dtype=np.float64),
    }


def run_box(box_length, coords, pseudos, n_bands, occ):
    backend = AdaptiveBackend(
        hartree_boundary_mode="uniform_exterior",
        kinetic_mode=KINETIC_MODE,
    )
    box = jnp.asarray([box_length, box_length, box_length], dtype=jnp.float32)
    grid = backend.create_grid(
        spacing=H_MIN,
        box_size=box,
        atom_coords=coords,
        h_min=H_MIN,
        h_max=H_MAX,
        r_core=R_CORE,
        stretch_beta=STRETCH_BETA,
    )
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid,
        coords,
        n_bands,
        occ,
        V_loc,
        pseudos,
        key=jax.random.PRNGKey(3200 + int(round(box_length))),
        backend=backend,
        **SCF_KWARGS,
    )
    energy_budget = decompose_energy(grid, backend, coords, pseudos, occ, rho, eigvals, eigvecs, V_loc, V_H, eps_xc, v_xc)
    energy_budget["box"] = box_length
    energy_budget["grid_shape"] = tuple(int(n) for n in grid.shape)
    return energy_budget


def build_drift_tables(results):
    ref = results[-1]
    rows = []
    for row in results:
        d_total = row["total_energy"] - ref["total_energy"]
        row_out = {
            "box": row["box"],
            "dE_total": d_total,
        }
        for key in ("Ts", "Eloc", "Enl", "Eh", "Exc"):
            d_key = row[key] - ref[key]
            row_out[f"d{key}"] = d_key
            row_out[f"abs_d{key}"] = abs(d_key)
            if abs(d_total) > SMALL:
                row_out[f"frac_{key}"] = abs(d_key) / abs(d_total)
                row_out[f"signed_frac_{key}"] = d_key / d_total
            else:
                row_out[f"frac_{key}"] = None
                row_out[f"signed_frac_{key}"] = None
        rows.append(row_out)
    return rows


def diagnose(drift_rows):
    max_d_total = max(abs(row["dE_total"]) for row in drift_rows[:-1]) if len(drift_rows) > 1 else 0.0
    max_d_ts = max(row["abs_dTs"] for row in drift_rows[:-1]) if len(drift_rows) > 1 else 0.0
    max_d_eloc = max(row["abs_dEloc"] for row in drift_rows[:-1]) if len(drift_rows) > 1 else 0.0
    max_d_enl = max(row["abs_dEnl"] for row in drift_rows[:-1]) if len(drift_rows) > 1 else 0.0
    max_d_eh = max(row["abs_dEh"] for row in drift_rows[:-1]) if len(drift_rows) > 1 else 0.0
    max_d_exc = max(row["abs_dExc"] for row in drift_rows[:-1]) if len(drift_rows) > 1 else 0.0
    max_d_electro = max(max_d_eloc, max_d_enl, max_d_eh)

    if max_d_ts >= 1.5 * max(max_d_electro, max_d_exc) and max_d_ts >= 0.4 * max_d_total:
        label = "kinetic-dominated"
    elif max_d_electro >= 1.5 * max(max_d_ts, max_d_exc) and max_d_electro >= 0.4 * max_d_total:
        label = "electrostatic-dominated"
    elif max_d_exc >= 1.5 * max(max_d_ts, max_d_electro) and max_d_exc >= 0.4 * max_d_total:
        label = "xc-dominated"
    else:
        label = "mixed"

    return {
        "label": label,
        "max_dE_total": max_d_total,
        "max_dTs": max_d_ts,
        "max_dEloc": max_d_eloc,
        "max_dEnl": max_d_enl,
        "max_dEh": max_d_eh,
        "max_dExc": max_d_exc,
        "max_dElectro": max_d_electro,
    }


def print_energy_table(results):
    print("=== SCF Energy Table ===")
    header = (
        f"{'box':>5} {'E_tot':>13} {'Ts':>13} {'Eloc':>13} {'Enl':>13} {'Eh':>13} {'Exc':>13} {'Eion':>13} "
        f"{'N':>9} {'norm_dev':>11} {'shape':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['box']:>5.1f} {fmt_float(row['total_energy'])} {fmt_float(row['Ts'])} {fmt_float(row['Eloc'])} {fmt_float(row['Enl'])} "
            f"{fmt_float(row['Eh'])} {fmt_float(row['Exc'])} {fmt_float(row['Eion'])} {fmt_float(row['electron_count'], 9, 6)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 11, 2)} {str(row['grid_shape']):>14}"
        )
    print()


def print_drift_table(drift_rows):
    print("=== Drift Table (vs box=30 reference) ===")
    header = (
        f"{'box':>5} {'dE_tot':>13} {'dTs':>13} {'dEloc':>13} {'dEnl':>13} {'dEh':>13} {'dExc':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in drift_rows:
        print(
            f"{row['box']:>5.1f} {fmt_float(row['dE_total'])} {fmt_float(row['dTs'])} {fmt_float(row['dEloc'])} {fmt_float(row['dEnl'])} "
            f"{fmt_float(row['dEh'])} {fmt_float(row['dExc'])}"
        )
    print()


def print_contribution_table(drift_rows):
    print("=== Contribution Table ===")
    header = (
        f"{'box':>5} {'|dTs|/|dE|':>13} {'dTs/dE':>13} {'|dEloc|/|dE|':>13} {'dEloc/dE':>13} {'|dEnl|/|dE|':>13} {'dEnl/dE':>13} "
        f"{'|dEh|/|dE|':>13} {'dEh/dE':>13} {'|dExc|/|dE|':>13} {'dExc/dE':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in drift_rows:
        print(
            f"{row['box']:>5.1f} {fmt_float(row['frac_Ts'])} {fmt_float(row['signed_frac_Ts'])} {fmt_float(row['frac_Eloc'])} {fmt_float(row['signed_frac_Eloc'])} "
            f"{fmt_float(row['frac_Enl'])} {fmt_float(row['signed_frac_Enl'])} {fmt_float(row['frac_Eh'])} {fmt_float(row['signed_frac_Eh'])} "
            f"{fmt_float(row['frac_Exc'])} {fmt_float(row['signed_frac_Exc'])}"
        )
    print()


def print_consistency_summary(results):
    print("=== Consistency Summary ===")
    for row in results:
        print(
            f"box={row['box']:.1f}: "
            f"E_total={row['total_energy']:.8f}, "
            f"E_sum={row['E_sum']:.8f}, "
            f"E_total-E_sum={row['consistency_error']:.3e}, "
            f"relative_consistency_error={row['relative_consistency_error']:.3e}"
        )
    print()


def print_diagnosis(summary):
    print("=== Diagnosis Summary ===")
    print(f"label: {summary['label']}")
    print(f"max_dE_total: {summary['max_dE_total']:.6e}")
    print(f"max_dTs: {summary['max_dTs']:.6e}")
    print(f"max_dEloc: {summary['max_dEloc']:.6e}")
    print(f"max_dEnl: {summary['max_dEnl']:.6e}")
    print(f"max_dEh: {summary['max_dEh']:.6e}")
    print(f"max_dExc: {summary['max_dExc']:.6e}")
    print(f"max_dElectro: {summary['max_dElectro']:.6e}")
    print()


def main():
    print("=== H2 Adaptive SCF Energy Budget Audit ===")
    print("This is not a final benchmark.")
    print("Goal: identify which energy component dominates the real adaptive SCF box drift.")
    print(f"kinetic_mode = {KINETIC_MODE}")
    print()

    pseudos = load_pseudopotentials(["H", "H"], "pbe")
    _, n_bands, occ = build_occ(pseudos)
    coords = jnp.asarray([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)

    results = []
    for box in BOXES:
        results.append(run_box(box, coords, pseudos, n_bands, occ))

    drift_rows = build_drift_tables(results)
    summary = diagnose(drift_rows)

    print_energy_table(results)
    print_drift_table(drift_rows)
    print_contribution_table(drift_rows)
    print_consistency_summary(results)
    print_diagnosis(summary)

    overall_ok = True
    for row in results:
        overall_ok &= check(
            f"electron_count_box_{int(round(row['box']))}",
            abs(row['electron_count'] - float(jnp.sum(occ))) <= 5e-3,
            f"N={row['electron_count']:.6f}, target={float(jnp.sum(occ)):.6f}",
        )
        overall_ok &= check(
            f"orbital_norm_box_{int(round(row['box']))}",
            row['orbital_norm_maxdev'] <= 2e-2,
            f"maxdev={row['orbital_norm_maxdev']:.3e}",
        )
        overall_ok &= check(
            f"energy_consistency_box_{int(round(row['box']))}",
            abs(row['consistency_error']) <= CONSISTENCY_TOL,
            f"E_total-E_sum={row['consistency_error']:.3e}, rel={row['relative_consistency_error']:.3e}",
        )

    overall_ok &= check(
        "diagnosis_label",
        summary['label'] in {"kinetic-dominated", "electrostatic-dominated", "xc-dominated", "mixed"},
        f"label={summary['label']}",
    )

    print("=== Overall Summary ===")
    print(f"label={summary['label']}")
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
