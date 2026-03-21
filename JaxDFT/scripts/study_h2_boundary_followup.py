"""Focused H2 follow-up study for adaptive Hartree boundary choices.

This is not a final benchmark.
The current goal is to understand whether H2 at fixed bond length is still
primarily box/boundary dominated, whether monopole(charge_center) is more stable
than monopole(box_center), and where the current default multipole path sits
relative to those monopole variants.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp

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
BOXES = [14.0, 18.0, 22.0]
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


def fmt_float(value, width=11, precision=6):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=12, precision=3):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def fmt_text(value, width=22):
    text = "-" if value is None else str(value)
    return text[:width].ljust(width)


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


def eigvals_repr(eigvals):
    if eigvals is None:
        return None
    arr = [float(x) for x in jnp.asarray(eigvals).reshape(-1)]
    return "[" + ", ".join(f"{x:.4f}" for x in arr[:3]) + ("]" if len(arr) <= 3 else ", ...]")


def run_mode(mode_spec, grid, coords, pseudos, zion, occ, n_bands, key_seed):
    backend = mode_spec["backend"]
    result = {
        "mode": mode_spec["name"],
        "box": float(grid.box_size[0]),
        "completed": False,
        "all_finite": False,
        "energy": None,
        "electron_count": None,
        "electron_error": None,
        "orbital_norm_maxdev": None,
        "eig0": None,
        "eigvals_str": None,
        "shape": tuple(int(n) for n in grid.shape),
        "rho_min": None,
        "rho_max": None,
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
        all_finite = bool(
            jnp.all(jnp.isfinite(rho))
            and jnp.all(jnp.isfinite(eigvals))
            and jnp.all(jnp.isfinite(eigvecs))
            and jnp.all(jnp.isfinite(V_H))
            and jnp.isfinite(energy)
            and jnp.all(jnp.isfinite(norms))
        )
        result.update({
            "completed": True,
            "all_finite": all_finite,
            "energy": float(energy),
            "electron_count": electron_count,
            "electron_error": abs(electron_count - float(jnp.sum(occ))),
            "orbital_norm_maxdev": norm_maxdev,
            "eig0": float(jnp.asarray(eigvals).reshape(-1)[0]),
            "eigvals_str": eigvals_repr(eigvals),
            "rho_min": float(jnp.min(rho)),
            "rho_max": float(jnp.max(rho)),
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

    mode_specs = [
        {"name": "zero_dirichlet", "backend": AdaptiveBackend(hartree_boundary_mode="zero_dirichlet")},
        {
            "name": "monopole_box_center",
            "backend": AdaptiveBackend(hartree_boundary_mode="monopole_dirichlet", hartree_center_mode="box_center"),
        },
        {
            "name": "monopole_charge_center",
            "backend": AdaptiveBackend(hartree_boundary_mode="monopole_dirichlet", hartree_center_mode="charge_center"),
        },
        {"name": "multipole_dirichlet", "backend": AdaptiveBackend(hartree_boundary_mode="multipole_dirichlet")},
    ]

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
        mode_key = int(jax.random.fold_in(base_key, box_idx)[0])
        for spec in mode_specs:
            results.append(run_mode(spec, grid, coords, pseudos, zion, occ, n_bands, mode_key))

    return results


def group_by_mode(results):
    grouped = {}
    for row in results:
        grouped.setdefault(row["mode"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["box"])
    return grouped


def print_result_table(results):
    print("=== Result Table ===")
    header = (
        f"{'Mode':<24} {'Box':>5} {'Done':<5} {'Finite':<6} {'Energy':>12} {'N':>8} "
        f"{'dN':>10} {'NormDev':>10} {'eig0':>11} {'rho_min':>10} {'rho_max':>10} {'Shape':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['mode']:<24} {fmt_float(row['box'], 5, 1)} "
            f"{('PASS' if row['completed'] else 'FAIL'):<5} {('PASS' if row['all_finite'] else 'FAIL'):<6} "
            f"{fmt_float(row['energy'], 12, 6)} {fmt_float(row['electron_count'], 8, 4)} "
            f"{fmt_sci(row['electron_error'], 10, 2)} {fmt_sci(row['orbital_norm_maxdev'], 10, 2)} "
            f"{fmt_float(row['eig0'], 11, 6)} {fmt_float(row['rho_min'], 10, 4)} {fmt_float(row['rho_max'], 10, 4)} {str(row['shape']):<14}"
        )
        if row['error'] is not None:
            print(f"  error: {row['error']}")


def build_box_drift_summary(results):
    grouped = group_by_mode(results)
    summary = []
    for mode, rows in grouped.items():
        ref_energy = rows[-1]['energy'] if rows and rows[-1]['energy'] is not None else None
        max_abs_drift = None
        for row in rows:
            drift = None if ref_energy is None or row['energy'] is None else row['energy'] - ref_energy
            if drift is not None:
                max_abs_drift = abs(drift) if max_abs_drift is None else max(max_abs_drift, abs(drift))
            summary.append({
                'mode': mode,
                'box': row['box'],
                'energy': row['energy'],
                'drift_vs_largest': drift,
                'max_abs_drift': max_abs_drift,
            })
        for item in summary:
            if item['mode'] == mode:
                item['max_abs_drift'] = max_abs_drift
    return summary


def print_box_drift_summary(summary):
    print("=== Box Drift Summary ===")
    header = f"{'Mode':<24} {'Box':>5} {'Energy':>12} {'dE_vs_Lmax':>14} {'Max|dE|':>12}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['mode']:<24} {fmt_float(row['box'], 5, 1)} {fmt_float(row['energy'], 12, 6)} "
            f"{fmt_sci(row['drift_vs_largest'], 14, 3)} {fmt_sci(row['max_abs_drift'], 12, 3)}"
        )


def build_boundary_summary(results):
    index = {(row['mode'], row['box']): row for row in results}
    summary = []
    for box in BOXES:
        zero = index[('zero_dirichlet', box)]
        mono_box = index[('monopole_box_center', box)]
        mono_charge = index[('monopole_charge_center', box)]
        multi = index[('multipole_dirichlet', box)]

        def delta(a, b):
            if a['energy'] is None or b['energy'] is None:
                return None
            return a['energy'] - b['energy']

        summary.append({
            'box': box,
            'mono_box_minus_zero': delta(mono_box, zero),
            'mono_charge_minus_zero': delta(mono_charge, zero),
            'multi_minus_zero': delta(multi, zero),
            'mono_charge_minus_mono_box': delta(mono_charge, mono_box),
            'multi_minus_mono_box': delta(multi, mono_box),
            'multi_minus_mono_charge': delta(multi, mono_charge),
        })
    return summary


def print_boundary_summary(summary):
    print("=== Boundary Comparison Summary ===")
    header = (
        f"{'Box':>5} {'MonoBox-Zero':>14} {'MonoCharge-Zero':>16} {'Multi-Zero':>14} "
        f"{'MonoCharge-MonoBox':>20} {'Multi-MonoBox':>16} {'Multi-MonoCharge':>19}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_sci(row['mono_box_minus_zero'], 14, 3)} "
            f"{fmt_sci(row['mono_charge_minus_zero'], 16, 3)} {fmt_sci(row['multi_minus_zero'], 14, 3)} "
            f"{fmt_sci(row['mono_charge_minus_mono_box'], 20, 3)} {fmt_sci(row['multi_minus_mono_box'], 16, 3)} "
            f"{fmt_sci(row['multi_minus_mono_charge'], 19, 3)}"
        )


def diagnose(results, drift_summary, boundary_summary):
    drift_by_mode = {}
    for row in drift_summary:
        drift_by_mode[row['mode']] = row['max_abs_drift']

    max_center_delta = max(abs(row['mono_charge_minus_mono_box']) for row in boundary_summary if row['mono_charge_minus_mono_box'] is not None)
    max_multi_vs_charge = max(abs(row['multi_minus_mono_charge']) for row in boundary_summary if row['multi_minus_mono_charge'] is not None)
    max_multi_vs_box = max(abs(row['multi_minus_mono_box']) for row in boundary_summary if row['multi_minus_mono_box'] is not None)
    max_boundary_vs_zero = max(abs(row['multi_minus_zero']) for row in boundary_summary if row['multi_minus_zero'] is not None)

    best_boundary_box_drift = min(
        drift_by_mode.get('monopole_box_center', float('inf')),
        drift_by_mode.get('monopole_charge_center', float('inf')),
        drift_by_mode.get('multipole_dirichlet', float('inf')),
    )

    if best_boundary_box_drift >= 0.20:
        label = 'box-dominated'
    elif max_boundary_vs_zero >= 0.20 or max_center_delta >= 0.10:
        label = 'boundary-dominated'
    elif best_boundary_box_drift <= 0.05 and max_center_delta <= 0.05 and max_multi_vs_charge <= 0.05 and max_multi_vs_box <= 0.05:
        label = 'kinetic-suspected-next'
    else:
        label = 'mixed'

    return {
        'label': label,
        'max_box_drift_zero': drift_by_mode.get('zero_dirichlet'),
        'max_box_drift_mono_box': drift_by_mode.get('monopole_box_center'),
        'max_box_drift_mono_charge': drift_by_mode.get('monopole_charge_center'),
        'max_box_drift_multi': drift_by_mode.get('multipole_dirichlet'),
        'max_center_mode_delta': max_center_delta,
        'max_multi_vs_mono_box': max_multi_vs_box,
        'max_multi_vs_mono_charge': max_multi_vs_charge,
        'max_boundary_vs_zero': max_boundary_vs_zero,
    }


def print_diagnosis(diag):
    print("=== Diagnosis Summary ===")
    print(f"label: {diag['label']}")
    print(f"max_box_drift_zero         = {diag['max_box_drift_zero']:.6f}")
    print(f"max_box_drift_mono_box     = {diag['max_box_drift_mono_box']:.6f}")
    print(f"max_box_drift_mono_charge  = {diag['max_box_drift_mono_charge']:.6f}")
    print(f"max_box_drift_multi        = {diag['max_box_drift_multi']:.6f}")
    print(f"max_center_mode_delta      = {diag['max_center_mode_delta']:.6f}")
    print(f"max_multi_vs_mono_box      = {diag['max_multi_vs_mono_box']:.6f}")
    print(f"max_multi_vs_mono_charge   = {diag['max_multi_vs_mono_charge']:.6f}")
    print(f"max_boundary_vs_zero       = {diag['max_boundary_vs_zero']:.6f}")
    print()
    print("Interpret conservatively:")
    print("- If the best boundary mode still has large box drift, keep improving boundary/exterior behavior.")
    print("- If charge_center reduces box drift relative to box_center, center choice is helping and worth keeping.")
    print("- If multipole and the better monopole center are already close and box drift is small, the next suspect is likely the adaptive kinetic/Laplacian discretization.")


def main() -> int:
    print("=== H2 Boundary Follow-up Study ===")
    print("Note: this is not a final benchmark.")
    print("Note: the current goal is to see at what box size H2 stops being box-dominated.")
    print("Note: the current goal is also to test whether monopole(charge_center) is more stable than monopole(box_center).")
    print("Note: the current default multipole_dirichlet path is included only as a structural comparison, not as a claim of final physical correctness.")
    print(f"Setup: d={DISTANCE} Bohr, boxes={BOXES}, h_min={H_MIN}, h_max={H_MAX}, r_core={R_CORE}, stretch_beta={STRETCH_BETA}")
    print(f"SCF: max_iter={SCF_KWARGS['max_iter']}, mix_alpha={SCF_KWARGS['mix_alpha']}, tolerance={SCF_KWARGS['tolerance']}")
    print()

    results = build_results()
    drift_summary = build_box_drift_summary(results)
    boundary_summary = build_boundary_summary(results)
    diagnosis = diagnose(results, drift_summary, boundary_summary)

    print_result_table(results)
    print()
    print_box_drift_summary(drift_summary)
    print()
    print_boundary_summary(boundary_summary)
    print()
    print_diagnosis(diagnosis)

    all_ok = True
    all_ok &= check(
        'runs_ok',
        all(row['completed'] and row['all_finite'] for row in results),
        'all four H2 boundary paths completed with finite outputs',
    )
    all_ok &= check(
        'electron_counts_ok',
        all((row['electron_error'] is not None and row['electron_error'] <= ELECTRON_TOL) for row in results),
        'electron counts stayed within tolerance',
    )
    all_ok &= check(
        'orbital_norms_ok',
        all((row['orbital_norm_maxdev'] is not None and row['orbital_norm_maxdev'] <= NORM_TOL) for row in results),
        'orbital norms stayed within tolerance',
    )

    if all_ok:
        print('OVERALL: PASS')
        return 0

    print('OVERALL: FAIL')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
