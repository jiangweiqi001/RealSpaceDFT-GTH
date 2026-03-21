"""Focused H2 finite-box truncation diagnosis on the adaptive backend.

This is not a final benchmark.
The current goal is to decide whether the remaining large H2 box drift under a
fixed adaptive Hartree boundary model looks more consistent with:
  1. Hartree/exterior boundary treatment, or
  2. finite-box wavefunction/density truncation (and the kinetic-side error it induces).

To keep the attribution clean, this script fixes a single adaptive Hartree path
(currently ``multipole_dirichlet``) and studies only how the SCF solution
changes as the box grows.
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
DRIFT_LARGE = 0.10
RATIO_SMALL = 1.0e-3
RATIO_MODERATE = 1.0e-2


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


def collect_face_values(field):
    arr = jnp.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D field, got shape {arr.shape}")
    return jnp.concatenate(
        [
            jnp.ravel(arr[0, 1:-1, 1:-1]),
            jnp.ravel(arr[-1, 1:-1, 1:-1]),
            jnp.ravel(arr[1:-1, 0, 1:-1]),
            jnp.ravel(arr[1:-1, -1, 1:-1]),
            jnp.ravel(arr[1:-1, 1:-1, 0]),
            jnp.ravel(arr[1:-1, 1:-1, -1]),
        ],
        axis=0,
    )


def face_stats(field, *, absolute: bool = False):
    face = collect_face_values(field)
    if absolute:
        face = jnp.abs(face)
    return float(jnp.max(face)), float(jnp.mean(face))


def run_case(grid, coords, pseudos, zion, occ, n_bands, key_seed):
    backend = AdaptiveBackend(hartree_boundary_mode="multipole_dirichlet")
    result = {
        "box": float(grid.box_size[0]),
        "completed": False,
        "all_finite": False,
        "energy": None,
        "electron_count": None,
        "electron_error": None,
        "orbital_norm_maxdev": None,
        "eig0": None,
        "shape": tuple(int(n) for n in grid.shape),
        "rho_face_max": None,
        "rho_face_mean": None,
        "rho_face_ratio": None,
        "psi0_face_max": None,
        "psi0_face_mean": None,
        "psi0_face_ratio": None,
        "vh_face_max": None,
        "vh_face_mean": None,
        "vh_face_ratio": None,
        "hartree_proxy": None,
        "rho_max": None,
        "psi0_max": None,
        "vh_max": None,
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
        psi0 = jnp.abs(eigvec_fields[0])
        electron_count = float(backend.integrate(grid, rho))
        norm_maxdev = float(jnp.max(jnp.abs(norms - 1.0)))
        hartree_proxy = float(0.5 * backend.integrate(grid, rho * V_H))

        rho_face_max, rho_face_mean = face_stats(rho)
        psi_face_max, psi_face_mean = face_stats(psi0)
        vh_face_max, vh_face_mean = face_stats(V_H)
        rho_max = float(jnp.max(rho))
        psi0_max = float(jnp.max(psi0))
        vh_max = float(jnp.max(V_H))

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
            "rho_face_max": rho_face_max,
            "rho_face_mean": rho_face_mean,
            "rho_face_ratio": rho_face_max / max(rho_max, 1.0e-30),
            "psi0_face_max": psi_face_max,
            "psi0_face_mean": psi_face_mean,
            "psi0_face_ratio": psi_face_max / max(psi0_max, 1.0e-30),
            "vh_face_max": vh_face_max,
            "vh_face_mean": vh_face_mean,
            "vh_face_ratio": vh_face_max / max(vh_max, 1.0e-30),
            "hartree_proxy": hartree_proxy,
            "rho_max": rho_max,
            "psi0_max": psi0_max,
            "vh_max": vh_max,
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


def print_result_table(results):
    print("=== Result Table ===")
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
        if row['error'] is not None:
            print(f"  error: {row['error']}")


def build_drift_summary(results):
    ref_energy = results[-1]['energy'] if results and results[-1]['energy'] is not None else None
    ref_hartree = results[-1]['hartree_proxy'] if results and results[-1]['hartree_proxy'] is not None else None
    summary = []
    max_energy_drift = None
    max_hartree_drift = None
    for row in results:
        dE = None if ref_energy is None or row['energy'] is None else row['energy'] - ref_energy
        dEH = None if ref_hartree is None or row['hartree_proxy'] is None else row['hartree_proxy'] - ref_hartree
        if dE is not None:
            max_energy_drift = abs(dE) if max_energy_drift is None else max(max_energy_drift, abs(dE))
        if dEH is not None:
            max_hartree_drift = abs(dEH) if max_hartree_drift is None else max(max_hartree_drift, abs(dEH))
        summary.append({
            'box': row['box'],
            'energy': row['energy'],
            'dE_vs_Lmax': dE,
            'hartree_proxy': row['hartree_proxy'],
            'dEH_vs_Lmax': dEH,
            'max_energy_drift': None,
            'max_hartree_drift': None,
        })
    for row in summary:
        row['max_energy_drift'] = max_energy_drift
        row['max_hartree_drift'] = max_hartree_drift
    return summary


def print_drift_summary(summary):
    print("=== Drift Summary ===")
    header = f"{'Box':>5} {'Energy':>12} {'dE_vs_Lmax':>14} {'E_H_proxy':>12} {'dEH_vs_Lmax':>14}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_float(row['energy'], 12, 6)} {fmt_sci(row['dE_vs_Lmax'], 14, 3)} "
            f"{fmt_float(row['hartree_proxy'], 12, 6)} {fmt_sci(row['dEH_vs_Lmax'], 14, 3)}"
        )


def print_boundary_amplitude_table(results):
    print("=== Boundary Amplitude Table ===")
    header = (
        f"{'Box':>5} {'rho_face_max':>12} {'rho_face_mean':>13} {'rho_ratio':>11} "
        f"{'psi_face_max':>12} {'psi_face_mean':>13} {'psi_ratio':>11} "
        f"{'VH_face_max':>12} {'VH_face_mean':>13} {'VH_ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_sci(row['rho_face_max'], 12, 3)} {fmt_sci(row['rho_face_mean'], 13, 3)} {fmt_sci(row['rho_face_ratio'], 11, 3)} "
            f"{fmt_sci(row['psi0_face_max'], 12, 3)} {fmt_sci(row['psi0_face_mean'], 13, 3)} {fmt_sci(row['psi0_face_ratio'], 11, 3)} "
            f"{fmt_sci(row['vh_face_max'], 12, 3)} {fmt_sci(row['vh_face_mean'], 13, 3)} {fmt_sci(row['vh_face_ratio'], 10, 3)}"
        )


def diagnose(results, drift_summary):
    max_energy_drift = drift_summary[0]['max_energy_drift'] if drift_summary else None
    max_hartree_drift = drift_summary[0]['max_hartree_drift'] if drift_summary else None
    largest = results[-1]

    rho_ratio = largest['rho_face_ratio']
    psi_ratio = largest['psi0_face_ratio']
    vh_ratio = largest['vh_face_ratio']

    if max_energy_drift is None:
        label = 'mixed'
    elif max_energy_drift >= DRIFT_LARGE and rho_ratio is not None and psi_ratio is not None and rho_ratio <= RATIO_SMALL and psi_ratio <= RATIO_SMALL:
        label = 'hartree-dominated'
    elif max_energy_drift >= DRIFT_LARGE and ((rho_ratio is not None and rho_ratio >= RATIO_MODERATE) or (psi_ratio is not None and psi_ratio >= RATIO_MODERATE)):
        label = 'wavefunction-truncation-dominated'
    else:
        label = 'mixed'

    return {
        'label': label,
        'max_energy_drift': max_energy_drift,
        'max_hartree_drift': max_hartree_drift,
        'largest_box': largest['box'],
        'rho_face_ratio_largest': rho_ratio,
        'psi_face_ratio_largest': psi_ratio,
        'vh_face_ratio_largest': vh_ratio,
    }


def print_diagnosis(diag):
    print("=== Diagnosis Summary ===")
    print(f"label: {diag['label']}")
    print(f"largest_box                = {diag['largest_box']:.1f}")
    print(f"max_energy_drift           = {diag['max_energy_drift']:.6f}")
    print(f"max_hartree_proxy_drift    = {diag['max_hartree_drift']:.6f}")
    print(f"rho_face_ratio_largest     = {diag['rho_face_ratio_largest']:.6e}")
    print(f"psi_face_ratio_largest     = {diag['psi_face_ratio_largest']:.6e}")
    print(f"vh_face_ratio_largest      = {diag['vh_face_ratio_largest']:.6e}")
    print()
    print("Interpret conservatively:")
    print("- If total-energy drift stays large while rho and |psi| are already tiny on the box faces, Hartree/exterior treatment remains the main suspect.")
    print("- If rho and |psi| are still visibly nonzero on the box faces and decay slowly with box size, finite-box wavefunction truncation is the stronger suspect.")
    print("- If both are true only partially, treat the result as mixed rather than over-claiming a single cause.")


def main() -> int:
    print("=== H2 Finite-Box Truncation Diagnosis ===")
    print("Note: this is not a final benchmark.")
    print("Note: this study fixes the adaptive Hartree path to multipole_dirichlet.")
    print("Note: the goal is to decide whether the remaining H2 box drift looks more Hartree/exterior-driven or wavefunction/density-truncation-driven.")
    print(f"Setup: d={DISTANCE} Bohr, boxes={BOXES}")
    print(f"Adaptive params: h_min={H_MIN}, h_max={H_MAX}, r_core={R_CORE}, stretch_beta={STRETCH_BETA}")
    print(f"SCF: max_iter={SCF_KWARGS['max_iter']}, mix_alpha={SCF_KWARGS['mix_alpha']}, tolerance={SCF_KWARGS['tolerance']}")
    print()

    results = build_results()
    drift_summary = build_drift_summary(results)
    diagnosis = diagnose(results, drift_summary)

    print_result_table(results)
    print()
    print_drift_summary(drift_summary)
    print()
    print_boundary_amplitude_table(results)
    print()
    print_diagnosis(diagnosis)

    all_ok = True
    all_ok &= check(
        'runs_ok',
        all(row['completed'] and row['all_finite'] for row in results),
        'all H2 finite-box runs completed with finite outputs',
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
