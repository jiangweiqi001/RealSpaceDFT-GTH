"""Single-point H2 (R=1.4 Bohr) energy-breakdown comparison.

Compares at the same geometry:
  - Adaptive RealSpace
  - Uniform RealSpace
  - PySCF reference

The goal is to identify which energy component first becomes abnormal in the
current adaptive H2 setup near equilibrium bond length.
"""

from __future__ import annotations

import argparse
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
    from JaxDFT.src.backends.adaptive import AdaptiveBackend
    from JaxDFT.src.backends.uniform import UniformBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends.adaptive import AdaptiveBackend
    from src.backends.uniform import UniformBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


SMALL = 1.0e-12


def case_key(seed: int, case_index: int) -> jax.Array:
    """Return a reproducible per-geometry key shared across verification scripts."""
    return jax.random.fold_in(jax.random.PRNGKey(seed), int(case_index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check H2 R=1.4 Bohr adaptive vs uniform vs PySCF energy breakdown.",
    )
    parser.add_argument("--R", type=float, default=1.4, help="H-H distance in Bohr. Default: 1.4")

    parser.add_argument("--box", type=float, default=30.0, help="Adaptive cubic box in Bohr. Default: 30.0")
    parser.add_argument("--h-min", type=float, default=0.25, help="Adaptive h_min. Default: 0.25")
    parser.add_argument("--h-max", type=float, default=0.80, help="Adaptive h_max. Default: 0.80")
    parser.add_argument("--r-core", type=float, default=1.0, help="Adaptive r_core. Default: 1.0")
    parser.add_argument("--stretch-beta", type=float, default=5.0, help="Adaptive stretch beta. Default: 5.0")
    parser.add_argument(
        "--hartree-boundary-mode",
        type=str,
        default="uniform_exterior",
        choices=["zero_dirichlet", "monopole_dirichlet", "multipole_dirichlet", "uniform_exterior"],
        help="Adaptive Hartree boundary mode. Default: uniform_exterior",
    )
    parser.add_argument(
        "--kinetic-mode",
        type=str,
        default="prototype_fd2",
        choices=["prototype_fd2", "symmetric_fv"],
        help="Adaptive kinetic mode. Default: prototype_fd2",
    )

    parser.add_argument("--uniform-box", type=float, default=18.0, help="Uniform cubic box in Bohr. Default follows verify_h2.py: 18.0")
    parser.add_argument("--uniform-spacing", type=float, default=0.18, help="Uniform spacing. Default follows verify_h2.py: 0.18")

    parser.add_argument("--max-iter", type=int, default=120, help="SCF max iterations. Default: 120")
    parser.add_argument("--mix-alpha", type=float, default=0.30, help="SCF mixing alpha. Default: 0.30")
    parser.add_argument("--tolerance", type=float, default=1.0e-5, help="SCF tolerance. Default: 1e-5")
    parser.add_argument("--seed", type=int, default=42, help="Base PRNG seed. Default: 42")
    return parser.parse_args()


def fmt_float(value: float | None, width: int = 13, precision: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value: float | None, width: int = 13, precision: int = 3) -> str:
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


def orbital_norm_maxdev(grid, backend, eigvecs_flat) -> float:
    states = orbital_fields_from_flat(grid, eigvecs_flat)
    norms = jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(states)
    return float(jnp.max(jnp.abs(norms - 1.0)))


def decompose_realspace(method: str, backend, grid, coords, pseudos, occ, rho, eigvals, eigvecs, V_loc, V_H, eps_xc, v_xc):
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    eion = float(ion_ion_energy(coords, zion))
    e_total = float(total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, eion, backend=backend))

    states = orbital_fields_from_flat(grid, eigvecs)
    cache = backend.precompute_nonlocal(grid, coords, pseudos)
    occ_np = np.asarray(jnp.asarray(occ), dtype=np.float64)
    ts = 0.0
    enl = 0.0
    for i in range(states.shape[0]):
        psi = states[i]
        occ_i = float(occ_np[i])
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
    electron_count = float(backend.integrate(grid, rho))
    norm_dev = orbital_norm_maxdev(grid, backend, eigvecs)

    return {
        "method": method,
        "E_total": e_total,
        "Ts": ts,
        "Eloc": eloc,
        "Enl": enl,
        "Eh": eh,
        "Exc": exc,
        "Eion": eion,
        "electron_count": electron_count,
        "norm_dev": norm_dev,
        "E_sum": e_sum,
        "consistency_error": consistency_error,
        "relative_consistency_error": consistency_error / max(abs(e_total), SMALL),
    }


def run_realspace_case(method: str, backend, grid, coords, pseudos, n_bands, occ, args, key):
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid,
        coords,
        n_bands,
        occ,
        V_loc,
        pseudos,
        max_iter=args.max_iter,
        mix_alpha=args.mix_alpha,
        tolerance=args.tolerance,
        key=key,
        backend=backend,
    )
    return decompose_realspace(method, backend, grid, coords, pseudos, occ, rho, eigvals, eigvecs, V_loc, V_H, eps_xc, v_xc)


def run_pyscf_total_only(R: float):
    try:
        from pyscf import dft, gto
    except Exception as exc:
        return {
            "method": "PySCF",
            "E_total": None,
            "status": f"import_failed:{type(exc).__name__}",
        }

    try:
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {R}",
            unit="Bohr",
            basis="gth-tzvp",
            pseudo="gth-lda",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "lda,pz"
        e_total = float(mf.kernel())
        return {
            "method": "PySCF",
            "E_total": e_total,
            "status": "ok" if np.isfinite(e_total) else "nonfinite",
        }
    except Exception as exc:
        return {
            "method": "PySCF",
            "E_total": None,
            "status": f"failed:{type(exc).__name__}",
        }


def print_main_table(adaptive, uniform, pyscf):
    print("=== Energy Breakdown Table ===")
    header = (
        f"{'method':<12} {'E_total':>13} {'Ts':>13} {'Eloc':>13} {'Enl':>13} {'Eh':>13} {'Exc':>13} {'Eion':>13} {'N':>10} {'norm_dev':>11} {'cons_err':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in (adaptive, uniform):
        print(
            f"{row['method']:<12} {fmt_float(row['E_total'])} {fmt_float(row['Ts'])} {fmt_float(row['Eloc'])} {fmt_float(row['Enl'])} {fmt_float(row['Eh'])} {fmt_float(row['Exc'])} {fmt_float(row['Eion'])} {fmt_float(row['electron_count'], 10, 6)} {fmt_sci(row['norm_dev'], 11, 2)} {fmt_sci(row['consistency_error'])}"
        )
    print(
        f"{pyscf['method']:<12} {fmt_float(pyscf['E_total'])} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None, 10, 6)} {fmt_sci(None, 11, 2)} {fmt_sci(None)}"
    )
    print()
    print("Note: PySCF total energy follows verify_h2.py exactly (isolated molecule, gth-tzvp, gth-lda, lda,pz).")
    print("Note: PySCF sub-components are not emitted here because a strict Ts/Eloc/Enl split is not directly available in the same pseudopotential decomposition.")
    print()


def print_diff_table(adaptive, uniform, pyscf):
    print("=== Difference Table ===")
    header = f"{'pair':<22} {'dE_total':>13} {'dTs':>13} {'dEloc':>13} {'dEnl':>13} {'dEh':>13} {'dExc':>13}"
    print(header)
    print("-" * len(header))

    def line(name, a, b):
        if a is None or b is None:
            print(f"{name:<22} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)}")
            return
        print(
            f"{name:<22} {fmt_float(a['E_total'] - b['E_total'])} {fmt_float(a['Ts'] - b['Ts'])} {fmt_float(a['Eloc'] - b['Eloc'])} {fmt_float(a['Enl'] - b['Enl'])} {fmt_float(a['Eh'] - b['Eh'])} {fmt_float(a['Exc'] - b['Exc'])}"
        )

    line("Adaptive - Uniform", adaptive, uniform)
    print(f"{'Adaptive - PySCF':<22} {fmt_float(None if pyscf['E_total'] is None else adaptive['E_total'] - pyscf['E_total'])} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)}")
    print(f"{'Uniform - PySCF':<22} {fmt_float(None if pyscf['E_total'] is None else uniform['E_total'] - pyscf['E_total'])} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)} {fmt_float(None)}")
    print()


def diagnose(adaptive, uniform, pyscf):
    component_names = ["Ts", "Eloc", "Enl", "Eh", "Exc"]
    diffs = {name: abs(adaptive[name] - uniform[name]) for name in component_names}
    worst_name = max(diffs, key=diffs.get)
    total_consistent = abs(adaptive['consistency_error']) < 5e-3 and abs(uniform['consistency_error']) < 5e-3

    print("=== Diagnosis Summary ===")
    print(f"largest_adaptive_vs_uniform_component_gap: {worst_name} ({diffs[worst_name]:.6e})")
    if pyscf['E_total'] is not None:
        print(f"Adaptive - PySCF total: {adaptive['E_total'] - pyscf['E_total']:.6e}")
        print(f"Uniform  - PySCF total: {uniform['E_total'] - pyscf['E_total']:.6e}")
    else:
        print("PySCF total energy unavailable for direct delta.")

    if total_consistent:
        print("energy_accounting: consistent (this does not look like a bookkeeping / total-energy formula error)")
    else:
        print("energy_accounting: inconsistent (decomposition and total-energy formula should be checked before deeper physical interpretation)")

    if worst_name == "Eloc":
        print("diagnosis: Adaptive is being pulled away from Uniform primarily through Eloc, so the current positive/strange total energy does not look like a pure kinetic or XC issue.")
    elif worst_name == "Ts":
        print("diagnosis: Adaptive is being pulled away from Uniform primarily through Ts, so the current pathology looks kinetic-led.")
    elif worst_name in {"Eh", "Enl"}:
        print("diagnosis: Adaptive is being pulled away from Uniform primarily through the electrostatic/nonlocal side.")
    elif worst_name == "Exc":
        print("diagnosis: Adaptive is being pulled away from Uniform primarily through Exc, likely via density-response distortion rather than bookkeeping.")
    else:
        print("diagnosis: No single component cleanly dominates.")
    print()
    return worst_name, total_consistent


def main() -> int:
    args = parse_args()
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    base_pseudos = load_pseudopotentials(["H"], pseudo_dir)
    pseudos = [base_pseudos[0], base_pseudos[0]]
    _, n_bands, occ = build_occ(pseudos)
    coords = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, args.R]], dtype=jnp.float32)

    print("\n=== H2 R=1.4 Energy Breakdown Audit ===")
    print(f"R = {args.R} Bohr")
    print(
        "Adaptive setup: "
        f"box={args.box}, h_min={args.h_min}, h_max={args.h_max}, r_core={args.r_core}, "
        f"stretch_beta={args.stretch_beta}, hartree={args.hartree_boundary_mode}, kinetic={args.kinetic_mode}"
    )
    print(
        "Uniform setup: "
        f"box={args.uniform_box}, spacing={args.uniform_spacing} (verify_h2.py style baseline)"
    )
    print()

    adaptive_backend = AdaptiveBackend(
        hartree_boundary_mode=args.hartree_boundary_mode,
        kinetic_mode=args.kinetic_mode,
    )
    adaptive_grid = adaptive_backend.create_grid(
        spacing=args.h_min,
        box_size=[args.box, args.box, args.box],
        atom_coords=coords,
        h_min=args.h_min,
        h_max=args.h_max,
        r_core=args.r_core,
        stretch_beta=args.stretch_beta,
    )
    adaptive = run_realspace_case(
        "Adaptive",
        adaptive_backend,
        adaptive_grid,
        coords,
        pseudos,
        n_bands,
        occ,
        args,
        case_key(args.seed, 0),
    )

    uniform_backend = UniformBackend()
    uniform_grid = uniform_backend.create_grid(args.uniform_spacing, [args.uniform_box, args.uniform_box, args.uniform_box])
    uniform = run_realspace_case(
        "Uniform",
        uniform_backend,
        uniform_grid,
        coords,
        pseudos,
        n_bands,
        occ,
        args,
        case_key(args.seed, 0),
    )

    pyscf = run_pyscf_total_only(args.R)

    print_main_table(adaptive, uniform, pyscf)
    print_diff_table(adaptive, uniform, pyscf)
    worst_name, total_consistent = diagnose(adaptive, uniform, pyscf)

    overall_ok = True
    overall_ok &= check("adaptive_electron_count", abs(adaptive['electron_count'] - float(jnp.sum(occ))) <= 5e-3, f"N={adaptive['electron_count']:.6f}")
    overall_ok &= check("uniform_electron_count", abs(uniform['electron_count'] - float(jnp.sum(occ))) <= 5e-3, f"N={uniform['electron_count']:.6f}")
    overall_ok &= check("adaptive_norm_dev", adaptive['norm_dev'] <= 2e-2, f"maxdev={adaptive['norm_dev']:.3e}")
    overall_ok &= check("uniform_norm_dev", uniform['norm_dev'] <= 2e-2, f"maxdev={uniform['norm_dev']:.3e}")
    overall_ok &= check("adaptive_consistency", abs(adaptive['consistency_error']) <= 5e-3, f"E_total-E_sum={adaptive['consistency_error']:.3e}")
    overall_ok &= check("uniform_consistency", abs(uniform['consistency_error']) <= 5e-3, f"E_total-E_sum={uniform['consistency_error']:.3e}")
    overall_ok &= check("worst_component_identified", worst_name in {"Ts", "Eloc", "Enl", "Eh", "Exc"}, f"worst={worst_name}")

    print("=== Overall Summary ===")
    print(f"largest_component_gap = {worst_name}")
    print(f"energy_accounting_consistent = {total_consistent}")
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
