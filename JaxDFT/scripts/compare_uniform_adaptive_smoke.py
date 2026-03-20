"""Very conservative uniform-vs-adaptive SCF smoke comparison.

This script is intentionally a smoke/stability check, not a benchmark.
The current adaptive Hartree path still uses a zero-Dirichlet box Poisson
prototype, so uniform and adaptive total energies must not be interpreted as a
strict physical accuracy comparison. The more meaningful checks here are:
- whether each tiny case runs to completion
- whether the outputs stay finite and normalized
- whether adaptive results vary smoothly when adaptive parameters are perturbed
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
    from JaxDFT.src.backends import AdaptiveBackend, UniformBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend, UniformBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


SCF_KWARGS = {
    "max_iter": 2,
    "mix_alpha": 0.25,
    "tolerance": 1.0e-3,
}
ELECTRON_TOL = 5.0e-3
NORM_TOL = 2.0e-2
SMOOTH_ENERGY_TOL = 2.0


def fmt_float(value, width=10, precision=6):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=10, precision=3):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def fmt_bool(value):
    return ("PASS" if value else "FAIL") if value is not None else "-"


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


def run_case(case_name, elements, coords, config, key_seed):
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    backend_name = config["backend"]
    backend = UniformBackend() if backend_name == "uniform" else AdaptiveBackend()
    result = {
        "case": case_name,
        "config": config["name"],
        "backend": backend.name,
        "completed": False,
        "all_finite": False,
        "passed": False,
        "energy": None,
        "electron_count": None,
        "electron_error": None,
        "orbital_norm_maxdev": None,
        "eigvals_finite": False,
        "rho_min": None,
        "rho_max": None,
        "shape": None,
        "error": None,
    }

    try:
        pseudos = load_pseudopotentials(elements, pseudo_dir)
        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        n_electrons, n_bands, occ = build_occ(pseudos)
        box = jnp.asarray(config["box"], dtype=jnp.float32)
        spacing = float(config["spacing"])

        if backend_name == "uniform":
            grid = backend.create_grid(spacing, box)
        else:
            adaptive_kwargs = dict(config["adaptive_kwargs"])
            grid = backend.create_grid(spacing, box, atom_coords=coords, **adaptive_kwargs)

        V_loc = backend.build_local_potential(grid, coords, pseudos)
        key = jax.random.PRNGKey(key_seed)
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
            grid,
            coords,
            n_bands,
            occ,
            V_loc,
            pseudos,
            key=key,
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
        electron_error = abs(electron_count - float(jnp.sum(occ)))
        passed = (
            all_finite
            and electron_error <= ELECTRON_TOL
            and norm_maxdev <= NORM_TOL
        )

        result.update({
            "completed": True,
            "all_finite": all_finite,
            "passed": passed,
            "energy": float(energy),
            "electron_count": electron_count,
            "electron_error": electron_error,
            "orbital_norm_maxdev": norm_maxdev,
            "eigvals_finite": bool(jnp.all(jnp.isfinite(eigvals))),
            "rho_min": float(jnp.min(rho)),
            "rho_max": float(jnp.max(rho)),
            "shape": tuple(int(n) for n in grid.shape),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def build_smoothness_rows(results):
    rows = []
    grouped = {}
    for row in results:
        grouped.setdefault(row["case"], []).append(row)

    for case_name, case_rows in grouped.items():
        adaptive_rows = [r for r in case_rows if r["config"].startswith("adaptive_")]
        adaptive_rows.sort(key=lambda r: r["config"])
        if len(adaptive_rows) < 2:
            continue
        for left, right in zip(adaptive_rows[:-1], adaptive_rows[1:]):
            if left["completed"] and right["completed"] and left["all_finite"] and right["all_finite"]:
                delta_energy = right["energy"] - left["energy"]
                delta_electron = right["electron_count"] - left["electron_count"]
                delta_norm = right["orbital_norm_maxdev"] - left["orbital_norm_maxdev"]
                stable = (
                    abs(delta_energy) <= SMOOTH_ENERGY_TOL
                    and abs(delta_electron) <= ELECTRON_TOL
                    and abs(delta_norm) <= NORM_TOL
                )
            else:
                delta_energy = None
                delta_electron = None
                delta_norm = None
                stable = False
            rows.append({
                "case": case_name,
                "pair": f"{left['config']} -> {right['config']}",
                "stable": stable,
                "delta_energy": delta_energy,
                "delta_electron": delta_electron,
                "delta_norm": delta_norm,
            })
    return rows


def print_result_table(results):
    print("=== Result Table ===")
    header = (
        f"{'Case':<10} {'Config':<20} {'Backend':<12} {'Done':<5} {'Finite':<6} "
        f"{'Eig':<5} {'Energy':>11} {'N':>8} {'dN':>10} {'NormDev':>10} "
        f"{'rho_min':>10} {'rho_max':>10} {'Shape':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        shape_str = str(row["shape"]) if row["shape"] is not None else "-"
        print(
            f"{row['case']:<10} {row['config']:<20} {row['backend']:<12} "
            f"{fmt_bool(row['completed']):<5} {fmt_bool(row['all_finite']):<6} {fmt_bool(row['eigvals_finite']):<5} "
            f"{fmt_float(row['energy'], 11, 6)} {fmt_float(row['electron_count'], 8, 4)} "
            f"{fmt_sci(row['electron_error'], 10, 2)} {fmt_sci(row['orbital_norm_maxdev'], 10, 2)} "
            f"{fmt_float(row['rho_min'], 10, 4)} {fmt_float(row['rho_max'], 10, 4)} {shape_str:<12}"
        )
        if row["error"] is not None:
            print(f"  error: {row['error']}")


def print_smoothness_table(rows):
    print("=== Adaptive Smoothness Table ===")
    header = f"{'Case':<10} {'Pair':<40} {'Stable':<6} {'dE':>12} {'dN':>12} {'dNorm':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<10} {row['pair']:<40} {fmt_bool(row['stable']):<6} "
            f"{fmt_sci(row['delta_energy'], 12, 3)} {fmt_sci(row['delta_electron'], 12, 3)} {fmt_sci(row['delta_norm'], 12, 3)}"
        )


def main() -> int:
    print("=== Uniform vs Adaptive Smoke Comparison ===")
    print("Note: this is not a formal benchmark.")
    print("Note: adaptive Hartree currently uses a zero-Dirichlet box Poisson prototype.")
    print("Note: uniform_ref and adaptive_uniformlike must not be interpreted as a strict physical one-to-one energy comparison.")
    print("Note: the key question here is whether tiny adaptive runs remain finite, normalized, and smooth under mild parameter changes.")
    print()

    cases = [
        {
            "name": "H_atom",
            "elements": ["H"],
            "coords": jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            "configs": [
                {
                    "name": "uniform_ref",
                    "backend": "uniform",
                    "spacing": 1.0,
                    "box": [4.0, 4.0, 4.0],
                },
                {
                    "name": "adaptive_uniformlike",
                    "backend": "adaptive",
                    "spacing": 1.0,
                    "box": [4.0, 4.0, 4.0],
                    "adaptive_kwargs": {
                        "h_max": 1.0,
                        "r_core": 0.5,
                        "stretch_beta": 0.0,
                    },
                },
                {
                    "name": "adaptive_mild",
                    "backend": "adaptive",
                    "spacing": 1.0,
                    "box": [4.0, 4.0, 4.0],
                    "adaptive_kwargs": {
                        "h_max": 1.2,
                        "r_core": 0.5,
                        "stretch_beta": 3.0,
                    },
                },
            ],
        },
        {
            "name": "H2_fixed",
            "elements": ["H", "H"],
            "coords": jnp.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=jnp.float32),
            "configs": [
                {
                    "name": "uniform_ref",
                    "backend": "uniform",
                    "spacing": 1.0,
                    "box": [5.0, 5.0, 5.0],
                },
                {
                    "name": "adaptive_uniformlike",
                    "backend": "adaptive",
                    "spacing": 1.0,
                    "box": [5.0, 5.0, 5.0],
                    "adaptive_kwargs": {
                        "h_max": 1.0,
                        "r_core": 0.7,
                        "stretch_beta": 0.0,
                    },
                },
                {
                    "name": "adaptive_mild",
                    "backend": "adaptive",
                    "spacing": 1.0,
                    "box": [5.0, 5.0, 5.0],
                    "adaptive_kwargs": {
                        "h_max": 1.2,
                        "r_core": 0.7,
                        "stretch_beta": 3.0,
                    },
                },
            ],
        },
    ]

    results = []
    key_seed = 0
    for case in cases:
        print(f"--- Running {case['name']} ---")
        for config in case["configs"]:
            result = run_case(case["name"], case["elements"], case["coords"], config, key_seed)
            key_seed += 1
            results.append(result)
            if result["error"] is None:
                print(
                    f"  {config['name']}: completed={result['completed']} finite={result['all_finite']} "
                    f"energy={result['energy']:.6f} N={result['electron_count']:.6f} shape={result['shape']}"
                )
            else:
                print(f"  {config['name']}: error={result['error']}")
        print()

    smoothness_rows = build_smoothness_rows(results)
    print_result_table(results)
    print()
    print_smoothness_table(smoothness_rows)
    print()

    runs_ok = all(row["passed"] for row in results)
    smooth_ok = all(row["stable"] for row in smoothness_rows)
    check("runs_ok", runs_ok, "all tiny uniform/adaptive runs completed with finite outputs and reasonable normalization")
    check("adaptive_smoothness_ok", smooth_ok, "adaptive_uniformlike -> adaptive_mild stayed finite without large jumps")

    print()
    print("=== Interpretation ===")
    print("Use this script as a stability/smoothness check only.")
    print("If adaptive runs complete, keep electron count and orbital norms under control, and change smoothly with mild adaptive parameters, the current adaptive path is behaving reasonably.")
    print("Do not read small uniform-vs-adaptive energy differences here as a formal physics benchmark, because the Hartree boundary conditions still differ.")

    if runs_ok and smooth_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
