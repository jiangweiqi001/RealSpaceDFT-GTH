#!/usr/bin/env python3
"""CO grid convergence: same-density Hartree vs spacing + full SCF (Pulay/Kerker diagnostic config).

Runs experiments only; does not modify Poisson, default mixer, or projector formulas.

Same-density: ``diagnose_same_density_components.run_same_density_diagnostic`` for CO.

Full SCF: ``benchmark_systems.run_benchmark`` with fixed Pulay/Kerker/RMS settings.

Warning: ``target_spacing=0.08`` yields ~225^3 grid points per SCF step; expect long runtime
and high memory. If outer SCF hits ``max_iter`` with ``density_converged=false``, raise
``--max-iter`` and often ``--orbital-max-iter`` (e.g. 500 and 60) without changing Poisson or
mixer *formulas*—only iteration budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _co_system():
    from JaxDFT.scripts.benchmark_systems import get_benchmark_systems

    for s in get_benchmark_systems():
        if s.name == "CO":
            return s
    raise RuntimeError("CO benchmark system missing")


def _run_same_density_rows(
    co,
    spacings: Sequence[float],
    box_size: float,
    pseudo_dir: str,
) -> list[dict[str, Any]]:
    from JaxDFT.scripts.diagnose_same_density_components import run_same_density_diagnostic

    rows: list[dict[str, Any]] = []
    for sp in spacings:
        t0 = time.time()
        try:
            raw = run_same_density_diagnostic(
                co,
                float(sp),
                float(box_size),
                pseudo_dir,
                calculation_dtype="float32",
                grid_phase=0.0,
            )
        except Exception as exc:
            rows.append(
                {
                    "target_spacing": float(sp),
                    "error": str(exc),
                    "seconds": time.time() - t0,
                }
            )
            continue
        rows.append(
            {
                "target_spacing": raw["target_spacing"],
                "actual_spacing": raw["actual_spacing"],
                "box_size": raw["box_size"],
                "grid_shape": list(raw["grid_shape"]),
                "seconds": time.time() - t0,
                "delta_hartree_vs_pyscf_coul_mHa": raw["delta_hartree_vs_pyscf_coul_mHa"],
                "delta_xc_vs_pyscf_exc_mHa": raw["delta_xc_vs_pyscf_exc_mHa"],
                "rho_electron_count_error": raw["rho_electron_count_error"],
                "pyscf_converged": raw["pyscf_converged"],
            }
        )
    return rows


def _run_scf_rows(
    co,
    spacings: Sequence[float],
    box_size: float,
    pseudo_dir: str,
    *,
    max_iter: int,
    mix_alpha: float,
    tolerance: float,
    orbital_max_iter: int,
    orbital_tolerance: float,
    energy_tolerance: float,
) -> list[dict[str, Any]]:
    from JaxDFT.scripts.benchmark_systems import run_benchmark

    rows: list[dict[str, Any]] = []
    for sp in spacings:
        r = run_benchmark(
            co,
            float(sp),
            float(box_size),
            max_iter=max_iter,
            mix_alpha=mix_alpha,
            tolerance=tolerance,
            pseudo_dir=pseudo_dir,
            orbital_max_iter=orbital_max_iter,
            orbital_tolerance=orbital_tolerance,
            orbital_preconditioner="none",
            orbital_preconditioner_shift=1.0,
            mixing_mode="pulay",
            anderson_regularization=1e-4,
            anderson_history=3,
            mixing_safeguard="none",
            mixing_safeguard_factor=1.0,
            scf_convergence_metric="rms",
            energy_tolerance=energy_tolerance,
            pulay_residual_metric="kerker",
            pulay_kerker_k0=2.0,
            laplacian_order=8,
            calculation_dtype="float32",
            grid_phase=0.0,
        )
        rows.append(
            {
                "spacing": r.actual_spacing,
                "target_spacing": r.target_spacing,
                "grid_shape": list(r.grid_shape),
                "runtime_seconds": r.seconds,
                "final_energy": r.jaxdft_energy_ha,
                "energy_last20_mean": r.energy_last20_mean,
                "energy_last20_std": r.energy_last20_std,
                "total_error_mHa": r.error_mha,
                "local_pseudopotential_energy": r.local_pseudopotential_energy,
                "nonlocal_pseudopotential_energy": r.nonlocal_pseudopotential_energy,
                "hartree_energy": r.hartree_energy,
                "xc_energy": r.xc_energy,
                "density_rms_diff": r.density_rms_diff,
                "energy_delta_last10_max": r.energy_delta_last10_max,
                "scf_status": r.scf_status,
                "scf_iterations": r.scf_iterations,
                "density_converged": r.density_converged,
                "energy_converged": r.energy_converged,
                "orbital_converged": r.orbital_converged,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-size", type=float, default=18.0)
    p.add_argument(
        "--pseudo-dir",
        default=os.path.join(REPO_ROOT, "JaxDFT", "data", "gth_potentials"),
    )
    p.add_argument(
        "--same-density-spacings",
        type=float,
        nargs="+",
        default=[0.14, 0.12, 0.10, 0.09, 0.08],
        help="Spacings for diagnose_same_density_components (default includes 0.09).",
    )
    p.add_argument(
        "--scf-spacings",
        type=float,
        nargs="+",
        default=[0.14, 0.12, 0.10, 0.08],
        help="Spacings for full JaxDFT SCF benchmark (default 0.14..0.08).",
    )
    p.add_argument("--skip-same-density", action="store_true")
    p.add_argument("--skip-scf", action="store_true")
    p.add_argument("--output", type=str, default="", help="Write JSON to this path (optional).")
    p.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="SCF outer max iterations (raise for fine grids, e.g. 500 for dx=0.08).",
    )
    p.add_argument(
        "--orbital-max-iter",
        type=int,
        default=30,
        help="Inner orbital / subspace iteration cap per SCF step.",
    )
    p.add_argument("--orbital-tolerance", type=float, default=1e-5, help="Inner orbital residual tolerance.")
    p.add_argument(
        "--tolerance",
        type=float,
        default=3e-5,
        help="Outer SCF density residual threshold (uses RMS metric in this harness).",
    )
    p.add_argument(
        "--energy-tolerance",
        type=float,
        default=5e-6,
        help="Outer SCF energy stability tolerance (Ha, last-10 max |ΔE|) passed to energy_and_forces.",
    )
    p.add_argument(
        "--mix-alpha",
        type=float,
        default=0.2,
        help="Linear/Pulay mixing alpha (diagnostic default 0.2).",
    )
    args = p.parse_args(argv)

    co = _co_system()
    report: dict[str, Any] = {
        "config": {
            "system": "CO",
            "box_size": args.box_size,
            "scf": {
                "mixing_mode": "pulay",
                "pulay_history": 3,
                "pulay_regularization": 1e-4,
                "mix_alpha": args.mix_alpha,
                "pulay_residual_metric": "kerker",
                "pulay_kerker_k0": 2.0,
                "scf_convergence_metric": "rms",
                "tolerance": args.tolerance,
                "max_iter": args.max_iter,
                "orbital_max_iter": args.orbital_max_iter,
                "orbital_tolerance": args.orbital_tolerance,
                "orbital_preconditioner": "none",
                "energy_tolerance": args.energy_tolerance,
            },
        },
        "same_density_hartree_mHa": [],
        "scf_spacing_convergence": [],
    }

    if not args.skip_same_density:
        report["same_density_hartree_mHa"] = _run_same_density_rows(
            co, args.same_density_spacings, args.box_size, args.pseudo_dir
        )
    if not args.skip_scf:
        report["scf_spacing_convergence"] = _run_scf_rows(
            co,
            args.scf_spacings,
            args.box_size,
            args.pseudo_dir,
            max_iter=args.max_iter,
            mix_alpha=args.mix_alpha,
            tolerance=args.tolerance,
            orbital_max_iter=args.orbital_max_iter,
            orbital_tolerance=args.orbital_tolerance,
            energy_tolerance=args.energy_tolerance,
        )

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
