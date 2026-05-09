#!/usr/bin/env python3
"""Two-stage SCF: coarse-grid solve, interpolate ``rho``, warm-start fine-grid SCF.

Uses trilinear interpolation (``JaxDFT.src.continuation.interpolate_rho_trilinear``) and
``energy_and_forces(..., initial_rho=...)``. Typical use: CO ``dx=0.14`` then ``dx=0.10`` to
reduce wall time versus a cold start on the fine grid.

This script intentionally benchmarks **one** coarse→fine hop (two-stage chain). Longer
multi-stage chains are not always more cost-effective; use JSON fields
``two_stage_chain_wall_seconds`` vs ``cold_fine_wall_seconds`` (with ``--compare-cold``)
to check whether **total** two-stage time beats a single cold fine solve at the same
diagnostic tolerances.

Example::

    python3 JaxDFT/scripts/scf_continuation_benchmark.py \\
      --compare-cold \\
      --output continuation_co_010.json
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


def _select_system(name: str):
    from JaxDFT.scripts.benchmark_systems import get_benchmark_systems

    u = name.upper()
    for s in get_benchmark_systems():
        if s.name.upper() == u:
            return s
    raise ValueError(f"Unknown system {name!r}")


def _make_grid(target_spacing: float, box_size: float, dtype, grid_phase: float):
    import jax.numpy as jnp

    from JaxDFT.src.hamiltonian import create_grid

    n_intervals = int(round(box_size / target_spacing))
    actual_spacing = box_size / n_intervals
    grid = create_grid(actual_spacing, [box_size, box_size, box_size], dtype=dtype, phase=grid_phase)
    return grid, float(actual_spacing)


def _n_electrons(pseudos) -> float:
    import jax.numpy as jnp

    return float(jnp.sum(jnp.asarray([p["q"] for p in pseudos], dtype=jnp.float32)))


def _run_stage(
    grid,
    coords,
    pseudos,
    *,
    max_iter: int,
    mix_alpha: float,
    tolerance: float,
    orbital_max_iter: int,
    orbital_tolerance: float,
    energy_tolerance: float,
    mixing_mode: str,
    anderson_history: int,
    anderson_regularization: float,
    scf_convergence_metric: str,
    pulay_residual_metric: str,
    pulay_kerker_k0: float,
    laplacian_order: int,
    initial_rho,
    key,
) -> tuple[float, dict[str, Any], float]:
    from JaxDFT.src.solver import energy_and_forces

    t0 = time.time()
    energy, _, info = energy_and_forces(
        grid,
        coords,
        pseudos,
        max_iter,
        mix_alpha,
        tolerance,
        key,
        return_info=True,
        orbital_max_iter=orbital_max_iter,
        orbital_tolerance=orbital_tolerance,
        orbital_preconditioner="none",
        orbital_preconditioner_shift=1.0,
        mixing_mode=mixing_mode,
        anderson_regularization=anderson_regularization,
        anderson_history=anderson_history,
        mixing_safeguard="none",
        mixing_safeguard_factor=1.0,
        scf_convergence_metric=scf_convergence_metric,
        energy_tolerance=energy_tolerance,
        pulay_residual_metric=pulay_residual_metric,
        pulay_kerker_k0=pulay_kerker_k0,
        laplacian_order=laplacian_order,
        initial_rho=initial_rho,
    )
    elapsed = time.time() - t0
    return float(energy), info, elapsed


def _slim_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "scf_iterations": int(info["scf_iterations"]),
        "density_converged": bool(info["density_converged"]),
        "energy_converged": bool(info["energy_converged"]),
        "scf_converged": bool(info["scf_converged"]),
        "orbital_converged": bool(info["orbital_converged"]),
        "density_rms_diff": float(info["density_rms_diff"]),
        "energy_delta_last10_max": float(info["energy_delta_last10_max"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import jax
    import jax.numpy as jnp

    from JaxDFT.scripts.benchmark_systems import run_pyscf_reference
    from JaxDFT.src.continuation import interpolate_rho_trilinear
    from JaxDFT.src.io import load_pseudopotentials

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", type=str, default="CO")
    p.add_argument("--box-size", type=float, default=18.0)
    p.add_argument("--coarse-dx", type=float, default=0.14)
    p.add_argument("--fine-dx", type=float, default=0.10)
    p.add_argument("--pseudo-dir", default=os.path.join(REPO_ROOT, "JaxDFT", "data", "gth_potentials"))
    p.add_argument("--coarse-max-iter", type=int, default=250)
    p.add_argument("--fine-max-iter", type=int, default=250)
    p.add_argument("--mix-alpha", type=float, default=0.2)
    p.add_argument("--tolerance", type=float, default=3e-5)
    p.add_argument(
        "--energy-tolerance",
        type=float,
        default=5e-6,
        help="Ha: same as solver energy_converged test vs energy_delta_last10_max.",
    )
    p.add_argument("--orbital-max-iter", type=int, default=30)
    p.add_argument("--orbital-tolerance", type=float, default=1e-5)
    p.add_argument("--anderson-history", type=int, default=3)
    p.add_argument("--anderson-regularization", type=float, default=1e-4)
    p.add_argument("--pulay-kerker-k0", type=float, default=2.0)
    p.add_argument("--calculation-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--grid-phase", type=float, default=0.0)
    p.add_argument("--compare-cold", action="store_true", help="Also run fine grid without warm-start.")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args(argv)

    jax.config.update("jax_enable_x64", args.calculation_dtype == "float64")
    dtype = jnp.float64 if args.calculation_dtype == "float64" else jnp.float32

    system = _select_system(args.system)
    grid_c, spacing_c = _make_grid(args.coarse_dx, args.box_size, dtype, args.grid_phase)
    grid_f, spacing_f = _make_grid(args.fine_dx, args.box_size, dtype, args.grid_phase)
    pseudos = load_pseudopotentials(list(system.symbols), args.pseudo_dir)
    coords = jnp.array(system.coords_bohr, dtype=dtype)
    n_elec = _n_electrons(pseudos)
    key_c = jax.random.PRNGKey(0)
    key_f = jax.random.PRNGKey(1)
    key_cold = jax.random.PRNGKey(2)

    common = dict(
        mix_alpha=args.mix_alpha,
        tolerance=args.tolerance,
        orbital_max_iter=args.orbital_max_iter,
        orbital_tolerance=args.orbital_tolerance,
        energy_tolerance=args.energy_tolerance,
        mixing_mode="pulay",
        anderson_history=args.anderson_history,
        anderson_regularization=args.anderson_regularization,
        scf_convergence_metric="rms",
        pulay_residual_metric="kerker",
        pulay_kerker_k0=args.pulay_kerker_k0,
        laplacian_order=8,
    )

    wall: dict[str, float] = {}
    t0 = time.time()
    e_coarse, info_c, wall_coarse = _run_stage(
        grid_c,
        coords,
        pseudos,
        max_iter=args.coarse_max_iter,
        initial_rho=None,
        key=key_c,
        **common,
    )
    wall["coarse_scf_seconds"] = wall_coarse

    t_interp0 = time.time()
    rho_f0 = interpolate_rho_trilinear(grid_c, info_c["density"], grid_f, n_elec)
    wall["interpolate_seconds"] = time.time() - t_interp0

    e_fine_w, info_fw, wall_fine_w = _run_stage(
        grid_f,
        coords,
        pseudos,
        max_iter=args.fine_max_iter,
        initial_rho=rho_f0,
        key=key_f,
        **common,
    )
    wall["fine_scf_warm_seconds"] = wall_fine_w

    two_stage_chain_wall = float(
        wall["coarse_scf_seconds"] + wall["interpolate_seconds"] + wall["fine_scf_warm_seconds"]
    )

    ref = float(run_pyscf_reference(system))
    err_w = 1000.0 * (e_fine_w - ref)

    report: dict[str, Any] = {
        "system": system.name,
        "box_size": args.box_size,
        "two_stage_chain_wall_seconds": two_stage_chain_wall,
        "coarse": {
            "target_dx": args.coarse_dx,
            "actual_dx": spacing_c,
            "grid_shape": list(grid_c.shape),
            "energy_ha": e_coarse,
            "wall_seconds": wall_coarse,
            "summary": _slim_info(info_c),
        },
        "fine": {
            "target_dx": args.fine_dx,
            "actual_dx": spacing_f,
            "grid_shape": list(grid_f.shape),
            "energy_ha_warm": e_fine_w,
            "total_error_mHa_warm": err_w,
            "pyscf_reference_ha": ref,
            "wall_seconds_warm": wall_fine_w,
            "summary_warm": _slim_info(info_fw),
        },
        "wall_breakdown_seconds": wall,
    }

    if args.compare_cold:
        e_fine_c, info_fc, wall_fine_c = _run_stage(
            grid_f,
            coords,
            pseudos,
            max_iter=args.fine_max_iter,
            initial_rho=None,
            key=key_cold,
            **common,
        )
        err_c = 1000.0 * (e_fine_c - ref)
        report["fine"]["energy_ha_cold"] = e_fine_c
        report["fine"]["total_error_mHa_cold"] = err_c
        report["fine"]["wall_seconds_cold"] = wall_fine_c
        report["fine"]["summary_cold"] = _slim_info(info_fc)
        report["fine"]["warm_vs_cold"] = {
            "scf_iterations_delta_warm_minus_cold": int(info_fw["scf_iterations"])
            - int(info_fc["scf_iterations"]),
            "wall_seconds_cold_minus_warm": wall_fine_c - wall_fine_w,
        }
        report["cold_fine_wall_seconds"] = wall_fine_c
        # Positive => two-stage total (coarse + interp + warm fine) is faster than cold fine only.
        report["cold_fine_minus_two_stage_chain_wall_seconds"] = wall_fine_c - two_stage_chain_wall
        report["accuracy_mHa"] = {
            "warm_total_error_mHa": err_w,
            "cold_total_error_mHa": err_c,
            "abs_total_error_mHa_diff_warm_vs_cold": abs(err_w - err_c),
        }

    report["benchmark_script_wall_seconds"] = time.time() - t0
    # Historical key: entire script wall time (includes optional cold fine when --compare-cold).
    report["total_wall_seconds"] = report["benchmark_script_wall_seconds"]

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
