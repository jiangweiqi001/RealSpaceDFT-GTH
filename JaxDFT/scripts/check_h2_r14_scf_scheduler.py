"""Compare baseline adaptive SCF against the adaptive-only eigensolver scheduler."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends import AdaptiveBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import energy_and_forces
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.io import load_pseudopotentials
    from src.solver import energy_and_forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline adaptive SCF against the adaptive-only SCF-aware eigensolver scheduler.",
    )
    parser.add_argument("--R", type=float, default=1.4, help="H-H distance in Bohr. Default: 1.4")
    parser.add_argument("--box", type=float, default=30.0, help="Adaptive cubic box length in Bohr. Default: 30.0")
    parser.add_argument("--h-min", type=float, default=0.16, help="Adaptive h_min. Default: 0.16")
    parser.add_argument("--h-max", type=float, default=0.32, help="Adaptive h_max. Default: 0.32")
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
    parser.add_argument("--max-iter", type=int, default=120, help="SCF max iterations. Default: 120")
    parser.add_argument("--mix-alpha", type=float, default=0.30, help="SCF mixing alpha. Default: 0.30")
    parser.add_argument("--tolerance", type=float, default=1.0e-5, help="SCF tolerance. Default: 1e-5")
    parser.add_argument("--seed", type=int, default=42, help="Base PRNG seed. Default: 42")
    parser.add_argument("--early-max-iter", type=int, default=4, help="Scheduled early-stage eigensolver max iterations. Default: 4")
    parser.add_argument("--late-max-iter", type=int, default=12, help="Scheduled late-stage eigensolver max iterations. Default: 12")
    parser.add_argument("--stage-threshold", type=float, default=1.0e-3, help="Stage switch threshold for max_abs_rho_new_minus_rho_in. Default: 1e-3")
    parser.add_argument("--late-tol", type=float, default=1.0e-5, help="Late-stage eigensolver residual tolerance. Default: 1e-5")
    return parser.parse_args()


def fmt_float(value: float | None, width: int = 8, precision: int = 2) -> str:
    if value is None:
        return " " * width
    return f"{value:{width}.{precision}f}"


def fmt_sci(value: float | None, width: int = 12, precision: int = 3) -> str:
    if value is None:
        return " " * width
    return f"{value:{width}.{precision}e}"


def build_coords(R: float) -> jnp.ndarray:
    half = 0.5 * R
    return jnp.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]], dtype=jnp.float32)


def run_case(name: str, backend: AdaptiveBackend, coords: jnp.ndarray, pseudos: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    state = backend.create_grid(
        spacing=args.h_min,
        box_size=[args.box, args.box, args.box],
        atom_coords=coords,
        h_min=args.h_min,
        h_max=args.h_max,
        r_core=args.r_core,
        stretch_beta=args.stretch_beta,
    )
    key = jax.random.PRNGKey(args.seed)
    t0 = time.perf_counter()
    energy, _, diagnostics = energy_and_forces(
        state,
        coords,
        pseudos,
        args.max_iter,
        args.mix_alpha,
        args.tolerance,
        key,
        backend=backend,
        return_diagnostics=True,
    )
    wall_s = time.perf_counter() - t0
    result = diagnostics.get("result", {})
    eig = diagnostics.get("eigensolver_diagnostics", {})
    return {
        "name": name,
        "energy": float(energy),
        "wall_s": wall_s,
        "final_iterations": int(result.get("final_iterations", -1)),
        "final_density_residual": float(result.get("final_density_residual", float("nan"))),
        "converged": bool(result.get("converged", False)),
        "total_inner_iterations": int(eig.get("total_inner_iterations", 0)),
        "total_hpsi_calls": int(eig.get("total_hpsi_calls", 0)),
        "stage_counts": eig.get("stage_counts", {}),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print("case           wall_s   final_iter   density_resid   conv   eig_iters   hpsi_calls   early   late   fixed")
    for row in rows:
        stages = row.get("stage_counts", {})
        print(
            f"{row['name']:<12}"
            f"{fmt_float(row['wall_s'])}   "
            f"{row['final_iterations']:>10d}   "
            f"{fmt_sci(row['final_density_residual'])}   "
            f"{str(row['converged']):<5}   "
            f"{row['total_inner_iterations']:>9d}   "
            f"{row['total_hpsi_calls']:>10d}   "
            f"{int(stages.get('early', 0)):>5d}   "
            f"{int(stages.get('late', 0)):>4d}   "
            f"{int(stages.get('fixed', 0)):>5d}"
        )


def main() -> int:
    args = parse_args()
    coords = build_coords(args.R)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    baseline_backend = AdaptiveBackend(
        hartree_boundary_mode=args.hartree_boundary_mode,
        kinetic_mode=args.kinetic_mode,
    )
    scheduled_backend = AdaptiveBackend(
        hartree_boundary_mode=args.hartree_boundary_mode,
        kinetic_mode=args.kinetic_mode,
        adaptive_scf_aware_eigensolver=True,
        adaptive_eigensolver_early_max_iter=args.early_max_iter,
        adaptive_eigensolver_late_max_iter=args.late_max_iter,
        adaptive_eigensolver_stage_residual_threshold=args.stage_threshold,
        adaptive_eigensolver_late_tol=args.late_tol,
    )
    rows = [
        run_case("baseline", baseline_backend, coords, pseudos, args),
        run_case("scheduled", scheduled_backend, coords, pseudos, args),
    ]
    print("=== Adaptive H2 R=1.4 SCF-Aware Eigensolver Scheduler ===")
    print(
        "metric=max_abs_rho_new_minus_rho_in, "
        f"early_max_iter={args.early_max_iter}, late_max_iter={args.late_max_iter}, "
        f"stage_threshold={args.stage_threshold:.3e}, late_tol={args.late_tol:.3e}"
    )
    print_table(rows)
    base, sched = rows
    print("\n=== Delta (scheduled - baseline) ===")
    print(f"wall_s: {sched['wall_s'] - base['wall_s']:+.2f}")
    print(f"final_density_residual: {sched['final_density_residual'] - base['final_density_residual']:+.3e}")
    print(f"total_inner_iterations: {sched['total_inner_iterations'] - base['total_inner_iterations']:+d}")
    print(f"total_hpsi_calls: {sched['total_hpsi_calls'] - base['total_hpsi_calls']:+d}")
    print(f"converged: baseline={base['converged']} scheduled={sched['converged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
