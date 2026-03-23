"""Adaptive H2 vs PySCF near-equilibrium verification.

This mirrors the intent and plotting style of verify_h2.py, but compares only:
  - Adaptive RealSpace
  - PySCF reference

The default scan is intentionally small and focused near equilibrium.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends.adaptive import AdaptiveBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import energy_and_forces
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends.adaptive import AdaptiveBackend
    from src.io import load_pseudopotentials
    from src.solver import energy_and_forces


DEFAULT_DISTANCES = [1.2, 1.4, 1.6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a small-range adaptive H2 curve against the PySCF reference used in verify_h2.py.",
    )
    parser.add_argument(
        "--dist",
        type=float,
        nargs="+",
        default=None,
        help="Bond lengths in Bohr. Default: 1.2 1.4 1.6",
    )
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
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="h2_adaptive_vs_pyscf_near_eq",
        help="Output prefix for CSV/PNG files. Default: h2_adaptive_vs_pyscf_near_eq",
    )
    return parser.parse_args()


def fmt_float(value: float | None, width: int = 14, precision: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def ensure_parent_dir(path_prefix: str) -> None:
    parent = os.path.dirname(path_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_pyscf_verify_h2_style(dist: float) -> dict[str, Any]:
    try:
        from pyscf import dft, gto
    except Exception as exc:
        return {
            "success": False,
            "status": f"pyscf_import_failed:{type(exc).__name__}",
            "energy": None,
        }

    try:
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {dist}",
            unit="Bohr",
            basis="gth-tzvp",
            pseudo="gth-lda",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "lda,pz"
        energy = float(mf.kernel())
        ok = bool(np.isfinite(energy))
        return {
            "success": ok,
            "status": "ok" if ok else "pyscf_nonfinite",
            "energy": energy if ok else None,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": f"pyscf_failed:{type(exc).__name__}",
            "energy": None,
        }


def run_adaptive_case(
    backend: AdaptiveBackend,
    coords: jnp.ndarray,
    pseudos: list[dict[str, Any]],
    args: argparse.Namespace,
    key: jax.Array,
) -> dict[str, Any]:
    try:
        state = backend.create_grid(
            spacing=args.h_min,
            box_size=[args.box, args.box, args.box],
            atom_coords=coords,
            h_min=args.h_min,
            h_max=args.h_max,
            r_core=args.r_core,
            stretch_beta=args.stretch_beta,
        )
        energy, _ = energy_and_forces(
            state,
            coords,
            pseudos,
            args.max_iter,
            args.mix_alpha,
            args.tolerance,
            key,
            backend=backend,
        )
        energy = float(energy)
        ok = bool(np.isfinite(energy))
        return {
            "success": ok,
            "status": "ok" if ok else "adaptive_nonfinite",
            "energy": energy if ok else None,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": f"adaptive_failed:{type(exc).__name__}",
            "energy": None,
            "error": str(exc),
        }


def combined_status(adaptive: dict[str, Any], pyscf: dict[str, Any]) -> str:
    if adaptive.get("success") and pyscf.get("success"):
        return "ok"
    if not adaptive.get("success") and not pyscf.get("success"):
        return "adaptive+pyscf_failed"
    if not adaptive.get("success"):
        return adaptive.get("status", "adaptive_failed")
    return pyscf.get("status", "pyscf_failed")


def plot_energy_curve(path: str, rows: list[dict[str, Any]]) -> None:
    adaptive_x = [row["R"] for row in rows if row["E_adaptive"] is not None]
    adaptive_y = [row["E_adaptive"] for row in rows if row["E_adaptive"] is not None]
    pyscf_x = [row["R"] for row in rows if row["E_pyscf"] is not None]
    pyscf_y = [row["E_pyscf"] for row in rows if row["E_pyscf"] is not None]

    plt.figure(figsize=(8, 5))
    if adaptive_x:
        plt.plot(adaptive_x, adaptive_y, "o-", linewidth=2, label="Adaptive (RealSpace)")
    if pyscf_x:
        plt.plot(pyscf_x, pyscf_y, "x--", linewidth=2, label="PySCF Reference")
    plt.xlabel("Bond Length (Bohr)")
    plt.ylabel("Total Energy (Hartree)")
    plt.title("H2 Near Equilibrium: Adaptive vs PySCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_error_curve(path: str, rows: list[dict[str, Any]]) -> None:
    dE_x = [row["R"] for row in rows if row["dE"] is not None]
    dE_y = [row["dE"] for row in rows if row["dE"] is not None]

    plt.figure(figsize=(8, 5))
    if dE_x:
        plt.plot(dE_x, dE_y, "o-", linewidth=2, label="dE = Adaptive - PySCF")
        plt.axhline(0.0, color="black", linestyle="--", alpha=0.6)
    plt.xlabel("Bond Length (Bohr)")
    plt.ylabel("Energy Error (Hartree)")
    plt.title("H2 Near Equilibrium: Adaptive Error vs PySCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def summarize(rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in rows if row["dE"] is not None]
    print("\n=== Summary ===")
    for row in rows:
        de = row["dE"]
        if de is None:
            print(f"R={row['R']:.2f}: dE = failed ({row['status']})")
        else:
            print(f"R={row['R']:.2f}: dE = {de:.6f} Ha ({de * 1000.0:.3f} mHa)")

    if not valid_rows:
        print("No valid adaptive/PySCF overlap points were produced.")
        return

    all_mha = all(abs(row["dE"]) < 1.0e-3 for row in valid_rows)
    adaptive_min_row = min(
        (row for row in rows if row["E_adaptive"] is not None),
        key=lambda row: row["E_adaptive"],
        default=None,
    )
    curve_smooth = (
        len(valid_rows) == 3
        and rows[1]["E_adaptive"] is not None
        and rows[0]["E_adaptive"] is not None
        and rows[2]["E_adaptive"] is not None
        and rows[1]["E_adaptive"] <= rows[0]["E_adaptive"]
        and rows[1]["E_adaptive"] <= rows[2]["E_adaptive"]
    )
    min_reasonable = adaptive_min_row is not None and 1.2 <= adaptive_min_row["R"] <= 1.6

    print(f"all points in mHa regime: {'yes' if all_mha else 'no'}")
    print(f"curve smooth near equilibrium: {'yes' if curve_smooth else 'no'}")
    if adaptive_min_row is not None:
        print(f"adaptive minimum among sampled points: R = {adaptive_min_row['R']:.2f} Bohr")
    print(f"minimum stays in reasonable interval [1.2, 1.6]: {'yes' if min_reasonable else 'no'}")


def main() -> int:
    args = parse_args()
    distances = [float(x) for x in (args.dist if args.dist is not None else DEFAULT_DISTANCES)]
    ensure_parent_dir(args.out_prefix)
    csv_path = f"{args.out_prefix}.csv"
    png_path = f"{args.out_prefix}.png"
    de_png_path = f"{args.out_prefix}_dE.png"

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    base_pseudos = load_pseudopotentials(["H"], pseudo_dir)
    pseudos = [base_pseudos[0], base_pseudos[0]]

    backend = AdaptiveBackend(
        hartree_boundary_mode=args.hartree_boundary_mode,
        kinetic_mode=args.kinetic_mode,
    )
    base_key = jax.random.PRNGKey(args.seed)

    print(f"\n{'=' * 20} Adaptive H2 vs PySCF {'=' * 20}")
    print(f"distances = {distances}")
    print(
        "adaptive setup: "
        f"box={args.box:.1f}, h_min={args.h_min}, h_max={args.h_max}, "
        f"r_core={args.r_core}, stretch_beta={args.stretch_beta}, "
        f"hartree={args.hartree_boundary_mode}, kinetic={args.kinetic_mode}"
    )
    print("PySCF setup matches verify_h2.py: isolated molecule, gth-tzvp basis, gth-lda pseudo, lda,pz XC.")
    print()

    rows: list[dict[str, Any]] = []
    for idx, dist in enumerate(distances):
        coords = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, dist]], dtype=jnp.float32)
        dist_key = jax.random.fold_in(base_key, idx)
        adaptive = run_adaptive_case(backend, coords, pseudos, args, dist_key)
        pyscf = run_pyscf_verify_h2_style(dist)
        status = combined_status(adaptive, pyscf)
        dE = None
        if adaptive.get("energy") is not None and pyscf.get("energy") is not None:
            dE = adaptive["energy"] - pyscf["energy"]
        rows.append(
            {
                "R": dist,
                "E_adaptive": adaptive.get("energy"),
                "E_pyscf": pyscf.get("energy"),
                "dE": dE,
                "status": status,
            }
        )

    print("=" * 78)
    print(f"{'R':>5} {'E_adaptive':>14} {'E_pyscf':>14} {'dE':>14} {'status':>24}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['R']:>5.2f} {fmt_float(row['E_adaptive'])} {fmt_float(row['E_pyscf'])} {fmt_float(row['dE'])} {row['status']:>24}"
        )
    print("=" * 78)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["R", "E_adaptive", "E_pyscf", "dE", "status"])
        writer.writeheader()
        writer.writerows(rows)

    plot_energy_curve(png_path, rows)
    plot_error_curve(de_png_path, rows)
    summarize(rows)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved energy PNG: {png_path}")
    print(f"Saved dE PNG: {de_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
