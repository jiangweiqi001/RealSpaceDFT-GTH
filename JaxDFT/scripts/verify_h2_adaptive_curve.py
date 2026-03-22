"""Adaptive H2 curve verification script.

This is the adaptive counterpart to verify_h2.py. It runs an H2 bond-length
scan with the current adaptive main path, prints a clear terminal table, writes
one CSV file, and saves one PNG curve plot.

Default adaptive settings:
  - hartree_boundary_mode = uniform_exterior
  - kinetic_mode = prototype_fd2

Optional overlays:
  - --compare-uniform
  - --compare-pyscf
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
    from JaxDFT.src.backends.uniform import UniformBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends.adaptive import AdaptiveBackend
    from src.backends.uniform import UniformBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


DEFAULT_DISTANCES = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.4, 2.8, 3.2]
SMALL = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive H2 potential-curve verification with CSV/PNG output.",
    )
    parser.add_argument(
        "--dist",
        type=float,
        nargs="+",
        default=None,
        help="Bond lengths in Bohr. Default: 0.8 1.0 1.2 1.4 1.6 2.0 2.4 2.8 3.2",
    )
    parser.add_argument("--box", type=float, default=30.0, help="Cubic box length in Bohr. Default: 30.0")
    parser.add_argument("--h-min", type=float, default=0.25, help="Adaptive minimum spacing. Default: 0.25")
    parser.add_argument("--h-max", type=float, default=0.80, help="Adaptive maximum spacing. Default: 0.80")
    parser.add_argument("--r-core", type=float, default=1.0, help="Adaptive core refinement radius. Default: 1.0")
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
    parser.add_argument("--max-iter", type=int, default=4, help="SCF max iterations. Default: 4")
    parser.add_argument("--mix-alpha", type=float, default=0.30, help="SCF mixing alpha. Default: 0.30")
    parser.add_argument("--tolerance", type=float, default=5.0e-4, help="SCF tolerance. Default: 5e-4")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="h2_adaptive_curve",
        help="Output prefix for CSV/PNG files. Default: h2_adaptive_curve",
    )
    parser.add_argument(
        "--compare-uniform",
        action="store_true",
        help="Also run a uniform-grid baseline and overlay it on the plot.",
    )
    parser.add_argument(
        "--uniform-spacing",
        type=float,
        default=None,
        help="Uniform-grid spacing used by --compare-uniform. Default: use --h-min.",
    )
    parser.add_argument(
        "--compare-pyscf",
        action="store_true",
        help="Also run a PySCF isolated-molecule curve and overlay it on the plot if available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base PRNG seed. Default: 42")
    return parser.parse_args()


def fmt_float(value: float | None, width: int = 12, precision: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value: float | None, width: int = 11, precision: int = 2) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def build_occ(pseudos: list[dict[str, Any]]):
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


def eigvals_to_str(eigvals, n_show: int = 4) -> str:
    arr = np.asarray(jnp.asarray(eigvals), dtype=np.float64)
    shown = arr[: min(n_show, arr.shape[0])]
    return ";".join(f"{float(v):.8f}" for v in shown)


def ensure_parent_dir(path_prefix: str) -> None:
    parent = os.path.dirname(path_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_one_dft_case(grid, backend, coords, pseudos, n_bands, occ, args, key, label: str):
    try:
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
        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        eion = float(ion_ion_energy(coords, zion))
        energy = float(total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, eion, backend=backend))
        electron_count = float(backend.integrate(grid, rho))
        norm_dev = orbital_norm_maxdev(grid, backend, eigvecs)
        eig0 = float(np.asarray(jnp.asarray(eigvals), dtype=np.float64)[0]) if eigvals.shape[0] else None
        finite = bool(np.isfinite(energy) and np.isfinite(electron_count) and np.isfinite(norm_dev))
        status = "ok" if finite else "nonfinite"
        return {
            "label": label,
            "success": finite,
            "finite": finite,
            "status": status,
            "energy": energy,
            "electron_count": electron_count,
            "norm_dev": norm_dev,
            "eig0": eig0,
            "eigvals": eigvals_to_str(eigvals),
            "grid_shape": tuple(int(n) for n in grid.shape),
        }
    except Exception as exc:
        return {
            "label": label,
            "success": False,
            "finite": False,
            "status": f"fail:{type(exc).__name__}",
            "energy": None,
            "electron_count": None,
            "norm_dev": None,
            "eig0": None,
            "eigvals": "",
            "grid_shape": tuple(int(n) for n in getattr(grid, "shape", ())),
            "error": str(exc),
        }


def try_run_pyscf(dist: float):
    try:
        from pyscf import dft, gto
    except Exception as exc:
        return {
            "success": False,
            "status": f"import_fail:{type(exc).__name__}",
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
        finite = bool(np.isfinite(energy))
        return {
            "success": finite,
            "status": "ok" if finite else "nonfinite",
            "energy": energy,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": f"fail:{type(exc).__name__}",
            "energy": None,
        }


def main() -> int:
    args = parse_args()
    distances = [float(x) for x in (args.dist if args.dist is not None else DEFAULT_DISTANCES)]
    ensure_parent_dir(args.out_prefix)
    csv_path = f"{args.out_prefix}.csv"
    png_path = f"{args.out_prefix}.png"

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    base_pseudos = load_pseudopotentials(["H"], pseudo_dir)
    pseudos = [base_pseudos[0], base_pseudos[0]]
    n_electrons, n_bands, occ = build_occ(pseudos)

    adaptive_backend = AdaptiveBackend(
        hartree_boundary_mode=args.hartree_boundary_mode,
        kinetic_mode=args.kinetic_mode,
    )
    uniform_backend = UniformBackend() if args.compare_uniform else None
    uniform_spacing = float(args.uniform_spacing if args.uniform_spacing is not None else args.h_min)

    print(f"\n{'=' * 20} Adaptive H2 Curve Verification {'=' * 20}")
    print(f"distances = {distances}")
    print(
        "adaptive setup: "
        f"box={args.box:.1f}, h_min={args.h_min}, h_max={args.h_max}, "
        f"r_core={args.r_core}, stretch_beta={args.stretch_beta}, "
        f"hartree={args.hartree_boundary_mode}, kinetic={args.kinetic_mode}"
    )
    if args.compare_uniform:
        print(f"uniform overlay enabled: spacing={uniform_spacing}")
    if args.compare_pyscf:
        print("PySCF overlay enabled")
    print("This script does not change solver/backend mainline behavior; it only runs curve verification.")
    print()

    rows: list[dict[str, Any]] = []
    base_key = jax.random.PRNGKey(args.seed)

    for idx, dist in enumerate(distances):
        print(f"--- Running adaptive SCF for R={dist:.2f} Bohr ---")
        coords = jnp.asarray([[0.0, 0.0, -dist / 2.0], [0.0, 0.0, dist / 2.0]], dtype=jnp.float32)
        dist_key = jax.random.fold_in(base_key, idx)
        adaptive_grid = adaptive_backend.create_grid(
            spacing=args.h_min,
            box_size=[args.box, args.box, args.box],
            atom_coords=coords,
            h_min=args.h_min,
            h_max=args.h_max,
            r_core=args.r_core,
            stretch_beta=args.stretch_beta,
        )
        adaptive_result = run_one_dft_case(
            adaptive_grid,
            adaptive_backend,
            coords,
            pseudos,
            n_bands,
            occ,
            args,
            dist_key,
            label="adaptive",
        )

        uniform_result = None
        if args.compare_uniform:
            print(f"--- Running uniform baseline for R={dist:.2f} Bohr ---")
            uniform_grid = uniform_backend.create_grid(uniform_spacing, [args.box, args.box, args.box])
            uniform_result = run_one_dft_case(
                uniform_grid,
                uniform_backend,
                coords,
                pseudos,
                n_bands,
                occ,
                args,
                dist_key,
                label="uniform",
            )

        pyscf_result = None
        if args.compare_pyscf:
            print(f"--- Running PySCF reference for R={dist:.2f} Bohr ---")
            pyscf_result = try_run_pyscf(dist)

        rows.append({
            "R": dist,
            "adaptive": adaptive_result,
            "uniform": uniform_result,
            "pyscf": pyscf_result,
        })

    adaptive_finite_energies = [row["adaptive"]["energy"] for row in rows if row["adaptive"]["energy"] is not None and np.isfinite(row["adaptive"]["energy"])]
    adaptive_ref = min(adaptive_finite_energies) if adaptive_finite_energies else None

    uniform_ref = None
    if args.compare_uniform:
        finite_uniform = [row["uniform"]["energy"] for row in rows if row["uniform"] and row["uniform"]["energy"] is not None and np.isfinite(row["uniform"]["energy"])]
        uniform_ref = min(finite_uniform) if finite_uniform else None

    pyscf_ref = None
    if args.compare_pyscf:
        finite_pyscf = [row["pyscf"]["energy"] for row in rows if row["pyscf"] and row["pyscf"]["energy"] is not None and np.isfinite(row["pyscf"]["energy"])]
        pyscf_ref = min(finite_pyscf) if finite_pyscf else None

    print("=" * 92)
    print(f"{'R':>5} {'E':>14} {'dE':>12} {'N':>10} {'norm_dev':>11} {'eig0':>14} {'status':>16}")
    print("-" * 92)
    for row in rows:
        adaptive = row["adaptive"]
        energy = adaptive["energy"]
        dE = None if energy is None or adaptive_ref is None else energy - adaptive_ref
        print(
            f"{row['R']:>5.2f} {fmt_float(energy, 14, 6)} {fmt_float(dE, 12, 6)} "
            f"{fmt_float(adaptive['electron_count'], 10, 6)} {fmt_sci(adaptive['norm_dev'], 11, 2)} "
            f"{fmt_float(adaptive['eig0'], 14, 6)} {adaptive['status']:>16}"
        )
    print("=" * 92)

    csv_rows = []
    for row in rows:
        adaptive = row["adaptive"]
        out = {
            "R_bohr": row["R"],
            "adaptive_energy": adaptive["energy"],
            "adaptive_dE": None if adaptive["energy"] is None or adaptive_ref is None else adaptive["energy"] - adaptive_ref,
            "adaptive_electron_count": adaptive["electron_count"],
            "adaptive_norm_dev": adaptive["norm_dev"],
            "adaptive_eig0": adaptive["eig0"],
            "adaptive_eigvals": adaptive["eigvals"],
            "adaptive_status": adaptive["status"],
            "adaptive_grid_shape": str(adaptive.get("grid_shape", "")),
        }
        if args.compare_uniform:
            uniform = row["uniform"]
            out.update({
                "uniform_energy": uniform["energy"],
                "uniform_dE": None if uniform is None or uniform["energy"] is None or uniform_ref is None else uniform["energy"] - uniform_ref,
                "uniform_electron_count": uniform["electron_count"] if uniform else None,
                "uniform_norm_dev": uniform["norm_dev"] if uniform else None,
                "uniform_eig0": uniform["eig0"] if uniform else None,
                "uniform_eigvals": uniform["eigvals"] if uniform else "",
                "uniform_status": uniform["status"] if uniform else None,
                "uniform_grid_shape": str(uniform.get("grid_shape", "")) if uniform else "",
                "delta_energy_same_box_new_minus_uniform": None if uniform is None or adaptive["energy"] is None or uniform["energy"] is None else adaptive["energy"] - uniform["energy"],
            })
        if args.compare_pyscf:
            pyscf = row["pyscf"]
            out.update({
                "pyscf_energy": pyscf["energy"] if pyscf else None,
                "pyscf_dE": None if pyscf is None or pyscf["energy"] is None or pyscf_ref is None else pyscf["energy"] - pyscf_ref,
                "pyscf_status": pyscf["status"] if pyscf else None,
                "delta_energy_same_box_new_minus_pyscf": None if pyscf is None or adaptive["energy"] is None or pyscf["energy"] is None else adaptive["energy"] - pyscf["energy"],
            })
        csv_rows.append(out)

    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    plt.figure(figsize=(10, 6))
    plt.plot(
        distances,
        [row["adaptive"]["energy"] if row["adaptive"]["energy"] is not None else np.nan for row in rows],
        "o-",
        linewidth=2,
        label=f"Adaptive ({args.hartree_boundary_mode}, {args.kinetic_mode})",
    )
    if args.compare_uniform:
        plt.plot(
            distances,
            [row["uniform"]["energy"] if row["uniform"] and row["uniform"]["energy"] is not None else np.nan for row in rows],
            "s--",
            linewidth=2,
            label=f"Uniform baseline (h={uniform_spacing})",
        )
    if args.compare_pyscf:
        plt.plot(
            distances,
            [row["pyscf"]["energy"] if row["pyscf"] and row["pyscf"]["energy"] is not None else np.nan for row in rows],
            "x:",
            linewidth=2,
            label="PySCF reference",
        )
    plt.xlabel("Bond Length (Bohr)")
    plt.ylabel("Total Energy (Hartree)")
    plt.title("H2 Potential Curve: Adaptive Verification")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
