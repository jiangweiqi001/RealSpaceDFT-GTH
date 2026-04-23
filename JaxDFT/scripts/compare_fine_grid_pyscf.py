"""Compare JaxDFT fine-grid mode against PySCF for an off-grid H2O stretch.

This script is intended to answer a narrow question:
on the same coarse real-space grid, does atom-centered fine-grid sampling move
JaxDFT closer to a PySCF GTH/LDA reference?

The default H2O geometry is shifted by half a grid spacing in all directions so
the oxygen nucleus does not sit exactly on a coarse grid point. That makes the
comparison sensitive to egg-box error, which is what the local atom patch is
designed to reduce.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JAXDFT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(JAXDFT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if JAXDFT_ROOT not in sys.path:
    sys.path.insert(0, JAXDFT_ROOT)

try:
    import JaxDFT.src.solver as solver
    from JaxDFT.src.hamiltonian import create_grid
    from JaxDFT.src.io import load_pseudopotentials
except ImportError:
    import src.solver as solver
    from src.hamiltonian import create_grid
    from src.io import load_pseudopotentials


PSEUDO_DIR = os.path.join(JAXDFT_ROOT, "data", "gth_potentials")


@dataclass
class ResultRow:
    distance: float
    pyscf_energy: float
    baseline_energy: float
    fine_grid_energy: float

    @property
    def baseline_error(self) -> float:
        return self.baseline_energy - self.pyscf_energy

    @property
    def fine_grid_error(self) -> float:
        return self.fine_grid_energy - self.pyscf_energy


def require_optional_dependencies():
    try:
        import matplotlib.pyplot as plt
        from pyscf import dft, gto
    except ImportError as exc:
        raise SystemExit(
            "This comparison requires optional packages: pyscf and matplotlib. "
            "Install them in your environment, then rerun this script."
        ) from exc
    return plt, gto, dft


def build_h2o_coords(distance: float, angle_deg: float, shift):
    theta = math.radians(angle_deg)
    hx = distance * math.sin(theta / 2.0)
    hz = distance * math.cos(theta / 2.0)
    coords = [
        [0.0, 0.0, 0.0],
        [hx, 0.0, hz],
        [-hx, 0.0, hz],
    ]
    return [[x + shift[0], y + shift[1], z + shift[2]] for x, y, z in coords]


def run_pyscf_h2o(coords, gto, dft):
    atom = (
        f"O {coords[0][0]} {coords[0][1]} {coords[0][2]}; "
        f"H {coords[1][0]} {coords[1][1]} {coords[1][2]}; "
        f"H {coords[2][0]} {coords[2][1]} {coords[2][2]}"
    )
    mol = gto.M(
        atom=atom,
        unit="Bohr",
        basis="gth-tzvp",
        pseudo="gth-lda",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,pz"
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    return float(mf.kernel())


def scf_settings(distance: float):
    if distance >= 2.8:
        return 0.05, 800
    if distance >= 2.4:
        return 0.10, 600
    return 0.30, 400


def run_jax_h2o(grid, coords, pseudos, key, use_fine_grid: bool, distance: float):
    alpha, max_iter = scf_settings(distance)
    kwargs = {}
    if use_fine_grid:
        kwargs.update(
            fine_grid_mode="auto",
            fine_subgrid=5,
            fine_grid_radius_factor=4.0,
        )
    energy, _ = solver.energy_and_forces(
        grid,
        jnp.asarray(coords, dtype=jnp.float32),
        pseudos,
        max_iter,
        alpha,
        1e-5,
        key,
        **kwargs,
    )
    return float(energy)


def write_csv(path: str, rows: list[ResultRow]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distance_bohr",
                "pyscf_ha",
                "jax_baseline_ha",
                "jax_fine_grid_auto_ha",
                "baseline_minus_pyscf_ha",
                "fine_grid_minus_pyscf_ha",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.distance,
                    row.pyscf_energy,
                    row.baseline_energy,
                    row.fine_grid_energy,
                    row.baseline_error,
                    row.fine_grid_error,
                ]
            )


def plot_results(path: str, rows: list[ResultRow], plt):
    distances = [row.distance for row in rows]
    pyscf = [row.pyscf_energy for row in rows]
    baseline = [row.baseline_energy for row in rows]
    fine = [row.fine_grid_energy for row in rows]
    baseline_abs = [abs(row.baseline_error) for row in rows]
    fine_abs = [abs(row.fine_grid_error) for row in rows]

    fig, (ax_energy, ax_error) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax_energy.plot(distances, pyscf, "k--", linewidth=2, label="PySCF gth-tzvp")
    ax_energy.plot(distances, baseline, "o-", label="JaxDFT baseline")
    ax_energy.plot(distances, fine, "s-", label='JaxDFT fine_grid_mode="auto"')
    ax_energy.set_ylabel("Total energy (Ha)")
    ax_energy.set_title("H2O symmetric stretch: same grid, baseline vs atom patch")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()

    ax_error.plot(distances, baseline_abs, "o-", label="|baseline - PySCF|")
    ax_error.plot(distances, fine_abs, "s-", label="|auto - PySCF|")
    ax_error.set_xlabel("O-H bond length (Bohr)")
    ax_error.set_ylabel("Absolute error (Ha)")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=0.40, help="Target real-space grid spacing in Bohr.")
    parser.add_argument("--box-size", type=float, default=9.6, help="Cubic box length in Bohr.")
    parser.add_argument("--angle-deg", type=float, default=104.5, help="H-O-H angle in degrees.")
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=[1.6, 1.8, 2.0, 2.4],
        help="Symmetric O-H bond lengths in Bohr.",
    )
    parser.add_argument(
        "--shift-fraction",
        type=float,
        default=0.5,
        help="Whole-molecule shift in units of the actual grid spacing.",
    )
    parser.add_argument("--output-prefix", default="h2o_fine_grid_compare", help="Output filename prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    plt, gto, dft = require_optional_dependencies()

    n_intervals = int(round(args.box_size / args.spacing))
    spacing = args.box_size / n_intervals
    grid = create_grid(spacing, [args.box_size, args.box_size, args.box_size])
    shift = [args.shift_fraction * float(grid.spacing)] * 3

    pseudos_loaded = load_pseudopotentials(["O", "H"], PSEUDO_DIR)
    pseudos = [pseudos_loaded[0], pseudos_loaded[1], pseudos_loaded[1]]
    key = jax.random.PRNGKey(42)

    print("H2O fine-grid comparison")
    print(f"actual spacing = {float(grid.spacing):.6f} Bohr, shape = {grid.shape}")
    print(f"box size = {args.box_size:.3f} Bohr, molecule shift = {shift[0]:.6f} Bohr")
    print("mode baseline: point-sampled local potential")
    print('mode auto: fine_grid_mode="auto" (local atom patch only)')
    print("-" * 96)
    print(f"{'d(OH)':>8} {'PySCF':>16} {'Baseline':>16} {'Auto':>16} {'Err base':>14} {'Err auto':>14}")

    rows: list[ResultRow] = []
    for distance in args.distances:
        coords = build_h2o_coords(distance, args.angle_deg, shift)
        pyscf_energy = run_pyscf_h2o(coords, gto, dft)
        baseline_energy = run_jax_h2o(grid, coords, pseudos, key, False, distance)
        fine_grid_energy = run_jax_h2o(grid, coords, pseudos, key, True, distance)
        row = ResultRow(distance, pyscf_energy, baseline_energy, fine_grid_energy)
        rows.append(row)
        print(
            f"{distance:8.3f} {row.pyscf_energy:16.8f} "
            f"{row.baseline_energy:16.8f} {row.fine_grid_energy:16.8f} "
            f"{row.baseline_error:14.6f} {row.fine_grid_error:14.6f}"
        )

    csv_path = args.output_prefix + ".csv"
    png_path = args.output_prefix + ".png"
    write_csv(csv_path, rows)
    plot_results(png_path, rows, plt)
    print("-" * 96)
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
