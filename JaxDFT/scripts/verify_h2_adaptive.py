"""Adaptive H2 dissociation verification against PySCF.

This script mirrors the existing verify_h2.py flow, but replaces the uniform
real-space path with the adaptive tensor-grid backend. The current adaptive
Hartree path uses a monopole-Dirichlet box Poisson prototype, so this script is
best interpreted as an adaptive-backend verification study rather than a final
physical benchmark.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyscf import dft, gto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import gto as pbcgto

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


def run_pyscf(dist, box_size=None):
    try:
        if box_size is not None:
            cell = pbcgto.Cell()
            cell.atom = f"H 0 0 0; H 0 0 {dist}"
            cell.a = jnp.eye(3) * box_size[0]
            cell.basis = "gth-tzvp"
            cell.pseudo = "gth-lda"
            cell.verbose = 0
            cell.build()
            mf = pbcdft.RKS(cell)
            mf.xc = "lda,pz"
            return mf.kernel()

        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {dist}",
            unit="Bohr",
            basis="gth-tzvp",
            pseudo="gth-lda",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "lda,pz"
        return mf.kernel()
    except Exception:
        return float("nan")


def main() -> int:
    print(f"\n{'=' * 20} Adaptive H2 Verification {'=' * 20}")

    box_size = [18.0, 18.0, 18.0]
    h_min = 0.10
    h_max = 0.60
    r_core = 1.00
    stretch_beta = 10.0
    max_iter = 500
    mix_alpha = 0.30
    tolerance = 1.0e-5
    distances = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H"], pseudo_dir)
    pseudos_for_calc = [pseudos[0], pseudos[0]]

    backend = AdaptiveBackend(hartree_boundary_mode="monopole_dirichlet")
    key = jax.random.PRNGKey(42)

    print(
        "Setup: "
        f"box={box_size}, h_min={h_min}, h_max={h_max}, "
        f"r_core={r_core}, stretch_beta={stretch_beta}"
    )
    print("Note: adaptive Hartree currently uses the monopole-Dirichlet box Poisson path.")

    jax_energies = []
    pyscf_energies = []

    print("-" * 75)
    print(f"{'Dist':<6} | {'JaxDFT (Adaptive)':<20} | {'PySCF (TZVP)':<15} | {'Diff'}")
    print("-" * 75)

    for d in distances:
        coords = jnp.array([
            [0.0, 0.0, -d / 2.0],
            [0.0, 0.0, d / 2.0],
        ], dtype=jnp.float32)

        try:
            state = backend.create_grid(
                spacing=h_min,
                box_size=box_size,
                atom_coords=coords,
                h_min=h_min,
                h_max=h_max,
                r_core=r_core,
                stretch_beta=stretch_beta,
            )
            e_jax, _ = energy_and_forces(
                state,
                coords,
                pseudos_for_calc,
                max_iter,
                mix_alpha,
                tolerance,
                key,
                backend=backend,
            )
            e_jax = float(e_jax)
        except Exception:
            e_jax = float("nan")

        e_pyscf = run_pyscf(d, box_size=None)

        jax_energies.append(e_jax)
        pyscf_energies.append(e_pyscf)

        diff = e_jax - e_pyscf
        print(f"{d:<6.2f} | {e_jax:<20.6f} | {e_pyscf:<15.6f} | {diff:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(distances, jax_energies, "o-", label="JaxDFT (Adaptive, h_min=0.1)", linewidth=2)
    plt.plot(distances, pyscf_energies, "x--", label="PySCF (TZVP Reference)", linewidth=2)
    plt.xlabel("Bond Length (Bohr)")
    plt.ylabel("Total Energy (Hartree)")
    plt.title("H2 Dissociation: Adaptive JaxDFT vs PySCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("h2_adaptive_verification.png", dpi=150)

    print("-" * 75)
    print("Verification complete. Figure: h2_adaptive_verification.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
