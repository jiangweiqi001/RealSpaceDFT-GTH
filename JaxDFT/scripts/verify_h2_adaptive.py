"""Reduced adaptive H2 boundary-diagnostic script.

This is a lightweight diagnostic study, not a final benchmark. The adaptive
Hartree path currently offers two box-Poisson boundary choices:
  - zero_dirichlet
  - monopole_dirichlet
The monopole variant is more isolated-like than zero Dirichlet, but still not
an exact isolated/open-boundary Hartree treatment.

The current adaptive SCF path still uses the conservative Python ``while`` loop
inside the solver. That is not because the grid changes during SCF for a fixed
geometry: at fixed geometry the adaptive grid is static. In principle, a single
fixed-geometry adaptive SCF could be wrapped in ``jax.lax.while_loop`` once the
remaining adaptive sparse/Hartree path is fully stabilized under JAX tracing.
That migration is intentionally out of scope for this script.
"""

from __future__ import annotations

import argparse
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


FULL_DISTANCES = [0.8, 1.4, 2.0, 2.8]
QUICK_DISTANCES = [1.4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive H2 boundary diagnostic: zero-Dirichlet vs monopole-Dirichlet vs PySCF."
    )
    parser.add_argument(
        "--dist",
        type=float,
        nargs="+",
        help="Run one or more specific H-H distances in Bohr, e.g. --dist 2.0 or --dist 1.4 2.0",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full reduced diagnostic set: 0.8, 1.4, 2.0, 2.8 Bohr.",
    )
    return parser.parse_args()


def select_distances(args: argparse.Namespace) -> list[float]:
    if args.all:
        return list(FULL_DISTANCES)
    if args.dist:
        return [float(d) for d in args.dist]
    return list(QUICK_DISTANCES)


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
    args = parse_args()
    distances = select_distances(args)

    print(f"\n{'=' * 20} Adaptive H2 Boundary Diagnostic {'=' * 20}")

    box_size = [18.0, 18.0, 18.0]
    h_min = 0.25
    h_max = 0.80
    r_core = 1.00
    stretch_beta = 5.0
    max_iter = 120
    mix_alpha = 0.30
    tolerance = 5.0e-4

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H"], pseudo_dir)
    pseudos_for_calc = [pseudos[0], pseudos[0]]

    backend_zero = AdaptiveBackend(hartree_boundary_mode="zero_dirichlet")
    backend_mono = AdaptiveBackend(hartree_boundary_mode="monopole_dirichlet")
    key = jax.random.PRNGKey(42)

    mode_label = "full reduced set (--all)" if args.all else ("user-specified --dist" if args.dist else "default quick mode")
    print(
        "Setup: "
        f"box={box_size}, h_min={h_min}, h_max={h_max}, "
        f"r_core={r_core}, stretch_beta={stretch_beta}"
    )
    print(f"Mode: {mode_label}; distances={distances}")
    print("Note: this is not a final benchmark.")
    print("Note: adaptive monopole-Dirichlet is still not an exact isolated/open-boundary Hartree treatment.")
    print("Note: the goal here is to diagnose boundary sensitivity, not to force exact agreement with uniform or PySCF.")
    print("Note: SCF settings are intentionally relaxed for speed in this reduced diagnostic script.")
    print("Note: for fixed geometry the adaptive grid is static across SCF iterations; the current Python while-loop path is a conservative solver choice, not a geometry-change requirement.")

    zero_energies = []
    mono_energies = []
    pyscf_energies = []

    print("-" * 120)
    print(
        f"{'Dist':<6} | {'PySCF':<14} | {'Adaptive Zero':<14} | {'Err Zero':<12} | "
        f"{'Adaptive Mono':<14} | {'Err Mono':<12} | {'Delta(M-Z)':<12}"
    )
    print("-" * 120)

    for idx, d in enumerate(distances):
        coords = jnp.array([
            [0.0, 0.0, -d / 2.0],
            [0.0, 0.0, d / 2.0],
        ], dtype=jnp.float32)
        dist_key = jax.random.fold_in(key, idx)
        state = None

        print(f"--- Building adaptive grid for d={d:.2f} Bohr ---")
        try:
            state = backend_zero.create_grid(
                spacing=h_min,
                box_size=box_size,
                atom_coords=coords,
                h_min=h_min,
                h_max=h_max,
                r_core=r_core,
                stretch_beta=stretch_beta,
            )
            print(f"--- Adaptive grid ready for d={d:.2f}; shape={state.shape} ---")
        except Exception as exc:
            print(f"--- Adaptive grid construction failed for d={d:.2f}: {exc} ---")

        print(f"--- Starting Zero Dirichlet SCF for d={d:.2f} ---")
        try:
            if state is None:
                raise RuntimeError("adaptive grid state was not created")
            e_zero, _ = energy_and_forces(
                state,
                coords,
                pseudos_for_calc,
                max_iter,
                mix_alpha,
                tolerance,
                dist_key,
                backend=backend_zero,
            )
            e_zero = float(e_zero)
            print(f"--- Finished Zero Dirichlet SCF for d={d:.2f}; E={e_zero:.6f} ---")
        except Exception as exc:
            e_zero = float("nan")
            print(f"--- Zero Dirichlet SCF failed for d={d:.2f}: {exc} ---")

        print(f"--- Starting Monopole Dirichlet SCF for d={d:.2f} ---")
        try:
            if state is None:
                raise RuntimeError("adaptive grid state was not created")
            e_mono, _ = energy_and_forces(
                state,
                coords,
                pseudos_for_calc,
                max_iter,
                mix_alpha,
                tolerance,
                dist_key,
                backend=backend_mono,
            )
            e_mono = float(e_mono)
            print(f"--- Finished Monopole Dirichlet SCF for d={d:.2f}; E={e_mono:.6f} ---")
        except Exception as exc:
            e_mono = float("nan")
            print(f"--- Monopole Dirichlet SCF failed for d={d:.2f}: {exc} ---")

        print(f"--- Starting PySCF reference for d={d:.2f} ---")
        e_pyscf = run_pyscf(d, box_size=None)
        print(f"--- Finished PySCF reference for d={d:.2f}; E={e_pyscf:.6f} ---")

        zero_energies.append(e_zero)
        mono_energies.append(e_mono)
        pyscf_energies.append(e_pyscf)

        err_zero = e_zero - e_pyscf
        err_mono = e_mono - e_pyscf
        delta_mz = e_mono - e_zero
        print(
            f"{d:<6.2f} | {e_pyscf:<14.6f} | {e_zero:<14.6f} | {err_zero:<12.4f} | "
            f"{e_mono:<14.6f} | {err_mono:<12.4f} | {delta_mz:<12.4f}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(distances, zero_energies, "s--", label="JaxDFT (Adaptive, zero Dirichlet)", linewidth=2)
    plt.plot(distances, mono_energies, "o-", label="JaxDFT (Adaptive, monopole Dirichlet)", linewidth=2)
    plt.plot(distances, pyscf_energies, "x:", label="PySCF (TZVP Reference)", linewidth=2)
    plt.xlabel("Bond Length (Bohr)")
    plt.ylabel("Total Energy (Hartree)")
    plt.title("H2 Boundary Diagnostic: Adaptive Hartree Modes vs PySCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("h2_adaptive_verification.png", dpi=150)

    print("-" * 120)
    print("Saved figure: h2_adaptive_verification.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
