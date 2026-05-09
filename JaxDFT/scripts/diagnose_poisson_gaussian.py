#!/usr/bin/env python3
"""Independent Gaussian Hartree / Poisson diagnostic (P0).

Uses the existing FFT zero-padding Poisson path from ``solver.py`` without
changing SCF, mixers, or Poisson implementation:

  rho(r) = Q * (alpha/pi)^(3/2) * exp(-alpha * r^2)
  V_H    = solve_poisson(rho, ...)
  E_H    = 0.5 * sum(rho * V_H) * dV

Analytic Hartree self-energy (infinite domain, same rho):

  E_H_exact = Q^2 * sqrt(alpha / (2*pi))
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Sequence

import jax.numpy as jnp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from JaxDFT.src.hamiltonian import create_grid  # noqa: E402
from JaxDFT.src.solver import precompute_poisson_kernel, solve_poisson  # noqa: E402


def gaussian_rho_on_grid(grid, charge: float, alpha: float) -> jnp.ndarray:
    """Normalized 3D Gaussian centered at the grid origin."""
    coords = grid.coords
    r2 = jnp.sum(coords * coords, axis=-1)
    pref = jnp.asarray(charge, dtype=coords.dtype) * (alpha / jnp.pi) ** 1.5
    return (pref * jnp.exp(-alpha * r2)).astype(coords.dtype)


def compute_row(
    spacing: float,
    box_size: float,
    alpha: float,
    charge: float,
) -> dict[str, Any]:
    """Run one (spacing, box, alpha, charge) case using current Poisson path."""
    grid = create_grid(spacing, [box_size, box_size, box_size])
    rho = gaussian_rho_on_grid(grid, charge, alpha)
    dV = float(grid.volume_element)
    spacing_used = float(grid.spacing)

    electron_count = float(jnp.sum(rho) * dV)

    kernel_k = precompute_poisson_kernel(grid.shape, spacing_used)
    V_H = solve_poisson(rho, kernel_k, spacing_used)

    E_H_num = 0.5 * float(jnp.sum(rho * V_H) * dV)
    E_H_exact = float(charge * charge * math.sqrt(alpha / (2.0 * math.pi)))
    error_ha = E_H_num - E_H_exact
    error_mha = error_ha * 1000.0
    if abs(E_H_exact) > 1e-20:
        relative_error = abs(error_ha) / abs(E_H_exact)
    else:
        relative_error = float("nan")

    return {
        "spacing": spacing_used,
        "box_size": float(box_size),
        "alpha": float(alpha),
        "charge": float(charge),
        "electron_count": electron_count,
        "E_H_num": E_H_num,
        "E_H_exact": E_H_exact,
        "error_Ha": error_ha,
        "error_mHa": error_mha,
        "relative_error": relative_error,
        "hartree_potential_min": float(jnp.min(V_H)),
        "hartree_potential_max": float(jnp.max(V_H)),
    }


def compute_all(
    spacings: Sequence[float],
    box_sizes: Sequence[float],
    alphas: Sequence[float],
    charge: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in spacings:
        for L in box_sizes:
            for a in alphas:
                rows.append(compute_row(s, L, a, charge))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Gaussian Hartree / Poisson diagnostic using the current solve_poisson path."
    )
    p.add_argument(
        "--spacings",
        type=float,
        nargs="+",
        default=[0.18, 0.14, 0.12, 0.10],
        help="Target grid spacings (Bohr); actual spacing follows create_grid preserve_box policy.",
    )
    p.add_argument(
        "--box-sizes",
        type=float,
        nargs="+",
        default=[18.0, 22.0],
        help="Cubic box edge length (Bohr).",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Gaussian exponent alpha in exp(-alpha r^2).",
    )
    p.add_argument("--charge", type=float, default=1.0, help="Total charge Q (electron count).")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print results as a JSON array (one object per case).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    rows = compute_all(args.spacings, args.box_sizes, args.alphas, args.charge)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    hdr = (
        f"{'spacing':>10} {'box':>6} {'alpha':>8} {'Q':>6} "
        f"{'N_e':>10} {'E_H_num':>14} {'E_H_exact':>14} {'err_mHa':>10} {'rel_err':>10} "
        f"{'Vmin':>10} {'Vmax':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['spacing']:10.6f} {r['box_size']:6.1f} {r['alpha']:8.4f} {r['charge']:6.2f} "
            f"{r['electron_count']:10.6f} {r['E_H_num']:14.8f} {r['E_H_exact']:14.8f} "
            f"{r['error_mHa']:10.3f} {r['relative_error']:10.3e} "
            f"{r['hartree_potential_min']:10.4f} {r['hartree_potential_max']:10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
