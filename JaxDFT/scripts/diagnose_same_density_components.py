#!/usr/bin/env python3
"""Same-density component diagnostic: PySCF rho on JaxDFT grid vs PySCF energy components.

Uses PySCF converged RKS density (gth-tzvp, gth-lda, lda,pz), evaluates rho on the
same Cartesian grid as JaxDFT ``create_grid``, then forms Hartree, local GTH
pseudopotential, and XC energies with JaxDFT operators.

Interpretation (see repository docs):

- If Hartree / XC / local match PySCF counterparts at fixed rho, JaxDFT grid
  operators are likely sound and remaining total-energy bias is dominated by
  self-consistent density differences (or nonlocal / kinetic paths not tested here).
- If large discrepancies remain at fixed rho, suspect the corresponding JaxDFT
  operator implementation or convention mismatch vs PySCF.

``pyscf_e1`` is a *composite* KS one-electron-like term in PySCF's ``scf_summary``;
it is not the same object as JaxDFT local pseudopotential energy alone. Prefer
``jax_hartree_same_density`` vs ``pyscf_coul`` and ``jax_xc_same_density`` vs
``pyscf_exc`` for direct operator alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from JaxDFT.scripts.benchmark_systems import (  # noqa: E402
    BenchmarkSystem,
    _format_pyscf_atoms,
    get_benchmark_systems,
)
from JaxDFT.src.functional import lda_xc  # noqa: E402
from JaxDFT.src.hamiltonian import build_local_potential, create_grid  # noqa: E402
from JaxDFT.src.io import load_pseudopotentials  # noqa: E402
from JaxDFT.src.solver import precompute_poisson_kernel, solve_poisson  # noqa: E402


def _system_by_name(name: str) -> BenchmarkSystem:
    for s in get_benchmark_systems():
        if s.name == name:
            return s
    raise ValueError(f"Unknown system {name!r}; choose from H2, H2O, CO")


def pyscf_converged_mf_and_components(system: BenchmarkSystem) -> tuple[Any, Any, dict[str, float]]:
    """Run PySCF RKS to convergence; return ``(mol, mf, scf_summary_components)``."""
    try:
        from pyscf import dft, gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required for same-density diagnostics. Install pyscf in this environment."
        ) from exc

    mol = gto.M(
        atom=_format_pyscf_atoms(system),
        unit="Bohr",
        basis="gth-tzvp",
        pseudo="gth-lda",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,pz"
    mf.kernel()
    summary = getattr(mf, "scf_summary", {}) or {}
    components = {
        "e1": float(summary.get("e1", float("nan"))),
        "coul": float(summary.get("coul", float("nan"))),
        "exc": float(summary.get("exc", float("nan"))),
    }
    return mol, mf, components


def eval_pyscf_rho_on_jax_grid(mol, mf, grid) -> jnp.ndarray:
    """Evaluate PySCF converged spin-summed density on JaxDFT grid nodes (Bohr^-3)."""
    from pyscf import dft

    coords = np.asarray(jnp.array(grid.coords).reshape(-1, 3), dtype=np.float64)
    ao = dft.numint.eval_ao(mol, coords)
    dm = mf.make_rdm1()
    if isinstance(dm, tuple):
        dm_tot = dm[0] + dm[1]
    else:
        dm_tot = dm
    rho_np = dft.numint.eval_rho(mol, ao, dm_tot, xctype="LDA", hermi=1)
    rho = jnp.asarray(rho_np.reshape(grid.shape), dtype=grid.coords.dtype)
    return rho


def make_jax_grid(target_spacing: float, box_size: float, dtype: jnp.dtype, grid_phase: float):
    """Match ``benchmark_systems.run_jaxdft_energy`` grid construction."""
    n_intervals = int(round(box_size / target_spacing))
    actual_spacing = box_size / n_intervals
    grid = create_grid(actual_spacing, [box_size, box_size, box_size], dtype=dtype, phase=grid_phase)
    return grid


def jax_operator_components_from_rho(
    rho: jnp.ndarray,
    grid,
    coords: jnp.ndarray,
    zion: jnp.ndarray,
    rloc: jnp.ndarray,
    c: jnp.ndarray,
) -> dict[str, Any]:
    """Hartree, total local GTH pseudo energy, XC using existing JaxDFT paths."""
    dtype = rho.dtype
    dV = grid.volume_element
    spacing = float(grid.spacing)

    rho_work = jnp.clip(jnp.nan_to_num(rho, nan=0.0), 1e-15, None).astype(dtype)

    kernel_k = precompute_poisson_kernel(grid.shape, spacing)
    V_H = solve_poisson(rho_work, kernel_k, spacing).astype(dtype)
    hartree = 0.5 * float(dV * jnp.sum(rho_work * V_H))

    V_loc = build_local_potential(coords, grid.coords, zion, rloc, c)
    local_pp = float(dV * jnp.sum(rho_work * V_loc))

    eps_xc, _ = lda_xc(rho_work)
    xc = float(dV * jnp.sum(eps_xc))

    return {
        "jax_hartree_same_density": hartree,
        "jax_local_pseudopotential_same_density": local_pp,
        "jax_xc_same_density": xc,
        "hartree_potential_min": float(jnp.min(V_H)),
        "hartree_potential_max": float(jnp.max(V_H)),
    }


def run_same_density_diagnostic(
    system: BenchmarkSystem,
    target_spacing: float,
    box_size: float,
    pseudo_dir: str,
    calculation_dtype: str = "float32",
    grid_phase: float = 0.0,
) -> dict[str, Any]:
    """One system: PySCF rho on Jax grid + Jax operators + PySCF reference components."""
    if calculation_dtype not in ("float32", "float64"):
        raise ValueError("calculation_dtype must be 'float32' or 'float64'")
    jax.config.update("jax_enable_x64", calculation_dtype == "float64")
    dtype = jnp.float64 if calculation_dtype == "float64" else jnp.float32

    mol, mf, pyscf_components = pyscf_converged_mf_and_components(system)
    nelec = int(mol.nelectron)

    pyscf_e1 = pyscf_components["e1"]
    pyscf_coul = pyscf_components["coul"]
    pyscf_exc = pyscf_components["exc"]
    pyscf_nuc = float(mol.energy_nuc())

    grid = make_jax_grid(target_spacing, box_size, dtype, grid_phase)
    pseudos = load_pseudopotentials(list(system.symbols), pseudo_dir)
    coords = jnp.array(system.coords_bohr, dtype=dtype)
    zion = jnp.array([p["zion"] for p in pseudos], dtype=dtype)
    rloc = jnp.array([p["rloc"] for p in pseudos], dtype=dtype)
    c = jnp.array([p["c"] for p in pseudos], dtype=dtype)

    rho_pyscf = eval_pyscf_rho_on_jax_grid(mol, mf, grid)
    dV = float(grid.volume_element)
    rho_integral_raw = float(dV * jnp.sum(rho_pyscf))
    rho_min = float(jnp.min(rho_pyscf))
    rho_max = float(jnp.max(rho_pyscf))

    jax_parts = jax_operator_components_from_rho(rho_pyscf, grid, coords, zion, rloc, c)

    def _dmha(a: float, b: float) -> float:
        return 1000.0 * (a - b)

    row: dict[str, Any] = {
        "system": system.name,
        "pyscf_converged": bool(getattr(mf, "converged", True)),
        "target_spacing": float(target_spacing),
        "actual_spacing": float(grid.spacing),
        "box_size": float(box_size),
        "grid_shape": tuple(int(x) for x in grid.shape),
        "n_electrons": nelec,
        "rho_integral_raw": rho_integral_raw,
        "rho_electron_count_error": rho_integral_raw - float(nelec),
        "rho_min": rho_min,
        "rho_max": rho_max,
        "pyscf_e1": pyscf_e1,
        "pyscf_coul": pyscf_coul,
        "pyscf_exc": pyscf_exc,
        "pyscf_nuc": pyscf_nuc,
        **jax_parts,
        "delta_hartree_vs_pyscf_coul_mHa": _dmha(jax_parts["jax_hartree_same_density"], pyscf_coul),
        "delta_xc_vs_pyscf_exc_mHa": _dmha(jax_parts["jax_xc_same_density"], pyscf_exc),
        "delta_local_vs_pyscf_e1_mHa": _dmha(jax_parts["jax_local_pseudopotential_same_density"], pyscf_e1),
        "note_pyscf_e1": (
            "PySCF scf_summary['e1'] is a composite KS term (not equal to local pseudopotential energy alone)."
        ),
    }
    return row


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--systems",
        nargs="+",
        default=["CO", "H2O"],
        help="Benchmark system names: CO, H2O, H2 (default: CO H2O).",
    )
    p.add_argument("--target-spacing", type=float, default=0.12, help="Target grid spacing (Bohr).")
    p.add_argument("--box-size", type=float, default=18.0, help="Cubic box edge (Bohr).")
    p.add_argument(
        "--pseudo-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "JaxDFT", "data", "gth_potentials"),
        help="GTH pseudopotential data directory.",
    )
    p.add_argument("--calculation-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--grid-phase", type=float, default=0.0)
    p.add_argument("--json", action="store_true", help="Print JSON array of result rows.")
    args = p.parse_args(list(argv) if argv is not None else None)

    rows = []
    for name in args.systems:
        system = _system_by_name(name)
        rows.append(
            run_same_density_diagnostic(
                system,
                args.target_spacing,
                args.box_size,
                args.pseudo_dir,
                calculation_dtype=args.calculation_dtype,
                grid_phase=args.grid_phase,
            )
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    for r in rows:
        print(json.dumps(r, indent=2))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
