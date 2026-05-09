"""Benchmark fixed molecular systems against PySCF references.

This script is intentionally small and explicit: it is the reproducible
benchmark harness described in the repository AGENTS.md optimization plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass(frozen=True)
class BenchmarkSystem:
    name: str
    symbols: tuple[str, ...]
    coords_bohr: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class BenchmarkResult:
    system: str
    target_spacing: float
    actual_spacing: float
    grid_shape: tuple[int, int, int]
    jaxdft_energy_ha: float
    pyscf_energy_ha: float
    error_mha: float
    seconds: float
    scf_max_iter: int
    scf_mix_alpha: float
    scf_tolerance: float
    scf_iterations: int | None
    density_diff: float | None
    density_rms_diff: float | None
    density_l2_diff: float | None
    scf_convergence_residual: float | None
    scf_convergence_metric: str
    energy_delta_last: float | None
    energy_delta_history_last10: tuple[float, ...]
    energy_delta_last10_max: float | None
    density_converged: bool | None
    energy_converged: bool | None
    orbital_residual: float | None
    local_pseudopotential_energy: float | None
    local_pseudopotential_energy_by_atom: tuple[float, ...] | None
    local_pseudopotential_min: float | None
    local_pseudopotential_max: float | None
    local_pseudopotential_integral: float | None
    local_pseudopotential_integral_by_atom: tuple[float, ...] | None
    nonlocal_pseudopotential_energy: float | None
    hartree_energy: float | None
    hartree_potential_min: float | None
    hartree_potential_max: float | None
    hartree_potential_integral: float | None
    projector_overlap_max_error: float | None
    pyscf_e1: float | None
    pyscf_coul: float | None
    pyscf_xc: float | None
    pyscf_nuc: float | None
    orbital_iterations: int | None
    scf_converged: bool | None
    orbital_converged: bool | None
    eigenvalues: tuple[float, ...] | None
    orbital_residuals: tuple[float, ...] | None
    orbital_max_iter: int
    orbital_tolerance: float
    orbital_preconditioner: str
    orbital_preconditioner_shift: float
    mixing_mode: str
    pulay_residual_metric: str
    pulay_kerker_k0: float
    laplacian_order: int
    calculation_dtype: str
    grid_phase: float
    energy_tolerance: float
    energy_history_last10: tuple[float, ...]
    density_diff_history_last10: tuple[float, ...]
    density_rms_diff_history_last10: tuple[float, ...]
    density_l2_diff_history_last10: tuple[float, ...]
    density_min: float
    density_integral: float
    anderson_regularization: float
    anderson_history: int
    mixing_safeguard: str
    mixing_safeguard_factor: float
    mixing_fallback_count: int
    xc_energy: float | None = None
    energy_last20_mean: float | None = None
    energy_last20_std: float | None = None
    scf_status: str | None = None


def _energy_last20_mean_std(energy_history, scf_iterations: int) -> tuple[float | None, float | None]:
    """Mean and sample std of the last up-to-20 finite total energies (includes ion-ion)."""
    vals: list[float] = []
    for i in range(int(scf_iterations)):
        v = float(energy_history[i])
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    tail = vals[-20:]
    mean = statistics.mean(tail)
    if len(tail) < 2:
        return mean, 0.0
    return mean, statistics.stdev(tail)


def get_benchmark_systems() -> list[BenchmarkSystem]:
    h2 = BenchmarkSystem(
        name="H2",
        symbols=("H", "H"),
        coords_bohr=((0.0, 0.0, -0.7), (0.0, 0.0, 0.7)),
    )

    oh_distance = 1.8
    angle_rad = math.radians(104.5)
    hx = oh_distance * math.sin(angle_rad / 2.0)
    hz = oh_distance * math.cos(angle_rad / 2.0)
    h2o = BenchmarkSystem(
        name="H2O",
        symbols=("O", "H", "H"),
        coords_bohr=((0.0, 0.0, 0.0), (hx, 0.0, hz), (-hx, 0.0, hz)),
    )

    co_distance = 2.132
    co = BenchmarkSystem(
        name="CO",
        symbols=("C", "O"),
        coords_bohr=((0.0, 0.0, -co_distance / 2.0), (0.0, 0.0, co_distance / 2.0)),
    )

    return [h2, h2o, co]


def error_mha(jaxdft_energy_ha: float, reference_energy_ha: float) -> float:
    return 1000.0 * (float(jaxdft_energy_ha) - float(reference_energy_ha))


def _format_pyscf_atoms(system: BenchmarkSystem) -> str:
    return "; ".join(
        f"{symbol} {coord[0]} {coord[1]} {coord[2]}"
        for symbol, coord in zip(system.symbols, system.coords_bohr)
    )


def run_pyscf_reference(system: BenchmarkSystem) -> float:
    energy, _ = run_pyscf_reference_with_components(system)
    return energy


def run_pyscf_reference_with_components(system: BenchmarkSystem) -> tuple[float, dict[str, float]]:
    try:
        from pyscf import dft, gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required for benchmark references. Install pyscf or run in the WSL environment that has it."
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
    energy = float(mf.kernel())
    summary = getattr(mf, "scf_summary", {}) or {}
    components = {
        "e1": float(summary.get("e1", math.nan)),
        "coul": float(summary.get("coul", math.nan)),
        "exc": float(summary.get("exc", math.nan)),
        "nuc": float(mol.energy_nuc()),
    }
    return energy, components


def run_jaxdft_energy(
    system: BenchmarkSystem,
    target_spacing: float,
    box_size: float,
    max_iter: int,
    mix_alpha: float,
    tolerance: float,
    pseudo_dir: str,
    orbital_max_iter: int = 30,
    orbital_tolerance: float = 1e-5,
    orbital_preconditioner: str = "none",
    orbital_preconditioner_shift: float = 1.0,
    mixing_mode: str = "anderson",
    anderson_regularization: float = 1e-10,
    anderson_history: int = 5,
    mixing_safeguard: str = "none",
    mixing_safeguard_factor: float = 1.0,
    scf_convergence_metric: str = "max",
    energy_tolerance: float = 5e-6,
    pulay_residual_metric: str = "euclidean",
    pulay_kerker_k0: float = 1.0,
    laplacian_order: int = 8,
    calculation_dtype: str = "float32",
    grid_phase: float = 0.0,
    initial_rho=None,
) -> tuple[float, tuple[int, int, int], float, dict]:
    import jax

    jax.config.update("jax_enable_x64", calculation_dtype == "float64")
    import jax.numpy as jnp

    from JaxDFT.src.hamiltonian import create_grid
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import energy_and_forces

    if calculation_dtype not in ("float32", "float64"):
        raise ValueError("calculation_dtype must be 'float32' or 'float64'")
    dtype = jnp.float64 if calculation_dtype == "float64" else jnp.float32
    n_intervals = int(round(box_size / target_spacing))
    actual_spacing = box_size / n_intervals
    grid = create_grid(actual_spacing, [box_size, box_size, box_size], dtype=dtype, phase=grid_phase)
    pseudos = load_pseudopotentials(list(system.symbols), pseudo_dir)
    coords = jnp.array(system.coords_bohr, dtype=dtype)
    energy, _, info = energy_and_forces(
        grid,
        coords,
        pseudos,
        max_iter,
        mix_alpha,
        tolerance,
        jax.random.PRNGKey(42),
        return_info=True,
        orbital_max_iter=orbital_max_iter,
        orbital_tolerance=orbital_tolerance,
        orbital_preconditioner=orbital_preconditioner,
        orbital_preconditioner_shift=orbital_preconditioner_shift,
        mixing_mode=mixing_mode,
        anderson_regularization=anderson_regularization,
        anderson_history=anderson_history,
        mixing_safeguard=mixing_safeguard,
        mixing_safeguard_factor=mixing_safeguard_factor,
        scf_convergence_metric=scf_convergence_metric,
        energy_tolerance=energy_tolerance,
        pulay_residual_metric=pulay_residual_metric,
        pulay_kerker_k0=pulay_kerker_k0,
        laplacian_order=laplacian_order,
        initial_rho=initial_rho,
    )
    return float(energy), tuple(int(v) for v in grid.shape), float(grid.spacing), info


def run_benchmark(
    system: BenchmarkSystem,
    target_spacing: float,
    box_size: float,
    max_iter: int,
    mix_alpha: float,
    tolerance: float,
    pseudo_dir: str,
    orbital_max_iter: int = 30,
    orbital_tolerance: float = 1e-5,
    orbital_preconditioner: str = "none",
    orbital_preconditioner_shift: float = 1.0,
    mixing_mode: str = "anderson",
    anderson_regularization: float = 1e-10,
    anderson_history: int = 5,
    mixing_safeguard: str = "none",
    mixing_safeguard_factor: float = 1.0,
    scf_convergence_metric: str = "max",
    energy_tolerance: float = 5e-6,
    pulay_residual_metric: str = "euclidean",
    pulay_kerker_k0: float = 1.0,
    laplacian_order: int = 8,
    calculation_dtype: str = "float32",
    grid_phase: float = 0.0,
) -> BenchmarkResult:
    start = time.time()
    reference_energy, pyscf_components = run_pyscf_reference_with_components(system)
    jaxdft_energy, grid_shape, actual_spacing, jaxdft_info = run_jaxdft_energy(
        system,
        target_spacing,
        box_size,
        max_iter,
        mix_alpha,
        tolerance,
        pseudo_dir,
        orbital_max_iter,
        orbital_tolerance,
        orbital_preconditioner,
        orbital_preconditioner_shift,
        mixing_mode,
        anderson_regularization,
        anderson_history,
        mixing_safeguard,
        mixing_safeguard_factor,
        scf_convergence_metric,
        energy_tolerance,
        pulay_residual_metric,
        pulay_kerker_k0,
        laplacian_order,
        calculation_dtype,
        grid_phase,
    )
    scf_iterations = int(jaxdft_info["scf_iterations"])
    hist_start = max(0, scf_iterations - 10)
    energy_history = jaxdft_info["energy_history"][hist_start:scf_iterations]
    energy_delta_history = jaxdft_info["energy_delta_history"][hist_start:scf_iterations]
    density_diff_history = jaxdft_info["density_diff_history"][hist_start:scf_iterations]
    density_rms_history = jaxdft_info["density_rms_diff_history"][hist_start:scf_iterations]
    density_l2_history = jaxdft_info["density_l2_diff_history"][hist_start:scf_iterations]
    projector_overlap_errors = [
        abs(float(v))
        for diag in jaxdft_info["projector_overlap_diagnostics"]
        for row in diag["overlap_error"]
        for v in row
    ]
    projector_overlap_max_error = max(projector_overlap_errors, default=0.0)
    e20_mean, e20_std = _energy_last20_mean_std(jaxdft_info["energy_history"], scf_iterations)
    scf_conv = bool(jaxdft_info["scf_converged"])
    scf_status = "converged" if scf_conv else "not_converged"
    xc_energy = float(jaxdft_info["energy_components"]["xc"])
    return BenchmarkResult(
        system=system.name,
        target_spacing=float(target_spacing),
        actual_spacing=actual_spacing,
        grid_shape=grid_shape,
        jaxdft_energy_ha=jaxdft_energy,
        pyscf_energy_ha=reference_energy,
        error_mha=error_mha(jaxdft_energy, reference_energy),
        seconds=time.time() - start,
        scf_max_iter=int(max_iter),
        scf_mix_alpha=float(mix_alpha),
        scf_tolerance=float(tolerance),
        scf_iterations=scf_iterations,
        density_diff=float(jaxdft_info["density_diff"]),
        density_rms_diff=float(jaxdft_info["density_rms_diff"]),
        density_l2_diff=float(jaxdft_info["density_l2_diff"]),
        scf_convergence_residual=float(jaxdft_info["scf_convergence_residual"]),
        scf_convergence_metric=str(jaxdft_info["scf_convergence_metric"]),
        energy_delta_last=float(jaxdft_info["energy_delta_last"]),
        energy_delta_history_last10=tuple(float(v) for v in energy_delta_history),
        energy_delta_last10_max=float(jaxdft_info["energy_delta_last10_max"]),
        density_converged=bool(jaxdft_info["density_converged"]),
        energy_converged=bool(jaxdft_info["energy_converged"]),
        orbital_residual=float(jaxdft_info["orbital_residual"]),
        local_pseudopotential_energy=float(jaxdft_info["energy_components"]["local_pseudopotential"]),
        local_pseudopotential_energy_by_atom=tuple(
            float(v) for v in jaxdft_info["energy_components"]["local_pseudopotential_by_atom"]
        ),
        local_pseudopotential_min=float(jaxdft_info["local_pseudopotential_min"]),
        local_pseudopotential_max=float(jaxdft_info["local_pseudopotential_max"]),
        local_pseudopotential_integral=float(jaxdft_info["local_pseudopotential_integral"]),
        local_pseudopotential_integral_by_atom=tuple(
            float(v) for v in jaxdft_info["local_pseudopotential_integral_by_atom"]
        ),
        nonlocal_pseudopotential_energy=float(jaxdft_info["energy_components"]["nonlocal_pseudopotential"]),
        hartree_energy=float(jaxdft_info["energy_components"]["hartree"]),
        hartree_potential_min=float(jaxdft_info["hartree_potential_min"]),
        hartree_potential_max=float(jaxdft_info["hartree_potential_max"]),
        hartree_potential_integral=float(jaxdft_info["hartree_potential_integral"]),
        projector_overlap_max_error=float(projector_overlap_max_error),
        pyscf_e1=float(pyscf_components["e1"]),
        pyscf_coul=float(pyscf_components["coul"]),
        pyscf_xc=float(pyscf_components["exc"]),
        pyscf_nuc=float(pyscf_components["nuc"]),
        orbital_iterations=int(jaxdft_info["orbital_iterations"]),
        scf_converged=bool(jaxdft_info["scf_converged"]),
        orbital_converged=bool(jaxdft_info["orbital_converged"]),
        eigenvalues=tuple(float(v) for v in jaxdft_info["eigenvalues"]),
        orbital_residuals=tuple(float(v) for v in jaxdft_info["orbital_residuals"]),
        orbital_max_iter=int(jaxdft_info["orbital_max_iter"]),
        orbital_tolerance=float(jaxdft_info["orbital_tolerance"]),
        orbital_preconditioner=str(jaxdft_info["orbital_preconditioner"]),
        orbital_preconditioner_shift=float(jaxdft_info["orbital_preconditioner_shift"]),
        mixing_mode=str(jaxdft_info["mixing_mode"]),
        pulay_residual_metric=str(jaxdft_info["pulay_residual_metric"]),
        pulay_kerker_k0=float(jaxdft_info["pulay_kerker_k0"]),
        laplacian_order=int(jaxdft_info["laplacian_order"]),
        calculation_dtype=str(calculation_dtype),
        grid_phase=float(grid_phase),
        energy_tolerance=float(energy_tolerance),
        energy_history_last10=tuple(float(v) for v in energy_history),
        density_diff_history_last10=tuple(float(v) for v in density_diff_history),
        density_rms_diff_history_last10=tuple(float(v) for v in density_rms_history),
        density_l2_diff_history_last10=tuple(float(v) for v in density_l2_history),
        density_min=float(jaxdft_info["density_min"]),
        density_integral=float(jaxdft_info["density_integral"]),
        anderson_regularization=float(jaxdft_info["anderson_regularization"]),
        anderson_history=int(jaxdft_info["anderson_history"]),
        mixing_safeguard=str(jaxdft_info["mixing_safeguard"]),
        mixing_safeguard_factor=float(jaxdft_info["mixing_safeguard_factor"]),
        mixing_fallback_count=int(jaxdft_info["mixing_fallback_count"]),
        xc_energy=xc_energy,
        energy_last20_mean=e20_mean,
        energy_last20_std=e20_std,
        scf_status=scf_status,
    )


def _select_systems(names: Iterable[str]) -> list[BenchmarkSystem]:
    requested = {name.upper() for name in names}
    systems = get_benchmark_systems()
    selected = [system for system in systems if system.name.upper() in requested]
    missing = requested - {system.name.upper() for system in selected}
    if missing:
        raise ValueError(f"Unknown benchmark system(s): {', '.join(sorted(missing))}")
    return selected


def _print_table(results: Sequence[BenchmarkResult]) -> None:
    header = (
        "system target_dx actual_dx grid_shape jaxdft_ha pyscf_ha error_mha seconds "
        "scf_iter density_diff density_rms density_l2 energy_delta_last10_max "
        "orbital_residual orbital_iter density_ok energy_ok scf_ok orbital_ok "
        "orbital_precond mixing_mode max_iter mix_alpha tolerance energy_tolerance"
    )
    print(header)
    for result in results:
        scf_iter = "NA" if result.scf_iterations is None else str(result.scf_iterations)
        print(
            f"{result.system} "
            f"{result.target_spacing:.6f} "
            f"{result.actual_spacing:.6f} "
            f"{result.grid_shape} "
            f"{result.jaxdft_energy_ha:.12f} "
            f"{result.pyscf_energy_ha:.12f} "
            f"{result.error_mha:.3f} "
            f"{result.seconds:.1f} "
            f"{scf_iter} "
            f"{result.density_diff:.3e} "
            f"{result.density_rms_diff:.3e} "
            f"{result.density_l2_diff:.3e} "
            f"{result.energy_delta_last10_max:.3e} "
            f"{result.orbital_residual:.3e} "
            f"{result.orbital_iterations} "
            f"{result.density_converged} "
            f"{result.energy_converged} "
            f"{result.scf_converged} "
            f"{result.orbital_converged} "
            f"{result.orbital_preconditioner} "
            f"{result.mixing_mode} "
            f"{result.scf_max_iter} "
            f"{result.scf_mix_alpha:.6f} "
            f"{result.scf_tolerance:.3e} "
            f"{result.energy_tolerance:.3e}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=["H2", "H2O", "CO"])
    parser.add_argument("--spacings", nargs="+", type=float, default=[0.18, 0.12])
    parser.add_argument("--box-size", type=float, default=18.0)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--mix-alpha", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--orbital-max-iter", type=int, default=30)
    parser.add_argument("--orbital-tolerance", type=float, default=1e-5)
    parser.add_argument("--orbital-preconditioner", choices=["none", "kinetic"], default="none")
    parser.add_argument("--orbital-preconditioner-shift", type=float, default=1.0)
    parser.add_argument("--mixing-mode", choices=["anderson", "linear", "pulay"], default="anderson")
    parser.add_argument("--pulay-residual-metric", choices=["euclidean", "kerker"], default="euclidean")
    parser.add_argument("--pulay-kerker-k0", type=float, default=1.0)
    parser.add_argument("--laplacian-order", type=int, choices=[4, 6, 8], default=8)
    parser.add_argument("--calculation-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--grid-phase", type=float, default=0.0)
    parser.add_argument("--anderson-regularization", type=float, default=1e-10)
    parser.add_argument("--anderson-history", type=int, default=5)
    parser.add_argument("--mixing-safeguard", choices=["none", "density_diff"], default="none")
    parser.add_argument("--mixing-safeguard-factor", type=float, default=1.0)
    parser.add_argument("--scf-convergence-metric", choices=["max", "rms", "l2"], default="max")
    parser.add_argument(
        "--energy-tolerance",
        type=float,
        default=5e-6,
        help="Ha: energy_converged iff max |E_i-E_{i-1}| over last 10 SCF iters <= this "
        "(same units as energy_delta_last10_max; CO fine grids often plateau ~1e-3 Ha).",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=os.path.join(REPO_ROOT, "JaxDFT", "data", "gth_potentials"),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        systems = _select_systems(args.systems)
        results = [
            run_benchmark(
                system,
                spacing,
                args.box_size,
                args.max_iter,
                args.mix_alpha,
                args.tolerance,
                args.pseudo_dir,
                args.orbital_max_iter,
                args.orbital_tolerance,
                args.orbital_preconditioner,
                args.orbital_preconditioner_shift,
                args.mixing_mode,
                args.anderson_regularization,
                args.anderson_history,
                args.mixing_safeguard,
                args.mixing_safeguard_factor,
                args.scf_convergence_metric,
                args.energy_tolerance,
                args.pulay_residual_metric,
                args.pulay_kerker_k0,
                args.laplacian_order,
                args.calculation_dtype,
                args.grid_phase,
            )
            for system in systems
            for spacing in args.spacings
        ]
    except (RuntimeError, ValueError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
