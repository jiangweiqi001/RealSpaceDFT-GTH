"""Unified fine-grid benchmark runner for O, H2O, and N2.

This script compares JaxDFT against a PySCF GTH/LDA reference while sweeping:

- system: O, H2O, N2
- coarse-grid spacing
- coarse-grid placement: centered vs half-grid shift
- fine-grid mode: off vs auto

It writes one CSV and one PNG per system so the benchmark panel stays stable
across future code changes. A dedicated projector-sweep mode extends the
comparison to explicit local-only vs local+projector-patch settings.

Mainline semantics:
- standard mode is the stable benchmark path: `off` vs `fine_grid_mode="auto"`
- `auto` means atom-centered `V_loc` patch only
- projector patch comparisons are experimental research runs and should not be
  interpreted as the stable default path
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

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
CSV_FIELDNAMES = [
    "system",
    "case",
    "geometry_param_name",
    "geometry_param",
    "spacing_bohr",
    "actual_spacing_bohr",
    "box_size_bohr",
    "shift_fraction",
    "mode_label",
    "fine_grid_mode",
    "local_subgrid",
    "local_mode",
    "local_patch_radius_factor",
    "projector_subgrid",
    "projector_mode",
    "projector_patch_radius_factor",
    "energy_jax_ha",
    "energy_pyscf_ha",
    "error_vs_pyscf_ha",
    "scf_steps",
    "runtime_sec",
    "notes",
]


class SystemSpec(NamedTuple):
    atom_symbols: list[str]
    geometry_param_name: str
    geometry_values: list[float]
    x_label: str
    title: str


class BenchmarkJob(NamedTuple):
    system: str
    case: str
    geometry_param_name: str
    geometry_param: float
    spacing_bohr: float
    box_size_bohr: float
    shift_fraction: float
    mode_label: str
    fine_grid_mode: str | None
    local_subgrid: int
    local_mode: str
    local_patch_radius_factor: float
    projector_subgrid: int
    projector_mode: str
    projector_patch_radius_factor: float


SYSTEM_SPECS = {
    "O": SystemSpec(
        atom_symbols=["O"],
        geometry_param_name="none",
        geometry_values=[0.0],
        x_label="Grid spacing (Bohr)",
        title="O atom fine-grid benchmark",
    ),
    "H2O": SystemSpec(
        atom_symbols=["O", "H", "H"],
        geometry_param_name="oh_distance",
        geometry_values=[1.6, 1.8, 2.0, 2.4],
        x_label="O-H bond length (Bohr)",
        title="H2O symmetric stretch fine-grid benchmark",
    ),
    "N2": SystemSpec(
        atom_symbols=["N", "N"],
        geometry_param_name="bond_length",
        geometry_values=[1.8, 2.0, 2.2, 2.4, 2.8],
        x_label="N-N bond length (Bohr)",
        title="N2 stretch fine-grid benchmark",
    ),
}


def require_optional_dependencies():
    try:
        import matplotlib.pyplot as plt
        from pyscf import dft, gto
    except ImportError as exc:
        raise SystemExit(
            "This benchmark requires optional packages: pyscf and matplotlib. "
            "Install them in your active environment, then rerun the script."
        ) from exc
    return plt, gto, dft


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-mode",
        choices=["standard", "projector_sweep"],
        default="standard",
        help="Run the stable off/auto mainline comparison or the experimental projector sweep.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=sorted(SYSTEM_SPECS.keys()),
        default=["O", "H2O", "N2"],
        help="Benchmark systems to run.",
    )
    parser.add_argument(
        "--spacings",
        nargs="+",
        type=float,
        default=[0.40, 0.32, 0.24],
        help="Target coarse-grid spacings in Bohr.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["center", "half_shift"],
        default=["center", "half_shift"],
        help="Grid-placement cases to benchmark.",
    )
    parser.add_argument("--box-size", type=float, default=9.6, help="Cubic box size in Bohr.")
    parser.add_argument("--shift-fraction", type=float, default=0.5, help="Shift in units of actual spacing.")
    parser.add_argument(
        "--fine-grid-modes",
        nargs="+",
        choices=["off", "auto"],
        default=["off", "auto"],
        help="Baseline fine-grid modes to compare.",
    )
    parser.add_argument(
        "--projector-subgrids",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Projector fine subgrid values for projector_sweep mode.",
    )
    parser.add_argument(
        "--projector-radii",
        nargs="+",
        type=float,
        default=[4.0, 6.0, 8.0],
        help="Projector patch radius factors for projector_sweep mode.",
    )
    parser.add_argument("--angle-deg", type=float, default=104.5, help="H-O-H angle in degrees for H2O.")
    parser.add_argument("--output-dir", default=".", help="Directory for CSV and PNG outputs.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation and only write CSV.")
    return parser.parse_args()


def build_shift_vector(actual_spacing: float, shift_fraction: float, case_name: str):
    if case_name == "center":
        return [0.0, 0.0, 0.0]
    if case_name == "half_shift":
        delta = float(actual_spacing) * float(shift_fraction)
        return [delta, delta, delta]
    raise ValueError(f"Unknown benchmark case: {case_name}")


def build_coords(system_name: str, geometry_param: float, shift, angle_deg: float):
    sx, sy, sz = shift
    if system_name == "O":
        return [[sx, sy, sz]]
    if system_name == "H2O":
        theta = math.radians(angle_deg)
        hx = geometry_param * math.sin(theta / 2.0)
        hz = geometry_param * math.cos(theta / 2.0)
        return [
            [sx, sy, sz],
            [sx + hx, sy, sz + hz],
            [sx - hx, sy, sz + hz],
        ]
    if system_name == "N2":
        return [
            [sx, sy, sz - geometry_param / 2.0],
            [sx, sy, sz + geometry_param / 2.0],
        ]
    raise ValueError(f"Unsupported system: {system_name}")


def _baseline_job(system_name: str, geometry_param_name: str, geometry_param: float, spacing: float, box_size: float, case: str, shift_fraction: float, fine_grid_mode: str):
    if fine_grid_mode == "off":
        return BenchmarkJob(
            system=system_name,
            case=case,
            geometry_param_name=geometry_param_name,
            geometry_param=geometry_param,
            spacing_bohr=float(spacing),
            box_size_bohr=float(box_size),
            shift_fraction=float(shift_fraction if case == "half_shift" else 0.0),
            mode_label="off",
            fine_grid_mode="off",
            local_subgrid=1,
            local_mode="cell_average",
            local_patch_radius_factor=6.0,
            projector_subgrid=1,
            projector_mode="cell_average",
            projector_patch_radius_factor=6.0,
        )
    if fine_grid_mode == "auto":
        return BenchmarkJob(
            system=system_name,
            case=case,
            geometry_param_name=geometry_param_name,
            geometry_param=geometry_param,
            spacing_bohr=float(spacing),
            box_size_bohr=float(box_size),
            shift_fraction=float(shift_fraction if case == "half_shift" else 0.0),
            mode_label="auto_local",
            fine_grid_mode="auto",
            local_subgrid=5,
            local_mode="patch",
            local_patch_radius_factor=4.0,
            projector_subgrid=1,
            projector_mode="cell_average",
            projector_patch_radius_factor=6.0,
        )
    raise ValueError(f"Unsupported baseline fine_grid_mode: {fine_grid_mode}")


def _projector_patch_job(system_name: str, geometry_param_name: str, geometry_param: float, spacing: float, box_size: float, case: str, shift_fraction: float, projector_subgrid: int, projector_radius: float):
    return BenchmarkJob(
        system=system_name,
        case=case,
        geometry_param_name=geometry_param_name,
        geometry_param=geometry_param,
        spacing_bohr=float(spacing),
        box_size_bohr=float(box_size),
        shift_fraction=float(shift_fraction if case == "half_shift" else 0.0),
        mode_label=f"local_proj_sg{projector_subgrid}_r{projector_radius:.1f}",
        fine_grid_mode=None,
        local_subgrid=5,
        local_mode="patch",
        local_patch_radius_factor=4.0,
        projector_subgrid=int(projector_subgrid),
        projector_mode="patch",
        projector_patch_radius_factor=float(projector_radius),
    )


def build_jobs(
    system_name: str,
    spacing_values,
    shift_fraction: float,
    benchmark_mode: str,
    fine_grid_modes,
    projector_subgrids,
    projector_radii,
    box_size: float,
    cases=("center", "half_shift"),
):
    spec = SYSTEM_SPECS[system_name]
    jobs = []
    for geometry_param in spec.geometry_values:
        for spacing in spacing_values:
            for case in cases:
                for fine_grid_mode in fine_grid_modes:
                    jobs.append(_baseline_job(
                        system_name,
                        spec.geometry_param_name,
                        geometry_param,
                        spacing,
                        box_size,
                        case,
                        shift_fraction,
                        fine_grid_mode,
                    ))
                if benchmark_mode == "projector_sweep":
                    for projector_subgrid in projector_subgrids:
                        for projector_radius in projector_radii:
                            jobs.append(
                                _projector_patch_job(
                                    system_name,
                                    spec.geometry_param_name,
                                    geometry_param,
                                    spacing,
                                    box_size,
                                    case,
                                    shift_fraction,
                                    projector_subgrid,
                                    projector_radius,
                                )
                            )
    return jobs


def resolve_grid(target_spacing: float, box_size: float):
    n_intervals = int(round(box_size / target_spacing))
    actual_spacing = box_size / n_intervals
    grid = create_grid(actual_spacing, [box_size, box_size, box_size])
    return grid, float(grid.spacing)


def run_pyscf_energy(atom_symbols, coords, gto, dft):
    atom = "; ".join(
        f"{symbol} {xyz[0]} {xyz[1]} {xyz[2]}"
        for symbol, xyz in zip(atom_symbols, coords)
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


def scf_settings(system_name: str, geometry_param: float):
    if system_name == "O":
        return 0.30, 400
    if system_name == "H2O":
        if geometry_param >= 2.8:
            return 0.05, 800
        if geometry_param >= 2.4:
            return 0.10, 600
        return 0.30, 400
    if system_name == "N2":
        if geometry_param >= 3.0:
            return 0.05, 700
        if geometry_param >= 2.4:
            return 0.10, 600
        return 0.30, 400
    raise ValueError(f"Unsupported system: {system_name}")


def run_jax_energy(grid, coords, pseudos, key, system_name: str, geometry_param: float, job: BenchmarkJob):
    alpha, max_iter = scf_settings(system_name, geometry_param)
    kwargs = {}
    if job.fine_grid_mode is not None:
        kwargs["fine_grid_mode"] = job.fine_grid_mode
    else:
        kwargs.update(
            local_subgrid=job.local_subgrid,
            local_mode=job.local_mode,
            local_patch_radius_factor=job.local_patch_radius_factor,
            projector_subgrid=job.projector_subgrid,
            projector_mode=job.projector_mode,
            projector_patch_radius_factor=job.projector_patch_radius_factor,
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


def job_notes(job: BenchmarkJob):
    notes = ["scf_steps unavailable in current solver API"]
    if job.projector_mode == "patch":
        notes.append("experimental projector patch; not part of the stable auto mainline")
    return "; ".join(notes)


def run_system(system_name: str, args, gto, dft):
    spec = SYSTEM_SPECS[system_name]
    jobs = build_jobs(
        system_name=system_name,
        spacing_values=args.spacings,
        shift_fraction=args.shift_fraction,
        benchmark_mode=args.benchmark_mode,
        fine_grid_modes=args.fine_grid_modes,
        projector_subgrids=args.projector_subgrids,
        projector_radii=args.projector_radii,
        cases=args.cases,
        box_size=args.box_size,
    )
    grids = {}
    pseudos = load_pseudopotentials(spec.atom_symbols, PSEUDO_DIR)
    rows = []

    print(f"\nSystem {system_name}")
    print("-" * 112)
    print(
        f"{'geom':>8} {'spacing':>8} {'case':>12} {'mode':>18} "
        f"{'PySCF':>16} {'JaxDFT':>16} {'Error':>14} {'runtime':>10}"
    )
    for job_idx, job in enumerate(jobs):
        grid_key = (job.spacing_bohr, job.box_size_bohr)
        if grid_key not in grids:
            grids[grid_key] = resolve_grid(job.spacing_bohr, job.box_size_bohr)
        grid, actual_spacing = grids[grid_key]
        shift = build_shift_vector(actual_spacing, args.shift_fraction, job.case)
        coords = build_coords(system_name, job.geometry_param, shift, args.angle_deg)
        pyscf_energy = run_pyscf_energy(spec.atom_symbols, coords, gto, dft)
        key = jax.random.fold_in(jax.random.PRNGKey(42), job_idx)
        start = time.perf_counter()
        jax_energy = run_jax_energy(grid, coords, pseudos, key, system_name, job.geometry_param, job)
        runtime_sec = time.perf_counter() - start
        error_vs_pyscf = jax_energy - pyscf_energy
        rows.append(
            {
                "system": system_name,
                "case": job.case,
                "geometry_param_name": job.geometry_param_name,
                "geometry_param": job.geometry_param,
                "spacing_bohr": job.spacing_bohr,
                "actual_spacing_bohr": actual_spacing,
                "box_size_bohr": job.box_size_bohr,
                "shift_fraction": job.shift_fraction,
                "mode_label": job.mode_label,
                "fine_grid_mode": "" if job.fine_grid_mode is None else job.fine_grid_mode,
                "local_subgrid": job.local_subgrid,
                "local_mode": job.local_mode,
                "local_patch_radius_factor": job.local_patch_radius_factor,
                "projector_subgrid": job.projector_subgrid,
                "projector_mode": job.projector_mode,
                "projector_patch_radius_factor": job.projector_patch_radius_factor,
                "energy_jax_ha": jax_energy,
                "energy_pyscf_ha": pyscf_energy,
                "error_vs_pyscf_ha": error_vs_pyscf,
                "scf_steps": "",
                "runtime_sec": runtime_sec,
                "notes": job_notes(job),
            }
        )
        print(
            f"{job.geometry_param:8.3f} {actual_spacing:8.3f} {job.case:>12} {job.mode_label:>18} "
            f"{pyscf_energy:16.8f} {jax_energy:16.8f} {error_vs_pyscf:14.6f} {runtime_sec:10.2f}"
        )
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _rows_for_case(rows, case_name: str, mode_label: str):
    return [
        row for row in rows
        if row["case"] == case_name and row["mode_label"] == mode_label
    ]


def _o_shift_by_spacing(rows, mode: str):
    center_rows = {
        row["spacing_bohr"]: row["energy_jax_ha"]
        for row in _rows_for_case(rows, "center", mode)
    }
    shifted_rows = {
        row["spacing_bohr"]: row["energy_jax_ha"]
        for row in _rows_for_case(rows, "half_shift", mode)
    }
    xs = sorted(center_rows)
    ys = [abs(shifted_rows[x] - center_rows[x]) for x in xs]
    return xs, ys


def _geometry_rows(rows, case_name: str, spacing: float, mode_label: str):
    matched = [
        row for row in rows
        if row["case"] == case_name
        and row["mode_label"] == mode_label
        and abs(row["spacing_bohr"] - spacing) < 1e-12
    ]
    return sorted(matched, key=lambda row: row["geometry_param"])


def _shift_span_by_spacing(rows, mode: str):
    spacings = sorted({row["spacing_bohr"] for row in rows})
    spans = []
    for spacing in spacings:
        center_rows = {
            row["geometry_param"]: row["energy_jax_ha"]
            for row in _geometry_rows(rows, "center", spacing, mode)
        }
        shifted_rows = {
            row["geometry_param"]: row["energy_jax_ha"]
            for row in _geometry_rows(rows, "half_shift", spacing, mode)
        }
        common = sorted(set(center_rows) & set(shifted_rows))
        span = max(abs(shifted_rows[g] - center_rows[g]) for g in common)
        spans.append(span)
    return spacings, spans


def _projector_mode_labels(rows):
    return sorted(
        {
            row["mode_label"] for row in rows
            if row["mode_label"] not in ("off", "auto_local")
        }
    )


def _summary_abs_error(rows, case_name: str, spacing: float, mode_label: str):
    matched = _geometry_rows(rows, case_name, spacing, mode_label)
    values = [abs(row["error_vs_pyscf_ha"]) for row in matched]
    return sum(values) / len(values)


def _plot_standard_results(path: Path, system_name: str, rows, plt):
    spec = SYSTEM_SPECS[system_name]
    if system_name == "O":
        fig, (ax_error, ax_shift) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        for mode, marker in (("off", "o"), ("auto_local", "s")):
            for case, linestyle in (("center", "-"), ("half_shift", "--")):
                matched = _rows_for_case(rows, case, mode)
                matched = sorted(matched, key=lambda row: row["spacing_bohr"])
                xs = [row["spacing_bohr"] for row in matched]
                ys = [abs(row["error_vs_pyscf_ha"]) for row in matched]
                ax_error.plot(xs, ys, marker=marker, linestyle=linestyle, label=f"{mode} {case}")
            xs, ys = _o_shift_by_spacing(rows, mode)
            ax_shift.plot(xs, ys, marker=marker, linestyle="-", label=f"{mode}")
        ax_error.set_ylabel("|JaxDFT - PySCF| (Ha)")
        ax_error.set_title(spec.title)
        ax_error.grid(True, alpha=0.3)
        ax_error.legend()
        ax_shift.set_xlabel(spec.x_label)
        ax_shift.set_ylabel("|E(half_shift) - E(center)| (Ha)")
        ax_shift.grid(True, alpha=0.3)
        ax_shift.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=False)
    ax_energy, ax_error, ax_shift = axes
    spacing_values = sorted({row["spacing_bohr"] for row in rows})
    reference_rows = _geometry_rows(rows, "half_shift", spacing_values[0], "off")
    ax_energy.plot(
        [row["geometry_param"] for row in reference_rows],
        [row["energy_pyscf_ha"] for row in reference_rows],
        "k--",
        linewidth=2,
        label="PySCF",
    )
    for spacing in spacing_values:
        for mode, marker in (("off", "o"), ("auto_local", "s")):
            matched = _geometry_rows(rows, "half_shift", spacing, mode)
            xs = [row["geometry_param"] for row in matched]
            energy_ys = [row["energy_jax_ha"] for row in matched]
            error_ys = [abs(row["error_vs_pyscf_ha"]) for row in matched]
            label = f"{mode}, h={spacing:.2f}"
            ax_energy.plot(xs, energy_ys, marker=marker, label=label)
            ax_error.plot(xs, error_ys, marker=marker, label=label)
    for mode, marker in (("off", "o"), ("auto_local", "s")):
        xs, ys = _shift_span_by_spacing(rows, mode)
        ax_shift.plot(xs, ys, marker=marker, linestyle="-", label=mode)

    ax_energy.set_ylabel("Total energy (Ha)")
    ax_energy.set_title(spec.title + " (half-shift curves)")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()
    ax_error.set_xlabel(spec.x_label)
    ax_error.set_ylabel("|JaxDFT - PySCF| (Ha)")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()
    ax_shift.set_xlabel("Grid spacing (Bohr)")
    ax_shift.set_ylabel("Max shift span (Ha)")
    ax_shift.grid(True, alpha=0.3)
    ax_shift.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def _plot_projector_sweep_results(path: Path, system_name: str, rows, plt):
    spec = SYSTEM_SPECS[system_name]
    spacing_values = sorted({row["spacing_bohr"] for row in rows})
    projector_labels = _projector_mode_labels(rows)
    if not projector_labels:
        return _plot_standard_results(path, system_name, rows, plt)

    fig, axes = plt.subplots(2, len(spacing_values), figsize=(5 * len(spacing_values), 8), sharey="row")
    if len(spacing_values) == 1:
        axes = [[axes[0]], [axes[1]]]
    else:
        axes = axes.tolist()

    radius_to_points = {}
    for label in projector_labels:
        parts = label.split("_")
        subgrid = int(parts[2][2:])
        radius = float(parts[3][1:])
        radius_to_points.setdefault(radius, []).append((subgrid, label))

    for col, spacing in enumerate(spacing_values):
        for row_idx, case_name in enumerate(("center", "half_shift")):
            ax = axes[row_idx][col]
            for radius, entries in sorted(radius_to_points.items()):
                entries = sorted(entries)
                xs = [subgrid for subgrid, _ in entries]
                ys = [_summary_abs_error(rows, case_name, spacing, label) for _, label in entries]
                ax.plot(xs, ys, marker="o", label=f"r={radius:.1f}")

            off_val = _summary_abs_error(rows, case_name, spacing, "off")
            auto_val = _summary_abs_error(rows, case_name, spacing, "auto_local")
            ax.axhline(off_val, color="tab:gray", linestyle="--", label="off" if row_idx == 0 and col == 0 else None)
            ax.axhline(auto_val, color="tab:green", linestyle="-.", label="auto_local" if row_idx == 0 and col == 0 else None)
            ax.set_title(f"{case_name}, h={spacing:.2f}")
            ax.set_xlabel("Projector subgrid")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("Mean |JaxDFT - PySCF| (Ha)")
            ax.legend()

    fig.suptitle(spec.title + " projector patch sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def plot_results(path: Path, system_name: str, rows, plt, benchmark_mode: str):
    if benchmark_mode == "projector_sweep":
        return _plot_projector_sweep_results(path, system_name, rows, plt)
    return _plot_standard_results(path, system_name, rows, plt)


def main():
    args = parse_args()
    plt, gto, dft = require_optional_dependencies()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for system_name in args.systems:
        rows = run_system(system_name, args, gto, dft)
        suffix = "" if args.benchmark_mode == "standard" else f"_{args.benchmark_mode}"
        csv_path = output_dir / f"benchmark_{system_name}{suffix}.csv"
        write_csv(csv_path, rows)
        print(f"Saved {csv_path}")
        if not args.no_plot:
            png_path = output_dir / f"benchmark_{system_name}{suffix}.png"
            plot_results(png_path, system_name, rows, plt, args.benchmark_mode)
            print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
