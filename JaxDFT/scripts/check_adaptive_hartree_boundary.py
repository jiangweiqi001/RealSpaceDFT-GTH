"""Validation checks for adaptive Hartree boundary-condition upgrades.

This script validates the first monopole Dirichlet boundary upgrade for the
adaptive tensor-grid Hartree prototype. It is intentionally conservative: the
goal is to check correctness and stability trends, not to claim exact
isolated/open-boundary agreement.

Notes:
- This is still not an exact isolated/open-boundary Hartree solver.
- Monopole Dirichlet is only a first upgrade over zero Dirichlet.
- The box-size study uses simple normalized Gaussian model densities for H
  and H2 to isolate boundary-condition behavior without entering SCF.
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.grids.adaptive_tensor import create_adaptive_grid
    from JaxDFT.src.grids.adaptive_poisson import (
        solve_poisson_dirichlet_3d,
        solve_hartree_dirichlet_3d,
        solve_hartree_monopole_dirichlet_3d,
    )
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.grids.adaptive_tensor import create_adaptive_grid
    from src.grids.adaptive_poisson import (
        solve_poisson_dirichlet_3d,
        solve_hartree_dirichlet_3d,
        solve_hartree_monopole_dirichlet_3d,
    )


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def exact_boundary_faces(field):
    return {
        "x_lo": field[0, 1:-1, 1:-1],
        "x_hi": field[-1, 1:-1, 1:-1],
        "y_lo": field[1:-1, 0, 1:-1],
        "y_hi": field[1:-1, -1, 1:-1],
        "z_lo": field[1:-1, 1:-1, 0],
        "z_hi": field[1:-1, 1:-1, -1],
    }


def max_face_mismatch(field_num, field_exact) -> float:
    faces_num = exact_boundary_faces(field_num)
    faces_exact = exact_boundary_faces(field_exact)
    return max(float(jnp.max(jnp.abs(faces_num[key] - faces_exact[key]))) for key in faces_num)


def interior_weighted_rms(grid, err) -> float:
    vol = grid.volume_weights[1:-1, 1:-1, 1:-1]
    err_int = err[1:-1, 1:-1, 1:-1]
    return float(jnp.sqrt(jnp.sum(vol * err_int * err_int) / jnp.sum(vol)))


def interior_max_abs(err) -> float:
    return float(jnp.max(jnp.abs(err[1:-1, 1:-1, 1:-1])))


def hartree_energy_proxy(grid, rho, V) -> float:
    return float(0.5 * grid.integrate(rho * V))


def build_model_density(grid, atom_coords, electrons: float, alpha: float = 1.2):
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for center in atom_coords:
        dr = grid.coords - center
        r2 = jnp.sum(dr * dr, axis=-1)
        rho = rho + jnp.exp(-alpha * r2)
    norm = grid.integrate(rho)
    return rho * (electrons / norm)


def run_linear_manufactured_solution():
    atom_coords = jnp.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float32)
    grid = create_adaptive_grid(jnp.array([8.0, 6.0, 5.0], dtype=jnp.float32), atom_coords, 0.35, 0.70, 0.80, 4.0)
    X = grid.coords[..., 0]
    Y = grid.coords[..., 1]
    Z = grid.coords[..., 2]
    u_exact = 1.0 + 0.17 * X - 0.11 * Y + 0.09 * Z
    rhs = jnp.zeros(grid.shape, dtype=jnp.float32)
    u_num, diag = solve_poisson_dirichlet_3d(grid, rhs, boundary_faces=exact_boundary_faces(u_exact))
    err = u_num - u_exact
    return {
        "grid": grid,
        "u_num": u_num,
        "u_exact": u_exact,
        "diagnostics": diag,
        "interior_rms": interior_weighted_rms(grid, err),
        "interior_max": interior_max_abs(err),
        "face_max": max_face_mismatch(u_num, u_exact),
    }


def run_shifted_sine_manufactured_solution():
    atom_coords = jnp.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float32)
    grid = create_adaptive_grid(jnp.array([8.0, 6.0, 5.0], dtype=jnp.float32), atom_coords, 0.35, 0.70, 0.80, 4.0)
    X = grid.coords[..., 0]
    Y = grid.coords[..., 1]
    Z = grid.coords[..., 2]
    sx = (X - grid.x[0]) / (grid.x[-1] - grid.x[0])
    sy = (Y - grid.y[0]) / (grid.y[-1] - grid.y[0])
    sz = (Z - grid.z[0]) / (grid.z[-1] - grid.z[0])
    sine = jnp.sin(jnp.pi * sx) * jnp.sin(2.0 * jnp.pi * sy) * jnp.sin(jnp.pi * sz)
    u_exact = 1.0 + sine
    lam = (
        (jnp.pi / (grid.x[-1] - grid.x[0])) ** 2
        + (2.0 * jnp.pi / (grid.y[-1] - grid.y[0])) ** 2
        + (jnp.pi / (grid.z[-1] - grid.z[0])) ** 2
    )
    rhs = lam * sine
    u_num, diag = solve_poisson_dirichlet_3d(grid, rhs, boundary_faces=exact_boundary_faces(u_exact))
    err = u_num - u_exact
    return {
        "grid": grid,
        "u_num": u_num,
        "u_exact": u_exact,
        "diagnostics": diag,
        "interior_rms": interior_weighted_rms(grid, err),
        "interior_max": interior_max_abs(err),
        "face_max": max_face_mismatch(u_num, u_exact),
    }


def run_box_series(case_name: str, atom_coords, electrons: float, box_lengths):
    rows = []
    for L in box_lengths:
        box = jnp.array([L, L, L], dtype=jnp.float32)
        grid = create_adaptive_grid(box, atom_coords, 0.45, 0.90, 0.80, 4.0)
        rho = build_model_density(grid, atom_coords, electrons)

        V_zero, diag_zero = solve_hartree_dirichlet_3d(grid, rho)
        V_mono, diag_mono = solve_hartree_monopole_dirichlet_3d(grid, rho)

        rows.append({
            "case": case_name,
            "mode": "zero_dirichlet",
            "L": float(L),
            "shape": tuple(int(n) for n in grid.shape),
            "finite": bool(jnp.all(jnp.isfinite(V_zero))),
            "rel_res": float(diag_zero["relative_residual"]),
            "E_H": hartree_energy_proxy(grid, rho, V_zero),
            "V_max": float(jnp.max(V_zero)),
        })
        rows.append({
            "case": case_name,
            "mode": "monopole_dirichlet",
            "L": float(L),
            "shape": tuple(int(n) for n in grid.shape),
            "finite": bool(jnp.all(jnp.isfinite(V_mono))),
            "rel_res": float(diag_mono["relative_residual"]),
            "E_H": hartree_energy_proxy(grid, rho, V_mono),
            "V_max": float(jnp.max(V_mono)),
        })

    for mode in ("zero_dirichlet", "monopole_dirichlet"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        ref_energy = mode_rows[-1]["E_H"]
        for row in mode_rows:
            row["dE_ref"] = abs(row["E_H"] - ref_energy)

    zero_max = max(row["dE_ref"] for row in rows if row["mode"] == "zero_dirichlet")
    mono_max = max(row["dE_ref"] for row in rows if row["mode"] == "monopole_dirichlet")
    summary = {
        "case": case_name,
        "zero_max_drift": zero_max,
        "monopole_max_drift": mono_max,
        "drift_ratio": mono_max / max(zero_max, 1.0e-30),
        "monopole_not_worse": mono_max <= 1.10 * zero_max + 1.0e-6,
    }
    return rows, summary


def print_box_table(rows):
    print("Case | Mode | L | Finite | rel_res | E_H | dE_ref | V_max | Shape")
    for row in rows:
        print(
            f"{row['case']:>6} | {row['mode']:<19} | {row['L']:>3.1f} | "
            f"{str(row['finite']):<6} | {row['rel_res']:.3e} | {row['E_H']:.6f} | "
            f"{row['dE_ref']:.6f} | {row['V_max']:.6f} | {row['shape']}"
        )


def main() -> int:
    print("Adaptive Hartree Boundary Validation")
    print("Notes:")
    print("- This is still not an exact isolated/open-boundary Hartree solver.")
    print("- Monopole Dirichlet is only a first upgrade over zero Dirichlet.")
    print("- The box study uses simple model densities to isolate boundary behavior.")
    print()

    all_ok = True

    print("=== A. Manufactured Solution: Linear Harmonic ===")
    linear = run_linear_manufactured_solution()
    all_ok &= check(
        "linear_face_match",
        linear["face_max"] <= 1.0e-7,
        f"face_max={linear['face_max']:.6e}",
    )
    all_ok &= check(
        "linear_interior_rms",
        linear["interior_rms"] <= 1.0e-6,
        f"interior_rms={linear['interior_rms']:.6e}",
    )
    all_ok &= check(
        "linear_interior_max",
        linear["interior_max"] <= 2.0e-6,
        f"interior_max={linear['interior_max']:.6e}",
    )
    all_ok &= check(
        "linear_residual",
        linear["diagnostics"]["relative_residual"] <= 1.0e-10,
        f"rel_res={linear['diagnostics']['relative_residual']:.6e}",
    )

    print()
    print("=== A. Manufactured Solution: Shifted Sine ===")
    shifted = run_shifted_sine_manufactured_solution()
    all_ok &= check(
        "shifted_face_match",
        shifted["face_max"] <= 1.0e-7,
        f"face_max={shifted['face_max']:.6e}",
    )
    all_ok &= check(
        "shifted_interior_rms",
        shifted["interior_rms"] <= 1.2e-2,
        f"interior_rms={shifted['interior_rms']:.6e}",
    )
    all_ok &= check(
        "shifted_interior_max",
        shifted["interior_max"] <= 3.0e-2,
        f"interior_max={shifted['interior_max']:.6e}",
    )
    all_ok &= check(
        "shifted_residual",
        shifted["diagnostics"]["relative_residual"] <= 1.0e-10,
        f"rel_res={shifted['diagnostics']['relative_residual']:.6e}",
    )

    print()
    print("=== B. Very Small Box-Size Sensitivity ===")
    h_rows, h_summary = run_box_series(
        "H",
        jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        1.0,
        [4.0, 5.0, 6.0],
    )
    h2_rows, h2_summary = run_box_series(
        "H2",
        jnp.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float32),
        2.0,
        [5.0, 6.0, 7.0],
    )
    rows = h_rows + h2_rows
    print_box_table(rows)

    all_ok &= check(
        "box_all_finite",
        all(row["finite"] for row in rows),
        "all Hartree solves returned finite potentials",
    )
    all_ok &= check(
        "box_residuals",
        all(row["rel_res"] <= 1.0e-10 for row in rows),
        f"max_rel_res={max(row['rel_res'] for row in rows):.6e}",
    )

    print()
    print("Sensitivity Summary")
    print("Case | zero_max_drift | monopole_max_drift | ratio | Verdict")
    for summary in (h_summary, h2_summary):
        verdict = "PASS" if summary["monopole_not_worse"] else "FAIL"
        print(
            f"{summary['case']:>4} | {summary['zero_max_drift']:.6f} | "
            f"{summary['monopole_max_drift']:.6f} | {summary['drift_ratio']:.3f} | {verdict}"
        )
        all_ok &= check(
            f"{summary['case']}_monopole_not_worse",
            summary["monopole_not_worse"],
            (
                f"zero_max_drift={summary['zero_max_drift']:.6f}, "
                f"monopole_max_drift={summary['monopole_max_drift']:.6f}, "
                f"ratio={summary['drift_ratio']:.3f}"
            ),
        )

    print()
    print("=== Summary ===")
    print("Interpretation:")
    print("- Manufactured-solution checks validate nonzero Dirichlet handling directly.")
    print("- The box study is a trend check only; it is not a formal isolated-boundary benchmark.")
    print("- A smaller drift for monopole Dirichlet suggests boundary behavior closer to isolated decay.")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
