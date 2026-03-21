"""Manufactured-solution validation for the adaptive tensor-grid Poisson prototype."""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.grids.adaptive_tensor import create_adaptive_axis, create_adaptive_grid
    from JaxDFT.src.grids.adaptive_poisson import (
        solve_poisson_dirichlet_1d,
        solve_poisson_dirichlet_3d,
    )
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.grids.adaptive_tensor import create_adaptive_axis, create_adaptive_grid
    from src.grids.adaptive_poisson import (
        solve_poisson_dirichlet_1d,
        solve_poisson_dirichlet_3d,
    )


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def weighted_rms_1d(weights, err):
    return float(jnp.sqrt(jnp.sum(weights * err * err) / jnp.sum(weights)))


def weighted_rms_3d(grid, err):
    volume = float(jnp.prod(grid.box_size))
    return float(jnp.sqrt(grid.integrate(err * err) / volume))


def main() -> int:
    all_ok = True

    atom_coords = jnp.array([
        [-0.7, 0.0, 0.0],
        [0.7, 0.0, 0.0],
    ], dtype=jnp.float32)

    print("=== 1D Poisson Manufactured Solution ===")
    Lx = 8.0
    x, wx, _ = create_adaptive_axis(Lx, atom_coords[:, 0], 0.35, 0.70, 0.80, 4.0)
    sx = (x - x[0]) / (x[-1] - x[0])
    u_exact_1d = jnp.sin(jnp.pi * sx)
    lam_1d = (jnp.pi / (x[-1] - x[0])) ** 2
    rhs_1d = lam_1d * u_exact_1d
    u_num_1d, diag_1d = solve_poisson_dirichlet_1d(x, wx, rhs_1d)
    err_1d = u_num_1d - u_exact_1d
    rms_1d = weighted_rms_1d(wx, err_1d)
    max_int_1d = float(jnp.max(jnp.abs(err_1d[1:-1])))

    all_ok &= check(
        "1d_finite",
        bool(jnp.all(jnp.isfinite(u_num_1d))),
        f"min={float(jnp.min(u_num_1d)):.6f}, max={float(jnp.max(u_num_1d)):.6f}",
    )
    all_ok &= check(
        "1d_weighted_rms",
        rms_1d <= 6.0e-3,
        f"rms={rms_1d:.6e}",
    )
    all_ok &= check(
        "1d_interior_max",
        max_int_1d <= 7.0e-3,
        f"max_interior={max_int_1d:.6e}",
    )
    all_ok &= check(
        "1d_residual",
        diag_1d["relative_residual"] <= 1.0e-10,
        f"rel_res={diag_1d['relative_residual']:.6e}, abs_res={diag_1d['residual_norm']:.6e}",
    )

    print("\n=== 3D Poisson Manufactured Solution ===")
    box = jnp.array([8.0, 6.0, 5.0], dtype=jnp.float32)
    grid = create_adaptive_grid(box, atom_coords, 0.35, 0.70, 0.80, 4.0)
    sx = (grid.coords[..., 0] - grid.x[0]) / (grid.x[-1] - grid.x[0])
    sy = (grid.coords[..., 1] - grid.y[0]) / (grid.y[-1] - grid.y[0])
    sz = (grid.coords[..., 2] - grid.z[0]) / (grid.z[-1] - grid.z[0])

    nx_mode = 1.0
    ny_mode = 2.0
    nz_mode = 1.0
    u_exact_3d = (
        jnp.sin(nx_mode * jnp.pi * sx)
        * jnp.sin(ny_mode * jnp.pi * sy)
        * jnp.sin(nz_mode * jnp.pi * sz)
    )
    lam_3d = (
        (nx_mode * jnp.pi / (grid.x[-1] - grid.x[0])) ** 2
        + (ny_mode * jnp.pi / (grid.y[-1] - grid.y[0])) ** 2
        + (nz_mode * jnp.pi / (grid.z[-1] - grid.z[0])) ** 2
    )
    rhs_3d = lam_3d * u_exact_3d
    u_num_3d, diag_3d = solve_poisson_dirichlet_3d(grid, rhs_3d)
    err_3d = u_num_3d - u_exact_3d
    rms_3d = weighted_rms_3d(grid, err_3d)
    max_int_3d = float(jnp.max(jnp.abs(err_3d[1:-1, 1:-1, 1:-1])))

    all_ok &= check(
        "3d_finite",
        bool(jnp.all(jnp.isfinite(u_num_3d))),
        f"min={float(jnp.min(u_num_3d)):.6f}, max={float(jnp.max(u_num_3d)):.6f}",
    )
    all_ok &= check(
        "3d_weighted_rms",
        rms_3d <= 1.2e-2,
        f"rms={rms_3d:.6e}",
    )
    all_ok &= check(
        "3d_interior_max",
        max_int_3d <= 3.0e-2,
        f"max_interior={max_int_3d:.6e}",
    )
    all_ok &= check(
        "3d_residual",
        diag_3d["relative_residual"] <= 5.0e-6,
        f"rel_res={diag_3d['relative_residual']:.6e}, abs_res={diag_3d['residual_norm']:.6e}",
    )

    print("\n=== Summary ===")
    print(f"1D rms={rms_1d:.6e}, 1D max_interior={max_int_1d:.6e}, 1D rel_res={diag_1d['relative_residual']:.6e}")
    print(f"3D shape={grid.shape}, unknowns={diag_3d['n_unknowns']}")
    print(f"3D rms={rms_3d:.6e}, 3D max_interior={max_int_3d:.6e}, 3D rel_res={diag_3d['relative_residual']:.6e}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

