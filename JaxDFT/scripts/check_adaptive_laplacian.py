"""Validation script for the prototype adaptive tensor-grid Laplacian.

This script stays below the SCF layer. It checks the 1D variable-spacing second
Derivative and the 3D tensor-product Laplacian against analytic test functions.
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
    from JaxDFT.src.grids.adaptive_tensor import (
        create_adaptive_axis,
        create_adaptive_grid,
        laplacian_nonuniform_3d,
        second_derivative_nonuniform_1d,
    )
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.grids.adaptive_tensor import (
        create_adaptive_axis,
        create_adaptive_grid,
        laplacian_nonuniform_3d,
        second_derivative_nonuniform_1d,
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
    box = jnp.array([20.0, 14.0, 14.0], dtype=jnp.float32)
    h_min = 0.18
    h_max = 0.45
    r_core = 1.2
    stretch_beta = 4.0

    print("=== 1D Second-Derivative Check ===")
    x, wx, meta_x = create_adaptive_axis(float(box[0]), atom_coords[:, 0], h_min, h_max, r_core, stretch_beta)
    alpha_1d = 0.12
    f_1d = jnp.exp(-alpha_1d * x ** 2)
    f2_exact = (4.0 * alpha_1d * alpha_1d * x ** 2 - 2.0 * alpha_1d) * f_1d
    f2_num = second_derivative_nonuniform_1d(x, f_1d)
    err_1d = f2_num - f2_exact
    interior_err_1d = err_1d[1:-1]
    interior_w_1d = wx[1:-1]
    rms_1d = weighted_rms_1d(wx, err_1d)
    rms_1d_interior = weighted_rms_1d(interior_w_1d, interior_err_1d)
    max_1d = float(jnp.max(jnp.abs(err_1d)))
    max_1d_interior = float(jnp.max(jnp.abs(interior_err_1d)))

    all_ok &= check(
        "1d_finite",
        bool(jnp.all(jnp.isfinite(f2_num))),
        f"shape={f2_num.shape}, min={float(jnp.min(f2_num)):.6f}, max={float(jnp.max(f2_num)):.6f}",
    )
    all_ok &= check(
        "1d_rms_error",
        rms_1d <= 1.5e-2,
        f"rms={rms_1d:.6e}, interior_rms={rms_1d_interior:.6e}",
    )
    all_ok &= check(
        "1d_max_interior_error",
        max_1d_interior <= 5.0e-2,
        f"max_interior={max_1d_interior:.6e}, max_full={max_1d:.6e}",
    )

    print("\n=== 3D Laplacian Check ===")
    grid = create_adaptive_grid(box, atom_coords, h_min, h_max, r_core, stretch_beta)
    alpha_3d = 0.10
    r2 = jnp.sum(grid.coords ** 2, axis=-1)
    f_3d = jnp.exp(-alpha_3d * r2)
    lap_exact = (4.0 * alpha_3d * alpha_3d * r2 - 6.0 * alpha_3d) * f_3d
    lap_num = laplacian_nonuniform_3d(grid, f_3d)
    err_3d = lap_num - lap_exact
    interior_err_3d = err_3d[1:-1, 1:-1, 1:-1]
    interior_volume = float(jnp.sum(grid.volume_weights[1:-1, 1:-1, 1:-1]))
    rms_3d = weighted_rms_3d(grid, err_3d)
    rms_3d_interior = float(jnp.sqrt(jnp.sum(grid.volume_weights[1:-1, 1:-1, 1:-1] * interior_err_3d * interior_err_3d) / interior_volume))
    max_3d = float(jnp.max(jnp.abs(err_3d)))
    max_3d_interior = float(jnp.max(jnp.abs(interior_err_3d)))

    all_ok &= check(
        "3d_shape",
        lap_num.shape == grid.shape,
        f"lap_num.shape={lap_num.shape}, grid.shape={grid.shape}",
    )
    all_ok &= check(
        "3d_finite",
        bool(jnp.all(jnp.isfinite(lap_num))),
        f"min={float(jnp.min(lap_num)):.6f}, max={float(jnp.max(lap_num)):.6f}",
    )
    all_ok &= check(
        "3d_rms_error",
        rms_3d <= 2.0e-2,
        f"rms={rms_3d:.6e}, interior_rms={rms_3d_interior:.6e}",
    )
    all_ok &= check(
        "3d_max_interior_error",
        max_3d_interior <= 8.0e-2,
        f"max_interior={max_3d_interior:.6e}, max_full={max_3d:.6e}",
    )

    print("\n=== Summary ===")
    print(f"1D nodes={x.size}, min_dx={float(jnp.min(meta_x['actual_min_spacing'])) if isinstance(meta_x['actual_min_spacing'], jnp.ndarray) else meta_x['actual_min_spacing']:.6f}, max_dx={float(meta_x['actual_max_spacing']):.6f}")
    print(f"1D rms={rms_1d:.6e}, 1D max_interior={max_1d_interior:.6e}")
    print(f"3D shape={grid.shape}")
    print(f"3D rms={rms_3d:.6e}, 3D max_interior={max_3d_interior:.6e}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
