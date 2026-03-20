"""Basic geometry checks for the adaptive tensor-axis generator.

This script performs only geometric and integration-weight checks. It does not
run SCF or any electronic-structure calculation.
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
    from JaxDFT.src.grids import build_volume_weights, create_adaptive_axis
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.grids import build_volume_weights, create_adaptive_axis


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def main() -> int:
    all_ok = True

    coords = jnp.array([
        [-0.7, 0.0, 0.0],
        [0.7, 0.0, 0.0],
    ], dtype=jnp.float32)
    box = jnp.array([20.0, 14.0, 14.0], dtype=jnp.float32)

    uniform_h = 0.18
    h_min = 0.18
    h_max = 0.45
    r_core = 1.2
    stretch_beta = 4.0

    print("=== Uniform Degeneracy Check ===")
    x_uni, wx_uni, meta_uni = create_adaptive_axis(
        float(box[0]),
        coords[:, 0],
        uniform_h,
        uniform_h,
        r_core,
        stretch_beta,
    )
    expected_n_intervals = max(1, int(round(float(box[0]) / uniform_h)))
    expected_axis = jnp.linspace(-0.5 * box[0], 0.5 * box[0], expected_n_intervals + 1, dtype=jnp.float32)
    expected_weights = jnp.zeros_like(expected_axis)
    expected_dx = expected_axis[1:] - expected_axis[:-1]
    expected_weights = expected_weights.at[0].set(0.5 * expected_dx[0])
    expected_weights = expected_weights.at[-1].set(0.5 * expected_dx[-1])
    expected_weights = expected_weights.at[1:-1].set(0.5 * (expected_axis[2:] - expected_axis[:-2]))

    all_ok &= check(
        "uniform_degenerate_flag",
        bool(meta_uni["is_uniform_degenerate"]),
        f"n_nodes={meta_uni['n_nodes']}, n_intervals={meta_uni['n_intervals']}",
    )
    all_ok &= check(
        "uniform_axis_match",
        bool(jnp.allclose(x_uni, expected_axis, atol=1e-7, rtol=0.0)),
        f"max|dx|={float(jnp.max(jnp.abs(x_uni - expected_axis))):.3e}",
    )
    all_ok &= check(
        "uniform_weights_match",
        bool(jnp.allclose(wx_uni, expected_weights, atol=1e-7, rtol=0.0)),
        f"max|dw|={float(jnp.max(jnp.abs(wx_uni - expected_weights))):.3e}",
    )

    print("\n=== Adaptive Axis Check ===")
    x, wx, meta_x = create_adaptive_axis(float(box[0]), coords[:, 0], h_min, h_max, r_core, stretch_beta)
    y, wy, meta_y = create_adaptive_axis(float(box[1]), coords[:, 1], h_min, h_max, r_core, stretch_beta)
    z, wz, meta_z = create_adaptive_axis(float(box[2]), coords[:, 2], h_min, h_max, r_core, stretch_beta)
    volume_weights = build_volume_weights(wx, wy, wz)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    box_volume = float(jnp.prod(box))
    volume_sum = float(jnp.sum(volume_weights))
    rel_volume_error = abs(volume_sum - box_volume) / box_volume

    all_ok &= check(
        "x_monotone",
        bool(jnp.all(dx > 0.0)),
        f"min_dx={float(jnp.min(dx)):.6f}, max_dx={float(jnp.max(dx)):.6f}",
    )
    all_ok &= check(
        "y_monotone",
        bool(jnp.all(dy > 0.0)),
        f"min_dy={float(jnp.min(dy)):.6f}, max_dy={float(jnp.max(dy)):.6f}",
    )
    all_ok &= check(
        "z_monotone",
        bool(jnp.all(dz > 0.0)),
        f"min_dz={float(jnp.min(dz)):.6f}, max_dz={float(jnp.max(dz)):.6f}",
    )
    all_ok &= check(
        "weights_positive",
        bool(jnp.all(wx > 0.0) and jnp.all(wy > 0.0) and jnp.all(wz > 0.0)),
        f"min(wx,wy,wz)=({float(jnp.min(wx)):.6f}, {float(jnp.min(wy)):.6f}, {float(jnp.min(wz)):.6f})",
    )
    all_ok &= check(
        "axis_weight_sums",
        bool(
            jnp.isclose(jnp.sum(wx), box[0], atol=1e-5, rtol=0.0)
            and jnp.isclose(jnp.sum(wy), box[1], atol=1e-5, rtol=0.0)
            and jnp.isclose(jnp.sum(wz), box[2], atol=1e-5, rtol=0.0)
        ),
        (
            f"sum(wx,wy,wz)=({float(jnp.sum(wx)):.6f}, {float(jnp.sum(wy)):.6f}, {float(jnp.sum(wz)):.6f})"
        ),
    )
    all_ok &= check(
        "volume_consistency",
        rel_volume_error <= 1e-6,
        f"sum(volume_weights)={volume_sum:.6f}, box_volume={box_volume:.6f}, rel_err={rel_volume_error:.3e}",
    )
    all_ok &= check(
        "adaptive_not_uniform",
        bool((jnp.min(dx) < jnp.max(dx)) and (jnp.min(dy) < jnp.max(dy)) and (jnp.min(dz) < jnp.max(dz))),
        (
            f"x[min,max]=({meta_x['actual_min_spacing']:.6f}, {meta_x['actual_max_spacing']:.6f}), "
            f"y[min,max]=({meta_y['actual_min_spacing']:.6f}, {meta_y['actual_max_spacing']:.6f}), "
            f"z[min,max]=({meta_z['actual_min_spacing']:.6f}, {meta_z['actual_max_spacing']:.6f})"
        ),
    )

    print("\n=== Summary ===")
    print(f"shape=({x.size}, {y.size}, {z.size})")
    print(f"sum(wx)={float(jnp.sum(wx)):.6f}, sum(wy)={float(jnp.sum(wy)):.6f}, sum(wz)={float(jnp.sum(wz)):.6f}")
    print(f"sum(volume_weights)={volume_sum:.6f}")
    print(f"expected_box_volume={box_volume:.6f}")
    print(f"relative_volume_error={rel_volume_error:.3e}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
