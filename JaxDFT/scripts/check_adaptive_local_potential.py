"""Compatibility check for evaluating the existing local potential on an adaptive tensor grid.

This script does not run SCF. It only verifies that the current pointwise
local-potential implementation can consume adaptive-grid coordinates directly.
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
    from JaxDFT.src.hamiltonian import build_local_potential
    from JaxDFT.src.io import load_pseudopotentials
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.grids.adaptive_tensor import create_adaptive_grid
    from src.hamiltonian import build_local_potential
    from src.io import load_pseudopotentials


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def masked_mean(arr, mask):
    return float(jnp.sum(jnp.where(mask, arr, 0.0)) / jnp.sum(mask))


def main() -> int:
    all_ok = True

    atom_coords = jnp.array([
        [-0.7, 0.0, 0.0],
        [0.7, 0.0, 0.0],
    ], dtype=jnp.float32)
    box = jnp.array([20.0, 14.0, 14.0], dtype=jnp.float32)
    grid = create_adaptive_grid(box, atom_coords, 0.18, 0.45, 1.2, 4.0)

    pseudo_dir = os.path.join(project_root, 'JaxDFT', 'data', 'gth_potentials')
    pseudos = load_pseudopotentials(['H', 'H'], pseudo_dir)
    zion = jnp.asarray([p['zion'] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p['rloc'] for p in pseudos], dtype=jnp.float32)
    c = jnp.asarray([p['c'] for p in pseudos], dtype=jnp.float32)

    V_loc = build_local_potential(atom_coords, grid.coords, zion, rloc, c)

    dist = jnp.linalg.norm(grid.coords[None, ...] - atom_coords[:, None, None, None, :], axis=-1)
    dist_to_nearest = jnp.min(dist, axis=0)
    near_mask = dist_to_nearest <= 0.4
    far_mask = dist_to_nearest >= 5.0

    v_min = float(jnp.min(V_loc))
    v_max = float(jnp.max(V_loc))
    near_mean = masked_mean(V_loc, near_mask)
    far_mean = masked_mean(V_loc, far_mask)
    near_min = float(jnp.min(jnp.where(near_mask, V_loc, jnp.inf)))
    far_max = float(jnp.max(jnp.where(far_mask, V_loc, -jnp.inf)))
    corner_value = float(V_loc[0, 0, 0])

    print("=== Adaptive Local Potential Check ===")
    all_ok &= check(
        "shape_matches_grid",
        V_loc.shape == grid.shape,
        f"V_loc.shape={V_loc.shape}, grid.shape={grid.shape}",
    )
    all_ok &= check(
        "finite_values",
        bool(jnp.all(jnp.isfinite(V_loc))),
        f"min={v_min:.6f}, max={v_max:.6f}",
    )
    all_ok &= check(
        "attractive_minimum",
        v_min < -0.5,
        f"v_min={v_min:.6f}",
    )
    all_ok &= check(
        "ordered_range",
        v_min < v_max,
        f"v_min={v_min:.6f}, v_max={v_max:.6f}",
    )
    all_ok &= check(
        "near_region_present",
        bool(jnp.any(near_mask)) and bool(jnp.any(far_mask)),
        f"near_count={int(jnp.sum(near_mask))}, far_count={int(jnp.sum(far_mask))}",
    )
    all_ok &= check(
        "near_deeper_than_far",
        near_mean < far_mean,
        f"near_mean={near_mean:.6f}, far_mean={far_mean:.6f}",
    )
    all_ok &= check(
        "far_region_decay",
        abs(far_mean) < abs(near_mean),
        f"|near_mean|={abs(near_mean):.6f}, |far_mean|={abs(far_mean):.6f}",
    )
    all_ok &= check(
        "near_min_deeper_than_corner",
        near_min < corner_value,
        f"near_min={near_min:.6f}, corner={corner_value:.6f}",
    )
    all_ok &= check(
        "far_max_matches_decay",
        far_max > near_min,
        f"far_max={far_max:.6f}, near_min={near_min:.6f}",
    )

    print("\n=== Summary ===")
    print(f"shape={grid.shape}")
    print(f"backend_name={grid.backend_name}")
    print(f"V_min={v_min:.6f}, V_max={v_max:.6f}")
    print(f"near_mean={near_mean:.6f}, far_mean={far_mean:.6f}")
    print(f"corner_value={corner_value:.6f}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
