"""Minimal capability check for the prototype AdaptiveBackend.

This script stays below the SCF layer. It verifies that the backend can create
an adaptive grid and expose the currently implemented pointwise and weighted
operations.
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
    from JaxDFT.src.backends import AdaptiveBackend
    from JaxDFT.src.io import load_pseudopotentials
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.io import load_pseudopotentials


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def expect_not_implemented(fn, name: str) -> bool:
    try:
        fn()
    except NotImplementedError as exc:
        print(f"[PASS] {name}: {exc}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: unexpected {type(exc).__name__}: {exc}")
        return False
    print(f"[FAIL] {name}: expected NotImplementedError")
    return False


def main() -> int:
    all_ok = True

    backend = AdaptiveBackend()
    coords = jnp.array([
        [-0.7, 0.0, 0.0],
        [0.7, 0.0, 0.0],
    ], dtype=jnp.float32)
    box = jnp.array([20.0, 14.0, 14.0], dtype=jnp.float32)

    grid = backend.create_grid(
        0.18,
        box,
        atom_coords=coords,
        h_max=0.45,
        r_core=1.2,
        stretch_beta=4.0,
    )

    ones = jnp.ones(grid.shape, dtype=jnp.float32)
    r2 = jnp.sum(grid.coords ** 2, axis=-1)
    alpha = 0.10
    psi = jnp.exp(-alpha * r2)

    pseudo_dir = os.path.join(project_root, 'JaxDFT', 'data', 'gth_potentials')
    pseudos = load_pseudopotentials(['H', 'H'], pseudo_dir)

    volume = float(jnp.prod(grid.box_size))
    integ = float(backend.integrate(grid, ones))
    inner = float(backend.inner_product(grid, ones, ones))
    v_loc = backend.build_local_potential(grid, coords, pseudos)
    kinetic = backend.apply_kinetic(grid, psi)
    kinetic_exact = -0.5 * (4.0 * alpha * alpha * r2 - 6.0 * alpha) * psi
    kinetic_err = kinetic - kinetic_exact
    kinetic_rms = float(jnp.sqrt(grid.integrate(kinetic_err * kinetic_err) / volume))

    print("=== AdaptiveBackend Check ===")
    all_ok &= check("backend_name", backend.name == 'adaptive_tensor', f"backend.name={backend.name}")
    all_ok &= check("grid_backend_name", getattr(grid, 'backend_name', None) == 'adaptive_tensor', f"grid.backend_name={getattr(grid, 'backend_name', None)}")
    all_ok &= check("grid_shape", grid.coords.shape[:-1] == grid.shape, f"coords.shape={grid.coords.shape}, shape={grid.shape}")
    all_ok &= check("integrate_constant", abs(integ - volume) / volume <= 1e-6, f"integrate(1)={integ:.6f}, volume={volume:.6f}")
    all_ok &= check("inner_product_constant", abs(inner - volume) / volume <= 1e-6, f"<1,1>={inner:.6f}, volume={volume:.6f}")
    all_ok &= check("local_potential_shape", v_loc.shape == grid.shape, f"V_loc.shape={v_loc.shape}, grid.shape={grid.shape}")
    all_ok &= check("local_potential_finite", bool(jnp.all(jnp.isfinite(v_loc))), f"min={float(jnp.min(v_loc)):.6f}, max={float(jnp.max(v_loc)):.6f}")
    all_ok &= check("kinetic_shape", kinetic.shape == grid.shape, f"kinetic.shape={kinetic.shape}, grid.shape={grid.shape}")
    all_ok &= check("kinetic_finite", bool(jnp.all(jnp.isfinite(kinetic))), f"min={float(jnp.min(kinetic)):.6f}, max={float(jnp.max(kinetic)):.6f}")
    all_ok &= check("kinetic_rms", kinetic_rms <= 1.0e-2, f"rms={kinetic_rms:.6e}")
    all_ok &= expect_not_implemented(lambda: backend.solve_hartree(grid, ones), 'hartree_not_implemented')
    all_ok &= expect_not_implemented(lambda: backend.precompute_nonlocal(grid, coords, pseudos), 'nonlocal_precompute_not_implemented')
    all_ok &= expect_not_implemented(lambda: backend.apply_nonlocal(grid, psi, None), 'nonlocal_apply_not_implemented')

    print("\n=== Summary ===")
    print(f"shape={grid.shape}")
    print(f"integrate(1)={integ:.6f}")
    print(f"<1,1>={inner:.6f}")
    print(f"V_loc[min,max]=({float(jnp.min(v_loc)):.6f}, {float(jnp.max(v_loc)):.6f})")
    print(f"kinetic_rms={kinetic_rms:.6e}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
