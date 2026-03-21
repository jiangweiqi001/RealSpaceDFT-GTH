"""Minimal capability check for the prototype AdaptiveBackend.

This script stays below the SCF layer. It verifies that the backend can create
an adaptive grid and expose the currently implemented pointwise, weighted,
Hartree, and nonlocal operations. The current Hartree default is a multipole
Dirichlet box solve: more isolated-like than the older monopole and zero-
Dirichlet prototypes, but still not an exact isolated/open-boundary treatment.
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


def main() -> int:
    all_ok = True

    backend = AdaptiveBackend()
    monopole_charge_backend = AdaptiveBackend(
        hartree_boundary_mode="monopole_dirichlet",
        hartree_center_mode="charge_center",
    )
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

    hartree_box = jnp.array([8.0, 6.0, 5.0], dtype=jnp.float32)
    hartree_grid = backend.create_grid(
        0.35,
        hartree_box,
        atom_coords=coords,
        h_max=0.70,
        r_core=0.80,
        stretch_beta=4.0,
    )
    hartree_r2 = jnp.sum(hartree_grid.coords ** 2, axis=-1)
    rho = jnp.exp(-0.25 * hartree_r2)
    shifted_rho = jnp.exp(
        -0.25 * (
            (hartree_grid.coords[..., 0] - 0.8) ** 2
            + hartree_grid.coords[..., 1] ** 2
            + hartree_grid.coords[..., 2] ** 2
        )
    )

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    nonlocal_atom_coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    nonlocal_pseudos = load_pseudopotentials(["O"], pseudo_dir)

    volume = float(jnp.prod(grid.box_size))
    integ = float(backend.integrate(grid, ones))
    inner = float(backend.inner_product(grid, ones, ones))
    v_loc = backend.build_local_potential(grid, coords, pseudos)
    kinetic = backend.apply_kinetic(grid, psi)
    v_h = backend.solve_hartree(hartree_grid, rho)
    v_h_mono_charge = monopole_charge_backend.solve_hartree(hartree_grid, shifted_rho)
    nl_cache = backend.precompute_nonlocal(grid, nonlocal_atom_coords, nonlocal_pseudos)
    v_nl = backend.apply_nonlocal(grid, psi, nl_cache)
    kinetic_exact = -0.5 * (4.0 * alpha * alpha * r2 - 6.0 * alpha) * psi
    kinetic_err = kinetic - kinetic_exact
    kinetic_rms = float(jnp.sqrt(grid.integrate(kinetic_err * kinetic_err) / volume))

    print("=== AdaptiveBackend Check ===")
    all_ok &= check("backend_name", backend.name == "adaptive_tensor", f"backend.name={backend.name}")
    all_ok &= check("grid_backend_name", getattr(grid, "backend_name", None) == "adaptive_tensor", f"grid.backend_name={getattr(grid, 'backend_name', None)}")
    all_ok &= check("grid_shape", grid.coords.shape[:-1] == grid.shape, f"coords.shape={grid.coords.shape}, shape={grid.shape}")
    all_ok &= check("integrate_constant", abs(integ - volume) / volume <= 1e-6, f"integrate(1)={integ:.6f}, volume={volume:.6f}")
    all_ok &= check("inner_product_constant", abs(inner - volume) / volume <= 1e-6, f"<1,1>={inner:.6f}, volume={volume:.6f}")
    all_ok &= check("local_potential_shape", v_loc.shape == grid.shape, f"V_loc.shape={v_loc.shape}, grid.shape={grid.shape}")
    all_ok &= check("local_potential_finite", bool(jnp.all(jnp.isfinite(v_loc))), f"min={float(jnp.min(v_loc)):.6f}, max={float(jnp.max(v_loc)):.6f}")
    all_ok &= check("kinetic_shape", kinetic.shape == grid.shape, f"kinetic.shape={kinetic.shape}, grid.shape={grid.shape}")
    all_ok &= check("kinetic_finite", bool(jnp.all(jnp.isfinite(kinetic))), f"min={float(jnp.min(kinetic)):.6f}, max={float(jnp.max(kinetic)):.6f}")
    all_ok &= check("kinetic_rms", kinetic_rms <= 1.0e-2, f"rms={kinetic_rms:.6e}")
    all_ok &= check("hartree_mode", getattr(backend, "hartree_boundary_mode", None) == "multipole_dirichlet", f"hartree_boundary_mode={getattr(backend, 'hartree_boundary_mode', None)}")
    all_ok &= check("hartree_center_mode", getattr(backend, "hartree_center_mode", None) == "box_center", f"hartree_center_mode={getattr(backend, 'hartree_center_mode', None)}")
    all_ok &= check("monopole_charge_center_mode", getattr(monopole_charge_backend, "hartree_center_mode", None) == "charge_center", f"hartree_center_mode={getattr(monopole_charge_backend, 'hartree_center_mode', None)}")
    all_ok &= check("hartree_shape", v_h.shape == hartree_grid.shape, f"V_H.shape={v_h.shape}, hartree_grid.shape={hartree_grid.shape}")
    all_ok &= check("hartree_finite", bool(jnp.all(jnp.isfinite(v_h))), f"min={float(jnp.min(v_h)):.6f}, max={float(jnp.max(v_h)):.6f}")
    all_ok &= check(
        "hartree_boundary_finite",
        bool(
            jnp.all(jnp.isfinite(v_h[0, 1:-1, 1:-1]))
            and jnp.all(jnp.isfinite(v_h[-1, 1:-1, 1:-1]))
            and jnp.all(jnp.isfinite(v_h[1:-1, 0, 1:-1]))
            and jnp.all(jnp.isfinite(v_h[1:-1, -1, 1:-1]))
            and jnp.all(jnp.isfinite(v_h[1:-1, 1:-1, 0]))
            and jnp.all(jnp.isfinite(v_h[1:-1, 1:-1, -1]))
        ),
        "monopole Dirichlet interior-aligned faces are finite",
    )
    all_ok &= check(
        "hartree_boundary_positive",
        bool(
            float(jnp.min(v_h[0, 1:-1, 1:-1])) > 0.0
            and float(jnp.min(v_h[-1, 1:-1, 1:-1])) > 0.0
            and float(jnp.min(v_h[1:-1, 0, 1:-1])) > 0.0
            and float(jnp.min(v_h[1:-1, -1, 1:-1])) > 0.0
            and float(jnp.min(v_h[1:-1, 1:-1, 0])) > 0.0
            and float(jnp.min(v_h[1:-1, 1:-1, -1])) > 0.0
        ),
        "monopole Dirichlet interior-aligned faces are positive for positive rho",
    )
    all_ok &= check("hartree_positive_peak", float(jnp.max(v_h)) > 0.0, f"max={float(jnp.max(v_h)):.6f}")
    all_ok &= check("monopole_charge_hartree_shape", v_h_mono_charge.shape == hartree_grid.shape, f"V_H_charge.shape={v_h_mono_charge.shape}, hartree_grid.shape={hartree_grid.shape}")
    all_ok &= check("monopole_charge_hartree_finite", bool(jnp.all(jnp.isfinite(v_h_mono_charge))), f"min={float(jnp.min(v_h_mono_charge)):.6f}, max={float(jnp.max(v_h_mono_charge)):.6f}")
    all_ok &= check("nonlocal_cache_present", nl_cache is not None, f"cache_is_none={nl_cache is None}")
    if nl_cache is not None:
        p_i, p_j, coeffs = nl_cache
        all_ok &= check("nonlocal_cache_shape", p_i.shape[1:] == grid.shape and p_j.shape[1:] == grid.shape, f"P_i.shape={p_i.shape}, P_j.shape={p_j.shape}")
        all_ok &= check("nonlocal_channels", p_i.shape[0] == coeffs.shape[0] and p_i.shape[0] > 0, f"n_channels={p_i.shape[0]}")
    all_ok &= check("nonlocal_shape", v_nl.shape == grid.shape, f"V_nl.shape={v_nl.shape}, grid.shape={grid.shape}")
    all_ok &= check("nonlocal_finite", bool(jnp.all(jnp.isfinite(v_nl))), f"min={float(jnp.min(v_nl)):.6f}, max={float(jnp.max(v_nl)):.6f}")

    print("\n=== Summary ===")
    print(f"shape={grid.shape}")
    print(f"hartree_shape={hartree_grid.shape}")
    print(f"integrate(1)={integ:.6f}")
    print(f"<1,1>={inner:.6f}")
    print(f"V_loc[min,max]=({float(jnp.min(v_loc)):.6f}, {float(jnp.max(v_loc)):.6f})")
    print(f"V_H[min,max]=({float(jnp.min(v_h)):.6f}, {float(jnp.max(v_h)):.6f})")
    print(f"V_H_charge[min,max]=({float(jnp.min(v_h_mono_charge)):.6f}, {float(jnp.max(v_h_mono_charge)):.6f})")
    print(f"V_nl[min,max]=({float(jnp.min(v_nl)):.6f}, {float(jnp.max(v_nl)):.6f})")
    print(f"kinetic_rms={kinetic_rms:.6e}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
