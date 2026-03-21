"""Validation script for the adaptive weighted-overlap nonlocal projector path."""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends import AdaptiveBackend, UniformBackend
    from JaxDFT.src.io import load_pseudopotentials
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend, UniformBackend
    from src.io import load_pseudopotentials


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def weighted_rel_rms(grid, err, ref):
    num = float(jnp.sqrt(grid.integrate(err * err)))
    den = float(jnp.sqrt(grid.integrate(ref * ref)))
    return num / max(den, 1.0e-30)


def main() -> int:
    all_ok = True

    adaptive_backend = AdaptiveBackend()
    uniform_backend = UniformBackend()
    atom_coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["O"], pseudo_dir)

    print("=== Adaptive Nonlocal Weighted-Overlap Check ===")
    box = jnp.array([12.0, 10.0, 10.0], dtype=jnp.float32)
    grid = adaptive_backend.create_grid(
        0.24,
        box,
        atom_coords=atom_coords,
        h_max=0.55,
        r_core=1.0,
        stretch_beta=4.0,
    )
    x = grid.coords[..., 0]
    y = grid.coords[..., 1]
    z = grid.coords[..., 2]
    r2 = x * x + y * y + z * z
    psi = jnp.exp(-0.14 * r2) * (1.0 + 0.10 * x - 0.05 * y)
    phi = jnp.exp(-0.11 * ((x - 0.15) ** 2 + 1.1 * (y + 0.10) ** 2 + 0.9 * (z - 0.05) ** 2))

    cache = adaptive_backend.precompute_nonlocal(grid, atom_coords, pseudos)
    all_ok &= check("projector_cache_present", cache is not None, f"cache_is_none={cache is None}")
    if cache is None:
        print("OVERALL: FAIL")
        return 1

    p_i, p_j, coeffs = cache
    all_ok &= check("projector_cache_shape", p_i.shape[1:] == grid.shape and p_j.shape[1:] == grid.shape, f"P_i.shape={p_i.shape}, P_j.shape={p_j.shape}")
    all_ok &= check("projector_channel_count", p_i.shape[0] == coeffs.shape[0] and p_i.shape[0] > 0, f"n_channels={p_i.shape[0]}")

    vpsi = adaptive_backend.apply_nonlocal(grid, psi, cache)
    vphi = adaptive_backend.apply_nonlocal(grid, phi, cache)
    all_ok &= check("apply_nonlocal_shape", vpsi.shape == grid.shape, f"V_nl.shape={vpsi.shape}, grid.shape={grid.shape}")
    all_ok &= check("apply_nonlocal_finite", bool(jnp.all(jnp.isfinite(vpsi))), f"min={float(jnp.min(vpsi)):.6f}, max={float(jnp.max(vpsi)):.6f}")

    lhs = float(adaptive_backend.inner_product(grid, phi, vpsi))
    rhs = float(adaptive_backend.inner_product(grid, vphi, psi))
    herm_rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)
    all_ok &= check("weighted_hermitian", herm_rel <= 5.0e-5, f"lhs={lhs:.6e}, rhs={rhs:.6e}, rel_diff={herm_rel:.6e}")

    print("\n=== Uniform Degeneracy Check ===")
    h = 0.35
    deg_box = jnp.array([12.0, 12.0, 12.0], dtype=jnp.float32)
    adaptive_uniform_grid = adaptive_backend.create_grid(
        h,
        deg_box,
        atom_coords=atom_coords,
        h_max=h,
        r_core=1.0,
        stretch_beta=4.0,
    )
    uniform_grid = uniform_backend.create_grid(h, deg_box)
    coord_diff = float(jnp.max(jnp.abs(adaptive_uniform_grid.coords - uniform_grid.coords)))
    all_ok &= check("uniform_coords_match", coord_diff <= 1.0e-7, f"max_coord_diff={coord_diff:.6e}")

    sx = (adaptive_uniform_grid.coords[..., 0] - adaptive_uniform_grid.x[0]) / (adaptive_uniform_grid.x[-1] - adaptive_uniform_grid.x[0])
    sy = (adaptive_uniform_grid.coords[..., 1] - adaptive_uniform_grid.y[0]) / (adaptive_uniform_grid.y[-1] - adaptive_uniform_grid.y[0])
    sz = (adaptive_uniform_grid.coords[..., 2] - adaptive_uniform_grid.z[0]) / (adaptive_uniform_grid.z[-1] - adaptive_uniform_grid.z[0])
    psi_deg = (
        jnp.exp(-0.12 * jnp.sum(adaptive_uniform_grid.coords ** 2, axis=-1))
        * jnp.sin(jnp.pi * sx)
        * jnp.sin(jnp.pi * sy)
        * jnp.sin(jnp.pi * sz)
    )

    cache_ad = adaptive_backend.precompute_nonlocal(adaptive_uniform_grid, atom_coords, pseudos)
    cache_un = uniform_backend.precompute_nonlocal(uniform_grid, atom_coords, pseudos)
    v_ad = adaptive_backend.apply_nonlocal(adaptive_uniform_grid, psi_deg, cache_ad)
    v_un = uniform_backend.apply_nonlocal(uniform_grid, psi_deg, cache_un)
    all_ok &= check("uniform_deg_shape", v_ad.shape == v_un.shape, f"adaptive.shape={v_ad.shape}, uniform.shape={v_un.shape}")
    all_ok &= check("uniform_deg_finite", bool(jnp.all(jnp.isfinite(v_ad)) and jnp.all(jnp.isfinite(v_un))), f"adaptive_max={float(jnp.max(jnp.abs(v_ad))):.6f}, uniform_max={float(jnp.max(jnp.abs(v_un))):.6f}")
    rel_rms = weighted_rel_rms(adaptive_uniform_grid, v_ad - v_un, v_un)
    max_abs = float(jnp.max(jnp.abs(v_ad - v_un)))
    all_ok &= check("uniform_deg_rel_rms", rel_rms <= 5.0e-3, f"rel_rms={rel_rms:.6e}")
    all_ok &= check("uniform_deg_max_abs", max_abs <= 2.0e-4, f"max_abs={max_abs:.6e}")

    print("\n=== Summary ===")
    print(f"adaptive_shape={grid.shape}")
    print(f"n_channels={p_i.shape[0]}")
    print(f"weighted_hermitian_rel_diff={herm_rel:.6e}")
    print(f"uniform_deg_rel_rms={rel_rms:.6e}")
    print(f"uniform_deg_max_abs={max_abs:.6e}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
