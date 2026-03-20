"""Minimal adaptive SCF smoke test.

This is intentionally a smoke test, not a benchmark. The current adaptive
Hartree path still uses a zero-Dirichlet box Poisson prototype, so the output
must not be interpreted as a uniform-vs-adaptive physical comparison.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends import AdaptiveBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def main() -> int:
    all_ok = True
    backend = AdaptiveBackend()
    key = jax.random.PRNGKey(0)

    coords = jnp.array([
        [0.0, 0.0, 0.0],
    ], dtype=jnp.float32)
    box = jnp.array([4.0, 4.0, 4.0], dtype=jnp.float32)
    grid = backend.create_grid(
        0.80,
        box,
        atom_coords=coords,
        h_max=1.00,
        r_core=0.50,
        stretch_beta=4.0,
    )

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H"], pseudo_dir)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)

    n_electrons = float(jnp.sum(jnp.asarray([p["q"] for p in pseudos], dtype=jnp.float32)))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float32)
    rem = n_electrons
    for i in range(n_bands):
        val = min(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val

    V_loc = backend.build_local_potential(grid, coords, pseudos)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid,
        coords,
        n_bands,
        occ,
        V_loc,
        pseudos,
        max_iter=2,
        mix_alpha=0.25,
        tolerance=1.0e-3,
        key=key,
        backend=backend,
    )
    ion_e = ion_ion_energy(coords, zion)
    energy = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, ion_e, backend=backend)

    eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_bands,)), -1, 0)
    norms = jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields)

    print("=== Adaptive SCF Smoke Test ===")
    print("Note: adaptive Hartree currently uses a zero-Dirichlet box Poisson prototype.")
    print("Note: this smoke test is not a physical uniform-vs-adaptive benchmark.")

    all_ok &= check("grid_shape", grid.coords.shape[:-1] == grid.shape, f"coords.shape={grid.coords.shape}, shape={grid.shape}")
    all_ok &= check("rho_finite", bool(jnp.all(jnp.isfinite(rho))), f"rho[min,max]=({float(jnp.min(rho)):.6f}, {float(jnp.max(rho)):.6f})")
    all_ok &= check("eigvals_finite", bool(jnp.all(jnp.isfinite(eigvals))), f"eigvals={jnp.asarray(eigvals)}")
    all_ok &= check("eigvecs_finite", bool(jnp.all(jnp.isfinite(eigvecs))), f"eigvecs.shape={eigvecs.shape}")
    all_ok &= check("hartree_finite", bool(jnp.all(jnp.isfinite(V_H))), f"V_H[min,max]=({float(jnp.min(V_H)):.6f}, {float(jnp.max(V_H)):.6f})")
    all_ok &= check("energy_finite", bool(jnp.isfinite(energy)), f"energy={float(energy):.6f}")
    all_ok &= check("electron_count", abs(float(backend.integrate(grid, rho)) - float(jnp.sum(occ))) <= 5e-3, f"N={float(backend.integrate(grid, rho)):.6f}, target={float(jnp.sum(occ)):.6f}")
    all_ok &= check("orbital_norms_finite", bool(jnp.all(jnp.isfinite(norms))), f"norms={jnp.asarray(norms)}")
    all_ok &= check("orbital_norms_reasonable", bool(jnp.all(jnp.abs(norms - 1.0) < 2e-2)), f"norms={jnp.asarray(norms)}")

    print("\n=== Summary ===")
    print(f"shape={grid.shape}")
    print(f"energy={float(energy):.6f}")
    print(f"eigvals={jnp.asarray(eigvals)}")
    print(f"electron_count={float(backend.integrate(grid, rho)):.6f}")
    print(f"orbital_norms={jnp.asarray(norms)}")

    if all_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
