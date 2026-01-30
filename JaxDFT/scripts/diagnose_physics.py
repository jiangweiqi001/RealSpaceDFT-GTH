import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from src.hamiltonian import create_grid, prepare_system
from src.io import load_pseudopotentials
from src.solver import energy_and_forces, ion_ion_energy, scf


def build_occupations(pseudos):
    electrons = jnp.sum(jnp.asarray([pp["q"] for pp in pseudos], dtype=jnp.float64))
    n_bands = int(jnp.ceil(electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float64)
    remaining = electrons
    for i in range(n_bands):
        occ = occ.at[i].set(jnp.minimum(2.0, remaining))
        remaining = remaining - occ[i]
    return electrons, n_bands, occ


def scf_diagnostics(coords, spacing, box_size, pseudos, max_iter=150, mix_alpha=0.1, tolerance=1e-5, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    coords = jnp.asarray(coords, dtype=jnp.float64)
    grid = create_grid(spacing, box_size)
    grid = prepare_system(grid, coords, pseudos)
    electrons, n_bands, occ = build_occupations(pseudos)
    V_loc = jnp.nan_to_num(grid.V_loc, nan=0.0, posinf=0.0, neginf=0.0)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid, coords, n_bands, occ, V_loc, grid.projectors, max_iter, mix_alpha, tolerance, key
    )
    rho = jnp.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    V_H = jnp.nan_to_num(V_H, nan=0.0, posinf=0.0, neginf=0.0)
    eps_xc = jnp.nan_to_num(eps_xc, nan=0.0, posinf=0.0, neginf=0.0)
    v_xc = jnp.nan_to_num(v_xc, nan=0.0, posinf=0.0, neginf=0.0)
    e_band = jnp.sum(eigvals * occ)
    e_h = 0.5 * grid.volume_element * jnp.sum(rho * V_H)
    e_loc = grid.volume_element * jnp.sum(rho * V_loc)
    e_xc = grid.volume_element * jnp.sum(eps_xc)
    e_vxc = grid.volume_element * jnp.sum(rho * v_xc)
    ion = ion_ion_energy(coords, jnp.asarray([pp["zion"] for pp in pseudos], dtype=jnp.float64))
    total = e_band - e_h + e_loc + e_xc - e_vxc + ion
    kinetic = e_band - e_loc - e_h - e_vxc
    rho_int = grid.volume_element * jnp.sum(rho)
    vloc_min = jnp.min(V_loc)
    return {
        "total": total,
        "kinetic": kinetic,
        "e_band": e_band,
        "e_h": e_h,
        "e_loc": e_loc,
        "e_xc": e_xc,
        "e_vxc": e_vxc,
        "ion": ion,
        "rho_int": rho_int,
        "vloc_min": vloc_min,
        "grid": grid,
        "coords": coords,
        "electrons": electrons,
    }


def print_diagnostics(tag, data):
    print(f"\n[{tag}]")
    print(f"Total Energy: {float(data['total']): .8f} Ha")
    print(f"Kinetic (E_band - E_loc - E_H - E_vxc): {float(data['kinetic']): .8f} Ha")
    print(f"E_band: {float(data['e_band']): .8f} Ha")
    print(f"E_H: {float(data['e_h']): .8f} Ha")
    print(f"E_loc: {float(data['e_loc']): .8f} Ha")
    print(f"E_xc: {float(data['e_xc']): .8f} Ha")
    print(f"E_vxc: {float(data['e_vxc']): .8f} Ha")
    print(f"Ion-Ion: {float(data['ion']): .8f} Ha")
    print(f"Charge Integral: {float(data['rho_int']): .8f} e")
    print(f"Expected Electrons: {float(data['electrons']): .8f} e")
    print(f"V_loc min: {float(data['vloc_min']): .8f} Ha")
    print(f"Grid shape: {data['grid'].shape}, spacing: {data['grid'].spacing}, dV: {data['grid'].volume_element}")


def main():
    box_size = [6.0, 6.0, 6.0]
    data_path = os.path.join(root, "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], data_path)

    d = 0.8
    offset = d / 2.0
    coords_sym = np.array([[0.0, 0.0, -offset], [0.0, 0.0, offset]], dtype=np.float64)
    coords_break = np.array([[0.0, 0.0, -0.4], [0.0, 0.0, 0.5]], dtype=np.float64)

    for spacing in [0.5, 0.3, 0.2]:
        key = jax.random.PRNGKey(int(spacing * 1000))
        data = scf_diagnostics(coords_sym, spacing, box_size, pseudos, key=key)
        print_diagnostics(f"H2 d=0.8 Bohr, spacing={spacing}", data)

    spacing = 0.5
    grid = create_grid(spacing, box_size)
    grid = prepare_system(grid, jnp.asarray(coords_break), pseudos)
    key = jax.random.PRNGKey(999)
    energy, forces = energy_and_forces(
        grid,
        jnp.asarray(coords_break),
        pseudos,
        max_iter=150,
        mix_alpha=0.1,
        tolerance=1e-5,
        key=key,
    )
    print("\n[Force Symmetry Check]")
    print(f"Coords: {coords_break.tolist()}")
    print(f"Energy: {float(energy): .8f} Ha")
    print(f"Forces: {np.asarray(forces).tolist()}")


if __name__ == "__main__":
    main()
