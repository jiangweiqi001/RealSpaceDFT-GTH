import os
import sys

import jax
import numpy as np


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from src.hamiltonian import create_grid, prepare_system
from src.io import PERIODIC_TABLE, load_config, load_pseudopotentials, write_hdf5
from src.solver import energy_and_forces
from src.structure import generate_random_cluster


def main():
    config_path = os.path.join(root, "config", "default.yaml")
    cfg = load_config(config_path)

    spacing = cfg["grid"]["spacing"]
    box_size = cfg["grid"]["box_size"]
    n_samples = int(cfg["sampling"]["n_samples"])
    min_distance = float(cfg["sampling"]["min_distance"])
    elements = cfg["elements"]
    scf_cfg = cfg["scf"]

    data_dir = os.path.join(root, "data", "gth_potentials")

    rng = np.random.default_rng(1234)
    coords_list = []
    z_list = []
    energy_list = []
    forces_list = []

    for idx in range(n_samples):
        n_atoms = int(rng.integers(2, 7))
        coords, symbols = generate_random_cluster(elements, n_atoms, box_size, min_distance, rng=rng)
        pseudos = load_pseudopotentials(symbols, data_dir)
        grid = create_grid(spacing, box_size)
        grid = prepare_system(grid, jax.numpy.asarray(coords, dtype=jax.numpy.float32), pseudos)
        key = jax.random.PRNGKey(idx)
        try:
            energy, forces = energy_and_forces(
                grid,
                jax.numpy.asarray(coords, dtype=jax.numpy.float32),
                pseudos,
                int(scf_cfg["max_iter"]),
                float(scf_cfg["mix_alpha"]),
                float(scf_cfg["tolerance"]),
                key,
            )
        except Exception as exc:
            print(f"[{idx + 1}/{n_samples}] SCF failed: {exc}")
            continue

        energy_val = float(energy)
        if not np.isfinite(energy_val):
            print(f"[{idx + 1}/{n_samples}] SCF returned NaN/Inf, skipped")
            continue

        coords_list.append(coords)
        z_list.append([PERIODIC_TABLE[s] for s in symbols])
        energy_list.append(energy_val)
        forces_list.append(np.asarray(forces))
        print(f"[{idx + 1}/{n_samples}] E={energy_val:.6f} Ha")

    output_path = os.path.join(root, "dataset.h5")
    write_hdf5(output_path, coords_list, z_list, energy_list, forces_list)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
