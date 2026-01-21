import numpy as np


def check_min_distance(coords, limit):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return True
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    iu = np.triu_indices(n, k=1)
    return np.all(d2[iu] >= float(limit) ** 2)


def generate_random_cluster(elements, n_atoms_range, box_size, min_distance, rng=None, max_tries=2000):
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(n_atoms_range, int):
        n_min, n_max = n_atoms_range, n_atoms_range
    else:
        n_min, n_max = n_atoms_range
    box_size = np.asarray(box_size, dtype=float)
    low = -0.5 * box_size
    high = 0.5 * box_size
    for _ in range(max_tries):
        n_atoms = int(rng.integers(n_min, n_max + 1))
        symbols = rng.choice(elements, size=n_atoms, replace=True).tolist()
        coords = rng.uniform(low, high, size=(n_atoms, 3))
        if check_min_distance(coords, min_distance):
            return coords, symbols
    raise RuntimeError("Failed to generate a valid cluster within max_tries")
