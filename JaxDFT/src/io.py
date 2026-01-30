import os
from dataclasses import dataclass

import h5py
import numpy as np
import yaml


PERIODIC_TABLE = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Fe": 26,
    "Cu": 29,
    "Au": 79,
}


VALENCE_Q = {
    "H": 1,
    "He": 2,
    "Li": 1,
    "Be": 2,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 1,
    "Mg": 2,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8,
    "K": 1,
    "Ca": 2,
    "Fe": 8,
    "Cu": 11,
    "Au": 11,
}


def _base_params(symbol):
    z = PERIODIC_TABLE[symbol]
    rloc = 0.18 + 0.006 * z
    c1 = -3.5 - 0.03 * z
    c2 = 0.5 + 0.01 * z
    proj_r = 0.22 + 0.004 * z
    h = [0.3 + 0.002 * z]
    return rloc, [c1, c2], proj_r, h


GTH_POTENTIALS = {}
for _sym in PERIODIC_TABLE:
    if _sym not in VALENCE_Q:
        continue
    _rloc, _c, _pr, _h = _base_params(_sym)
    GTH_POTENTIALS[_sym] = {
        "q": VALENCE_Q[_sym],
        "rloc": _rloc,
        "c": _c,
        "projectors": [
            {"l": 0, "r": _pr, "h": _h},
        ],
    }


@dataclass
class GTHLocal:
    zion: float
    rloc: float
    c: np.ndarray


@dataclass
class GTHProjector:
    l: int
    r: float
    h: np.ndarray
    poly: np.ndarray


@dataclass
class GTHPP:
    symbol: str
    q: int
    local: GTHLocal
    projectors: list


def gth_to_text(symbol, params):
    q = int(params["q"])
    rloc = float(params["rloc"])
    c = params["c"]
    projectors = params.get("projectors", [])
    lines = [
        f"{symbol} GTH-LDA-q{q} GTH-LDA",
        f"    {q}",
        "    " + " ".join([f"{rloc:.8f}", str(len(c))] + [f"{v:.8f}" for v in c]),
        f"    {len(projectors)}",
    ]
    for proj in projectors:
        r = float(proj["r"])
        h = proj["h"]
        lines.append("    " + " ".join([f"{r:.8f}", str(len(h))]))
        for val in h:
            lines.append(f"      {float(val):.8f}")
    return "\n".join(lines) + "\n"


def initialize_gth_potentials(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    existing = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if existing:
        return
    for symbol, params in GTH_POTENTIALS.items():
        text = gth_to_text(symbol, params)
        path = os.path.join(data_dir, f"{symbol}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def parse_gth_text(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    header = lines[0].split()
    symbol = header[0]
    q = int(lines[1].split()[0])
    local_tokens = lines[2].split()
    rloc = float(local_tokens[0])
    n_gauss = int(local_tokens[1])
    c = np.array([float(x) for x in local_tokens[2:2 + n_gauss]], dtype=float)
    c = np.pad(c, (0, max(0, 4 - c.size)), constant_values=0.0)
    zion = float(q)
    local = GTHLocal(zion=zion, rloc=rloc, c=c)
    projectors = []
    idx = 3
    n_channels = int(lines[idx].split()[0])
    idx += 1
    l_cur = 0
    for _ in range(n_channels):
        tokens = lines[idx].split()
        r = float(tokens[0])
        n_proj = int(tokens[1])
        idx += 1
        hvals = []
        for _ in range(n_proj):
            toks = lines[idx].split()
            hvals.append(float(toks[0]))
            idx += 1
        h = np.array(hvals, dtype=float)
        poly = np.arange(n_proj, dtype=int)
        projectors.append(GTHProjector(l=l_cur, r=r, h=h, poly=poly))
        l_cur += 1
    return GTHPP(symbol=symbol, q=q, local=local, projectors=projectors)


def load_pseudopotentials(symbols, data_dir):
    initialize_gth_potentials(data_dir)
    pseudos = []
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        pp = parse_gth_text(text)
        pseudos.append(
            {
                "symbol": pp.symbol,
                "q": pp.q,
                "zion": pp.local.zion,
                "rloc": pp.local.rloc,
                "c": pp.local.c.astype(float),
                "projectors": [
                    {
                        "l": pj.l,
                        "r": pj.r,
                        "h": pj.h.astype(float),
                    }
                    for pj in pp.projectors
                ],
            }
        )
    return pseudos


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_hdf5(path, coords_list, z_list, energy_list, forces_list):
    coords = np.asarray(coords_list, dtype=float)
    zvals = np.asarray(z_list, dtype=int)
    energies = np.asarray(energy_list, dtype=float)
    forces = np.asarray(forces_list, dtype=float)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("R", data=coords)
        h5.create_dataset("Z", data=zvals)
        h5.create_dataset("E", data=energies)
        h5.create_dataset("F", data=forces)
