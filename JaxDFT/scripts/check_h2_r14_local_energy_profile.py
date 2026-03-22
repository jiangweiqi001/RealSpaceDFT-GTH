"""H2 R=1.4 Bohr local-energy and density-profile audit.

This script compares Adaptive and Uniform real-space calculations at a single
geometry and asks where the local-potential energy difference comes from in
real space. The focus is the spatial decomposition of

    e_loc(r) = rho(r) * V_loc(r)

into near-core, bond-region, and outer contributions.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends.adaptive import AdaptiveBackend
    from JaxDFT.src.backends.uniform import UniformBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends.adaptive import AdaptiveBackend
    from src.backends.uniform import UniformBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


SMALL = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit H2 R=1.4 local-energy profile: Adaptive vs Uniform.",
    )
    parser.add_argument("--R", type=float, default=1.4, help="H-H distance in Bohr. Default: 1.4")
    parser.add_argument("--out-prefix", type=str, default="h2_r14_local_energy_profile", help="Output prefix. Default: h2_r14_local_energy_profile")

    parser.add_argument("--box", type=float, default=30.0, help="Adaptive cubic box in Bohr. Default: 30.0")
    parser.add_argument("--h-min", type=float, default=0.25, help="Adaptive h_min. Default: 0.25")
    parser.add_argument("--h-max", type=float, default=0.80, help="Adaptive h_max. Default: 0.80")
    parser.add_argument("--r-core", type=float, default=1.0, help="Adaptive r_core. Default: 1.0")
    parser.add_argument("--stretch-beta", type=float, default=5.0, help="Adaptive stretch beta. Default: 5.0")
    parser.add_argument(
        "--hartree-boundary-mode",
        type=str,
        default="uniform_exterior",
        choices=["zero_dirichlet", "monopole_dirichlet", "multipole_dirichlet", "uniform_exterior"],
        help="Adaptive Hartree boundary mode. Default: uniform_exterior",
    )
    parser.add_argument(
        "--kinetic-mode",
        type=str,
        default="prototype_fd2",
        choices=["prototype_fd2", "symmetric_fv"],
        help="Adaptive kinetic mode. Default: prototype_fd2",
    )

    parser.add_argument("--uniform-box", type=float, default=18.0, help="Uniform cubic box in Bohr. Default: 18.0")
    parser.add_argument("--uniform-spacing", type=float, default=0.18, help="Uniform spacing. Default: 0.18")

    parser.add_argument("--max-iter", type=int, default=120, help="SCF max iterations. Default: 120")
    parser.add_argument("--mix-alpha", type=float, default=0.30, help="SCF mixing alpha. Default: 0.30")
    parser.add_argument("--tolerance", type=float, default=1.0e-5, help="SCF tolerance. Default: 1e-5")
    parser.add_argument("--seed", type=int, default=42, help="Base PRNG seed. Default: 42")

    parser.add_argument("--core-radius", type=float, default=None, help="Core-region radius in Bohr. Default: min(r_core, 0.45 * R)")
    parser.add_argument("--bond-radius", type=float, default=0.8, help="Bond-cylinder radius in Bohr. Default: 0.8")
    return parser.parse_args()


def fmt_float(value: float | None, width: int = 13, precision: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value: float | None, width: int = 13, precision: int = 3) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_occ(pseudos):
    n_electrons = float(jnp.sum(jnp.asarray([p["q"] for p in pseudos], dtype=jnp.float32)))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float32)
    rem = n_electrons
    for i in range(n_bands):
        val = min(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val
    return n_electrons, n_bands, occ


def orbital_fields_from_flat(grid, eigvecs_flat):
    n_bands = int(eigvecs_flat.shape[1])
    return jnp.moveaxis(eigvecs_flat.reshape(grid.shape + (n_bands,)), -1, 0)


def orbital_norm_maxdev(grid, backend, eigvecs_flat) -> float:
    states = orbital_fields_from_flat(grid, eigvecs_flat)
    norms = jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(states)
    return float(jnp.max(jnp.abs(norms - 1.0)))


def _sample_axis_trilinear(axis_values, coord):
    axis_np = np.asarray(axis_values, dtype=np.float64)
    coord_np = np.asarray(coord, dtype=np.float64)
    upper = np.searchsorted(axis_np, coord_np, side="right")
    lower = upper - 1
    lower = np.clip(lower, 0, axis_np.size - 2)
    upper = np.clip(upper, 1, axis_np.size - 1)
    x0 = axis_np[lower]
    x1 = axis_np[upper]
    denom = np.maximum(x1 - x0, 1.0e-12)
    frac = np.clip((coord_np - x0) / denom, 0.0, 1.0)
    return lower, upper, frac


def _assemble_trilinear(field_np, ix0, ix1, tx, iy0, iy1, ty, iz0, iz1, tz):
    c000 = field_np[ix0, iy0, iz0]
    c001 = field_np[ix0, iy0, iz1]
    c010 = field_np[ix0, iy1, iz0]
    c011 = field_np[ix0, iy1, iz1]
    c100 = field_np[ix1, iy0, iz0]
    c101 = field_np[ix1, iy0, iz1]
    c110 = field_np[ix1, iy1, iz0]
    c111 = field_np[ix1, iy1, iz1]

    wx0 = 1.0 - tx
    wy0 = 1.0 - ty
    wz0 = 1.0 - tz
    wx1 = tx
    wy1 = ty
    wz1 = tz

    sampled = (
        c000 * wx0 * wy0 * wz0
        + c001 * wx0 * wy0 * wz1
        + c010 * wx0 * wy1 * wz0
        + c011 * wx0 * wy1 * wz1
        + c100 * wx1 * wy0 * wz0
        + c101 * wx1 * wy0 * wz1
        + c110 * wx1 * wy1 * wz0
        + c111 * wx1 * wy1 * wz1
    )
    return jnp.asarray(sampled, dtype=jnp.float32)


def sample_adaptive_trilinear(grid, field, probe_points):
    field_np = np.asarray(jnp.asarray(field), dtype=np.float64)
    points_np = np.asarray(jnp.asarray(probe_points), dtype=np.float64)
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    z = np.asarray(grid.z, dtype=np.float64)

    outside = (
        (points_np[:, 0] < x[0]) | (points_np[:, 0] > x[-1]) |
        (points_np[:, 1] < y[0]) | (points_np[:, 1] > y[-1]) |
        (points_np[:, 2] < z[0]) | (points_np[:, 2] > z[-1])
    )
    if np.any(outside):
        bad_idx = int(np.argmax(outside))
        raise ValueError(f"probe point {points_np[bad_idx].tolist()} lies outside adaptive box")

    ix0, ix1, tx = _sample_axis_trilinear(x, points_np[:, 0])
    iy0, iy1, ty = _sample_axis_trilinear(y, points_np[:, 1])
    iz0, iz1, tz = _sample_axis_trilinear(z, points_np[:, 2])
    return _assemble_trilinear(field_np, ix0, ix1, tx, iy0, iy1, ty, iz0, iz1, tz)


def sample_uniform_trilinear(grid, field, probe_points):
    field_np = np.asarray(jnp.asarray(field), dtype=np.float64)
    points_np = np.asarray(jnp.asarray(probe_points), dtype=np.float64)
    coords = np.asarray(grid.coords, dtype=np.float64)
    x = coords[:, 0, 0, 0]
    y = coords[0, :, 0, 1]
    z = coords[0, 0, :, 2]

    outside = (
        (points_np[:, 0] < x[0]) | (points_np[:, 0] > x[-1]) |
        (points_np[:, 1] < y[0]) | (points_np[:, 1] > y[-1]) |
        (points_np[:, 2] < z[0]) | (points_np[:, 2] > z[-1])
    )
    if np.any(outside):
        bad_idx = int(np.argmax(outside))
        raise ValueError(f"probe point {points_np[bad_idx].tolist()} lies outside uniform box")

    ix0, ix1, tx = _sample_axis_trilinear(x, points_np[:, 0])
    iy0, iy1, ty = _sample_axis_trilinear(y, points_np[:, 1])
    iz0, iz1, tz = _sample_axis_trilinear(z, points_np[:, 2])
    return _assemble_trilinear(field_np, ix0, ix1, tx, iy0, iy1, ty, iz0, iz1, tz)


def run_realspace_case(method: str, backend, grid, coords, pseudos, occ, n_bands, args, key):
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid,
        coords,
        n_bands,
        occ,
        V_loc,
        pseudos,
        max_iter=args.max_iter,
        mix_alpha=args.mix_alpha,
        tolerance=args.tolerance,
        key=key,
        backend=backend,
    )
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    eion = float(ion_ion_energy(coords, zion))
    e_total = float(total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, eion, backend=backend))
    e_loc = rho * V_loc
    eloc_total = float(backend.integrate(grid, e_loc))
    electron_count = float(backend.integrate(grid, rho))
    norm_dev = orbital_norm_maxdev(grid, backend, eigvecs)
    return {
        "method": method,
        "backend": backend,
        "grid": grid,
        "coords": coords,
        "rho": rho,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "V_loc": V_loc,
        "V_H": V_H,
        "eps_xc": eps_xc,
        "v_xc": v_xc,
        "E_total": e_total,
        "Eloc_total": eloc_total,
        "electron_count": electron_count,
        "norm_dev": norm_dev,
        "e_loc": e_loc,
    }


def build_region_masks(grid, R: float, core_radius: float, bond_radius: float):
    coords = jnp.asarray(grid.coords, dtype=jnp.float32)
    left = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)
    right = jnp.asarray([0.0, 0.0, R], dtype=jnp.float32)
    d_left = jnp.linalg.norm(coords - left, axis=-1)
    d_right = jnp.linalg.norm(coords - right, axis=-1)
    radial = jnp.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2)

    core_left = (d_left < core_radius) & (d_left <= d_right)
    core_right = (d_right < core_radius) & (d_right < d_left)
    bond_region = (coords[..., 2] >= 0.0) & (coords[..., 2] <= R) & (radial < bond_radius)
    bond_region = bond_region & (~core_left) & (~core_right)
    outer = ~(core_left | core_right | bond_region)
    return {
        "core_left": core_left,
        "core_right": core_right,
        "bond_region": bond_region,
        "outer": outer,
    }


def integrate_masked(backend, grid, field, mask) -> float:
    masked = jnp.where(mask, field, 0.0)
    return float(backend.integrate(grid, masked))


def attach_region_stats(case, masks):
    backend = case["backend"]
    grid = case["grid"]
    rho = case["rho"]
    e_loc = case["e_loc"]
    case["Eloc_core_left"] = integrate_masked(backend, grid, e_loc, masks["core_left"])
    case["Eloc_core_right"] = integrate_masked(backend, grid, e_loc, masks["core_right"])
    case["Eloc_bond_region"] = integrate_masked(backend, grid, e_loc, masks["bond_region"])
    case["Eloc_outer"] = integrate_masked(backend, grid, e_loc, masks["outer"])

    case["N_core_left"] = integrate_masked(backend, grid, rho, masks["core_left"])
    case["N_core_right"] = integrate_masked(backend, grid, rho, masks["core_right"])
    case["N_bond_region"] = integrate_masked(backend, grid, rho, masks["bond_region"])
    case["N_outer"] = integrate_masked(backend, grid, rho, masks["outer"])

    case["Eloc_region_sum"] = (
        case["Eloc_core_left"] + case["Eloc_core_right"] + case["Eloc_bond_region"] + case["Eloc_outer"]
    )
    case["N_region_sum"] = (
        case["N_core_left"] + case["N_core_right"] + case["N_bond_region"] + case["N_outer"]
    )
    case["Eloc_region_error"] = case["Eloc_total"] - case["Eloc_region_sum"]
    case["N_region_error"] = case["electron_count"] - case["N_region_sum"]
    return case


def build_profile_points(R: float, pad: float = 3.0, n_points: int = 801):
    z_axis = jnp.linspace(0.5 * R - pad, 0.5 * R + pad, n_points, dtype=jnp.float32)
    points = jnp.stack(
        [
            jnp.zeros_like(z_axis),
            jnp.zeros_like(z_axis),
            z_axis,
        ],
        axis=-1,
    )
    return z_axis, points


def build_xz_slice_points(R: float, x_bound: float = 2.5, z_pad: float = 1.5, nx: int = 201, nz: int = 201):
    x_axis = jnp.linspace(-x_bound, x_bound, nx, dtype=jnp.float32)
    z_axis = jnp.linspace(-z_pad, R + z_pad, nz, dtype=jnp.float32)
    xx, zz = jnp.meshgrid(x_axis, z_axis, indexing="ij")
    yy = jnp.zeros_like(xx)
    points = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return x_axis, z_axis, points


def sample_case_profile(case, profile_points, slice_points, slice_shape):
    sampler = sample_adaptive_trilinear if case["method"] == "Adaptive" else sample_uniform_trilinear
    rho_profile = sampler(case["grid"], case["rho"], profile_points)
    eloc_profile = sampler(case["grid"], case["e_loc"], profile_points)
    rho_slice = sampler(case["grid"], case["rho"], slice_points).reshape(slice_shape)
    return {
        "rho_profile": np.asarray(rho_profile, dtype=np.float64),
        "eloc_profile": np.asarray(eloc_profile, dtype=np.float64),
        "rho_slice": np.asarray(rho_slice, dtype=np.float64),
    }


def print_summary_table(adaptive, uniform):
    print("=== Summary Table ===")
    header = (
        f"{'method':<10} {'E_total':>13} {'Eloc_total':>13} {'Eloc_core_L':>13} {'Eloc_core_R':>13} "
        f"{'Eloc_bond':>13} {'Eloc_outer':>13} {'N':>10} {'norm_dev':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in (adaptive, uniform):
        print(
            f"{row['method']:<10} {fmt_float(row['E_total'])} {fmt_float(row['Eloc_total'])} "
            f"{fmt_float(row['Eloc_core_left'])} {fmt_float(row['Eloc_core_right'])} "
            f"{fmt_float(row['Eloc_bond_region'])} {fmt_float(row['Eloc_outer'])} "
            f"{fmt_float(row['electron_count'], 10, 6)} {fmt_sci(row['norm_dev'], 11, 2)}"
        )
    print()


def print_difference_table(delta):
    print("=== Difference Table (Adaptive - Uniform) ===")
    header = (
        f"{'dEloc_total':>13} {'dEloc_core_L':>13} {'dEloc_core_R':>13} "
        f"{'dEloc_bond':>13} {'dEloc_outer':>13}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{fmt_float(delta['dEloc_total'])} {fmt_float(delta['dEloc_core_left'])} "
        f"{fmt_float(delta['dEloc_core_right'])} {fmt_float(delta['dEloc_bond_region'])} "
        f"{fmt_float(delta['dEloc_outer'])}"
    )
    print()


def print_charge_table(adaptive, uniform, delta):
    print("=== Region Charge Table ===")
    header = f"{'method':<18} {'N_core_L':>12} {'N_core_R':>12} {'N_bond':>12} {'N_outer':>12}"
    print(header)
    print("-" * len(header))
    for row in (adaptive, uniform):
        print(
            f"{row['method']:<18} {fmt_float(row['N_core_left'], 12, 6)} {fmt_float(row['N_core_right'], 12, 6)} "
            f"{fmt_float(row['N_bond_region'], 12, 6)} {fmt_float(row['N_outer'], 12, 6)}"
        )
    print(
        f"{'Adaptive - Uniform':<18} {fmt_float(delta['dN_core_left'], 12, 6)} {fmt_float(delta['dN_core_right'], 12, 6)} "
        f"{fmt_float(delta['dN_bond_region'], 12, 6)} {fmt_float(delta['dN_outer'], 12, 6)}"
    )
    print()


def summarize_deltas(adaptive, uniform):
    delta = {
        "dE_total": adaptive["E_total"] - uniform["E_total"],
        "dEloc_total": adaptive["Eloc_total"] - uniform["Eloc_total"],
        "dEloc_core_left": adaptive["Eloc_core_left"] - uniform["Eloc_core_left"],
        "dEloc_core_right": adaptive["Eloc_core_right"] - uniform["Eloc_core_right"],
        "dEloc_bond_region": adaptive["Eloc_bond_region"] - uniform["Eloc_bond_region"],
        "dEloc_outer": adaptive["Eloc_outer"] - uniform["Eloc_outer"],
        "dN_core_left": adaptive["N_core_left"] - uniform["N_core_left"],
        "dN_core_right": adaptive["N_core_right"] - uniform["N_core_right"],
        "dN_bond_region": adaptive["N_bond_region"] - uniform["N_bond_region"],
        "dN_outer": adaptive["N_outer"] - uniform["N_outer"],
    }
    delta["dEloc_core_total"] = delta["dEloc_core_left"] + delta["dEloc_core_right"]
    delta["dN_core_total"] = delta["dN_core_left"] + delta["dN_core_right"]
    return delta


def diagnose(delta):
    contributions = {
        "core": abs(delta["dEloc_core_total"]),
        "bond": abs(delta["dEloc_bond_region"]),
        "outer": abs(delta["dEloc_outer"]),
    }
    ordered = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    dominant_name, dominant_val = ordered[0]
    second_val = ordered[1][1]
    total_abs = max(abs(delta["dEloc_total"]), SMALL)
    if dominant_val >= 1.5 * max(second_val, SMALL) and dominant_val >= 0.4 * total_abs:
        dominant_region = dominant_name
    else:
        dominant_region = "mixed"

    if dominant_region == "core":
        if delta["dN_core_total"] > 0.0:
            density_label = "near-core over-concentrated / over-deepened"
            next_step = "near-core grid strategy"
        else:
            density_label = "near-core shifted, but not via simple net core accumulation"
            next_step = "spacing sweep"
    elif dominant_region == "bond":
        if delta["dN_bond_region"] > 0.0:
            density_label = "bond-region density too concentrated"
        else:
            density_label = "bond-region density depleted or redistributed oddly"
        next_step = "bond-region refinement"
    elif dominant_region == "outer":
        density_label = "outer-tail / vacuum redistribution dominates"
        next_step = "spacing sweep"
    else:
        density_label = "mixed core/bond/outer redistribution"
        next_step = "mixed; do spacing sweep first"

    print("=== Diagnosis Summary ===")
    print(f"dEloc_dominant_region: {dominant_region}")
    print(f"rho_pattern: {density_label}")
    print(f"recommended_next_step: {next_step}")
    print(
        "detail: "
        f"|core|={abs(delta['dEloc_core_total']):.6e}, "
        f"|bond|={abs(delta['dEloc_bond_region']):.6e}, "
        f"|outer|={abs(delta['dEloc_outer']):.6e}, "
        f"|dEloc_total|={abs(delta['dEloc_total']):.6e}"
    )
    print(
        "charge_shift: "
        f"dN_core_total={delta['dN_core_total']:.6e}, "
        f"dN_bond={delta['dN_bond_region']:.6e}, "
        f"dN_outer={delta['dN_outer']:.6e}"
    )
    print()
    return dominant_region, density_label, next_step


def save_summary_csv(path, adaptive, uniform, delta):
    ensure_parent_dir(path)
    fieldnames = [
        "method",
        "E_total",
        "Eloc_total",
        "Eloc_core_left",
        "Eloc_core_right",
        "Eloc_bond_region",
        "Eloc_outer",
        "electron_count",
        "N_core_left",
        "N_core_right",
        "N_bond_region",
        "N_outer",
        "norm_dev",
        "Eloc_region_error",
        "N_region_error",
    ]
    rows = []
    for row in (adaptive, uniform):
        rows.append({name: row.get(name) for name in fieldnames})
    rows.append(
        {
            "method": "Adaptive-Uniform",
            "E_total": delta["dE_total"],
            "Eloc_total": delta["dEloc_total"],
            "Eloc_core_left": delta["dEloc_core_left"],
            "Eloc_core_right": delta["dEloc_core_right"],
            "Eloc_bond_region": delta["dEloc_bond_region"],
            "Eloc_outer": delta["dEloc_outer"],
            "electron_count": adaptive["electron_count"] - uniform["electron_count"],
            "N_core_left": delta["dN_core_left"],
            "N_core_right": delta["dN_core_right"],
            "N_bond_region": delta["dN_bond_region"],
            "N_outer": delta["dN_outer"],
            "norm_dev": adaptive["norm_dev"] - uniform["norm_dev"],
            "Eloc_region_error": adaptive["Eloc_region_error"] - uniform["Eloc_region_error"],
            "N_region_error": adaptive["N_region_error"] - uniform["N_region_error"],
        }
    )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_profile_csv(path, z_axis, adaptive_profile, uniform_profile):
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "z",
                "rho_adaptive",
                "rho_uniform",
                "delta_rho",
                "eloc_adaptive",
                "eloc_uniform",
                "delta_eloc",
            ]
        )
        for z, rho_a, rho_u, eloc_a, eloc_u in zip(
            np.asarray(z_axis, dtype=np.float64),
            adaptive_profile["rho_profile"],
            uniform_profile["rho_profile"],
            adaptive_profile["eloc_profile"],
            uniform_profile["eloc_profile"],
        ):
            writer.writerow([z, rho_a, rho_u, rho_a - rho_u, eloc_a, eloc_u, eloc_a - eloc_u])


def plot_profile(path, z_axis, adaptive_profile, uniform_profile, R):
    ensure_parent_dir(path)
    z = np.asarray(z_axis, dtype=np.float64)
    rho_a = adaptive_profile["rho_profile"]
    rho_u = uniform_profile["rho_profile"]
    eloc_a = adaptive_profile["eloc_profile"]
    eloc_u = uniform_profile["eloc_profile"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(z, rho_a, label="Adaptive rho", linewidth=2)
    ax.plot(z, rho_u, label="Uniform rho", linewidth=2, linestyle="--")
    ax.set_ylabel(r"$\rho(0,0,z)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(z, rho_a - rho_u, color="black", alpha=0.7, label=r"$\Delta \rho$")
    ax2.set_ylabel(r"$\Delta \rho$")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(R, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(0.5 * R, color="gray", linestyle="--", alpha=0.5)

    ax = axes[1]
    ax.plot(z, eloc_a, label="Adaptive e_loc", linewidth=2)
    ax.plot(z, eloc_u, label="Uniform e_loc", linewidth=2, linestyle="--")
    ax.set_xlabel("z (Bohr)")
    ax.set_ylabel(r"$e_{\mathrm{loc}}(0,0,z)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(z, eloc_a - eloc_u, color="black", alpha=0.7, label=r"$\Delta e_{\mathrm{loc}}$")
    ax2.set_ylabel(r"$\Delta e_{\mathrm{loc}}$")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(R, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(0.5 * R, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("H2 R=1.4: axis profile audit (Adaptive vs Uniform)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_xz_slice(path, x_axis, z_axis, adaptive_slice, uniform_slice, R):
    ensure_parent_dir(path)
    x = np.asarray(x_axis, dtype=np.float64)
    z = np.asarray(z_axis, dtype=np.float64)
    rho_a = adaptive_slice
    rho_u = uniform_slice
    delta = rho_a - rho_u

    extent = [z[0], z[-1], x[0], x[-1]]
    log_floor = 1.0e-8
    delta_lim = np.max(np.abs(delta)) + 1.0e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    im0 = axes[0].imshow(np.log10(rho_a + log_floor), origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].set_title("log10 rho Adaptive")
    im1 = axes[1].imshow(np.log10(rho_u + log_floor), origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[1].set_title("log10 rho Uniform")
    im2 = axes[2].imshow(delta, origin="lower", extent=extent, aspect="auto", cmap="coolwarm", vmin=-delta_lim, vmax=delta_lim)
    axes[2].set_title("delta rho (Adaptive-Uniform)")

    for ax in axes:
        ax.scatter([0.0, R], [0.0, 0.0], color="white", edgecolor="black", s=25, zorder=5)
        ax.set_xlabel("z (Bohr)")
    axes[0].set_ylabel("x (Bohr)")

    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    fig.suptitle("H2 R=1.4: x-z slice at y=0")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    core_radius = float(min(args.r_core, 0.45 * args.R) if args.core_radius is None else args.core_radius)
    bond_radius = float(args.bond_radius)

    summary_csv = f"{args.out_prefix}_summary.csv"
    profile_csv = f"{args.out_prefix}_profile.csv"
    profile_png = f"{args.out_prefix}_profile.png"
    slice_png = f"{args.out_prefix}_xz_slice.png"

    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    base_pseudos = load_pseudopotentials(["H"], pseudo_dir)
    pseudos = [base_pseudos[0], base_pseudos[0]]
    _, n_bands, occ = build_occ(pseudos)
    coords = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, args.R]], dtype=jnp.float32)

    print("\n=== H2 R=1.4 Local-Energy Profile Audit ===")
    print(f"R = {args.R} Bohr (molecular axis: z)")
    print(
        "Adaptive setup: "
        f"box={args.box}, h_min={args.h_min}, h_max={args.h_max}, "
        f"r_core={args.r_core}, stretch_beta={args.stretch_beta}, "
        f"hartree={args.hartree_boundary_mode}, kinetic={args.kinetic_mode}"
    )
    print(
        "Uniform setup: "
        f"box={args.uniform_box}, spacing={args.uniform_spacing}"
    )
    print(f"Region setup: core_radius={core_radius:.3f}, bond_radius={bond_radius:.3f}")
    print()

    adaptive_backend = AdaptiveBackend(
        hartree_boundary_mode=args.hartree_boundary_mode,
        kinetic_mode=args.kinetic_mode,
    )
    adaptive_grid = adaptive_backend.create_grid(
        spacing=args.h_min,
        box_size=[args.box, args.box, args.box],
        atom_coords=coords,
        h_min=args.h_min,
        h_max=args.h_max,
        r_core=args.r_core,
        stretch_beta=args.stretch_beta,
    )
    adaptive = run_realspace_case(
        "Adaptive",
        adaptive_backend,
        adaptive_grid,
        coords,
        pseudos,
        occ,
        n_bands,
        args,
        jax.random.PRNGKey(args.seed),
    )
    adaptive = attach_region_stats(adaptive, build_region_masks(adaptive_grid, args.R, core_radius, bond_radius))

    uniform_backend = UniformBackend()
    uniform_grid = uniform_backend.create_grid(args.uniform_spacing, [args.uniform_box, args.uniform_box, args.uniform_box])
    uniform = run_realspace_case(
        "Uniform",
        uniform_backend,
        uniform_grid,
        coords,
        pseudos,
        occ,
        n_bands,
        args,
        jax.random.PRNGKey(args.seed),
    )
    uniform = attach_region_stats(uniform, build_region_masks(uniform_grid, args.R, core_radius, bond_radius))

    delta = summarize_deltas(adaptive, uniform)
    print_summary_table(adaptive, uniform)
    print_difference_table(delta)
    print_charge_table(adaptive, uniform, delta)
    dominant_region, density_label, next_step = diagnose(delta)

    z_axis, profile_points = build_profile_points(args.R)
    x_axis, z_slice_axis, slice_points = build_xz_slice_points(args.R)
    slice_shape = (x_axis.size, z_slice_axis.size)
    adaptive_profile = sample_case_profile(adaptive, profile_points, slice_points, slice_shape)
    uniform_profile = sample_case_profile(uniform, profile_points, slice_points, slice_shape)

    save_summary_csv(summary_csv, adaptive, uniform, delta)
    save_profile_csv(profile_csv, z_axis, adaptive_profile, uniform_profile)
    plot_profile(profile_png, z_axis, adaptive_profile, uniform_profile, args.R)
    plot_xz_slice(slice_png, x_axis, z_slice_axis, adaptive_profile["rho_slice"], uniform_profile["rho_slice"], args.R)

    overall_ok = True
    overall_ok &= check(
        "adaptive_eloc_partition",
        abs(adaptive["Eloc_region_error"]) <= 1e-3,
        f"Eloc_total-Eloc_regions={adaptive['Eloc_region_error']:.3e}",
    )
    overall_ok &= check(
        "uniform_eloc_partition",
        abs(uniform["Eloc_region_error"]) <= 1e-3,
        f"Eloc_total-Eloc_regions={uniform['Eloc_region_error']:.3e}",
    )
    overall_ok &= check(
        "adaptive_charge_partition",
        abs(adaptive["N_region_error"]) <= 1e-4,
        f"N_total-N_regions={adaptive['N_region_error']:.3e}",
    )
    overall_ok &= check(
        "uniform_charge_partition",
        abs(uniform["N_region_error"]) <= 1e-4,
        f"N_total-N_regions={uniform['N_region_error']:.3e}",
    )
    overall_ok &= check(
        "profile_finite",
        np.all(np.isfinite(adaptive_profile["rho_profile"]))
        and np.all(np.isfinite(uniform_profile["rho_profile"]))
        and np.all(np.isfinite(adaptive_profile["eloc_profile"]))
        and np.all(np.isfinite(uniform_profile["eloc_profile"])),
        "axis profiles finite",
    )
    overall_ok &= check(
        "slice_finite",
        np.all(np.isfinite(adaptive_profile["rho_slice"])) and np.all(np.isfinite(uniform_profile["rho_slice"])),
        "x-z slice finite",
    )
    overall_ok &= check(
        "adaptive_electron_count",
        abs(adaptive["electron_count"] - float(jnp.sum(occ))) <= 5e-3,
        f"N={adaptive['electron_count']:.6f}",
    )
    overall_ok &= check(
        "uniform_electron_count",
        abs(uniform["electron_count"] - float(jnp.sum(occ))) <= 5e-3,
        f"N={uniform['electron_count']:.6f}",
    )

    print("=== Outputs ===")
    print(f"summary_csv: {summary_csv}")
    print(f"profile_csv: {profile_csv}")
    print(f"profile_png: {profile_png}")
    print(f"xz_slice_png: {slice_png}")
    print()

    print("=== Overall Summary ===")
    print(f"dEloc_dominant_region = {dominant_region}")
    print(f"rho_pattern = {density_label}")
    print(f"recommended_next_step = {next_step}")
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
