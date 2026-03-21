"""H2 adaptive orbital localization / boundary-amplitude sanity study.

This is not a final benchmark.
The current goal is to determine whether the previously observed
``boundary/global ratio ~= 1`` for H2 on the adaptive grid means that the
orbital/density maxima are genuinely being pushed to the boundary, or whether
that ratio is partly an artifact of the current boundary-face statistic.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

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


DISTANCE = 1.4
BOXES = [14.0, 18.0, 22.0, 26.0, 30.0]
H_MIN = 0.25
H_MAX = 0.80
R_CORE = 1.0
STRETCH_BETA = 5.0
SCF_KWARGS = {
    "max_iter": 4,
    "mix_alpha": 0.30,
    "tolerance": 5.0e-4,
}
ELECTRON_TOL = 5.0e-3
NORM_TOL = 2.0e-2
BOUNDARY_NEAR_TOL = 0.25
INTERIOR_FAR_TOL = 1.0
GAP_REL_TOL = 1.0e-6


def fmt_float(value, width=11, precision=6):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=12, precision=3):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def fmt_vec3(value, width=26, precision=3):
    if value is None:
        return "-".rjust(width)
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    text = f"({arr[0]:.{precision}f},{arr[1]:.{precision}f},{arr[2]:.{precision}f})"
    return text.ljust(width)


def check(name: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return condition


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


def collect_face_values(field):
    arr = jnp.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D field, got shape {arr.shape}")
    return jnp.concatenate(
        [
            jnp.ravel(arr[0, 1:-1, 1:-1]),
            jnp.ravel(arr[-1, 1:-1, 1:-1]),
            jnp.ravel(arr[1:-1, 0, 1:-1]),
            jnp.ravel(arr[1:-1, -1, 1:-1]),
            jnp.ravel(arr[1:-1, 1:-1, 0]),
            jnp.ravel(arr[1:-1, 1:-1, -1]),
        ],
        axis=0,
    )


def face_stats(field, *, absolute: bool = False):
    face = collect_face_values(field)
    if absolute:
        face = jnp.abs(face)
    return float(jnp.max(face)), float(jnp.mean(face))


def nearest_boundary_info(coord, grid):
    x0, x1 = float(grid.x[0]), float(grid.x[-1])
    y0, y1 = float(grid.y[0]), float(grid.y[-1])
    z0, z1 = float(grid.z[0]), float(grid.z[-1])
    x, y, z = (float(coord[0]), float(coord[1]), float(coord[2]))
    candidates = {
        "x_lo": abs(x - x0),
        "x_hi": abs(x1 - x),
        "y_lo": abs(y - y0),
        "y_hi": abs(y1 - y),
        "z_lo": abs(z - z0),
        "z_hi": abs(z1 - z),
    }
    face = min(candidates, key=candidates.get)
    return float(candidates[face]), face


def nearest_nucleus_distance(coord, nuclei):
    coord_arr = np.asarray(coord, dtype=np.float64).reshape(1, 3)
    nuclei_arr = np.asarray(nuclei, dtype=np.float64).reshape(-1, 3)
    return float(np.min(np.linalg.norm(nuclei_arr - coord_arr, axis=1)))


def global_max_info(field, grid, nuclei, *, absolute: bool = False):
    arr = np.asarray(jnp.abs(field) if absolute else field, dtype=np.float64)
    flat_idx = int(np.argmax(arr))
    idx = tuple(int(i) for i in np.unravel_index(flat_idx, arr.shape))
    value = float(arr[idx])
    coord = np.asarray(grid.coords[idx], dtype=np.float64)
    d_nuc = nearest_nucleus_distance(coord, nuclei)
    d_bnd, face = nearest_boundary_info(coord, grid)
    on_boundary = idx[0] in (0, arr.shape[0] - 1) or idx[1] in (0, arr.shape[1] - 1) or idx[2] in (0, arr.shape[2] - 1)
    return {
        "value": value,
        "index": idx,
        "coord": coord,
        "dist_to_nucleus": d_nuc,
        "dist_to_boundary": d_bnd,
        "nearest_boundary_face": face,
        "on_boundary": bool(on_boundary),
    }


def z0_plane_summary(field, grid, *, absolute: bool = False):
    arr = jnp.abs(field) if absolute else jnp.asarray(field)
    z = np.asarray(grid.z, dtype=np.float64)
    iz = int(np.argmin(np.abs(z)))
    plane = np.asarray(arr[:, :, iz], dtype=np.float64)
    return {
        "z_coord": float(z[iz]),
        "index": iz,
        "max": float(np.max(plane)),
        "mean": float(np.mean(plane)),
    }


def run_case(grid, coords, pseudos, zion, occ, n_bands, key_seed):
    backend = AdaptiveBackend(hartree_boundary_mode="multipole_dirichlet")
    result = {
        "box": float(grid.box_size[0]),
        "completed": False,
        "all_finite": False,
        "energy": None,
        "electron_count": None,
        "electron_error": None,
        "orbital_norm_maxdev": None,
        "eig0": None,
        "shape": tuple(int(n) for n in grid.shape),
        "rho_face_max": None,
        "rho_face_mean": None,
        "rho_face_ratio": None,
        "rho_face_gap": None,
        "psi_face_max": None,
        "psi_face_mean": None,
        "psi_face_ratio": None,
        "psi_face_gap": None,
        "vh_face_max": None,
        "vh_face_mean": None,
        "vh_face_ratio": None,
        "vh_face_gap": None,
        "rho_max_info": None,
        "psi_max_info": None,
        "z0_rho": None,
        "z0_psi": None,
        "error": None,
    }

    try:
        V_loc = backend.build_local_potential(grid, coords, pseudos)
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
            grid,
            coords,
            n_bands,
            occ,
            V_loc,
            pseudos,
            key=jax.random.PRNGKey(key_seed),
            backend=backend,
            **SCF_KWARGS,
        )
        ion_e = ion_ion_energy(coords, zion)
        energy = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, ion_e, backend=backend)
        eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_bands,)), -1, 0)
        norms = jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields)
        psi0 = jnp.abs(eigvec_fields[0])
        electron_count = float(backend.integrate(grid, rho))
        norm_maxdev = float(jnp.max(jnp.abs(norms - 1.0)))

        rho_face_max, rho_face_mean = face_stats(rho)
        psi_face_max, psi_face_mean = face_stats(psi0)
        vh_face_max, vh_face_mean = face_stats(V_H)
        rho_max_info = global_max_info(rho, grid, coords)
        psi_max_info = global_max_info(psi0, grid, coords)
        rho_face_ratio = rho_face_max / max(rho_max_info["value"], 1.0e-30)
        psi_face_ratio = psi_face_max / max(psi_max_info["value"], 1.0e-30)
        vh_global_max = float(jnp.max(V_H))
        vh_face_ratio = vh_face_max / max(vh_global_max, 1.0e-30)

        all_finite = bool(
            jnp.all(jnp.isfinite(rho))
            and jnp.all(jnp.isfinite(eigvals))
            and jnp.all(jnp.isfinite(eigvecs))
            and jnp.all(jnp.isfinite(V_H))
            and jnp.isfinite(energy)
            and jnp.all(jnp.isfinite(norms))
        )
        result.update({
            "completed": True,
            "all_finite": all_finite,
            "energy": float(energy),
            "electron_count": electron_count,
            "electron_error": abs(electron_count - float(jnp.sum(occ))),
            "orbital_norm_maxdev": norm_maxdev,
            "eig0": float(jnp.asarray(eigvals).reshape(-1)[0]),
            "rho_face_max": rho_face_max,
            "rho_face_mean": rho_face_mean,
            "rho_face_ratio": rho_face_ratio,
            "rho_face_gap": rho_max_info["value"] - rho_face_max,
            "psi_face_max": psi_face_max,
            "psi_face_mean": psi_face_mean,
            "psi_face_ratio": psi_face_ratio,
            "psi_face_gap": psi_max_info["value"] - psi_face_max,
            "vh_face_max": vh_face_max,
            "vh_face_mean": vh_face_mean,
            "vh_face_ratio": vh_face_ratio,
            "vh_face_gap": vh_global_max - vh_face_max,
            "rho_max_info": rho_max_info,
            "psi_max_info": psi_max_info,
            "z0_rho": z0_plane_summary(rho, grid),
            "z0_psi": z0_plane_summary(psi0, grid, absolute=False),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def build_results():
    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(["H", "H"], pseudo_dir)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    _, n_bands, occ = build_occ(pseudos)

    grid_builder = AdaptiveBackend()
    results = []
    base_key = jax.random.PRNGKey(42)

    for box_idx, box_length in enumerate(BOXES):
        box = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
        grid = grid_builder.create_grid(
            spacing=H_MIN,
            box_size=box,
            atom_coords=coords,
            h_min=H_MIN,
            h_max=H_MAX,
            r_core=R_CORE,
            stretch_beta=STRETCH_BETA,
        )
        key_seed = int(jax.random.fold_in(base_key, box_idx)[0])
        results.append(run_case(grid, coords, pseudos, zion, occ, n_bands, key_seed))
    return results


def print_result_table(results):
    print("=== Result Table ===")
    header = (
        f"{'Box':>5} {'Done':<5} {'Finite':<6} {'Energy':>12} {'N':>8} {'dN':>10} "
        f"{'NormDev':>10} {'eig0':>11} {'Shape':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{fmt_float(row['box'], 5, 1)} {('PASS' if row['completed'] else 'FAIL'):<5} "
            f"{('PASS' if row['all_finite'] else 'FAIL'):<6} {fmt_float(row['energy'], 12, 6)} "
            f"{fmt_float(row['electron_count'], 8, 4)} {fmt_sci(row['electron_error'], 10, 2)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 10, 2)} {fmt_float(row['eig0'], 11, 6)} {str(row['shape']):<14}"
        )
        if row['error'] is not None:
            print(f"  error: {row['error']}")


def print_localization_table(results, key, label):
    print(f"=== {label} Localization Table ===")
    header = (
        f"{'Box':>5} {'Value':>12} {'Coord':<26} {'d_nearest_nuc':>14} {'d_nearest_bnd':>14} {'NearestFace':<8} {'OnBnd':<6}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        info = row[key]
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_sci(info['value'], 12, 3)} {fmt_vec3(info['coord'], 26, 3)} "
            f"{fmt_float(info['dist_to_nucleus'], 14, 6)} {fmt_float(info['dist_to_boundary'], 14, 6)} {info['nearest_boundary_face']:<8} {str(info['on_boundary']):<6}"
        )


def print_boundary_amplitude_table(results):
    print("=== Boundary Amplitude Table ===")
    header = (
        f"{'Box':>5} {'rho_face_max':>12} {'rho_face_mean':>13} {'rho_ratio':>11} {'rho_gap':>11} "
        f"{'psi_face_max':>12} {'psi_face_mean':>13} {'psi_ratio':>11} {'psi_gap':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_sci(row['rho_face_max'], 12, 3)} {fmt_sci(row['rho_face_mean'], 13, 3)} {fmt_sci(row['rho_face_ratio'], 11, 3)} {fmt_sci(row['rho_face_gap'], 11, 3)} "
            f"{fmt_sci(row['psi_face_max'], 12, 3)} {fmt_sci(row['psi_face_mean'], 13, 3)} {fmt_sci(row['psi_face_ratio'], 11, 3)} {fmt_sci(row['psi_face_gap'], 11, 3)}"
        )


def print_vh_boundary_table(results):
    print("=== V_H Boundary Table ===")
    header = f"{'Box':>5} {'VH_face_max':>12} {'VH_face_mean':>13} {'VH_ratio':>11} {'VH_gap':>11}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_sci(row['vh_face_max'], 12, 3)} {fmt_sci(row['vh_face_mean'], 13, 3)} {fmt_sci(row['vh_face_ratio'], 11, 3)} {fmt_sci(row['vh_face_gap'], 11, 3)}"
        )


def print_center_slice_table(results):
    print("=== z?0 Plane Summary ===")
    header = f"{'Box':>5} {'z_plane':>10} {'rho_z0_max':>12} {'rho_z0_mean':>13} {'psi_z0_max':>12} {'psi_z0_mean':>13}"
    print(header)
    print("-" * len(header))
    for row in results:
        rho_z0 = row['z0_rho']
        psi_z0 = row['z0_psi']
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_float(rho_z0['z_coord'], 10, 4)} {fmt_sci(rho_z0['max'], 12, 3)} {fmt_sci(rho_z0['mean'], 13, 3)} {fmt_sci(psi_z0['max'], 12, 3)} {fmt_sci(psi_z0['mean'], 13, 3)}"
        )


def diagnose(results):
    largest = results[-1]
    rho = largest['rho_max_info']
    psi = largest['psi_max_info']
    rho_gap = largest['rho_face_gap']
    psi_gap = largest['psi_face_gap']
    rho_tied = rho_gap is not None and abs(rho_gap) <= max(GAP_REL_TOL * max(rho['value'], 1.0), 1.0e-12)
    psi_tied = psi_gap is not None and abs(psi_gap) <= max(GAP_REL_TOL * max(psi['value'], 1.0), 1.0e-12)

    if rho['dist_to_boundary'] <= BOUNDARY_NEAR_TOL or psi['dist_to_boundary'] <= BOUNDARY_NEAR_TOL or rho['on_boundary'] or psi['on_boundary']:
        label = 'boundary-peaked-real'
    elif (rho_tied or psi_tied) and rho['dist_to_boundary'] >= INTERIOR_FAR_TOL and psi['dist_to_boundary'] >= INTERIOR_FAR_TOL:
        label = 'ratio-artifact-suspected'
    elif rho['dist_to_nucleus'] <= INTERIOR_FAR_TOL and psi['dist_to_nucleus'] <= INTERIOR_FAR_TOL:
        label = 'interior-localized'
    else:
        label = 'mixed'

    return {
        'label': label,
        'largest_box': largest['box'],
        'rho_dist_to_nucleus': rho['dist_to_nucleus'],
        'rho_dist_to_boundary': rho['dist_to_boundary'],
        'rho_on_boundary': rho['on_boundary'],
        'rho_nearest_face': rho['nearest_boundary_face'],
        'rho_face_ratio': largest['rho_face_ratio'],
        'rho_face_gap': largest['rho_face_gap'],
        'psi_dist_to_nucleus': psi['dist_to_nucleus'],
        'psi_dist_to_boundary': psi['dist_to_boundary'],
        'psi_on_boundary': psi['on_boundary'],
        'psi_nearest_face': psi['nearest_boundary_face'],
        'psi_face_ratio': largest['psi_face_ratio'],
        'psi_face_gap': largest['psi_face_gap'],
    }


def print_diagnosis(diag):
    print("=== Diagnosis Summary ===")
    print(f"label: {diag['label']}")
    print(f"largest_box              = {diag['largest_box']:.1f}")
    print(f"rho_d_nucleus           = {diag['rho_dist_to_nucleus']:.6f}")
    print(f"rho_d_boundary          = {diag['rho_dist_to_boundary']:.6f}")
    print(f"rho_nearest_face        = {diag['rho_nearest_face']}")
    print(f"rho_on_boundary         = {diag['rho_on_boundary']}")
    print(f"rho_face_ratio          = {diag['rho_face_ratio']:.6e}")
    print(f"rho_face_gap            = {diag['rho_face_gap']:.6e}")
    print(f"psi_d_nucleus           = {diag['psi_dist_to_nucleus']:.6f}")
    print(f"psi_d_boundary          = {diag['psi_dist_to_boundary']:.6f}")
    print(f"psi_nearest_face        = {diag['psi_nearest_face']}")
    print(f"psi_on_boundary         = {diag['psi_on_boundary']}")
    print(f"psi_face_ratio          = {diag['psi_face_ratio']:.6e}")
    print(f"psi_face_gap            = {diag['psi_face_gap']:.6e}")
    print()
    print("Interpret conservatively:")
    print("- If the global maxima really sit on or very near a boundary face, finite-box orbital truncation is a real physical/numerical issue.")
    print("- If the global maxima stay well inside the box and near the nuclei while face/global ratio is still ~1, then the old ratio is more likely a statistic/indexing artifact or a flat plateau effect.")
    print("- If the largest-box maxima are still near the nuclei and far from the boundary, it becomes more reasonable to shift attention back toward Hartree/exterior or later kinetic/Laplacian questions.")


def main() -> int:
    print("=== H2 Orbital Localization / Boundary-Amplitude Sanity Study ===")
    print("Note: this is not a final benchmark.")
    print("Note: the current goal is to verify whether boundary/global ratio ~ 1 really means rho / |psi| are being pulled to the boundary.")
    print("Note: Hartree is fixed to multipole_dirichlet here to avoid mixing multiple boundary modes.")
    print(f"Setup: d={DISTANCE} Bohr, boxes={BOXES}")
    print(f"Adaptive params: h_min={H_MIN}, h_max={H_MAX}, r_core={R_CORE}, stretch_beta={STRETCH_BETA}")
    print(f"SCF: max_iter={SCF_KWARGS['max_iter']}, mix_alpha={SCF_KWARGS['mix_alpha']}, tolerance={SCF_KWARGS['tolerance']}")
    print()

    results = build_results()
    diagnosis = diagnose(results)

    print_result_table(results)
    print()
    print_localization_table(results, 'rho_max_info', 'rho')
    print()
    print_localization_table(results, 'psi_max_info', '|psi0|')
    print()
    print_boundary_amplitude_table(results)
    print()
    print_vh_boundary_table(results)
    print()
    print_center_slice_table(results)
    print()
    print_diagnosis(diagnosis)

    all_ok = True
    all_ok &= check(
        'runs_ok',
        all(row['completed'] and row['all_finite'] for row in results),
        'all H2 localization runs completed with finite outputs',
    )
    all_ok &= check(
        'electron_counts_ok',
        all((row['electron_error'] is not None and row['electron_error'] <= ELECTRON_TOL) for row in results),
        'electron counts stayed within tolerance',
    )
    all_ok &= check(
        'orbital_norms_ok',
        all((row['orbital_norm_maxdev'] is not None and row['orbital_norm_maxdev'] <= NORM_TOL) for row in results),
        'orbital norms stayed within tolerance',
    )

    if all_ok:
        print('OVERALL: PASS')
        return 0

    print('OVERALL: FAIL')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
