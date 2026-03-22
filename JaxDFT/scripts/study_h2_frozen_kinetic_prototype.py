"""H2 frozen-Veff kinetic prototype comparison.

This is not a final benchmark.
The goal is to compare the current adaptive kinetic operator against the new
symmetric finite-volume-style prototype under the same frozen effective
potential, and check whether the new kinetic stabilizes the low-energy H2
subspace across box sizes.

We freeze the effective potential from the largest-box reference SCF and then,
for each smaller box, solve the lowest k states with
  - kinetic_mode='prototype_fd2'
  - kinetic_mode='symmetric_fv'
under the same frozen V_eff.

The acceptance focus is not total energy. We compare:
  - matched_overlap_ref0_old/new
  - matched_psi_probe_rms_old/new
  - matched_rho_probe_rms_old/new
  - min_sigma_m2_old/new
  - projector_dist_m2_old/new
  - gap01_old/new
"""

from __future__ import annotations

import itertools
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
    from JaxDFT.src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import scf, solve_orbitals_subspace
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend
    from src.hamiltonian import build_local_potential as build_local_potential_pointwise
    from src.io import load_pseudopotentials
    from src.solver import scf, solve_orbitals_subspace


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
K_EIG = 4
M_PRIMARY = 2
FIXED_SOLVE_MAX_ITER = 16
FIXED_SOLVE_TOL = 1.0e-5
MAIN_PROBE_BOUND = 4.0
PROBE_SPACING = 0.3
NORM_TOL = 2.0e-2
SMALL = 1.0e-12
MODES = {
    'old': 'prototype_fd2',
    'new': 'symmetric_fv',
}


def fmt_float(value, width=11, precision=6):
    if value is None:
        return '-'.rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=12, precision=3):
    if value is None:
        return '-'.rjust(width)
    return f"{value:{width}.{precision}e}"


def check(name: str, condition: bool, detail: str) -> bool:
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {name}: {detail}')
    return condition


def build_occ(pseudos):
    n_electrons = float(jnp.sum(jnp.asarray([p['q'] for p in pseudos], dtype=jnp.float32)))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float32)
    rem = n_electrons
    for i in range(n_bands):
        val = min(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val
    return n_electrons, n_bands, occ


def build_probe_axis(bound: float, spacing: float) -> jnp.ndarray:
    n = int(np.floor(bound / spacing))
    return spacing * jnp.arange(-n, n + 1, dtype=jnp.float32)


def build_probe_points(bound: float, spacing: float):
    axis = build_probe_axis(bound, spacing)
    xx, yy, zz = jnp.meshgrid(axis, axis, axis, indexing='ij')
    return axis, jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def _sample_axis_trilinear(axis_values, coord):
    axis_np = np.asarray(axis_values, dtype=np.float64)
    coord_np = np.asarray(coord, dtype=np.float64)
    upper = np.searchsorted(axis_np, coord_np, side='right')
    lower = upper - 1
    lower = np.clip(lower, 0, axis_np.size - 2)
    upper = np.clip(upper, 1, axis_np.size - 1)
    x0 = axis_np[lower]
    x1 = axis_np[upper]
    denom = np.maximum(x1 - x0, SMALL)
    frac = np.clip((coord_np - x0) / denom, 0.0, 1.0)
    return lower, upper, frac


def sample_adaptive_trilinear(grid, field, target_points):
    field_np = np.asarray(jnp.asarray(field), dtype=np.float64)
    points_np = np.asarray(jnp.asarray(target_points), dtype=np.float64)
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    z = np.asarray(grid.z, dtype=np.float64)

    ix0, ix1, tx = _sample_axis_trilinear(x, points_np[:, 0])
    iy0, iy1, ty = _sample_axis_trilinear(y, points_np[:, 1])
    iz0, iz1, tz = _sample_axis_trilinear(z, points_np[:, 2])

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


def sample_field_to_grid(source_grid, field, target_grid):
    sampled = sample_adaptive_trilinear(source_grid, field, target_grid.coords.reshape(-1, 3))
    return sampled.reshape(target_grid.shape)


def field_metrics(current, reference):
    cur = jnp.asarray(current, dtype=jnp.float32).reshape(-1)
    ref = jnp.asarray(reference, dtype=jnp.float32).reshape(-1)
    diff = cur - ref
    return {
        'rms': float(jnp.sqrt(jnp.mean(diff * diff))),
    }


def normalize_field(backend, grid, psi):
    psi = jnp.asarray(psi, dtype=jnp.float32) * grid.mask
    norm = jnp.sqrt(jnp.maximum(backend.inner_product(grid, psi, psi), SMALL))
    psi = psi / norm
    psi = psi * grid.mask
    norm2 = jnp.sqrt(jnp.maximum(backend.inner_product(grid, psi, psi), SMALL))
    psi = psi / norm2
    return psi


def sample_state_block_to_grid(source_grid, state_block, target_grid, backend):
    sampled_states = []
    target_points = target_grid.coords.reshape(-1, 3)
    for state in state_block:
        sampled = sample_adaptive_trilinear(source_grid, state, target_points).reshape(target_grid.shape)
        sampled_states.append(normalize_field(backend, target_grid, sampled))
    return jnp.stack(sampled_states, axis=0)


def solve_fixed_spectrum(backend, grid, pseudos, coords, V_eff, n_states, key_seed, x_init_states=None):
    proj_data = backend.precompute_nonlocal(grid, coords, pseudos)

    def apply_h(psi_flat):
        psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
        psi = psi * grid.mask
        kinetic_psi = backend.apply_kinetic(grid, psi)
        v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
        hpsi = kinetic_psi + V_eff * psi + v_nonlocal
        hpsi = hpsi * grid.mask
        return hpsi.reshape(-1)

    if x_init_states is None:
        x_init = None
    else:
        x_init = jnp.moveaxis(jnp.asarray(x_init_states, dtype=jnp.float32), 0, -1).reshape(-1, n_states)

    eigvals, eigvecs = solve_orbitals_subspace(
        apply_h,
        int(np.prod(grid.shape)),
        n_states,
        x_init=x_init,
        max_iter=FIXED_SOLVE_MAX_ITER,
        tol=FIXED_SOLVE_TOL,
        key=jax.random.PRNGKey(key_seed),
        grid=grid,
        backend=backend,
    )
    states = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_states,)), -1, 0)
    states = jnp.stack([normalize_field(backend, grid, psi) for psi in states], axis=0)
    norm_dev = float(max(abs(float(backend.inner_product(grid, psi, psi)) - 1.0) for psi in states))
    return {
        'eigvals': np.asarray(eigvals, dtype=float),
        'states': states,
        'norm_dev': norm_dev,
    }


def compute_overlap_matrix(backend, grid, states_a, states_b):
    k_a = states_a.shape[0]
    k_b = states_b.shape[0]
    mat = np.zeros((k_a, k_b), dtype=np.float64)
    for i in range(k_a):
        for j in range(k_b):
            mat[i, j] = float(jnp.abs(backend.inner_product(grid, states_a[i], states_b[j])))
    return mat


def best_assignment(overlap_matrix):
    k = overlap_matrix.shape[0]
    best_perm = None
    best_score = -1.0
    best_overlaps = None
    for perm in itertools.permutations(range(k)):
        overlaps = [float(overlap_matrix[i, perm[i]]) for i in range(k)]
        score = float(sum(overlaps))
        if score > best_score:
            best_score = score
            best_perm = perm
            best_overlaps = overlaps
    inverse = [0] * k
    for i, j in enumerate(best_perm):
        inverse[j] = i
    return {
        'perm': best_perm,
        'current_index_for_ref0': int(inverse[0]),
        'overlap_for_ref0': float(overlap_matrix[inverse[0], 0]),
        'total_assignment_overlap': best_score / max(k, 1),
        'min_matched_overlap': min(best_overlaps) if best_overlaps else 0.0,
    }


def metric_gram(backend, grid, block_a, block_b):
    m = block_a.shape[0]
    n = block_b.shape[0]
    gram = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            gram[i, j] = float(backend.inner_product(grid, block_a[i], block_b[j]))
    return gram


def metric_orthonormalize_block(backend, grid, block):
    block = jnp.asarray(block, dtype=jnp.float32)
    flat = block.reshape(block.shape[0], -1)
    gram = metric_gram(backend, grid, block, block)
    gram = 0.5 * (gram + gram.T)
    evals, evecs = np.linalg.eigh(gram)
    evals = np.clip(evals, 1.0e-8, None)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    ortho_flat = inv_sqrt @ np.asarray(flat, dtype=np.float64)
    ortho = jnp.asarray(ortho_flat, dtype=jnp.float32).reshape(block.shape)
    ortho = jnp.stack([normalize_field(backend, grid, psi) for psi in ortho], axis=0)
    return ortho


def subspace_metrics(backend, grid, current_states, ref_states, m):
    cur = metric_orthonormalize_block(backend, grid, current_states[:m])
    ref = metric_orthonormalize_block(backend, grid, ref_states[:m])
    s = metric_gram(backend, grid, cur, ref)
    sigma = np.linalg.svd(s, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    min_sigma = float(np.min(sigma))
    projector_dist = float(np.sqrt(max(2.0 * m - 2.0 * float(np.sum(sigma * sigma)), 0.0)) / np.sqrt(2.0 * m))
    return {
        'min_sigma': min_sigma,
        'projector_dist': projector_dist,
    }


def build_reference(coords, pseudos):
    _, n_occ_bands, occ = build_occ(pseudos)
    ref_box = jnp.array([BOXES[-1], BOXES[-1], BOXES[-1]], dtype=jnp.float32)
    scf_backend = AdaptiveBackend(hartree_boundary_mode='uniform_exterior', kinetic_mode='prototype_fd2')
    ref_grid = scf_backend.create_grid(
        spacing=H_MIN,
        box_size=ref_box,
        atom_coords=coords,
        h_min=H_MIN,
        h_max=H_MAX,
        r_core=R_CORE,
        stretch_beta=STRETCH_BETA,
    )
    V_loc_ref = scf_backend.build_local_potential(ref_grid, coords, pseudos)
    rho_ref, eigvals_ref_occ, eigvecs_ref_occ, V_H_ref, eps_xc_ref, v_xc_ref = scf(
        ref_grid,
        coords,
        n_occ_bands,
        occ,
        V_loc_ref,
        pseudos,
        key=jax.random.PRNGKey(202),
        backend=scf_backend,
        **SCF_KWARGS,
    )
    _ = eigvals_ref_occ, eigvecs_ref_occ, eps_xc_ref
    V_eff_ref = V_loc_ref + V_H_ref + v_xc_ref

    ref_specs = {}
    for offset, (label, kinetic_mode) in enumerate(MODES.items()):
        backend = AdaptiveBackend(hartree_boundary_mode='uniform_exterior', kinetic_mode=kinetic_mode)
        ref_specs[label] = solve_fixed_spectrum(
            backend,
            ref_grid,
            pseudos,
            coords,
            V_eff_ref,
            K_EIG,
            key_seed=300 + offset,
            x_init_states=None,
        )
    return {
        'grid': ref_grid,
        'V_H_ref': V_H_ref,
        'v_xc_ref': v_xc_ref,
        'V_eff_ref': V_eff_ref,
        'ref_specs': ref_specs,
    }


def analyze_boxes(coords, pseudos, reference):
    _, probe_points = build_probe_points(MAIN_PROBE_BOUND, PROBE_SPACING)
    probe_points = jnp.asarray(probe_points, dtype=jnp.float32)
    ref_grid = reference['grid']
    ref_specs = reference['ref_specs']
    grid_builder = AdaptiveBackend()
    zion = jnp.asarray([p['zion'] for p in pseudos], dtype=jnp.float32)
    rloc = jnp.asarray([p['rloc'] for p in pseudos], dtype=jnp.float32)
    coeff = jnp.asarray([p['c'] for p in pseudos], dtype=jnp.float32)

    rows = []
    for box_idx, box_length in enumerate(BOXES[:-1]):
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
        V_loc_current = build_local_potential_pointwise(coords, grid.coords, zion, rloc, coeff)
        V_H_ref_on_current = sample_field_to_grid(ref_grid, reference['V_H_ref'], grid)
        v_xc_ref_on_current = sample_field_to_grid(ref_grid, reference['v_xc_ref'], grid)
        V_eff_frozen = V_loc_current + V_H_ref_on_current + v_xc_ref_on_current

        row = {'box': box_length}
        for offset, (label, kinetic_mode) in enumerate(MODES.items()):
            backend = AdaptiveBackend(hartree_boundary_mode='uniform_exterior', kinetic_mode=kinetic_mode)
            ref_states_on_current = sample_state_block_to_grid(ref_grid, ref_specs[label]['states'], grid, backend)
            spec = solve_fixed_spectrum(
                backend,
                grid,
                pseudos,
                coords,
                V_eff_frozen,
                K_EIG,
                key_seed=400 + 10 * box_idx + offset,
                x_init_states=ref_states_on_current,
            )
            overlap = compute_overlap_matrix(backend, grid, spec['states'], ref_states_on_current)
            match = best_assignment(overlap)
            matched_idx = match['current_index_for_ref0']
            ref_state0_probe = sample_adaptive_trilinear(ref_grid, jnp.abs(ref_specs[label]['states'][0]), probe_points)
            matched_probe = sample_adaptive_trilinear(grid, jnp.abs(spec['states'][matched_idx]), probe_points)
            ref_rho_probe = 2.0 * (ref_state0_probe ** 2)
            matched_rho_probe = 2.0 * (matched_probe ** 2)
            psi_m = field_metrics(matched_probe, ref_state0_probe)
            rho_m = field_metrics(matched_rho_probe, ref_rho_probe)
            m2 = subspace_metrics(backend, grid, spec['states'], ref_states_on_current, M_PRIMARY)
            row.update({
                f'eigvals_{label}': spec['eigvals'],
                f'gap01_{label}': float(spec['eigvals'][1] - spec['eigvals'][0]),
                f'matched_idx_{label}': matched_idx,
                f'matched_overlap_ref0_{label}': match['overlap_for_ref0'],
                f'matched_psi_probe_rms_{label}': psi_m['rms'],
                f'matched_rho_probe_rms_{label}': rho_m['rms'],
                f'min_sigma_m2_{label}': m2['min_sigma'],
                f'projector_dist_m2_{label}': m2['projector_dist'],
                f'norm_dev_{label}': spec['norm_dev'],
            })
        rows.append(row)
    return rows


def fmt_eigs(arr):
    vals = [f'{float(v):.4f}' for v in arr]
    return '[' + ', '.join(vals) + ']'


def print_table(rows):
    print('=== Frozen Kinetic Prototype Table ===')
    header = (
        f"{'Box':>5} {'gap01_old':>10} {'gap01_new':>10} {'ovlp_old':>10} {'ovlp_new':>10} "
        f"{'psi_old':>10} {'psi_new':>10} {'rho_old':>10} {'rho_new':>10} {'sig2_old':>10} {'sig2_new':>10} {'proj_old':>10} {'proj_new':>10}"
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{fmt_float(row['box'], 5, 1)} {fmt_float(row['gap01_old'], 10, 6)} {fmt_float(row['gap01_new'], 10, 6)} "
            f"{fmt_float(row['matched_overlap_ref0_old'], 10, 6)} {fmt_float(row['matched_overlap_ref0_new'], 10, 6)} "
            f"{fmt_sci(row['matched_psi_probe_rms_old'], 10, 2)} {fmt_sci(row['matched_psi_probe_rms_new'], 10, 2)} "
            f"{fmt_sci(row['matched_rho_probe_rms_old'], 10, 2)} {fmt_sci(row['matched_rho_probe_rms_new'], 10, 2)} "
            f"{fmt_float(row['min_sigma_m2_old'], 10, 6)} {fmt_float(row['min_sigma_m2_new'], 10, 6)} "
            f"{fmt_float(row['projector_dist_m2_old'], 10, 6)} {fmt_float(row['projector_dist_m2_new'], 10, 6)}"
        )
    print()
    print('Reference-adjusted eigvals old/new by box:')
    for row in rows:
        print(
            f"box={row['box']:.1f} old={fmt_eigs(row['eigvals_old'])} new={fmt_eigs(row['eigvals_new'])}"
        )


def summarize(rows):
    min_sigma_old = min(row['min_sigma_m2_old'] for row in rows)
    min_sigma_new = min(row['min_sigma_m2_new'] for row in rows)
    max_proj_old = max(row['projector_dist_m2_old'] for row in rows)
    max_proj_new = max(row['projector_dist_m2_new'] for row in rows)
    min_overlap_old = min(row['matched_overlap_ref0_old'] for row in rows)
    min_overlap_new = min(row['matched_overlap_ref0_new'] for row in rows)
    max_psi_old = max(row['matched_psi_probe_rms_old'] for row in rows)
    max_psi_new = max(row['matched_psi_probe_rms_new'] for row in rows)
    max_rho_old = max(row['matched_rho_probe_rms_old'] for row in rows)
    max_rho_new = max(row['matched_rho_probe_rms_new'] for row in rows)

    sigma_improved = min_sigma_new > min_sigma_old + 0.02
    proj_improved = max_proj_new < max_proj_old - 0.02
    overlap_improved = min_overlap_new > min_overlap_old + 0.02
    improvement_count = sum([sigma_improved, proj_improved, overlap_improved])

    if improvement_count >= 2:
        label = 'improved'
    elif improvement_count == 1:
        label = 'mixed'
    else:
        label = 'no-clear-gain'

    print('=== Summary ===')
    print(f'label: {label}')
    print(f'min_sigma_m2_old = {min_sigma_old:.6f}')
    print(f'min_sigma_m2_new = {min_sigma_new:.6f}')
    print(f'max_projector_dist_m2_old = {max_proj_old:.6f}')
    print(f'max_projector_dist_m2_new = {max_proj_new:.6f}')
    print(f'worst_matched_overlap_ref0_old = {min_overlap_old:.6f}')
    print(f'worst_matched_overlap_ref0_new = {min_overlap_new:.6f}')
    print(f'max_matched_psi_probe_rms_old = {max_psi_old:.6e}')
    print(f'max_matched_psi_probe_rms_new = {max_psi_new:.6e}')
    print(f'max_matched_rho_probe_rms_old = {max_rho_old:.6e}')
    print(f'max_matched_rho_probe_rms_new = {max_rho_new:.6e}')
    print(f'sigma_improved = {sigma_improved}')
    print(f'projector_dist_improved = {proj_improved}')
    print(f'matched_overlap_improved = {overlap_improved}')
    print(f'improvement_count = {improvement_count}')
    return {
        'label': label,
        'improvement_count': improvement_count,
        'sigma_improved': sigma_improved,
        'proj_improved': proj_improved,
        'overlap_improved': overlap_improved,
    }


def main():
    print('H2 frozen-Veff kinetic prototype comparison')
    print('This is not a final benchmark.')
    print('The goal is to compare old vs new adaptive kinetic operators under the same frozen Veff.')
    print()

    coords = jnp.array([[0.0, 0.0, -DISTANCE / 2.0], [0.0, 0.0, DISTANCE / 2.0]], dtype=jnp.float32)
    pseudo_dir = os.path.join(project_root, 'JaxDFT', 'data', 'gth_potentials')
    pseudos = load_pseudopotentials(['H', 'H'], pseudo_dir)
    reference = build_reference(coords, pseudos)
    rows = analyze_boxes(coords, pseudos, reference)
    print_table(rows)
    summary = summarize(rows)

    checks = []
    checks.append(check(
        'norms_reasonable',
        all(row['norm_dev_old'] <= NORM_TOL and row['norm_dev_new'] <= NORM_TOL for row in rows),
        f"max_norm_dev={max(max(row['norm_dev_old'], row['norm_dev_new']) for row in rows):.3e}",
    ))
    checks.append(check(
        'metrics_finite',
        all(np.isfinite(row['matched_overlap_ref0_old']) and np.isfinite(row['matched_overlap_ref0_new']) for row in rows),
        f"min_overlap={min(min(row['matched_overlap_ref0_old'], row['matched_overlap_ref0_new']) for row in rows):.6f}",
    ))
    checks.append(check(
        'summary_label_valid',
        summary['label'] in {'improved', 'mixed', 'no-clear-gain'},
        f"label={summary['label']}",
    ))

    overall = all(checks)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == '__main__':
    main()
