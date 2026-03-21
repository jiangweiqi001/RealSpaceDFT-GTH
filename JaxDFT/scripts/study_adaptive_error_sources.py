"""Study likely adaptive-backend error sources on very small systems.

This is not a final benchmark.
Adaptive monopole-Dirichlet Hartree is still not an exact isolated/open-
boundary treatment.
The goal here is to localize likely error sources (boundary conditions, box
size, or adaptive parameters), not to force adaptive results to match uniform or
PySCF one-for-one.

Default mode is intentionally lightweight:
- H_atom runs the full light matrix
- H2_fixed runs a reduced set
- two boxes by default
- adaptive parameters default to uniformlike + mild

Use ``--full`` to expand to a somewhat richer matrix.
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
from pyscf import dft, gto

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from JaxDFT.src.backends import AdaptiveBackend, UniformBackend
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.solver import ion_ion_energy, scf, total_energy
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src.backends import AdaptiveBackend, UniformBackend
    from src.io import load_pseudopotentials
    from src.solver import ion_ion_energy, scf, total_energy


SCF_KWARGS = {
    "max_iter": 2,
    "mix_alpha": 0.25,
    "tolerance": 1.0e-3,
}
ELECTRON_TOL = 5.0e-3
NORM_TOL = 2.0e-2


def fmt_float(value, width=10, precision=6):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_sci(value, width=10, precision=3):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}e}"


def fmt_text(value, width=12):
    text = "-" if value is None else str(value)
    return text[:width].ljust(width)


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


def make_case_specs(full: bool):
    if full:
        return [
            {
                "name": "H_atom",
                "elements": ["H"],
                "coords": jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
                "uniform_boxes": [4.0, 5.0, 6.0],
                "uniform_spacings": [1.0, 0.8, 0.6],
                "adaptive_boxes": [4.0, 5.0, 6.0],
                "adaptive_spacing": 1.0,
                "adaptive_params": [
                    {"name": "uniformlike", "h_max": 1.0, "r_core": 0.5, "stretch_beta": 0.0},
                    {"name": "mild", "h_max": 1.2, "r_core": 0.5, "stretch_beta": 3.0},
                    {"name": "medium", "h_max": 1.4, "r_core": 0.5, "stretch_beta": 4.0},
                ],
                "pyscf_spin": 1,
                "pyscf_restricted": False,
            },
            {
                "name": "H2_fixed",
                "elements": ["H", "H"],
                "coords": jnp.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float32),
                "uniform_boxes": [5.0, 6.0, 7.0],
                "uniform_spacings": [1.0, 0.8],
                "adaptive_boxes": [5.0, 6.0, 7.0],
                "adaptive_spacing": 1.0,
                "adaptive_params": [
                    {"name": "uniformlike", "h_max": 1.0, "r_core": 0.7, "stretch_beta": 0.0},
                    {"name": "mild", "h_max": 1.2, "r_core": 0.7, "stretch_beta": 3.0},
                    {"name": "medium", "h_max": 1.4, "r_core": 0.7, "stretch_beta": 4.0},
                ],
                "pyscf_spin": 0,
                "pyscf_restricted": True,
            },
        ]

    return [
        {
            "name": "H_atom",
            "elements": ["H"],
            "coords": jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            "uniform_boxes": [4.0, 6.0],
            "uniform_spacings": [1.0, 0.8, 0.6],
            "adaptive_boxes": [4.0, 6.0],
            "adaptive_spacing": 1.0,
            "adaptive_params": [
                {"name": "uniformlike", "h_max": 1.0, "r_core": 0.5, "stretch_beta": 0.0},
                {"name": "mild", "h_max": 1.2, "r_core": 0.5, "stretch_beta": 3.0},
            ],
            "pyscf_spin": 1,
            "pyscf_restricted": False,
        },
        {
            "name": "H2_fixed",
            "elements": ["H", "H"],
            "coords": jnp.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float32),
            "uniform_boxes": [5.0, 7.0],
            "uniform_spacings": [1.0, 0.8],
            "adaptive_boxes": [5.0, 7.0],
            "adaptive_spacing": 1.0,
            "adaptive_params": [
                {"name": "uniformlike", "h_max": 1.0, "r_core": 0.7, "stretch_beta": 0.0},
                {"name": "mild", "h_max": 1.2, "r_core": 0.7, "stretch_beta": 3.0},
            ],
            "pyscf_spin": 0,
            "pyscf_restricted": True,
        },
    ]


def prepare_case(case_spec):
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudos = load_pseudopotentials(case_spec["elements"], pseudo_dir)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    n_electrons, n_bands, occ = build_occ(pseudos)
    return {
        "pseudos": pseudos,
        "zion": zion,
        "n_electrons": n_electrons,
        "n_bands": n_bands,
        "occ": occ,
    }


def eigvals_repr(eigvals):
    if eigvals is None:
        return None
    arr = [float(x) for x in jnp.asarray(eigvals).reshape(-1)]
    return "[" + ", ".join(f"{x:.4f}" for x in arr[:3]) + ("]" if len(arr) <= 3 else ", ...]")


def run_grid_case(case_spec, case_data, backend, label, box_length, spacing, key_seed, adaptive_kwargs=None):
    result = {
        "case": case_spec["name"],
        "label": label,
        "backend": backend.name,
        "hartree_mode": getattr(backend, "hartree_boundary_mode", None),
        "box": float(box_length),
        "spacing": float(spacing),
        "param_name": None if adaptive_kwargs is None else adaptive_kwargs.get("name"),
        "completed": False,
        "all_finite": False,
        "energy": None,
        "electron_count": None,
        "electron_error": None,
        "orbital_norm_maxdev": None,
        "eig0": None,
        "eigvals_str": None,
        "shape": None,
        "rho_min": None,
        "rho_max": None,
        "error": None,
    }

    try:
        box = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
        coords = case_spec["coords"]
        if backend.name == "uniform":
            grid = backend.create_grid(float(spacing), box)
        else:
            kwargs = dict(adaptive_kwargs)
            kwargs.pop("name", None)
            grid = backend.create_grid(float(spacing), box, atom_coords=coords, **kwargs)

        V_loc = backend.build_local_potential(grid, coords, case_data["pseudos"])
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
            grid,
            coords,
            case_data["n_bands"],
            case_data["occ"],
            V_loc,
            case_data["pseudos"],
            key=jax.random.PRNGKey(key_seed),
            backend=backend,
            **SCF_KWARGS,
        )
        ion_e = ion_ion_energy(coords, case_data["zion"])
        energy = total_energy(
            rho,
            eigvals,
            case_data["occ"],
            V_loc,
            V_H,
            eps_xc,
            v_xc,
            grid,
            ion_e,
            backend=backend,
        )
        eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (case_data["n_bands"],)), -1, 0)
        norms = jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields)
        electron_count = float(backend.integrate(grid, rho))
        norm_maxdev = float(jnp.max(jnp.abs(norms - 1.0)))
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
            "electron_error": abs(electron_count - float(jnp.sum(case_data["occ"]))),
            "orbital_norm_maxdev": norm_maxdev,
            "eig0": float(jnp.asarray(eigvals).reshape(-1)[0]),
            "eigvals_str": eigvals_repr(eigvals),
            "shape": tuple(int(n) for n in grid.shape),
            "rho_min": float(jnp.min(rho)),
            "rho_max": float(jnp.max(rho)),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def run_pyscf_reference(case_spec):
    result = {
        "case": case_spec["name"],
        "completed": False,
        "energy": None,
        "eig0": None,
        "eigvals_str": None,
        "error": None,
    }
    try:
        atom = "; ".join(
            f"{element} {float(coord[0]):.8f} {float(coord[1]):.8f} {float(coord[2]):.8f}"
            for element, coord in zip(case_spec["elements"], case_spec["coords"])
        )
        mol = gto.M(
            atom=atom,
            unit="Bohr",
            basis="gth-tzvp",
            pseudo="gth-lda",
            spin=int(case_spec["pyscf_spin"]),
            verbose=0,
        )
        mf = dft.RKS(mol) if case_spec["pyscf_restricted"] else dft.UKS(mol)
        mf.xc = "lda,pz"
        mf.conv_tol = 1e-10
        mf.max_cycle = 200
        energy = mf.kernel()
        mo_energy = jnp.asarray(mf.mo_energy)
        flat = jnp.ravel(mo_energy)
        result.update({
            "completed": True,
            "energy": float(energy),
            "eig0": float(flat[0]) if flat.size else None,
            "eigvals_str": eigvals_repr(flat),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_uniform_results(case_spec, case_data, full, key_counter):
    results = []
    base_spacing = case_spec["uniform_spacings"][0]
    small_box = case_spec["uniform_boxes"][0]
    large_box = case_spec["uniform_boxes"][-1]

    results.append(run_grid_case(case_spec, case_data, UniformBackend(), "uniform_box_small_base", small_box, base_spacing, key_counter[0]))
    key_counter[0] += 1

    for spacing in case_spec["uniform_spacings"]:
        label = f"uniform_box_large_dx{spacing:.2f}"
        results.append(run_grid_case(case_spec, case_data, UniformBackend(), label, large_box, spacing, key_counter[0]))
        key_counter[0] += 1

    if full and len(case_spec["uniform_boxes"]) > 2:
        mid_box = case_spec["uniform_boxes"][1]
        results.append(run_grid_case(case_spec, case_data, UniformBackend(), "uniform_box_mid_base", mid_box, base_spacing, key_counter[0]))
        key_counter[0] += 1

    return results


def build_adaptive_results(case_spec, case_data, full, key_counter):
    results = []
    seen = set()
    spacing = case_spec["adaptive_spacing"]
    boxes = case_spec["adaptive_boxes"]
    params = case_spec["adaptive_params"]

    def add_run(boundary_mode, box_length, param):
        key = (boundary_mode, float(box_length), param["name"])
        if key in seen:
            return
        seen.add(key)
        backend = AdaptiveBackend(hartree_boundary_mode=boundary_mode)
        label = f"adaptive_{boundary_mode}_{param['name']}_L{box_length:.1f}"
        results.append(run_grid_case(case_spec, case_data, backend, label, box_length, spacing, key_counter[0], adaptive_kwargs=param))
        key_counter[0] += 1

    uniformlike = params[0]
    for box_length in boxes:
        add_run("zero_dirichlet", box_length, uniformlike)
        add_run("monopole_dirichlet", box_length, uniformlike)

    large_box = boxes[-1]
    for param in params:
        add_run("monopole_dirichlet", large_box, param)

    if full and len(params) > 2:
        add_run("zero_dirichlet", large_box, params[1])

    return results


def index_results(rows, keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def summarize_case(case_spec, reference, uniform_rows, adaptive_rows):
    base_spacing = case_spec["uniform_spacings"][0]
    large_box = case_spec["uniform_boxes"][-1]
    small_box = case_spec["uniform_boxes"][0]
    uniform_index = index_results(uniform_rows, ["box", "spacing", "label"])
    adaptive_index = index_results(adaptive_rows, ["hartree_mode", "box", "param_name"])

    metrics = {
        "case": case_spec["name"],
        "pyscf_energy": reference["energy"],
        "uniform_box_drift": None,
        "uniform_spacing_drift": None,
        "adaptive_boundary_delta": None,
        "adaptive_box_drift_zero": None,
        "adaptive_box_drift_mono": None,
        "adaptive_param_delta": None,
        "likely_source": "inconclusive",
    }

    try:
        small_row = next(row for row in uniform_rows if row["box"] == float(small_box) and row["spacing"] == float(base_spacing))
        large_base_row = next(row for row in uniform_rows if row["box"] == float(large_box) and row["spacing"] == float(base_spacing))
        if small_row["energy"] is not None and large_base_row["energy"] is not None:
            metrics["uniform_box_drift"] = abs(small_row["energy"] - large_base_row["energy"])
    except StopIteration:
        pass

    large_uniform = [row for row in uniform_rows if row["box"] == float(large_box) and row["energy"] is not None]
    if len(large_uniform) >= 2:
        energies = [row["energy"] for row in large_uniform]
        metrics["uniform_spacing_drift"] = max(energies) - min(energies)

    try:
        zero_small = adaptive_index[("zero_dirichlet", float(small_box), "uniformlike")]
        mono_small = adaptive_index[("monopole_dirichlet", float(small_box), "uniformlike")]
        zero_large = adaptive_index[("zero_dirichlet", float(large_box), "uniformlike")]
        mono_large = adaptive_index[("monopole_dirichlet", float(large_box), "uniformlike")]
        if all(row["energy"] is not None for row in (zero_small, mono_small, zero_large, mono_large)):
            metrics["adaptive_boundary_delta"] = max(
                abs(zero_small["energy"] - mono_small["energy"]),
                abs(zero_large["energy"] - mono_large["energy"]),
            )
            metrics["adaptive_box_drift_zero"] = abs(zero_small["energy"] - zero_large["energy"])
            metrics["adaptive_box_drift_mono"] = abs(mono_small["energy"] - mono_large["energy"])
    except KeyError:
        pass

    mono_params = [
        row for row in adaptive_rows
        if row["hartree_mode"] == "monopole_dirichlet" and row["box"] == float(large_box) and row["energy"] is not None
    ]
    if len(mono_params) >= 2:
        energies = [row["energy"] for row in mono_params]
        metrics["adaptive_param_delta"] = max(energies) - min(energies)

    boundary_metric = metrics["adaptive_boundary_delta"] or -1.0
    box_metric = max(
        [value for value in (metrics["uniform_box_drift"], metrics["adaptive_box_drift_mono"], metrics["adaptive_box_drift_zero"]) if value is not None] or [-1.0]
    )
    param_metric = metrics["adaptive_param_delta"] if metrics["adaptive_param_delta"] is not None else -1.0

    if boundary_metric >= max(box_metric, param_metric) and metrics["adaptive_box_drift_mono"] is not None and metrics["adaptive_box_drift_zero"] is not None and metrics["adaptive_box_drift_mono"] < metrics["adaptive_box_drift_zero"]:
        metrics["likely_source"] = "boundary-dominated"
    elif box_metric >= max(boundary_metric, param_metric):
        metrics["likely_source"] = "box-dominated"
    elif param_metric >= max(boundary_metric, box_metric):
        metrics["likely_source"] = "adaptive-parameter-dominated"
    else:
        metrics["likely_source"] = "mixed"

    return metrics


def print_reference_table(rows):
    print("=== PySCF Reference Table ===")
    header = f"{'Case':<10} {'Done':<5} {'Energy':>12} {'eig0':>12} {'eigvals':<24}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<10} {fmt_text(row['completed'], 5)} {fmt_float(row['energy'], 12, 6)} "
            f"{fmt_float(row['eig0'], 12, 6)} {fmt_text(row['eigvals_str'], 24)}"
        )
        if row["error"] is not None:
            print(f"  error: {row['error']}")
    print()


def print_uniform_table(rows, reference_by_case):
    print("=== Uniform Study Table ===")
    header = (
        f"{'Case':<10} {'Label':<24} {'Box':>5} {'dx':>6} {'Energy':>12} {'dE_PySCF':>12} "
        f"{'N':>8} {'NormDev':>10} {'eig0':>12} {'Shape':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ref = reference_by_case.get(row["case"])
        dE = None if ref is None or ref["energy"] is None or row["energy"] is None else row["energy"] - ref["energy"]
        print(
            f"{row['case']:<10} {row['label']:<24} {fmt_float(row['box'], 5, 1)} {fmt_float(row['spacing'], 6, 2)} "
            f"{fmt_float(row['energy'], 12, 6)} {fmt_float(dE, 12, 6)} {fmt_float(row['electron_count'], 8, 4)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 10, 2)} {fmt_float(row['eig0'], 12, 6)} {fmt_text(row['shape'], 12)}"
        )
        if row['error'] is not None:
            print(f"  error: {row['error']}")
    print()


def print_adaptive_boundary_box_table(rows, reference_by_case):
    print("=== Adaptive Boundary/Box Table ===")
    header = (
        f"{'Case':<10} {'Boundary':<20} {'Box':>5} {'Param':<12} {'Energy':>12} {'dE_PySCF':>12} "
        f"{'N':>8} {'NormDev':>10} {'eig0':>12} {'Shape':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ref = reference_by_case.get(row["case"])
        dE = None if ref is None or ref["energy"] is None or row["energy"] is None else row["energy"] - ref["energy"]
        print(
            f"{row['case']:<10} {fmt_text(row['hartree_mode'], 20)} {fmt_float(row['box'], 5, 1)} {fmt_text(row['param_name'], 12)} "
            f"{fmt_float(row['energy'], 12, 6)} {fmt_float(dE, 12, 6)} {fmt_float(row['electron_count'], 8, 4)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 10, 2)} {fmt_float(row['eig0'], 12, 6)} {fmt_text(row['shape'], 12)}"
        )
        if row['error'] is not None:
            print(f"  error: {row['error']}")
    print()


def print_adaptive_param_table(rows, reference_by_case):
    print("=== Adaptive Parameter Table ===")
    header = (
        f"{'Case':<10} {'Boundary':<20} {'Box':>5} {'Param':<12} {'Energy':>12} {'dE_PySCF':>12} "
        f"{'N':>8} {'NormDev':>10} {'eig0':>12} {'Shape':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ref = reference_by_case.get(row["case"])
        dE = None if ref is None or ref["energy"] is None or row["energy"] is None else row["energy"] - ref["energy"]
        print(
            f"{row['case']:<10} {fmt_text(row['hartree_mode'], 20)} {fmt_float(row['box'], 5, 1)} {fmt_text(row['param_name'], 12)} "
            f"{fmt_float(row['energy'], 12, 6)} {fmt_float(dE, 12, 6)} {fmt_float(row['electron_count'], 8, 4)} "
            f"{fmt_sci(row['orbital_norm_maxdev'], 10, 2)} {fmt_float(row['eig0'], 12, 6)} {fmt_text(row['shape'], 12)}"
        )
        if row['error'] is not None:
            print(f"  error: {row['error']}")
    print()


def print_diagnosis_table(rows):
    print("=== Diagnosis Summary ===")
    header = (
        f"{'Case':<10} {'u_box':>10} {'u_dx':>10} {'a_boundary':>12} {'a_box_zero':>12} "
        f"{'a_box_mono':>12} {'a_param':>10} {'Likely Source':<28}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<10} {fmt_float(row['uniform_box_drift'], 10, 6)} {fmt_float(row['uniform_spacing_drift'], 10, 6)} "
            f"{fmt_float(row['adaptive_boundary_delta'], 12, 6)} {fmt_float(row['adaptive_box_drift_zero'], 12, 6)} "
            f"{fmt_float(row['adaptive_box_drift_mono'], 12, 6)} {fmt_float(row['adaptive_param_delta'], 10, 6)} {row['likely_source']:<28}"
        )
    print()


def print_notes(args):
    print("=== Adaptive Error-Source Study ===")
    print("Note: this is not a final benchmark.")
    print("Note: adaptive monopole-Dirichlet Hartree is still not an exact isolated/open-boundary treatment.")
    print("Note: the current goal is to localize likely error sources, not to force adaptive to match uniform or PySCF one-for-one.")
    print(f"Mode: {'full' if args.full else 'default-light'}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Study likely adaptive-backend error sources on very small systems.")
    parser.add_argument("--full", action="store_true", help="Run a somewhat richer matrix instead of the lightweight default.")
    args = parser.parse_args(argv)

    print_notes(args)
    case_specs = make_case_specs(args.full)
    key_counter = [0]

    reference_rows = []
    uniform_rows = []
    adaptive_rows = []

    for case_spec in case_specs:
        print(f"--- Running {case_spec['name']} ---")
        case_data = prepare_case(case_spec)

        ref = run_pyscf_reference(case_spec)
        reference_rows.append(ref)
        print(f"  PySCF: completed={ref['completed']} energy={ref['energy']} eig0={ref['eig0']}")

        case_uniform = build_uniform_results(case_spec, case_data, args.full, key_counter)
        uniform_rows.extend(case_uniform)
        print(f"  uniform runs: {len(case_uniform)}")

        case_adaptive = build_adaptive_results(case_spec, case_data, args.full, key_counter)
        adaptive_rows.extend(case_adaptive)
        print(f"  adaptive runs: {len(case_adaptive)}")
        print()

    reference_by_case = {row['case']: row for row in reference_rows}

    boundary_box_rows = [
        row for row in adaptive_rows
        if row['param_name'] == 'uniformlike'
    ]
    param_rows = []
    for case_spec in case_specs:
        large_box = float(case_spec['adaptive_boxes'][-1])
        param_rows.extend([
            row for row in adaptive_rows
            if row['hartree_mode'] == 'monopole_dirichlet' and row['box'] == large_box
        ])

    diagnoses = [
        summarize_case(
            case_spec,
            reference_by_case.get(case_spec['name']),
            [row for row in uniform_rows if row['case'] == case_spec['name']],
            [row for row in adaptive_rows if row['case'] == case_spec['name']],
        )
        for case_spec in case_specs
    ]

    print_reference_table(reference_rows)
    print_uniform_table(uniform_rows, reference_by_case)
    print_adaptive_boundary_box_table(boundary_box_rows, reference_by_case)
    print_adaptive_param_table(param_rows, reference_by_case)
    print_diagnosis_table(diagnoses)

    all_grid_ok = all(
        row['completed'] and row['all_finite'] and row['electron_error'] is not None and row['electron_error'] <= ELECTRON_TOL and row['orbital_norm_maxdev'] is not None and row['orbital_norm_maxdev'] <= NORM_TOL
        for row in uniform_rows + adaptive_rows
    )
    refs_ok = all(row['completed'] and row['energy'] is not None for row in reference_rows)

    print("=== Sanity Checks ===")
    ok = True
    ok &= check('pyscf_refs_ok', refs_ok, 'PySCF references completed and returned finite energies')
    ok &= check('grid_runs_ok', all_grid_ok, 'uniform/adaptive runs completed with finite outputs and reasonable normalization')

    print()
    print("=== Interpretation Guide ===")
    print("- Compare boundary-mode deltas against box and parameter drifts; a large boundary delta points to Hartree boundary conditions as a dominant error source.")
    print("- If monopole box drift is much smaller than zero-Dirichlet box drift, the boundary upgrade is helping even if absolute energies still differ.")
    print("- Uniform drifts provide a baseline for how much plain box/spacing error already exists before blaming adaptive settings.")
    print("- PySCF is only an external anchor here; the diagnosis summary is based primarily on uniform/adaptive drift patterns.")

    if ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
