# RealSpaceDFT-GTH

RealSpaceDFT-GTH is a prototype real-space Kohn-Sham DFT implementation using GTH pseudopotentials. It is intended for algorithm development, diagnostics, and small benchmark experiments, not as a production DFT package.

The current code focuses on energy-only isolated molecular calculations on uniform Cartesian grids. Forces are not implemented: `energy_and_forces(...)` still returns `(energy, forces)`, but `forces` is a zero placeholder with the same shape as the input coordinates.

For algorithm details, see [ALGORITHM.md](ALGORITHM.md). For current handoff status and next debugging steps, see [docs/STATUS_AND_HANDOFF.md](docs/STATUS_AND_HANDOFF.md).

## Current Status

- H2 reaches the current target error range against the fixed PySCF reference protocol.
- CO and H2O still show systematic roughly `15-25 mHa` total-energy bias in current practical runs. The source is not fully explained.
- **Long-term direction**: a **mature coarse-to-fine** workflow (density continuation + interpolation warm-start) so **CHON**-class systems can be run **accurate and fast** on fine grids; see [docs/STATUS_AND_HANDOFF.md](docs/STATUS_AND_HANDOFF.md) and [AGENTS.md](AGENTS.md).
- **Gaussian Poisson** and **same-density** diagnostics show Poisson/XC are well behaved at fixed density; Hartree vs PySCF `coul` gaps **shrink with finer spacing**; total SCF error **trends down** with spacing when iteration budgets allow.
- CO/H2O SCF results should usually be treated as practical plateaus or partial convergence unless the energy-stability diagnostics pass.
- The default public API remains `energy, forces = energy_and_forces(...)`. `return_info=True` includes **`density`** for continuation.
- The default density mixer remains Anderson mixing.
- Pulay/DIIS, Kerker-style residual metrics, RMS/L2 convergence metrics, and kinetic orbital preconditioning are optional experimental controls, not the default path.
- The benchmark harness can report PySCF reference components and JaxDFT local/nonlocal/Hartree components for diagnostics. **Continuation benchmark**: `JaxDFT/scripts/scf_continuation_benchmark.py`.

## Known Limitations

- Forces are zero placeholders and are out of scope for the current optimization phase.
- CO/H2O are not yet within the `<=10 mHa` first-stage target.
- SCF strict convergence is not guaranteed for C/O systems; report `energy_delta_last10_max`, density residuals, and ideally last-window statistics when comparing runs.
- The remaining CO/H2O bias is likely in component-level numerical physics: Hartree/Poisson, local GTH potential convention/discretization, or remaining nonlocal operator details.
- This repository should not be described as production-ready or high-accuracy DFT software.

## Quick Start

```python
import jax
import jax.numpy as jnp

from JaxDFT.src.hamiltonian import create_grid
from JaxDFT.src.io import load_pseudopotentials
from JaxDFT.src.solver import energy_and_forces

spacing = 0.18
box_size = [18.0, 18.0, 18.0]
grid = create_grid(spacing, box_size)

pseudos = load_pseudopotentials(["H", "H"], "JaxDFT/data/gth_potentials")
coords = jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32)

energy, forces = energy_and_forces(
    grid,
    coords,
    pseudos,
    max_iter=120,
    mix_alpha=0.3,
    tolerance=1e-5,
    key=jax.random.PRNGKey(0),
)

print(float(energy))
print(forces)  # zero placeholder, not physical forces
```

For diagnostics without changing the default return shape:

```python
energy, forces, info = energy_and_forces(
    grid,
    coords,
    pseudos,
    max_iter=120,
    mix_alpha=0.3,
    tolerance=1e-5,
    key=jax.random.PRNGKey(0),
    return_info=True,
)

print(info["density_rms_diff"])
print(info["energy_delta_last10_max"])
print(info["energy_components"])
```

## Benchmarks

The reproducible benchmark harness is:

```bash
python3 JaxDFT/scripts/benchmark_systems.py --systems H2 H2O CO --spacings 0.18 0.12 --box-size 18
```

Current recommended diagnostic configuration for CO/H2O stability checks:

```bash
python3 JaxDFT/scripts/benchmark_systems.py \
  --systems CO \
  --spacings 0.12 \
  --box-size 18 \
  --max-iter 200 \
  --mixing-mode pulay \
  --mix-alpha 0.2 \
  --anderson-history 3 \
  --anderson-regularization 1e-4 \
  --pulay-residual-metric kerker \
  --pulay-kerker-k0 2.0 \
  --scf-convergence-metric rms \
  --tolerance 3e-5 \
  --orbital-max-iter 30 \
  --orbital-tolerance 1e-5 \
  --orbital-preconditioner none \
  --json
```

This is a diagnostic reference configuration, not a new default algorithm.

## Development Notes

- Prefer reading the current code and benchmark output over relying on old comments or historical verification scripts.
- **Continuation first**: coarse-to-fine and reproducible continuation chains are the main lever for fine-grid cost and stability; mixer sweeps are secondary.
- Before changing speed/JIT/caching paths, first validate Hartree/Poisson and local GTH potential components independently and document continuation baselines.
