# RealSpaceDFT-GTH Algorithm Notes

This document describes the current implementation, diagnostics, and known limitations. It is intentionally conservative: the code is a real-space Kohn-Sham DFT prototype with GTH pseudopotentials, not a production DFT package.

## Scope

- Target calculations: small isolated molecular systems on uniform Cartesian grids.
- Current primary output: total energy.
- Current API: `energy_and_forces(...)` returns `(energy, forces)` by default.
- Forces are not implemented. The returned `forces` array is a zero placeholder and is not part of the current algorithmic validation loop.

## Real-Space Grid

The SCF path uses a uniform Cartesian real-space grid created by `create_grid(spacing, box_size)`.

- Coordinates are centered around the origin.
- The cell volume is `dV = spacing**3`.
- Optional diagnostic controls exist for `dtype` and `grid_phase`.
- The grid remains uniform; true nonuniform adaptive Kohn-Sham grids are not used.

## Density continuation (coarse-to-fine)

For expensive fine grids, the intended workflow is **SCF on a coarser spacing**, then **interpolate the self-consistent density** onto a finer grid (trilinear remap + electron-count renorm), and call `energy_and_forces(..., initial_rho=...)` or `scf(..., initial_rho=...)` as a warm start. Implementation: `JaxDFT/src/continuation.py`; orchestration and timing: `JaxDFT/scripts/scf_continuation_benchmark.py`.

This is the **primary strategic path** toward **accurate and fast** runs for **CHON**-class benchmarks once multi-stage chains and harness coverage are mature. It does not replace independent validation of Hartree/local components at each spacing.

## Kinetic Operator

The default SCF path uses the 8th-order centered finite-difference Laplacian. The code also contains 4th- and 6th-order stencils for controlled diagnostics.

The kinetic contribution applied to an orbital is:

```text
T psi = -0.5 * laplacian(psi)
```

Boundary behavior follows the current finite-difference implementation and grid masking. Do not describe this as a fully validated strict Dirichlet production solver.

## Potentials

The effective potential is:

```text
V_eff = V_local_GTH + V_Hartree + V_xc
```

### GTH Local Potential

The local GTH pseudopotential is evaluated directly on the real-space grid. Current diagnostics expose:

- local pseudopotential energy
- local pseudopotential energy by atom
- local potential min/max
- local potential integral
- local potential integral by atom

The local GTH formula, `r=0` handling, `erf` convention, and polynomial convention still need independent validation against a trusted reference.

### GTH Nonlocal Projectors

The nonlocal GTH pseudopotential is applied through real-space projector overlaps. Current diagnostics include:

- nonlocal pseudopotential energy
- projector overlap matrices
- projector overlap error versus identity-like normalization checks

C/O s-channel projector overlap normalization has been observed at roughly `1e-6` error and is unlikely to explain the remaining CO/H2O total-energy bias. Full angular and p-channel validation, especially for N and other systems, is not complete.

### Hartree / Poisson

The Hartree potential is computed with an FFT zero-padding Poisson/Hockney-style convolution path.

Current diagnostics expose:

- Hartree energy `0.5 * integral rho(r) V_H(r) dr`
- Hartree potential min/max
- Hartree potential integral

The current CO component diagnostics show a Hartree/Coulomb difference relative to PySCF on the order of tens of mHa, but that comparison is not same-density. A same-density or analytic Gaussian Poisson test is required before changing the Poisson implementation.

### Exchange-Correlation

The implemented exchange-correlation model is LDA/PZ81:

- Slater exchange
- Perdew-Zunger 1981 correlation

Benchmark references use PySCF with `gth-tzvp + gth-lda + lda,pz`.

## Orbital Solver

The default SCF orbital solve uses a block subspace expansion with Rayleigh-Ritz projection. Historical names may mention LOBPCG, but the current default path should not be described as a strict textbook LOBPCG implementation.

Optional diagnostics/controls include:

- orbital residuals
- orbital iteration counts
- eigenvalues
- occupations
- optional kinetic orbital preconditioner

Experiments so far indicate that simply increasing `orbital_max_iter` is not enough to remove the CO/H2O systematic bias.

## SCF Mixing

Default density mixing remains Anderson mixing.

Optional experimental paths:

- linear mixing
- Pulay/DIIS-style mixing
- charge-neutral Pulay residual
- Kerker-style residual metric for Pulay
- density stabilization and normalization

These optional paths improve diagnostics and sometimes plateau stability, but they are not the default algorithm and have not removed the systematic CO/H2O energy bias.

## Energy Components

Diagnostics and energy bookkeeping expose the following components:

- band/eigenvalue sum
- kinetic contribution indirectly through band-energy bookkeeping
- local pseudopotential energy
- nonlocal pseudopotential energy
- Hartree energy
- exchange-correlation energy
- exchange-correlation potential correction
- ion-ion energy

The total energy is computed in the usual Kohn-Sham eigenvalue-sum form:

```text
E_total = E_band - E_Hartree + E_xc - E_vxc + E_ion-ion
```

The local and nonlocal pseudopotential component outputs are diagnostics for error attribution; they are not separate terms in this eigenvalue-sum expression because they are already included in `E_band`.

## Convergence Diagnostics

`return_info=True` exposes convergence diagnostics including:

- max density residual
- RMS density residual
- L2 density residual
- `energy_delta_last`
- `energy_delta_history`
- `energy_delta_last10_max`
- `density_converged`
- `energy_converged`
- `scf_converged`
- orbital residuals and orbital convergence flag

Important interpretation:

- `density_converged` depends on the selected density residual metric.
- `energy_converged` depends on the configured energy tolerance (Hartree): the maximum of `|E_i - E_{i-1}|` over the last 10 SCF iterations must be at most `energy_tolerance`. Default in code is `5e-6` Ha; difficult CO/H2O fine-grid runs often show `energy_delta_last10_max` around `1e-3` Ha at practical iteration caps, so this flag can remain false until the run is tighter or the tolerance is loosened deliberately for reporting.
- `scf_converged` is a diagnostic combination of density and energy stability.
- Current CO/H2O runs often represent practical plateaus or partial convergence, not strict SCF convergence.

## Benchmark Harness

The fixed benchmark systems are:

- H2, `d=1.4 Bohr`
- H2O, `O-H=1.8 Bohr`, angle `104.5 deg`
- CO, `d=2.132 Bohr`

The fixed reference protocol is PySCF:

```text
gth-tzvp + gth-lda + lda,pz
```

The benchmark harness can report:

- JaxDFT energy
- PySCF energy
- error in mHa
- spacing and grid shape
- runtime
- SCF and orbital diagnostics
- JaxDFT local/nonlocal/Hartree components
- PySCF one-electron/Coulomb/XC/nuclear components

## Current Interpretation

- H2 is within the current target error range.
- CO/H2O still have systematic `15-25 mHa`-level bias in many practical runs; **same-density** diagnostics show **XC** matches PySCF very closely and **Hartree** vs PySCF `coul` gaps **shrink with finer spacing**; **Gaussian Poisson** tests validate the Poisson path at sub-mHa to ~1 mHa on scanned grids.
- **Coarse-to-fine continuation** reduces fine-grid SCF iterations and wall time in tested CO cases; productizing multi-stage chains is the main roadmap item for **CHON** accuracy + speed.
- Mixer changes can improve stability but are secondary to continuation and grid/iteration budgets for the current roadmap.
- C/O s-projector normalization is not the main explanation for large total bias.
- Next numerical targets: **continuation harness maturity**, then **local GTH** audit and **nonlocal / N** coverage as benchmarks expand.

## Out of Scope

- Real forces.
- Production-level geometry optimization or molecular dynamics.
- Treating old verification scripts as authoritative without checking current code.
- Speed/JIT/caching work before continuation baselines and component diagnostics are documented for the grids you care about.
