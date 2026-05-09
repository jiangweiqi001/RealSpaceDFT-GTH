# Current Best Understanding

The project now has a reasonably complete diagnostic loop. The main unresolved issue is not benchmark infrastructure or SCF mixer plumbing.

- H2 is within target.
- CO/H2O still have systematic `15-25 mHa` errors.
- Mixer improvements improve stability but do not remove systematic bias.
- Projector overlap normalization for C/O s-projectors is accurate to roughly `1e-6` and is unlikely to explain the error.
- CO component diagnostics show Hartree differs from PySCF Coulomb by roughly `44 mHa` at `spacing=0.12`, `box=18`, but this may include density differences; same-density Poisson tests are needed.
- For C/O systems, the remaining bias is likely in component-level numerical physics: Hartree/Poisson, local GTH potential convention/discretization, or remaining nonlocal angular/operator details.
- The next agent should start with an independent Gaussian Hartree analytic test before changing SCF mixing again.

# Completed Work

- `JaxDFT/scripts/benchmark_systems.py` benchmark harness additions for H2/H2O/CO.
- Fixed PySCF reference protocol: `gth-tzvp + gth-lda + lda,pz`.
- `return_info=True` diagnostics without changing the default `(energy, forces)` API.
- Density stabilization and electron-count renormalization.
- RMS/L2 density residual diagnostics in addition to max density residual.
- Energy stability diagnostics: `energy_delta_last`, `energy_delta_history`, `energy_delta_last10_max`, `density_converged`, `energy_converged`, `scf_converged`.
- Optional Pulay/DIIS mixer.
- Charge-neutral Pulay residual.
- Optional Kerker residual metric.
- Optional kinetic orbital preconditioner.
- C/O GTH parser and `n_proj=0` checks.
- Projector overlap diagnostics.
- Local/nonlocal/Hartree component output.
- Local GTH potential min/max/integral/by-atom diagnostics.
- Hartree/Poisson potential min/max/integral diagnostics.
- PySCF reference component output: one-electron, Coulomb, XC, nuclear.
- Dtype, grid phase, and laplacian-order diagnostic controls.

# Important Experimental Results

| Experiment | Observation | Conclusion |
| --- | --- | --- |
| CO `orbital_max_iter=30/60/100` | Larger inner iteration counts did not remove the systematic energy bias. | The issue is not simply insufficient orbital iterations. |
| Kinetic orbital preconditioner | Orbital residual dropped substantially, but SCF energy drift did not improve decisively. | Do not default-enable kinetic preconditioning as a fix. |
| RMS/L2 diagnostics | Max density residual can be dominated by localized grid spikes. | RMS/L2 are useful diagnostics, but they do not prove strict SCF convergence alone. |
| Pulay/DIIS | Often better plateau behavior than Anderson for CO. | Useful diagnostic path, not a final default and not a full accuracy fix. |
| Charge-neutral Pulay residual | Did not decisively remove CO/H2O bias. | Not enough by itself. |
| Projector overlap | C/O s-projector overlap error around `1e-6`. | s-channel normalization is unlikely to explain `15-25 mHa` error. |
| dtype/stencil/phase | mHa-level effects observed. | These do not explain the full CO bias. |
| box size `18 -> 22 Bohr` | Not a dominant error source in tested cases. | Do not prioritize larger boxes before component validation. |

CO `spacing=0.12`, `box=18` component snapshot from a Pulay/Kerker diagnostic run:

| Quantity | Value |
| --- | ---: |
| Total error | `-20.8 mHa` |
| JaxDFT local pseudopotential energy | `-74.8767 Ha` |
| JaxDFT nonlocal pseudopotential energy | `+1.8133 Ha` |
| JaxDFT Hartree energy | `28.8171 Ha` |
| PySCF Coulomb component | `28.7727 Ha` |
| Projector overlap max error | `~1.7e-6` |

# Current Recommended Diagnostic Config

CO diagnostic reference config:

```text
mixing_mode=pulay
pulay_history=3
pulay_regularization=1e-4
mix_alpha=0.2
pulay_residual_metric=kerker
kerker_k0=2.0
scf_convergence_metric=rms
tolerance=3e-5
max_iter=200 or 250
orbital_max_iter=30
orbital_preconditioner=none
```

This is a diagnostic reference config, not the final default. The default mixer remains Anderson.

When reporting CO/H2O results, include:

- energy
- error mHa
- density max/RMS/L2 residuals
- `energy_delta_last10_max`
- orbital residual
- `scf_converged`
- `orbital_converged`
- last-window energy mean/std/range if available

# Known Limitations

- Forces are not implemented; returned forces are zero placeholders.
- SCF strict convergence is not always reached for CO/H2O.
- Pulay/Kerker are optional and experimental.
- Hartree/Poisson and local GTH potential still need validation.
- Current final energy may depend on SCF plateau phase; report last20 mean/std where possible.
- Full C/O/N angular projector validation is incomplete, especially p-channel / N.
- CO/H2O remain outside the `<=10 mHa` target.
- This project should not be described as production-ready or high-accuracy DFT software.

# Next Recommended Task

Implement an independent Hartree/Poisson Gaussian analytic diagnostic:

```text
rho(r) = Q * (alpha/pi)^(3/2) * exp(-alpha r^2)
E_H_exact = Q^2 * sqrt(alpha / (2*pi))
E_H_num = 0.5 * sum(rho * V_H) * dV
```

Scan:

```text
spacing = 0.18, 0.14, 0.12, 0.10
box = 18, 22
alpha = 0.25, 0.5, 1.0
```

Output:

```text
electron_count
E_H_num
E_H_exact
error_mHa
relative_error
self_term_used
```

Do not change the Poisson implementation in the doc-only handoff task. For the next coding task, implement this as an isolated diagnostic test before changing SCF mixing again.

# Do Not Do Next

- Do not continue blind Anderson/Pulay parameter sweeps.
- Do not keep increasing `orbital_max_iter`.
- Do not default-enable kinetic preconditioning.
- Do not start with speed/JIT/caching.
- Do not start with coarse-to-fine continuation.
- Do not treat CO/H2O final energy as strictly converged unless energy stability passes.

# Verification Commands

Run these before handing off changes:

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q JaxDFT tests
git diff --check
```
