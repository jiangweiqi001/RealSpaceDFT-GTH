# Status and Handoff

This file tracks **current evidence**, **completed tooling**, and **what to do next**. It is kept in sync with [AGENTS.md](../AGENTS.md). Authoritative rules for agents remain in `AGENTS.md`.

## North Star

**Mature coarse-to-fine (density continuation)** for **CHON**-class organics and small molecules: **accurate and fast** under the fixed PySCF reference protocol (`gth-tzvp + gth-lda + lda,pz`), with **reproducible** scripts and JSON reports—not ad-hoc cold starts on the finest grid only.

Scope path: stabilize **H2 / H2O / CO** + continuation harness → extend benchmarks and continuation chains to **N-containing** systems once GTH data and harness rows exist.

---

## Current Best Understanding

1. **Poisson / Hartree (implementation path)**  
   The independent **Gaussian Hartree** diagnostic (`JaxDFT/scripts/diagnose_poisson_gaussian.py`) shows **sub-mHa to ~1 mHa**-level agreement with the analytic self-energy on moderate grids, improving monotonically as spacing refines. This does **not** support treating the FFT Poisson path as the primary bug for CO/H2O bias.

2. **Same-density components**  
   `JaxDFT/scripts/diagnose_same_density_components.py`: **XC** matches PySCF `exc` to **sub-0.02 mHa** in tested CO/H2O cases. **Hartree** vs PySCF `coul` gap **shrinks strongly with finer spacing** (same PySCF density on the JaxDFT grid), so **discretization** is a major contributor to the Hartree component gap. Do **not** compare JaxDFT local pseudopotential energy to PySCF `e1` (composite KS term).

3. **Total energy vs spacing (full SCF)**  
   CO benchmark sweeps show **total error vs PySCF improves as spacing refines**; very fine grids (e.g. dx≈0.08) may hit **`max_iter` caps** with `density_converged=false` unless outer-iteration budget and/or **continuation** is used—treat truncated runs as **trend points only**.

4. **Continuation (coarse-to-fine)**  
   `energy_and_forces(..., initial_rho=...)` + `JaxDFT/src/continuation.py` + `JaxDFT/scripts/scf_continuation_benchmark.py`. Example **CO 0.2 → 0.15**: warm fine SCF used **fewer outer iterations and less wall time** than cold fine SCF on the same fine grid; **total** wall time can still exceed cold-only when the coarse stage is cheap enough that cold competes—continuation pays most when the **fine** stage is expensive.

5. **Mixer**  
   Pulay/Kerker remains a **useful diagnostic configuration** for CO/H2O stability, but **blind mixer sweeps** are not the primary lever. **Targeted** mixer / inner-loop tuning may follow once **continuation + budgets** are baseline.

---

## Completed Work (recent + historical)

- Benchmark harness `JaxDFT/scripts/benchmark_systems.py` (H2, H2O, CO); PySCF reference protocol fixed as above.
- Full `return_info=True` diagnostics; `BenchmarkResult` includes **`xc_energy`**, **`energy_last20_mean` / `energy_last20_std`**, **`scf_status`**; `return_info` includes **`density`** for continuation and post-processing.
- **Gaussian Poisson diagnostic**: `JaxDFT/scripts/diagnose_poisson_gaussian.py`.
- **Same-density component diagnostic**: `JaxDFT/scripts/diagnose_same_density_components.py`.
- **Grid convergence report**: `JaxDFT/scripts/co_grid_convergence_report.py` (tunable `max_iter`, `orbital_max_iter`, etc.).
- **Continuation**: `JaxDFT/src/continuation.py` (trilinear `rho` remap + charge renorm); `scf_continuation_benchmark.py` (coarse+fine+optional cold comparison).
- Earlier completed items: density stabilization, RMS/L2 residuals, energy stability fields, Pulay/Kerker optional paths, projector overlap diagnostics, local/nonlocal/Hartree diagnostics, dtype/phase/laplacian-order controls, etc. (see `AGENTS.md` Completed).

---

## Important Experimental Results (updated)

| Topic | Observation | Conclusion |
| --- | --- | --- |
| Gaussian Hartree | Errors **well below 10 mHa** on scanned grids; improve with finer spacing. | Poisson/Hartree **implementation** is not the first suspect for large same-density gaps at moderate dx. |
| Same-density XC | Sub-0.02 mHa vs PySCF `exc` (CO/H2O samples). | XC operator alignment is **excellent** at fixed PySCF ρ. |
| Same-density Hartree vs `coul` | Gap **~14 → ~7 → ~4.7 mHa** as spacing tightens (CO examples). | **Discretization + grid sampling** explains much of the Hartree gap; not only self-consistent ρ differences. |
| Full SCF CO spacing sweep | Total error improves with spacing; dx≈0.08 can hit iter cap. | Need **larger `max_iter`** and/or **continuation** for “stable” fine-grid numbers. |
| CO 0.2→0.15 continuation | Warm fine: **112** iters, **~49.6 s**; cold fine: **170** iters, **~65.3 s**. | Continuation **reduces fine-grid work**; total time vs cold-only depends on coarse cost. |
| Mixer / `orbital_max_iter` sweeps (historical) | Did not remove systematic CO/H2O bias alone. | Not the primary narrative vs grid + continuation. |
| Kinetic preconditioner (historical) | Lower orbital residual; did not fix energy drift decisively. | Do not default-enable as the main fix. |
| Projector s overlap (C/O) | ~1e-6 level. | Unlikely to explain **10+ mHa** bias. |

---

## Current Recommended Diagnostic Config

Same as `AGENTS.md` (Pulay + Kerker + RMS density metric). For **fine** grids, increase **`max_iter`** (e.g. 500+) and/or **`orbital_max_iter`** when `density_converged` fails or iterations cap.

When reporting CO/H2O (and future CHON) runs, include: energy, error mHa, density max/RMS/L2, `energy_delta_last10_max`, orbital residual, `scf_converged`, `orbital_converged`, **`energy_last20_mean`/`std`** when available, and whether **`max_iter`** was hit.

---

## Next Recommended Tasks

**P0 — Productize coarse-to-fine**

- Multi-stage chains in script or thin driver (e.g. **0.14 → 0.12 → 0.10 → 0.08**), documented defaults, JSON schema for regression.
- Optional **looser tolerances / smaller coarse `max_iter` only in preheat stages**; **final** stage uses the **same** diagnostic tolerances as published numbers.

**P1 — Regression maintenance**

- Re-run Gaussian + same-density + selected continuation smoke tests when touching grid/Poisson/XC.

**P2 — Local GTH audit**

- After same-density Hartree gap is small at fine dx, audit local potential convention (`r=0`, erf, polynomial) for any **residual mHa** plateau.

**P3 — Nonlocal angular / p-channel**

- Especially N and other p-active cases when CHON harness expands.

**P4 — Mixer / inner loop (secondary)**

- Small, justified tweaks on top of continuation baselines—not blind sweeps.

---

## Do Not Do Next

- Blind Anderson/Pulay parameter sweeps as the main strategy.
- Default-enable kinetic preconditioning as the primary fix.
- Claim strict CO/H2O convergence without energy-stability and iteration-cap evidence.
- Treat **old** “do not do coarse-to-fine” guidance as current—it is **superseded**: continuation is now **encouraged** (see `AGENTS.md`).

---

## Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q JaxDFT tests
git diff --check
```
