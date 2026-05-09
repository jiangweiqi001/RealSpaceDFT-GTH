# RealSpaceDFT-GTH Agent Instructions

## Repository Rules

- 禁止批量删除文件或者目录。
- 优先看代码，不盲信 README、注释或旧验证脚本描述。
- 优化 DFT 内核前，必须先跑或更新可复现 benchmark；不要靠单点印象判断精度或性能。
- 除非用户明确要求，不要把文档计划直接理解为可以同时修改算法代码。
- 当前阶段不要声称 forces 已实现；`energy_and_forces(...)` 返回的 forces 仍是 zero placeholder。
- 当前阶段不要声称 CO/H2O 已严格 SCF 收敛；多数结果应视为 practical plateau 或 partial convergence。

## Current Implementation Facts

- 当前 SCF 使用均匀 Cartesian grid、默认 8 阶中心有限差分、FFT 补零 Poisson、block subspace eigensolver。
- 默认 mixer 仍是 Anderson。
- Pulay/DIIS、Kerker residual metric、RMS/L2 convergence metric、kinetic orbital preconditioner 都是可选实验入口，不是默认路径。
- `energy_and_forces(...)` 默认仍返回 `(energy, forces)`；`return_info=True` 才返回诊断信息。
- Forces 是零占位，不是真实力；真实 forces 暂不作为当前优化目标。
- H2 已达到当前目标误差范围。
- H2O 和 CO 仍有约 `15-25 mHa` 级系统偏差，尚未完全解释。

## Completed

- 新增固定体系 benchmark harness：H2、H2O、CO。
- 固定 PySCF 参考协议：`gth-tzvp + gth-lda + lda,pz`。
- 增加 `return_info=True` 诊断，不改变默认 API。
- 暴露 SCF iterations、density residual、orbital residual、eigenvalues、occupations。
- 增加 density stabilization 和电子数重新归一化。
- 增加 max/RMS/L2 density residual 诊断和可选收敛 metric。
- 增加 energy stability diagnostics：`energy_delta_last`、history、`energy_delta_last10_max`、`density_converged`、`energy_converged`、`scf_converged`。
- 修正/诊断 C/O GTH parser 和 `n_proj=0` 行为。
- 增加 projector overlap / normalization diagnostics。
- 增加 local/nonlocal/Hartree energy component output。
- 增加 local GTH potential min/max/integral/by-atom diagnostics。
- 增加 Hartree/Poisson potential min/max/integral diagnostics。
- 增加 PySCF reference component output：one-electron、Coulomb、XC、nuclear。
- 增加可选 Pulay/DIIS mixer。
- 增加 charge-neutral Pulay residual。
- 增加可选 Kerker-style residual metric。
- 增加可选 kinetic orbital preconditioner。
- 增加 dtype、grid phase、laplacian order 的诊断入口。

## Excluded Or Lower Priority Based On Experiments

- 不要继续盲扫 Anderson/Pulay 小参数；mixer 不是当前最可疑的主误差源。
- 不要继续单独增加 `orbital_max_iter`；30/60/100 对照说明这不是简单内层迭代不够。
- 不要默认启用 kinetic preconditioner；它显著降低 orbital residual，但没有决定性改善 SCF energy drift。
- C/O s-channel projector normalization 误差约 `1e-6`，不是当前 `15-25 mHa` 偏差的主要解释。
- box size `18 -> 22 Bohr` 不是主要误差来源。
- dtype、stencil order、grid phase 会造成 mHa 级影响，但没有解释完整 CO 偏差。
- 不要先做 speed/JIT/caching。
- 不要先做 coarse-to-fine continuation。
- 不要把 CO/H2O final energy 当严格收敛能量，除非 energy stability 过关。

## Current Recommended Diagnostic Config

这是 diagnostic reference config，不是最终默认：

```text
system=CO
mixing_mode=pulay
anderson_history=3
anderson_regularization=1e-4
mix_alpha=0.2
pulay_residual_metric=kerker
pulay_kerker_k0=2.0
scf_convergence_metric=rms
tolerance=3e-5
max_iter=200 or 250
orbital_max_iter=30
orbital_tolerance=1e-5
orbital_preconditioner=none
```

对比能量时必须同时报告 `energy_delta_last10_max`、density max/RMS/L2、orbital residual、last-window energy behavior。

## Current Best Understanding

The project now has a reasonably complete diagnostic loop. The main unresolved issue is not benchmark infrastructure or SCF mixer plumbing.

For C/O systems, the remaining `15-25 mHa` bias is likely in component-level numerical physics: Hartree/Poisson, local GTH potential convention/discretization, or remaining nonlocal angular/operator details.

Projector s-channel normalization for C/O has been tested and is not the main explanation. The next agent should start with an independent Gaussian Hartree analytic test before changing SCF mixing again.

## Next Priorities

P0: Hartree/Poisson independent analytic Gaussian test.

P1: Same-density Hartree comparison, ideally PySCF density sampled on the JaxDFT grid.

P2: Local GTH potential formula audit, including `r=0`, `erf` convention, and polynomial convention.

P3: Component-level comparison against PySCF.

P4: p-channel / angular nonlocal projector diagnostic for C/O/N, especially N and any active p-channel cases.

P5: Platform energy reporting: last20 mean/std/range/slope.

P6: Only after the numerical error source is identified, revisit speed/JIT/caching.

## Planned Hartree/Poisson Diagnostic

Do not change the Poisson implementation before this test exists.

Use a normalized Gaussian density:

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

## Acceptance Targets

- H2 error remains `<=5 mHa`.
- H2O and CO eventually reach `<=10 mHa`, but this is not yet achieved.
- H2O/CO single-point runtime target `<=2 minutes` is only meaningful after the accuracy target is met.
- `python3 -m unittest discover -s tests` passes.
- `python3 -m compileall -q JaxDFT tests` passes.
- PySCF-missing environments should fail/skip with a clear message.

## Out of Scope For Current Phase

- `run_sampling.py`, due to unrelated `prepare_system` and ragged HDF5 issues.
- Real forces.
- HCON/N2 as first-stage targets.
- Production-grade DFT claims.
