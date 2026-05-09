# RealSpaceDFT-GTH Agent Instructions

## Repository Rules

- 禁止批量删除文件或者目录。
- 优先看代码，不盲信 README、注释或旧验证脚本描述。
- 优化 DFT 内核或报告高精度数字前，必须先跑或更新可复现 benchmark / 诊断脚本；不要靠单点印象判断精度或性能。
- 除非用户明确要求，不要把文档计划直接理解为可以同时修改算法核心（Poisson、mixer 默认、projector 公式等）。
- 当前阶段不要声称 forces 已实现；`energy_and_forces(...)` 返回的 forces 仍是 zero placeholder。
- 当前阶段不要声称 CO/H2O 已严格 SCF 收敛；多数结果应视为 practical plateau 或 partial convergence，除非 `density_converged`、`energy_converged`、`scf_converged` 与能量窗口诊断一致地支持。

## Current Implementation Facts

- 当前 SCF 使用均匀 Cartesian grid、默认 8 阶中心有限差分、FFT 补零 Poisson、block subspace eigensolver。
- 默认 mixer 仍是 Anderson；benchmark / 诊断常用 Pulay + Kerker（可选）。
- Pulay/DIIS、Kerker residual metric、RMS/L2 convergence metric、kinetic orbital preconditioner 都是可选实验入口，不是默认路径。
- `energy_and_forces(...)` 默认仍返回 `(energy, forces)`；`return_info=True` 时返回诊断信息，且包含 **`density`**（自洽密度，供延续法与后处理）。
- **延续法（continuation）**：支持 `energy_and_forces(..., initial_rho=...)` 与 `scf(..., initial_rho=...)`；粗→细密度用 `JaxDFT/src/continuation.py` 三线性插值 + 电荷重归一化。编排脚本：`JaxDFT/scripts/scf_continuation_benchmark.py`。
- Forces 是零占位，不是真实力；真实 forces 暂不作为当前优化目标。
- H2 在常用验证配置下整体接近目标；最短键距等极端几何仍可能略超旧版 `<=5 mHa` 表述，以当前 benchmark / `verify_h2` 输出为准。
- CO/H2O：总能量相对 PySCF 仍存在 **mHa～十 mHa 量级** 的可改善空间；其中 **空间离散（spacing）** 与 **SCF 是否充分迭代/收敛** 已证明是重要杠杆，**不能**再只归因于单一 mixer 或单一 Poisson bug 叙事。

## Completed（相对上一版 AGENTS 新增或已兑现）

- 固定体系 benchmark harness：H2、H2O、CO；PySCF 参考协议 `gth-tzvp + gth-lda + lda,pz`。
- 完整诊断链：`return_info=True`、分量能量、密度/轨道/能量稳定性、projector overlap、`BenchmarkResult` 中 **`xc_energy`**、**`energy_last20_mean` / `energy_last20_std`**、**`scf_status`** 等。
- **P0 高斯解析 Hartree/Poisson**：`JaxDFT/scripts/diagnose_poisson_gaussian.py`（解析高斯电荷下，当前 FFT/Hockney Poisson 路径在已测 spacing 上可达 **亚 mHa～约 1 mHa** 量级误差；**分子自洽密度**上 Hartree 与 PySCF 仍有 **spacing-dependent** 差异，见 same-density 诊断，勿把「高斯很好」误读成 Poisson 与分子总误差无关）。
- **Same-density 分量诊断**：`JaxDFT/scripts/diagnose_same_density_components.py`（PySCF 收敛密度 → JaxDFT 格点 `eval_rho` → 同密度下 Hartree / local / XC 与 PySCF 分量对照）。
- **经验结论（已由数据支持）**：在 PySCF 固定密度下，**XC 与 PySCF `exc` 对齐极好**；**Hartree 与 PySCF `coul` 的差随 spacing 变细显著下降**，说明离散误差是 Hartree 差的重要成分；**总能量误差随 spacing 变细亦明显下降**。`pyscf_e1` 不是局域赝势单项，禁止用它对齐 `local_pseudopotential`。
- **延续法工具链**：`continuation.py` + `scf_continuation_benchmark.py`；粗→细在 **CO 0.2→0.15** 等案例上已验证 **细格外层迭代与墙钟相对冷启动减少**（总墙钟是否划算取决于粗段成本与细格难度，见脚本 JSON）。
- **网格收敛汇总脚本**：`JaxDFT/scripts/co_grid_convergence_report.py`（可调 `max_iter` / `orbital_max_iter` 等）。

## Excluded Or Lower Priority（按当前证据更新）

- **不要**把「盲扫 Anderson/Pulay 小参数」当作第一生产力；mixer **可能仍有针对性优化空间**，但应放在 **延续法 + 足够 SCF 预算** 的基线之后。
- 不要默认启用 kinetic preconditioner 作为「主修复」；它改善轨道残差但未决定性消除 CO 能量平台问题（历史结论仍成立）。
- C/O s-channel projector normalization 误差约 `1e-6`，不是 CO/H2O 总偏差的主叙事。
- 盒子 `18 -> 22 Bohr` 在已测 Gaussian / same-density 下**不是** Hartree 差的主杠杆。
- dtype、stencil order、grid phase 会有 mHa 级影响，但**单独**不能解释全部 CO 平台行为。
- 不要把 CO/H2O 单次 `final_energy` 当严格收敛值，除非 `energy_delta_last10_max`、last-window 统计与 `scf_converged` 支持。
- **~~不要先做 coarse-to-fine continuation~~** **已废止**：延续法现为**推荐主路径**之一（见下节）。仍应避免无诊断的「为快而快」黑箱参数海搜。

## Current Recommended Diagnostic Config（最终报告仍应用统一判据）

CO / H2O 稳定性与对比仍推荐（与 `docs/STATUS_AND_HANDOFF.md` 一致；**非**代码默认路径）：

```text
mixing_mode=pulay
# 注意：API/CLI 在 Pulay 模式下仍使用参数名 anderson_history / anderson_regularization
#（分别对应 pulay_mixing 的历史长度 m 与 regularization；并非误用 Anderson）。
anderson_history=3
anderson_regularization=1e-4
mix_alpha=0.2
pulay_residual_metric=kerker
pulay_kerker_k0=2.0
scf_convergence_metric=rms
tolerance=3e-5
max_iter=200 or 250（细格可加到 500+ 若密度/能量未过关）
orbital_max_iter=30（细格可适度加大）
orbital_tolerance=1e-5
orbital_preconditioner=none
energy_tolerance：代码默认 5e-6 Ha（与 energy_delta_last10_max 同单位）。
  细格 CO/H2O 常见 energy_delta_last10_max ~1e-3～5e-3 Ha，故 energy_converged 仍常 false。
  「平台可读」诊断可显式试 1e-4～1e-3 Ha，**须在报告中写明所用阈值**；「严收敛」小体系对照才可收紧到 1e-6 量级并说明意图。
```

报告总能量时同时给出：`energy_delta_last10_max`、density max/RMS/L2、`energy_last20_mean`/`std`（若可得）、`scf_status`、轨道残差与是否打满 `max_iter`。

## Current Best Understanding

1. **Poisson / Hartree 主路径**：解析高斯检验表明当前 FFT/Hockney Poisson 对**光滑高斯电荷**在已测 spacing 上可达 **亚 mHa～约 1 mHa** 误差；**分子密度**上仍有 spacing-dependent 的 Hartree 分量差（same-density 已量化）。  
2. **XC**：same-density 下与 PySCF `exc` 高度一致，**不是**当前 CO/H2O 总偏差的主因。  
3. **总能量 vs PySCF**：CO 等案例上 **spacing 变细** 可显著降低 `total_error_mHa`；**细格（如 dx≈0.08）** 往往需要 **更大 `max_iter` 和/或延续法**，否则只能作趋势点。**勿**在 dx≈0.08 等上宣称已达「最终目标精度」，除非该次运行满足密度/能量窗口诊断，或**明确标注**为未充分收敛下的平台估计。  
4. **Mixer**：可改善稳定性与迭代数，但**不应**再被当作解释十 mHa 级系统偏差的唯一故事；在延续法与网格收敛之后再做 **有针对性的** mixer / 内层迭代调参。

## Next Priorities（重排：延续法为主目标）

**P0 — 延续法（continuation）作为高精度 / 细网格验证的主流程**

- **目标**：不要在最细网格上从天真初猜冷启动；先在较粗 `dx` 上跑到合理自洽密度，**三线性插值 ρ** 到细格，再 `initial_rho` 启动细格 SCF；必要时粗段用略少 `max_iter` 或（仅预计算阶段）略松判据，**最终报告**仍用统一 diagnostic 配置与判据。  
- **优先验证用例**：**CO 0.14→0.10** 与 **0.10→0.08**（`scf_continuation_benchmark.py --compare-cold`），对照 JSON 中 `two_stage_chain_wall_seconds` vs `cold_fine_wall_seconds` 与误差。**已有 CO 0.2→0.15** 等结果只证明 **暖启动减少细格外层迭代/细段墙钟**；**不足以**证明在「目标精度格点」上 **总墙钟** 一定优于冷启动——总时间是否划算须按参数单独量。  
- **落地**：先固化 **二段链** 与 JSON 字段；**多段链**（如 0.14→0.12→0.10→0.08）可选，须单独证明相对二段链的**性价比**，勿默认多段更优。与 `co_grid_convergence_report.py` 的衔接及 `docs/STATUS_AND_HANDOFF.md` 中的推荐链路保持更新。  
- **验收**：在 CO（及 H2O）上，在声明关心的 `(coarse_dx, fine_dx)` 上，相对「同配置冷启动细格」，**总链墙钟或细段墙钟**之一有**可复现**收益，或能在合理墙钟内达到先前达不到的密度/能量平台质量（报告写明阈值与是否打满 `max_iter`）。

**P1 — Same-density + 网格收敛维护**

- 保留 `diagnose_same_density_components.py` / `co_grid_convergence_report.py` 作为回归；换基组/势数据或改格点生成时需重跑。

**P2 — 局域 GTH 与分量定义**

- 在 same-density Hartree 差已随网格显著缩小后，对 **仍残留的 mHa 级** 差，审计局域势公式、`r=0`、erf 与多项式约定；必要时与 PySCF 做 **更严格同义** 的 Coulomb/Hartree 对照（定义对齐）。

**P3 — 非局域角动量 / p 通道诊断**

- 尤其对 C/O/N 活跃 p 通道；在 P0～P2 之后。

**P4 — Mixer / 内层求解（次要）**

- 在延续法基线上，针对仍难收敛的个案做 **小规模、可解释** 的 mixer 或 `orbital_max_iter` / `orbital_tolerance` 调整；禁止无诊断的盲扫。

**P5 — 性能工程**

- 仅在精度路径清晰后，再系统考虑 JIT 缓存、算子融合、多 GPU 等（GPU JAX：环境具备时 JaxDFT 主体自动受益；PySCF 仍在 CPU）。

## North Star（长期产品目标）

- **成熟的 coarse-to-fine（延续法）工作流**：对含 **C / H / O / N** 的典型有机/小分子体系，在 **统一物理与参考协议**（GTH + LDA/PZ、与 PySCF 可对照）下，做到 **既准又快**——细网格上 **不靠天真初猜硬磨**，粗段预条件 + 插值 warm-start + 可控 SCF 预算；必要时链式 **0.14 → 0.12 → 0.10 → 0.08** 等，并以 JSON / benchmark **可复现** 记录墙钟与误差。  
- **CHON 覆盖**：在 H2 / H2O / CO 基准稳定后，将 **N 及含 N 小体系**纳入同一 harness 与延续策略（需 GTH 势与 benchmark 定义齐备后再扩）。  
- **Mixer 等**：在延续法基线上做 **增量** 优化，而非替代 coarse-to-fine 作为主路线。

## Acceptance Targets

- H2：维持与当前 harness 一致的目标量级（见 Completed 说明）。  
- H2O / CO：**向** `<=10 mHa`（总能量相对固定 PySCF 协议）推进；细格需配合 **延续法 + 足够迭代** 再宣称接近目标；dx≈0.08 等细格若未满足密度/能量窗口诊断，只能作**趋势或平台估计**，不得写成已达最终目标。  
- `python3 -m unittest discover -s tests` 与 `python3 -m compileall -q JaxDFT tests` 保持通过。  
- 缺 PySCF 的环境应 skip 或明确报错信息。

## Out of Scope For Current Phase

- `run_sampling.py`（历史问题：`prepare_system`、HDF5 等）。  
- 真实 forces、HCON/N2 作为首阶段目标、生产级 DFT 宣传。  

## Historical Note（原 P0/P1 计划）

原「先 Gaussian 再 same-density 再动 Poisson」的阻塞式顺序**已完成第一、二步**；后续 Poisson **实现** 的修改仍须以 **诊断脚本回归** 为前提，但**当前证据**不支持把 CO/H2O 主偏差首要归因于 Poisson bug。
