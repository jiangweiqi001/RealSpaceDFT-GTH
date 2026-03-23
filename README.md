# RealSpaceDFT-GTH

## 1. 项目简介

RealSpaceDFT-GTH 是一个基于 **JAX** 的实空间 Kohn–Sham DFT 原型仓库，当前同时维护两条数值路线：

- **Uniform Grid**：均匀笛卡尔网格，作为稳定基线与验证参考。
- **Adaptive Tensor Grid**：按坐标轴自适应重分布节点的张量积网格，用更少的真空点逼近均匀网格与 PySCF 的结果。

当前仓库的主要工作不是“再加一条新算法”，而是把 **adaptive 路线的精度、稳定性和性能** 推到接近 uniform / PySCF 可接受的水平，并用一组 H2 单点、局域能量和小范围曲线脚本持续回归。

## 2. 当前状态

### 2.1 main 分支已经包含的关键能力

- **Uniform 实空间主线**
  - 高阶均匀网格 Laplacian
  - FFT Hartree / Poisson 路径
  - 作为 PySCF 对比与 spacing sweep 的稳定基线
- **Adaptive Tensor Grid 主线**
  - 自适应 1D 轴重分布与 3D 张量积网格
  - 加权积分、加权内积、加权非局域投影
  - 多种 Hartree 边界提供器：`zero_dirichlet`、`monopole_dirichlet`、`multipole_dirichlet`、`uniform_exterior`
  - 两种 kinetic 模式：`prototype_fd2`（当前主线）、`symmetric_fv`（实验性）
- **H2 相关验证与审计脚本**
  - adaptive vs PySCF 小范围曲线验证
  - `R=1.4 Bohr` 单点总能分解审计
  - `R=1.4 Bohr` 的 `Eloc` 空间分解与密度 profile 审计
- **性能优化（已保留在 main）**
  - adaptive SCF 外层 `jax.lax.while_loop`
  - adaptive 子空间 eigensolver `jax.lax.while_loop`
  - adaptive Hartree warm-start
  - `uniform_exterior` 相关缓存与 host-side 元数据缓存
  - trace-safe 的 adaptive Poisson / tensor-grid 热路径

### 2.2 当前默认行为、常用验证配置、实验路径

| 类别 | 当前状态 |
|---|---|
| 代码默认后端 | 如果 `backend=None`，`solver` 仍走 **UniformBackend** |
| `AdaptiveBackend` 默认 | `hartree_boundary_mode="multipole_dirichlet"`，`kinetic_mode="prototype_fd2"` |
| 当前常用 H2 adaptive 验证配置 | 通常显式改为 `hartree_boundary_mode="uniform_exterior"`，`kinetic_mode="prototype_fd2"` |
| 实验性路径 | `kinetic_mode="symmetric_fv"` 仍保留，但**不是当前主线推荐配置** |

### 2.3 当前已经确认的结论

以下结论是当前 main 上最可信、且脚本之间已经基本对齐的：

- `R=1.4 Bohr`、`box=30`、`h_min=0.16`、`h_max=0.32`、`hartree_boundary_mode=uniform_exterior`、`kinetic_mode=prototype_fd2`、`max_iter=120`、`tolerance=1e-5` 时：
  - adaptive 相对 PySCF 的单点误差目前大约在 **`-4.9 mHa`** 量级。
- 同一单点上，adaptive 相对 uniform 的总能偏差主要不是 bookkeeping 误差；总能分解与总能公式在当前脚本里是一致的。
- `R=1.4` 的能量分解与局域 profile 审计表明：
  - adaptive 相对 uniform 的主要偏差集中在 **`Eloc`**。
  - 更具体地说，偏差主要发生在 **核附近（near-core）局域势强区**，并伴随 outer 区域的反向补偿。
- 更细的 adaptive 参数（如 `h_min=0.16, h_max=0.32`）相比更粗参数已经明显改善单点误差，但还没有把整条 H2 曲线完全收敛到 uniform / PySCF 水平。

### 2.4 当前还没有解决的问题

- H2 **整条曲线** 还没有像 uniform 路线那样完全稳定收敛。
- adaptive 路线的 wall time 仍然偏高，尤其是较细网格和较严格 SCF 参数下。
- 某些性能优化曾经带来精度回退；其中 adaptive metric eigensolver 的 `HX/HR` 复用已经回退，不作为当前主线保留。

## 3. 安装与环境

### 3.1 建议环境

- Python 3.10+
- JAX / jaxlib
- NumPy
- SciPy
- matplotlib
- PySCF（用于参考总能与曲线对比）

### 3.2 最小安装

在仓库根目录下，至少保证这些依赖可用：

```bash
python -m pip install numpy scipy matplotlib jax jaxlib pyscf
```

如果只做不带 PySCF 的内部 smoke test，可以暂时不装 PySCF；但 README 里列出的主要 H2 对比脚本默认都建议装上 PySCF。

## 4. 快速开始

下面这些命令是当前最常用的入口。

### 4.1 Uniform H2 基线验证

```bash
python3 JaxDFT/scripts/verify_h2.py
```

输出：
- H2 均匀网格曲线
- 与 PySCF 的总能对比图
- 适合看 uniform baseline 是否正常

### 4.2 Adaptive vs PySCF 小范围 H2 曲线

```bash
python3 JaxDFT/scripts/verify_h2_adaptive_vs_pyscf.py
```

当前脚本默认：
- `R = 1.2, 1.4, 1.6 Bohr`
- `box = 30`
- `h_min = 0.16`
- `h_max = 0.32`
- `r_core = 1.0`
- `stretch_beta = 5.0`
- `hartree_boundary_mode = uniform_exterior`
- `kinetic_mode = prototype_fd2`
- `max_iter = 120`
- `tolerance = 1e-5`

输出：
- 终端表格：`R, E_adaptive, E_pyscf, dE, status`
- CSV
- 总能对比图
- `dE(R)` 误差图

### 4.3 `R=1.4 Bohr` 单点能量分解审计

```bash
python3 JaxDFT/scripts/check_h2_r14_energy_breakdown.py \
  --box 30 --h-min 0.16 --h-max 0.32 \
  --r-core 1.0 --stretch-beta 5.0 \
  --hartree-boundary-mode uniform_exterior \
  --kinetic-mode prototype_fd2 \
  --max-iter 120 --tolerance 1e-5
```

输出：
- `E_total, Ts, Eloc, Enl, Eh, Exc, Eion`
- adaptive / uniform / PySCF 差值表
- 总能一致性检查

### 4.4 `R=1.4 Bohr` 局域能量与密度 profile 审计

```bash
python3 JaxDFT/scripts/check_h2_r14_local_energy_profile.py \
  --box 30 --h-min 0.16 --h-max 0.32 \
  --r-core 1.0 --stretch-beta 5.0 \
  --hartree-boundary-mode uniform_exterior \
  --kinetic-mode prototype_fd2 \
  --max-iter 120 --tolerance 1e-5
```

输出：
- region-resolved `Eloc`
- 轴向 profile CSV / PNG
- `x-z` 截面图

### 4.5 快速 sanity 配置（更快，但不是高精度口径）

```bash
python3 JaxDFT/scripts/verify_h2_adaptive_vs_pyscf.py \
  --dist 1.2 1.4 1.6 \
  --box 30 --h-min 0.20 --h-max 0.40 \
  --r-core 1.0 --stretch-beta 5.0 \
  --hartree-boundary-mode uniform_exterior \
  --kinetic-mode prototype_fd2 \
  --max-iter 60 --tolerance 1e-4 --mix-alpha 0.30 \
  --out-prefix h2_adaptive_vs_pyscf_quick
```

用途：
- 先确认 adaptive H2 代码链路能跑通
- 看总能是否仍为负、最低点是否落在合理区间
- 不用于宣称 mHa 级精度

## 5. 仓库结构

```text
JaxDFT/
  src/
    backends/        UniformBackend / AdaptiveBackend
    grids/           uniform / adaptive grid 与 Poisson 相关实现
    hamiltonian.py   局域势、非局域投影、uniform 核心算子
    solver.py        SCF、子空间 eigensolver、总能
  scripts/
    verify_*.py      面向验证的脚本
    check_*.py       面向单点审计/组件检查的脚本
    study_*.py       面向特定问题的研究/诊断脚本
```

### 5.1 目前最重要的验证脚本

- `verify_h2.py`
  - uniform H2 vs PySCF 基线
- `verify_h2_adaptive_vs_pyscf.py`
  - adaptive H2 vs PySCF 近平衡小范围曲线
- `check_h2_r14_energy_breakdown.py`
  - `R=1.4` 单点能量账
- `check_h2_r14_local_energy_profile.py`
  - `R=1.4` 的 `Eloc` 空间分解与密度 profile
- `check_adaptive_scf_smoke.py`
  - adaptive 路径 smoke / 回归检查

### 5.2 `study_*.py` 的定位

`study_h2_*` 这一组脚本主要保留为**历史诊断与专题研究工具**。它们帮助定位过边界条件、局域势、branch switching、frozen V_eff、性能优化回退等问题，但不应直接视为“README 级别的当前结论入口”。

## 6. 当前最重要的验证结论

- `R=1.4 Bohr` 单点上，当前最可信的 adaptive vs PySCF 误差量级约为 **`-4.9 mHa`**。
- 这个结论是在：
  - 回退了带来精度回退的 `HX/HR` 复用之后，
  - 并统一了关键验证脚本的随机初始条件口径之后得到的。
- 同一单点上，adaptive 相对 uniform 的主要偏差不是 `Enl`，而是 **`Eloc` 主导、`Ts` 部分补偿**。
- `Eloc` 的空间审计显示，偏差主要来自 **near-core 区域**，不是简单的真空尾部或 bookkeeping 偏差。

这些表述都只针对**当前单点与小范围近平衡验证**，不等价于“整条 H2 曲线已经完全收敛”。

## 7. 已知限制

- 当前 adaptive 曲线级别验证仍不算完全收敛。
- adaptive 路线的 wall time 仍然显著高于理想状态；性能仍是当前仓库的重要问题。
- `symmetric_fv` kinetic 模式仍在代码中，但目前没有作为主线默认配置推广。
- 若只改脚本参数而不统一初始条件口径，adaptive 单点结果会表现出一定初始值敏感性。
- 部分 `study_*.py` 脚本记录了已经被后续审计修正或弱化的解释，不应把它们逐条当作当前 main 的最终结论。

## 8. 下一步

当前最值得推进的工作有三类：

1. **继续降低 adaptive 性能成本**
   - 尤其是较细网格与较严格 SCF 参数下的 wall time
2. **继续定位剩余几 mHa 单点误差来源**
   - 当前证据更支持 near-core resolution / `Eloc` 偏差，而不是总能公式问题
3. **把单点结论扩展到更可靠的曲线级验证**
   - 先在平衡附近验证，再逐步扩到更完整的 H2 曲线

## 9. 说明

本文档只描述当前 main 分支上已经存在、并且仍被认为有参考价值的实现与结论。若 README 与旧脚本注释、旧图或旧结论冲突，请以当前代码和当前审计脚本结果为准。
