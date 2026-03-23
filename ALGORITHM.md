# ALGORITHM

## 1. 整体目标

本仓库实现的是一个面向**孤立分子体系**的实空间 Kohn–Sham DFT 原型。当前同时维护两条路线：

- **Uniform Grid**：稳定、易解释、便于与 PySCF 对齐的基线方案。
- **Adaptive Tensor Grid**：在保持张量积结构的前提下，把节点集中到原子附近，减少真空区不必要的自由度。

当前 adaptive 路线的目标不是“先把功能堆齐”，而是：

1. 逼近 uniform 路线的数值稳定性；
2. 在 H2 这类小分子基准上逼近 PySCF 的总能与曲线趋势；
3. 在不破坏精度的前提下逐步降低 wall time。

因此，本文档只记录**当前 main 分支真实存在的实现**，并明确区分：

- 当前默认行为；
- 当前常用验证配置；
- 实验性/可选路径；
- 已尝试但当前不作为主线推进的路线。

## 2. 离散对象与主要模块

### 2.1 Uniform grid

对应模块与入口：
- `JaxDFT/src/backends/uniform.py`
- `JaxDFT/src/hamiltonian.py`

主要特征：
- 均匀笛卡尔网格；
- `laplacian_8th` 作为 kinetic 主算子；
- FFT Hartree / Poisson 路径；
- 作为 PySCF 对比、spacing sweep 与很多回归检查的基线。

### 2.2 Adaptive tensor grid

对应模块与入口：
- `JaxDFT/src/grids/adaptive_tensor.py`
- `JaxDFT/src/backends/adaptive.py`

主要特征：
- 每个坐标轴单独重分布；
- 仍保持 3D 张量积结构，不是非结构网格；
- 节点、积分权重、体元权重都由 adaptive 轴构造；
- adaptive 路线所有积分、内积、非局域投影重叠都显式使用权重。

### 2.3 Local / nonlocal pseudopotential

对应模块：
- `JaxDFT/src/hamiltonian.py`
- `JaxDFT/src/io.py`

当前主要使用：
- GTH-LDA 赝势；
- 局域势直接在网格点坐标上评估；
- 非局域势通过 projector 预计算并在 SCF 内重用。

### 2.4 Hartree / Poisson

- Uniform 路线：FFT Poisson。
- Adaptive 路线：`JaxDFT/src/grids/adaptive_poisson.py` 中的 box-Poisson 原型。

adaptive Hartree 当前支持多种边界提供器：
- `zero_dirichlet`
- `monopole_dirichlet`
- `multipole_dirichlet`
- `uniform_exterior`

### 2.5 SCF loop

对应模块：
- `JaxDFT/src/solver.py`

SCF 主循环负责：
- 初始化 `rho`
- 构造 `V_H + v_xc`
- 调用轨道求解器
- 重建新密度
- Anderson mixing
- 收敛判定

### 2.6 Orbital solver / subspace eigensolver

对应模块：
- `JaxDFT/src/solver.py::solve_orbitals_subspace`

当前使用的是一个 block subspace + Rayleigh–Ritz 的迭代本征求解器，不是严格教科书意义上的完整 LOBPCG，但保留了：
- 当前子空间内 Rayleigh–Ritz
- 残差构造
- 子空间扩展与再正交化
- 更新后的低维 Ritz 截断

## 3. 当前 adaptive 方法的真实实现

## 3.1 Adaptive grid 的参数化

当前 adaptive 轴的核心参数是：
- `h_min`
- `h_max`
- `r_core`
- `stretch_beta`

它们的角色如下：

- `h_min`
  - 近核区域允许达到的最细间距；
- `h_max`
  - 远离原子的背景间距上限；
- `r_core`
  - 近核细化的空间尺度；
- `stretch_beta`
  - 控制从核附近向外过渡时的 spacing profile 强度。

实现上，adaptive 轴不是手工分段网格，而是：
1. 在一条致密参考轴上构造 local spacing profile；
2. 用类似 monitor function 的累计积分重分布节点；
3. 得到严格单调的 1D adaptive 轴；
4. 组合成 3D tensor grid。

## 3.2 权重、积分和密度口径

当前 adaptive 方案里的几个关键量：

- `wx, wy, wz`
  - 1D nodal trapezoidal weights；
- `volume_weights`
  - 三维张量积体元权重；
- `grid.integrate(field)`
  - 统一用 `volume_weights` 积分；
- `grid.inner_product(x, y)`
  - 统一用加权内积；
- 非局域 projector overlap
  - 也走加权积分口径。

因此，adaptive 路线下：
- `rho` 的积分、电子数、Hartree 能量、局域能量、XC 能量、非局域能量，都必须显式经过 backend 的加权积分接口；
- 不能把 adaptive 场当作“只是坐标不均匀，但可以用普通求和代替积分”。

## 3.3 Adaptive kinetic 的当前实现

`AdaptiveBackend` 当前暴露两种 kinetic mode：

### 3.3.1 `prototype_fd2`（当前主线）
- 对应 `state.laplacian(...)`
- 是当前 adaptive 路线的主线 kinetic 配置
- 当前所有主要 H2 验证脚本都以它为默认/推荐配置

### 3.3.2 `symmetric_fv`（实验性）
- 对应 `state.laplacian_symmetric(...)`
- 是一个更偏 finite-volume / lumped-FEM 风格、在加权内积下更接近自伴的尝试
- 目前保留在代码中，但没有作为主线默认配置推进
- 原因是：它在 frozen-Veff 下出现过正向信号，但未稳定转化为真实 adaptive SCF 的明确增益

## 3.4 Adaptive Hartree 的当前实现

`AdaptiveBackend.solve_hartree(...)` 当前支持：
- `multipole_dirichlet`
- `monopole_dirichlet`
- `zero_dirichlet`
- `uniform_exterior`

### 3.4.1 代码默认

`AdaptiveBackend()` 的默认构造参数仍是：
- `hartree_boundary_mode="multipole_dirichlet"`
- `kinetic_mode="prototype_fd2"`

### 3.4.2 当前常用 H2 验证配置

虽然 backend 默认还是 `multipole_dirichlet`，但当前 H2 主验证通常显式使用：
- `hartree_boundary_mode="uniform_exterior"`
- `kinetic_mode="prototype_fd2"`

原因不是说 `multipole_dirichlet` 完全不可用，而是：
- `uniform_exterior` 在当前 H2 上通常更接近我们想要的 isolated-like adaptive Hartree 路径；
- 它是目前 adaptive H2 主验证里更常用的边界提供器。

### 3.4.3 `uniform_exterior` 的实现口径

`uniform_exterior` 不是把 adaptive interior operator 换掉，而是：
- 保留 adaptive interior Poisson operator；
- 通过一个更大、更粗的 auxiliary uniform exterior free-space-like 求解来提供 adaptive box 六个面的 Dirichlet 边界值；
- 然后把这些 face values 喂回 adaptive interior solve。

因此它仍是一个 **finite-box adaptive interior + exterior face provider** 的路线，不是精确的全空间 Green’s-function Hartree。

## 3.5 SCF 主循环的当前实现特征

当前 `solver.py::scf(...)` 的 adaptive 路径有几个必须写清楚的实现特征：

1. **初始密度**
   - 按原子位置生成 Gaussian-like 初猜；
   - 再归一化到总电子数；
   - 再投影到 Dirichlet mask。

2. **Hartree warm-start**
   - 当前 main 默认保留；
   - 每一轮 adaptive Hartree 求解允许使用上一轮 `V_H` 作为 `v_init`。

3. **XC**
   - 当前主线是 `lda_xc(rho)`。

4. **轨道求解器调用**
   - 每轮 SCF 都调用 `solve_orbitals_subspace(...)`；
   - `x_init` 使用上一轮 `eigvecs`；
   - adaptive 路径当前常用内部参数大致是：
     - `orbital_max_iter = 8`
     - `orbital_tol = 1e-4`

5. **密度 mixing**
   - 当前使用 Anderson mixing；
   - 历史长度是固定的小窗口；
   - 不是专门为 adaptive 另写的一套 mixing。

6. **Dirichlet 投影**
   - 当前 main 已经把 orbitals、残差、`Hpsi`、`rho` 等关键量都投影到 mask 上；
   - 这是修掉早期 adaptive 边界异常贴边问题之后的实现状态。

## 3.6 当前保留的性能优化

以下性能改动目前保留在 main：

- adaptive SCF 外层 `jax.lax.while_loop`
- adaptive metric eigensolver `jax.lax.while_loop`
- adaptive Hartree warm-start
- `uniform_exterior` 相关缓存
- adaptive grid host metadata 缓存
- trace/debug 默认关闭，热路径尽量避免 Python-side finite checks

需要特别说明的是：
- **adaptive metric eigensolver 中的 `HX/HR` 复用优化已经回退**；
- 它曾经带来可见的精度回退，因此不是当前 main 保留的优化。

## 4. 边界条件问题的演化

## 4.1 早期问题

adaptive 路线早期最显著的问题包括：
- Hartree 边界模型太粗；
- 自适应轨道在边界上出现不合理行为；
- H2 的 box sensitivity 很强。

这些问题曾触发过大量边界条件、center choice、multipole、exterior treatment 的专项审计。

## 4.2 后续修正

后续 main 上已经落实的关键修正包括：
- solver 内的真 Dirichlet 投影机制；
- `uniform_exterior` boundary provider；
- 多个 H2 单点审计脚本，用来把 Hartree、`V_loc`、`v_xc`、SCF 口径问题彼此拆开。

## 4.3 当前边界条件的真实限制

当前代码状态下，边界问题不能再简单概括成“边界 bug 还没修”。更准确地说：

- adaptive 边界错误的最严重实现 bug 已经修过；
- Hartree provider 也不再是唯一主要嫌疑；
- 但 `uniform_exterior` 仍然是实用型近似，不是最终 free-space 精确解；
- 对于较细 adaptive 参数，当前剩余误差已经更多表现为 near-core / `Eloc` 偏差，而不是单纯“边界条件没开对”。

## 5. H2 验证链条

仓库里 H2 相关脚本很多，但并不是每一个都应当视为“当前结论脚本”。

### 5.1 当前最重要的 H2 脚本

#### `verify_h2.py`
- uniform H2 vs PySCF 基线曲线
- 回答：uniform 路线是否稳定、当前 PySCF 对照口径是什么

#### `verify_h2_adaptive_vs_pyscf.py`
- adaptive H2 vs PySCF 近平衡小范围曲线
- 当前最直接的 adaptive 曲线入口
- 回答：adaptive 在 `R=1.2,1.4,1.6` 这类小范围上误差量级如何

#### `check_h2_r14_energy_breakdown.py`
- `R=1.4` 单点总能分解
- 回答：`Ts / Eloc / Enl / Eh / Exc / Eion` 中谁在拉偏 adaptive 总能

#### `check_h2_r14_local_energy_profile.py`
- `R=1.4` 的 `e_loc(r)=rho(r)V_loc(r)` 空间分解与密度 profile
- 回答：`dEloc` 主要来自 near-core、bond region 还是 outer region

### 5.2 仍保留但不应直接当作主结论的脚本

`study_h2_*` 这组脚本覆盖过：
- boundary follow-up
- frozen Veff
- state matching
- solver parity
- branch switching
- local-field consistency
- kinetic prototype

它们在开发过程中非常有用，但当前 main 的 README / 方法文档不应把这些阶段性工具逐条升级成“最终结论”。

## 6. 当前最可信的数值结论

## 6.1 单点结论（较可信）

在当前 main、并且已经统一关键验证脚本随机初始条件口径之后：

- `R = 1.4 Bohr`
- `box = 30`
- `h_min = 0.16`
- `h_max = 0.32`
- `r_core = 1.0`
- `stretch_beta = 5.0`
- `hartree_boundary_mode = uniform_exterior`
- `kinetic_mode = prototype_fd2`
- `max_iter = 120`
- `tolerance = 1e-5`

得到的 adaptive vs PySCF 单点误差大约为：
- **`-4.9 mHa`**

这是当前最可信、也最值得写入文档的 adaptive 单点结果量级。

## 6.2 曲线结论（仍需克制）

当前可以比较稳地说：
- adaptive 在近平衡小范围上已经能够给出负总能和合理的最低点区间；
- 但还不能把整条 H2 曲线说成“完全收敛到 uniform / PySCF”。

因此：
- **单点结论比整条曲线结论更可信**；
- README 与本文档都不应把单点进展夸大成“adaptive H2 曲线问题已经解决”。

## 7. 当前主要误差来源理解

根据当前 main 上最可信的审计结果：

1. `R=1.4` 单点的 adaptive vs uniform 总能差，不是总能公式错误；
2. 最大的分量差来自 **`Eloc`**；
3. `Ts` 明显参与，但更像部分补偿项；
4. `Enl` 很小，不是主矛盾；
5. `Eloc` 的空间分解进一步显示：
   - 偏差主要集中在 **near-core 区域**；
   - outer 区域有明显反向补偿；
   - 更细参数可以减小这一偏差，但尚未完全消除。

因此，当前更合理的表述是：
- 剩余几 mHa 误差更像是 **near-core resolution / density redistribution** 问题；
- 而不是 Hartree bookkeeping、`V_loc` 公式本身错误，或简单的总能口径错误。

## 8. 当前性能状态

## 8.1 已优化的热路径

当前保留在 main 的性能优化主要包括：
- adaptive SCF 与 adaptive eigensolver 的 `while_loop` 化；
- adaptive Hartree warm-start；
- adaptive `uniform_exterior` 缓存；
- 若干 trace-safe 改造，减少 JAX tracing 时的 Python 热路径开销。

## 8.2 当前真正的瓶颈

即便如此，adaptive 路线当前仍然慢，主要瓶颈仍在：
- 细网格下的 adaptive SCF 迭代成本；
- adaptive Hartree / CG 路径；
- near-core 分辨率提高后带来的网格与算子成本。

需要明确的是：
- “已经比以前快一些”
- **不等于**
- “性能问题已经解决”。

## 8.3 已被放弃的优化

最近一个重要教训是：
- adaptive metric eigensolver 中的 `HX/HR` 复用虽然能省掉一次 `apply_h`，
- 但它在 H2 单点上引入了可见精度回退，
- 因而已从当前 main 的主线配置中回退。

这也是为什么本文档强调：
- 当前 main 的“性能优化”只包含仍被保留的部分；
- 已经被证明会带来精度回退的实验优化，不应再写成当前主线特性。

## 9. 当前默认推荐验证配置

## 9.1 快速 sanity 配置

用途：
- 确认 adaptive H2 链路能跑通；
- 看总能是否为负；
- 看最低点是否落在合理区间；
- 不追求 mHa 级精度。

建议参数：
- `R = 1.2, 1.4, 1.6`
- `box = 30`
- `h_min = 0.20`
- `h_max = 0.40`
- `r_core = 1.0`
- `stretch_beta = 5.0`
- `hartree_boundary_mode = uniform_exterior`
- `kinetic_mode = prototype_fd2`
- `max_iter = 60`
- `tolerance = 1e-4`

## 9.2 更可信的 `R=1.4` 单点配置

用途：
- 当前最可信的 adaptive 单点审计配置；
- 用于 energy budget、local profile 和 adaptive vs PySCF 单点对齐。

建议参数：
- `R = 1.4`
- `box = 30`
- `h_min = 0.16`
- `h_max = 0.32`
- `r_core = 1.0`
- `stretch_beta = 5.0`
- `hartree_boundary_mode = uniform_exterior`
- `kinetic_mode = prototype_fd2`
- `max_iter = 120`
- `tolerance = 1e-5`

## 10. 后续路线

当前更值得推进的方向，不是再去重写 README 里已经被推翻的旧解释，而是：

1. **继续压 adaptive 性能**
   - 但保持“先不伤精度”的原则；
2. **继续定位剩余几 mHa 误差的 near-core 来源**
   - 尤其是 `Eloc` 与核附近密度分布；
3. **把单点结论稳步扩展到曲线级验证**
   - 先近平衡，再逐步扩展扫描范围；
4. **保持 uniform 基线与 PySCF 对照口径稳定**
   - 避免把脚本调用差异误判成物理结论。

本文档比 README 更技术，但仍然只记录当前 main 分支的真实实现状态。若与旧实验笔记、旧图、旧注释冲突，请以当前代码、当前脚本默认行为和当前审计结果为准。
