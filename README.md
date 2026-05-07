# JaxDFT: Real-Space DFT with GTH Pseudopotentials

**JaxDFT** 是一个基于 **JAX** 实现的实空间 (Real-Space) 密度泛函理论 (DFT) 计算包。它专为高性能计算设计，支持自动微分，并实现了标准的 **GTH (Goedecker-Teter-Hutter)** 赝势。

**当前版本专为孤立体系 (Isolated Systems / Open Boundary Conditions) 设计，采用严格的 Dirichlet 边界条件。**

> 📖 **[点击此处查看完整的 JaxDFT 核心算法与物理公式文档 (ALGORITHM.md)](ALGORITHM.md)**

---

## 🚀 核心特性 (Key Features)

- **实空间求解 (Real-Space Grid)**: 
  - 采用**非周期性** 4 阶中心有限差分计算动能 (Dirichlet 边界)。
  - 采用 **Hockney 补零 FFT 法**求解泊松方程，彻底消除镜像相互作用，实现纯正的孤立体系。
- **Kohn-Sham 求解器**:
  - 内置 LOBPCG 迭代求解器（带 Rayleigh-Ritz 子空间对角化的安全梯度下降法）和适用于小规模网格的稠密矩阵求解器。
- **GTH 赝势 (GTH Pseudopotentials)**:
  - 完整实现了标准的 GTH-LDA 局域势，以及支持全矩阵计算的非局域势 ($s$, $p$ 通道)。
- **物理精度对齐 (Physics Benchmark)**:
  - **XC 泛函**: LDA (Slater Exchange + Perdew-Zunger 1981 Correlation)。
  - **验证**: 与 PySCF 孤立体系高精度基组 (`gth-tzvp`) 进行了严格的绝对能量对齐。

---

## 🛠️ 安装与环境建议 (Installation)

推荐环境：

- `WSL2 Ubuntu` + `Python 3.11` 或 `Python 3.12`
- CPU 先跑通即可；后续再根据 JAX 安装方式切到 GPU

当前项目和对比脚本更推荐在 Linux/WSL Python 环境下运行，而不是直接用 Windows 原生 Python。原因是 `PySCF` 在 Linux/WSL 下更容易直接安装到可用 wheel；在 Windows 尤其较新的 Python 版本上，常会退回源码编译并要求额外的 C/C++ 构建工具。

不推荐环境：

- `Windows + Python 3.13` 作为主要开发/验证环境

一个稳妥的 WSL 安装示例：

```bash
cd /home/footman/RealSpaceDFT_Project/RealSpaceDFT-GTH
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r JaxDFT/requirements.txt
```

如果只运行 JaxDFT 主程序，上面的核心依赖通常就够了。
如果还要运行和 PySCF 的对比脚本，再额外安装：

```bash
python -m pip install pyscf matplotlib
```

建议先确认解释器和依赖：

```bash
python --version
python -m pip list | grep -E "jax|jaxlib|pyscf|matplotlib"
```

---

## 💻 快速开始 (Quick Start)

JaxDFT 提供了极简的 API 来运行分子体系的 SCF 计算（所有物理量采用 **原子单位制 Hartree Atomic Units**）：

```python
import jax
import jax.numpy as jnp
from JaxDFT.src.hamiltonian import create_grid
from JaxDFT.src.io import load_pseudopotentials
from JaxDFT.src.solver import energy_and_forces

# 1. 创建网格 (单位: Bohr)
spacing = 0.18
box_size = [18.0, 18.0, 18.0]
grid = create_grid(spacing, box_size)

# 2. 加载 GTH 赝势
pseudos = load_pseudopotentials(['H', 'Cl'], 'JaxDFT/data/gth_potentials')

# 3. 设置分子坐标 (居中对称拉伸，单位: Bohr)
d = 3.2
coords = jnp.array([[0.0, 0.0, -d/2.0], [0.0, 0.0, d/2.0]])

# 4. 运行 SCF 计算
key = jax.random.PRNGKey(42)
energy, forces = energy_and_forces(
    grid, coords, pseudos,
    max_iter=500, mix_alpha=0.3, tolerance=1e-5, key=key
)
print(f"总能量: {energy:.6f} Hartree")
```

---

## 🔬 双重网格与 PySCF 对比 (Fine Grid Benchmark)

对于随机生成的 C/H/O/N 构型，原子核通常不会正好落在主网格中心。当前推荐用法是在原子核附近启用局部细网格采样，以缓解 GTH 局域赝势带来的 egg-box 误差：

```python
energy, forces = energy_and_forces(
    grid, coords, pseudos,
    max_iter=500, mix_alpha=0.3, tolerance=1e-5, key=key,
    fine_grid_mode="auto",
    fine_subgrid=5,
    fine_grid_radius_factor=4.0,
)
```

当前统一接口的推荐理解如下：

- `fine_grid_mode="auto"`:
  - 当前推荐自动策略。
  - 只对局域赝势 `V_loc` 启用 atom-centered patch fine grid。
  - 非局域 projector 当前不会被 `auto` 自动切到细网格，因为这部分还在继续校准。
  - 可以把它理解成当前唯一稳定的主线路径。
- `fine_grid_mode="atom_patch"`:
  - 语义上等同于显式要求原子核附近的 patch 细网格。
  - 适合做可控实验或和 `auto` 对照。
- `fine_grid_mode="off"`:
  - 关闭细网格，回到主网格上的普通点采样/默认粗网格处理。
- `fine_subgrid`:
  - 每个主网格单元内部使用的细分数。
  - 例如 `fine_subgrid=5` 表示每个 coarse cell 沿每个方向再细分 5 份。
- `fine_grid_radius_factor`:
  - 控制每个原子核附近 patch 的作用半径。
  - 当前实现是按原子的局域赝势长度尺度 `rloc` 扩张，并在主网格单元尺度上再补一个小余量。

当前 `fine_grid_mode="auto"` 的实际行为可以概括为：
对每个原子，各自在核附近自动打开一个 `V_loc` 细网格 patch；离开这些 patch 后，仍然回到主网格计算。也就是说，细网格不是全盒子开启，而是只在原子核附近局部触发，这正是它兼顾精度和效率的关键。

关于 nonlocal projector：

- `projector_mode="patch"` 目前保留为**实验性研究能力**。
- 它不会被 `fine_grid_mode="auto"` 自动启用。
- 当前主线 benchmark 和推荐结论只基于 `off` 与 `fine_grid_mode="auto"` 的比较。
- 如果要研究 projector patch，建议单独在专门脚本或 `projector_sweep` 模式下使用，不要把它和主线精度结论混在一起。

可以用 H2O 对称伸缩作为一个小而敏感的基准体系：整体平移半个主网格 spacing，使 O/H 原子都不在主网格点上，然后比较同一主网格下 baseline 与 `fine_grid_mode="auto"` 是否更接近 PySCF。

```bash
MPLBACKEND=Agg python3 JaxDFT/scripts/compare_fine_grid_pyscf.py \
  --spacing 0.40 \
  --box-size 9.6 \
  --output-prefix h2o_fine_grid_compare
```

该脚本使用 PySCF 的 `gth-tzvp` 基组、`gth-lda` 赝势和 `lda,pz` 泛函作为参考，并输出：

- `h2o_fine_grid_compare.csv`
- `h2o_fine_grid_compare.png`

在默认 H2O off-grid 测试中，baseline 相对 PySCF 的误差约为 `0.65-3.61 Ha`，而 `fine_grid_mode="auto"` 后误差降到约 `0.029-0.042 Ha`。这说明在相同主网格下，原子核附近的 `V_loc` 细网格 patch 能显著改善与 PySCF 的一致性。

---

## 🧪 核心验证结果 (Verification Results)

本代码库包含多组自动化测试脚本，证明了底层的物理严谨性：

### 1. 强极性双原子分子极限解离曲线 (`verify_hcl.py`, `verify_h2.py`)
- **发现与解决**: 针对 HCl 等分子在长键长 (d > 3.0 Bohr) 大真空盒内易发生 **“严重电荷震荡 (Charge Sloshing)”** 的数值难题，引入了智能的 **三段式自适应 SCF 收敛策略** (针对高危解离区降低 `mix_alpha` 并增加 `max_iter`)。
- **结果**: JaxDFT 的解离曲线与 PySCF (`gth-tzvp`) **实现了严丝合缝的高度平行**，极限解离点的尖刺被完美抹平，彻底消除了单边边界碰撞（Boundary Squeezing）导致的能量异常。

### 2. 盒子大小独立性测试 (`check_box.py`)
- **结论**: 扫描了从 `L=10.0` 到 `L=36.0` Bohr 的超大盒子。结果显示，只要边界留有足够的衰减空间，体系总能量即保持恒定。这从数值上绝对证明了内核已成功实现 **孤立开边界体系**，规避了传统平面波 (Plane-Wave) 代码面临的周期性镜像排斥问题。

---

## Current Status / Known Limitations

- ??????? fine_grid_mode="auto"???????? V_loc ?? atom-centered patch fine grid?
- projector_mode="patch" ?? experimental ??????????????????? auto ?????
- ? spacing >= 0.2 Bohr ???????? PySCF ??????? 10-100 mHa ??????? 1-5 mHa ???
- forces ????????????????????????????????????????????

---

## V5 / Enriched Galerkin Requirements

This section defines the requirements for the next high-accuracy local
representation branch.  The current stable V4 tools are useful as
post-SCF diagnostics, but they must not be treated as the final route to
1-5 mHa accuracy.

### Goal

Build an atom-centered enriched Galerkin representation that can improve
H2O/N2/C/H/O/N systems at coarse-grid spacing `>= 0.2 Bohr` without relying on
global grid refinement.

The target is not a heuristic projector/local-potential patch.  The target is a
single mixed representation where:

- the coarse grid basis and atom-centered local basis form one Galerkin space;
- `H`, `S`, density reconstruction, and total energy all come from the same
continuous bilinear forms and the same physical mapping;
- patch/local basis functions are allowed to improve occupied states strongly
enough to matter, while preserving the meaning of the coarse main block.

### Hard Requirements

- The enriched basis must remain coarse-preserving.  Embedding a pure coarse
  vector `[c; 0]` must reproduce the coarse baseline Rayleigh quotient under the
  chosen baseline operator, up to normal solver/discretization tolerance.
- Patch/local basis functions may be orthogonalized against the coarse space,
  but they must not be constrained so strongly that occupied patch metric
  fractions stay at `1e-6` by construction.  If the enriched basis never carries
  meaningful occupied weight, it cannot deliver the target accuracy.
- `H` and `S` must be assembled from one consistent Galerkin projection.  Do not
  mix residual blocks, old coarse operators, and patch-local matrices unless the
  residual is explicitly derived from the same mapping.
- `rho` must be reconstructed from the same mixed basis used by the eigenproblem.
  Density back-projection must be conservative and must not double count the
  coarse contribution near nuclei.
- `V_loc`, `V_nl`, Hartree, and XC contributions must use a consistent basis and
  density interpretation.  A patch density contribution must not enter the
  energy functional unless the corresponding Hamiltonian contribution is defined
  with the same semantics.
- Total energy must have a mixed-basis audit path.  Reporting only
  `E_band - E_H + E_xc - int rho v_xc + E_ion` with coarse-grid assumptions is
  not sufficient until the mixed density and mixed Hamiltonian are proven to be
  one closed discretization.

### Explicit Non-Goals / Forbidden Shortcuts

- Do not make `delta_v_grid=0` post-SCF semantics the final SCF model.  That is a
  diagnostic-safe choice, not a complete enriched DFT functional.
- Do not tune `patch_radius_factor`, `patch_subgrid`, penalties, or stiffness
  constants to hide a basis/energy inconsistency.
- Do not let patch-local `V_loc` or projector information silently rewrite the
  coarse `cc` block.  Any `cc` change must be a deliberate Galerkin projection
  choice and must be tested against the `[c; 0]` preservation check.
- Do not claim PySCF-level accuracy from eigenvalue shifts alone.  The total
  energy decomposition and density feedback loop must also be consistent.
- Do not promote the experimental branch to the mainline until H2O and N2 both
  pass the fixed-potential, density-feedback, and total-energy audits below.

### Required Validation Order

1. Fixed-`V_eff` matrix audit.
   Check symmetry, positive `S`, `[c; 0]` preservation, band decomposition, and
   patch/coarse metric fractions for H2O and N2.
2. Density reconstruction audit.
   Check electron conservation, coarse/patch density split, nuclear-region
   weight changes, and absence of local double counting.
3. Energy decomposition audit.
   Report `E_band`, `E_H`, `E_xc`, `int rho v_xc`, `E_ion`, and total energy
   using the same mixed representation.  Re-solving at final `V_H + v_xc` must
   not change the reported total energy beyond the chosen tolerance.
4. Damped density-feedback trace.
   Start from a converged baseline state.  Use small fixed damping such as
   `alpha=0.05`.  Record `rho_l1`, `V_H + v_xc` change, occupied-band
   decomposition, patch fraction, and nuclear weights for 2-5 steps.
5. Small SCF bridge.
   Only after the previous audits pass, run a baseline-initialized damped SCF
   bridge.  Anderson or more aggressive mixing comes later.
6. PySCF comparison.
   Compare H2O and N2 against PySCF `gth-tzvp`, `gth-lda`, `lda,pz` using the
   same geometry.  Report both total error and decomposition, not only total
   energy.

### Exit Criteria Before Mainline Consideration

- H2O and N2 fixed-`V_eff` tests show meaningful but non-collapsing patch
  response.
- Patch metric fractions are physically interpretable rather than fixed near
  zero by construction.
- Density feedback does not create runaway nuclear localization.
- Total energy is invariant under a final same-`V_H/v_xc` re-solve to within the
  solver tolerance.
- H2O and N2 total-energy errors move toward the 5 mHa target for the right
  reason, confirmed by energy decomposition and density audits.

### V5b Rewrite Plan

The current V5 implementation is useful as a diagnostic branch, but it does not
yet satisfy the requirements above.  The next implementation should be treated
as a V5b rewrite rather than a parameter-tuning pass on the current branch.
Every step below must include an explicit compliance check before moving to the
next step.

#### Current V5 Compliance Snapshot

| Requirement | Current status | Reason |
| --- | --- | --- |
| Coarse-preserving `[c; 0]` semantics | Mostly satisfied | The coarse block is preserved and has tests/audits for embedded coarse Rayleigh quotients. |
| One Galerkin `H/S` space | Partially satisfied | `S`, `T`, and `V_loc` have patch/cross blocks, but `V_nl` is still coarse-only. |
| Mixed density from the same physical mapping | Partially satisfied | Density uses a patch increment `total_patch^2 - coarse_trace^2`, but this is not yet a fully closed mixed-basis density functional. |
| Hxc and total-energy consistency | Not satisfied | The total-energy audit still uses a standard coarse-grid double-counting form and is explicitly diagnostic-only. |
| No heuristic stabilization | Not satisfied | `vloc_aware_constraint` is a protective constraint that can suppress N2 over-localization but also weakens H2O corrections. |
| Better H2O/N2 accuracy than mainline | Not satisfied | Recent H2O checks show V5 is not more accurate than the current `V_loc` auto-patch mainline. |

#### Step 1: Define One Physical Mapping

Goal: define a single physical map `Phi` from mixed coefficients to values on
the selected evaluation support.

Implementation requirements:

- `Phi([c; a])` must include the coarse contribution and the atom-centered
  enriched contribution in one representation.
- The support, interpolation rule, volume weights, and boundary convention must
  be explicit and reused by all operators.
- A pure coarse state `[c; 0]` must reproduce the selected coarse baseline
  semantics exactly within solver tolerance.

Compliance checks:

- Pass `[c; 0]` mass/Rayleigh preservation tests for H2O and N2.
- Report `||Phi([c;0]) - coarse_trace(c)||` on every patch.
- Fail the step if any operator uses a different support or interpolation rule.

Current implementation status:

- Started in `JaxDFT/src/mixed_basis_enriched_v5b.py` as a fixed-`V_eff` dense
  prototype. The first slice keeps the coarse block frozen to the baseline
  semantics and builds atom-centered patch coordinates on the same patch
  interpolation support.
- Covered by `MixedBasisEnrichedV5bTest`; full H2O/N2 `[c;0]` reporting is
  still pending before any SCF bridge is attempted.

#### Step 2: Rebuild `S`, `T`, `V_loc`, and `V_nl` From `Phi`

Goal: assemble every Hamiltonian and metric block from the same Galerkin map.

Implementation requirements:

- `S = <Phi_i | Phi_j>`.
- `T = 1/2 <grad Phi_i | grad Phi_j>` or the documented finite-difference
  equivalent on the same support.
- `V_loc = <Phi_i | V_loc | Phi_j>` using the same local-potential semantics as
  the density/energy path.
- `V_nl = sum_ij h_ij <Phi_i|p_i><p_j|Phi_j>` using the same projector support,
  interpolation, and normalization as the rest of V5b.
- No residual block such as `H_cc - T_cc - Vloc_cc` may be used unless it is
  derived from the same `Phi` mapping and documented.

Compliance checks:

- Dense and matrix-free operators agree on a small grid.
- `H = T + V_loc + V_nl` recomposes eigenvalues band-by-band.
- `S` is symmetric positive definite after removing numerically null modes.
- H2O and N2 fixed-`V_eff` audits show nonzero but non-collapsing patch
  fractions.

Current implementation status:

- Dense fixed-`V_eff` `S/T/V_loc/V_nl` assembly is implemented in
  `build_fixed_veff_enriched_components_v5b()`.
- Unlike V5, `V_nl` now has explicit coarse-patch and patch-patch projector
  blocks on the patch support.
- Verified so far: matrix symmetry, nonzero projector patch blocks, and
  band-by-band `H = T + V_loc + V_nl` recomposition on tiny tests.
- Pending: matrix-free parity and real H2O/N2 fixed-`V_eff` audits.

#### Step 3: Rebuild Density From the Same Mixed State

Goal: reconstruct `rho` from `Phi(psi)` rather than from a separate heuristic
patch correction.

Implementation requirements:

- Compute density on the same physical support used by `Phi`.
- Conservative back-projection to the coarse grid must preserve total electron
  count.
- The coarse contribution near nuclei must not be double counted.
- The density routine must expose coarse, patch, and cross/increment
  contributions for auditing.

Compliance checks:

- Electron count error stays below the chosen tolerance for H2O and N2.
- Nuclear-region density weights do not show runaway localization.
- If patch coefficients are zero, density reduces to the baseline coarse
  density.
- The density split explains the sign and magnitude of Hxc feedback.

#### Step 4: Define A Consistent Mixed-Basis Energy Audit

Goal: evaluate total energy using the same density and Hamiltonian semantics.

Implementation requirements:

- Report `E_band`, `E_H`, `E_xc`, `int rho v_xc`, `E_ion`, and total energy.
- The density used by `E_H`, `E_xc`, and `int rho v_xc` must be the V5b density.
- A final same-`V_H/v_xc` re-solve must leave the reported total energy stable
  within solver tolerance.
- Do not claim accuracy from eigenvalue shifts alone.

Compliance checks:

- Same-Hxc re-solve delta is small for H2O and N2.
- Energy decomposition identifies whether changes come from `T`, `V_loc`,
  `V_nl`, or Hxc feedback.
- Total-energy trends match the density audit; if not, stop and fix the
  energy/density semantics before SCF.

#### Step 5: Run Damped Feedback Before Full SCF

Goal: test self-consistent feedback without hiding instabilities behind
aggressive mixing.

Implementation requirements:

- Start from a converged mainline baseline state.
- Use fixed damping, initially `alpha=0.05`.
- Record per-step density change, `V_H + v_xc` change, band decomposition,
  patch fraction, and nuclear weights.

Compliance checks:

- H2O and N2 do not show positive feedback into near-core localization.
- Patch fraction remains physically interpretable and is not forced to zero by
  constraints.
- `vloc_aware_constraint` remains off by default; if used, it must be reported
  as a diagnostic variant, not a production fix.

#### Step 6: Only Then Run A Small SCF Bridge

Goal: run a baseline-initialized damped V5b SCF bridge only after the fixed
operator, density, and energy audits pass.

Implementation requirements:

- Reuse the same V5b `Phi`, operators, density, and energy audit from previous
  steps.
- Keep fixed damping first; Anderson mixing comes later.
- Compare both constrained and unconstrained diagnostics if constraints still
  exist, but do not promote a constrained result unless it has a physical basis.

Compliance checks:

- H2O and N2 converge or show bounded monotonic feedback under fixed damping.
- Total energy is stable under same-Hxc re-solve at the final state.
- The bridge improves or at least does not degrade the mainline trend before any
  larger benchmark is attempted.

#### Step 7: Compare Against Mainline And PySCF

Goal: decide whether V5b has a real advantage over the stable mainline.

Implementation requirements:

- Compare H2O and N2 using the same geometries against:
  - mainline baseline;
  - mainline `fine_grid_mode="auto"` (`V_loc` atom patch only);
  - V5b;
  - PySCF `gth-tzvp`, `gth-lda`, `lda,pz`.
- Report total energies, errors, energy decomposition, density metrics, patch
  fractions, and runtime.

Compliance checks:

- V5b must improve the total-energy trend for H2O and N2 for the right reason.
- If V5b is worse than `fine_grid_mode="auto"`, keep it experimental and return
  to the failed audit step.
- Do not promote V5b unless both systems pass fixed-potential, feedback, and
  total-energy checks.
