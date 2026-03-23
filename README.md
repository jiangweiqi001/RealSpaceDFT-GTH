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

## 🧪 核心验证结果 (Verification Results)

本代码库包含多组自动化测试脚本，证明了底层的物理严谨性：

### 1. 强极性双原子分子极限解离曲线 (`verify_hcl.py`, `verify_h2.py`)
- **发现与解决**: 针对 HCl 等分子在长键长 (d > 3.0 Bohr) 大真空盒内易发生 **“严重电荷震荡 (Charge Sloshing)”** 的数值难题，引入了智能的 **三段式自适应 SCF 收敛策略** (针对高危解离区降低 `mix_alpha` 并增加 `max_iter`)。
- **结果**: JaxDFT 的解离曲线与 PySCF (`gth-tzvp`) **实现了严丝合缝的高度平行**，极限解离点的尖刺被完美抹平，彻底消除了单边边界碰撞（Boundary Squeezing）导致的能量异常。

### 2. 盒子大小独立性测试 (`check_box.py`)
- **结论**: 扫描了从 `L=10.0` 到 `L=36.0` Bohr 的超大盒子。结果显示，只要边界留有足够的衰减空间，体系总能量即保持恒定。这从数值上绝对证明了内核已成功实现 **孤立开边界体系**，规避了传统平面波 (Plane-Wave) 代码面临的周期性镜像排斥问题。


---

## Numerical Schemes

JaxDFT now contains **two real-space solution routes**:

### 1. Uniform Grid

The original route uses a **pure uniform Cartesian grid**:
- fixed spacing in all three directions
- standard real-space finite-difference kinetic operator
- FFT-based Hartree / Poisson treatment for the uniform grid path
- this remains the simplest baseline for verification and regression

This path is still the easiest way to do controlled spacing sweeps and compare against PySCF-style reference curves.

### 2. Adaptive Tensor Grid

The newer route uses an **Adaptive Tensor Grid**:
- each Cartesian axis is redistributed independently
- the grid remains tensor-product structured, but spacing is no longer uniform
- fine spacing is concentrated near nuclei and coarser spacing is used farther away
- integration, inner products, and nonlocal projector overlaps are all evaluated with adaptive volume weights

This path is intended to reduce unnecessary vacuum cost while keeping a real-space discretization.

## Adaptive Tensor Grid

The adaptive route is not an unstructured mesh. It is a **tensor-product nonuniform grid** built from three 1D adaptive axes:
- x = create_adaptive_axis(...)
- y = create_adaptive_axis(...)
- z = create_adaptive_axis(...)

Key properties:
- the grid is still logically Cartesian
- the spacing can vary strongly near the ionic cores
- the SCF layer still works with a full 3D field representation
- Dirichlet masking is used on the outer boundary of the adaptive box

In practice, this means JaxDFT currently supports both:
- a simple, easier-to-interpret **uniform-grid baseline**
- a more compact but more numerically delicate **adaptive tensor-grid route**

## CG Solver

For the adaptive Poisson problem, JaxDFT assembles a sparse finite-volume-style box operator for

-Laplace(V_H) = 4 pi rho

on the adaptive tensor grid.

Algorithmically, this adaptive Hartree solve is formulated as a sparse linear system and supports a **CG-based solver path** for the box Poisson problem. In addition, implementation-level acceleration may reuse cached sparse operators / factorizations so that repeated SCF Hartree solves do not rebuild the same linear algebra objects every iteration.

The key idea is that the **adaptive Hartree operator is geometry-dependent but density-independent**, so the expensive matrix construction can be cached once per grid.

## Multipole Boundaries

The adaptive Hartree route supports several boundary-data models for the finite box:
- zero_dirichlet
- monopole_dirichlet
- multipole_dirichlet
- uniform_exterior

multipole_dirichlet is the natural extension of the older monopole fallback: instead of using only the total charge, it uses low-order multipole information to provide more isolated-like boundary values on the six box faces.

uniform_exterior is the current practical adaptive mainline for H2-style studies: the adaptive interior operator is kept, but boundary values are obtained from a larger, coarse auxiliary uniform exterior solve.

So at a high level, the codebase now contains two complementary solution families:
- **Uniform Grid + FFT Poisson**
- **Adaptive Tensor Grid + Sparse/CG-style Poisson + Multipole/Exterior boundary models**

