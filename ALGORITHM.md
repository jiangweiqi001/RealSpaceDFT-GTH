# JaxDFT 工具包算法文档

## 概述

JaxDFT 是一个基于 **JAX** 实现的实空间 (Real-Space) 密度泛函理论 (DFT) 计算包，专为孤立体系 (Isolated Systems) 设计，采用开边界条件 (Open Boundary Conditions) 和 Dirichlet 边界条件。

---

## 1. DFT计算方法

### 1.1 整体流程

JaxDFT 采用标准的 Kohn-Sham DFT 自洽场 (SCF) 迭代方法：

```
┌─────────────────────────────────────────────────────────────┐
│                     SCF 迭代循环                             │
├─────────────────────────────────────────────────────────────┤
│  1. 初始化电子密度 ρ(r)                                       │
│  2. 构建有效势 V_eff = V_loc + V_H + V_xc + V_nonlocal       │
│  3. 求解 Kohn-Sham 方程: Hψ_i = ε_iψ_i                       │
│  4. 计算新密度: ρ_new = Σ|ψ_i|² × occ_i                      │
│  5. Anderson 密度混合 → 更新密度                              │
│  6. 检查收敛性 (|ρ_new - ρ| < tolerance)                     │
│  7. 未收敛则返回步骤 2                                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 实空间网格 (Real-Space Grid)

- **网格生成**: 使用 `create_grid(spacing, box_size)` 创建均匀三维网格
- **网格坐标**: 以原点为中心，`x, y, z ∈ [-box_size/2, box_size/2]`
- **体素体积**: `volume_element = spacing³` (单位: Bohr³)

### 1.3 动能算符 - 四阶有限差分

采用 **4阶中心差分** 计算拉普拉斯算符（动能项）:

```
∇²ψ ≈ c₀·ψ(i,j,k)
    + c₁·[ψ(i±1,j,k) + ψ(i,j±1,k) + ψ(i,j,k±1)]
    + c₂·[ψ(i±2,j,k) + ψ(i,j±2,k) + ψ(i,j,k±2)]

其中:
  c₀ = -2.5 / h²
  c₁ = (4/3) / h²
  c₂ = (-1/12) / h²
  h = grid spacing
```

**特点**:
- 非周期性边界，采用 **Dirichlet 边界条件** (边界处 ψ = 0)
- 通过 `shift_array` 函数处理边界截断

### 1.4 势能组成部分

#### 1.4.1 局域离子势 (GTH Local Potential)

GTH (Goedecker-Teter-Hutter) 局域赝势形式:

```
V_loc(r) = -Z_ion/r · erf(r/(√2·r_loc)) + exp(-(r/r_loc)²/2) × Σ c_i·(r/r_loc)^(2i)
```

参数说明:
- `Z_ion`: 离子有效电荷 (价电子数)
- `r_loc`: 局域势半径参数
- `c = [c₁, c₂, c₃, c₄]`: 高斯多项式系数

#### 1.4.2 非局域势 (GTH Nonlocal Potential)

支持 l=0 (s通道) 和 l=1 (p通道) 的非局域投影:

```
V_nonlocal ψ = Σ_{i,j,l,m} |p_i^{lm}⟩ h_{ij}^l ⟨p_j^{lm}|ψ⟩
```

投影函数形式:
```
p_i^l(r) = N_{il} · r^(l+2i-2) · exp(-r²/(2r_p²))

归一化系数:
N_{il} = √2 / (r_p^(l+(4i-1)/2) · √Γ(l+(4i-1)/2))
```

支持全矩阵 `h_{ij}` 计算（多个投影器之间的耦合）。

#### 1.4.3 Hartree 势 (库仑势)

采用 **Hockney 补零 FFT 法** 求解泊松方程:

```python
# 1. 对密度进行 2 倍零填充
rho_pad = pad(rho, 2x size)

# 2. 构建库仑核并 FFT
kernel(r) = 1/r  (r>0),  2.38/spacing  (r≈0)
V_H = IFFT(FFT(rho_pad) × FFT(kernel)) × spacing³

# 3. 截取有效区域
V_H = V_H[:nx, :ny, :nz]
```

**关键特性**:
- 消除周期性镜像相互作用
- 实现真正的孤立体系计算

#### 1.4.4 交换相关势 (XC)

采用 **LDA (Local Density Approximation)**:
- **交换项**: Slater 交换 (`lda_exchange_vxc`)
  ```
  V_x = -(3/π)^(1/3) · ρ^(1/3)
  ε_x = 3/4 · V_x · ρ
  ```

- **相关项**: Perdew-Zunger 1981 (PZ81) 参数化
  ```
  高密度 (r_s < 1): 对数多项式形式
  低密度 (r_s ≥ 1): 有理分式形式

  r_s = (3/(4πρ))^(1/3)  # Wigner-Seitz 半径
  ```

### 1.5 Kohn-Sham 方程求解

#### 1.5.1 LOBPCG 迭代求解器

使用带 Rayleigh-Ritz 子空间对角化的安全梯度下降法:

```
1. 初始化随机波函数 X (加微小噪声打破对称性)
2. 正交化: X = QR(X)
3. 迭代直到收敛:
   a. 计算 HX = H(X)
   b. Rayleigh-Ritz: 在子空间 [X] 中对角化 H_sub = XᵀHX
   c. 更新波函数: X = X · V_sub (特征向量矩阵)
   d. 计算残差: R = HX - X·E
   e. 梯度更新: X_new = X - α·R  (α=0.002)
   f. 重新正交化
```

#### 1.5.2 稠密矩阵求解器 (备选)

对于小规模网格 (N ~ 20,000)，可构建完整哈密顿矩阵:
```
H_dense = [H·e₁, H·e₂, ..., H·e_N]  (N×N 矩阵)
对角化: H_dense · ψ = E · ψ
```

### 1.6 密度混合 (Anderson Mixing)

使用 Anderson 混合加速 SCF 收敛:

```
f = ρ_new - ρ  (残差)

第一次迭代: 简单线性混合
  ρ_next = ρ + α·f

后续迭代 (存储 m=5 步历史):
  最小化 ||f_next||² 得到最优系数 θ
  ρ_next = ρ_new - α·Σ θ_i·(f_i - f_current)
```

### 1.7 总能量计算

```
E_total = E_band - E_Hartree + E_xc - E_vxc + E_ion-ion

其中:
  E_band = Σ ε_i · occ_i                    (能带能量和)
  E_Hartree = 0.5 ∫ ρ·V_H dr                (Hartree 能量)
  E_xc = ∫ ε_xc(ρ)·ρ dr                     (XC 能量)
  E_vxc = ∫ V_xc(ρ)·ρ dr                    (XC 势修正)
  E_ion-ion = Σ_{i<j} Z_i·Z_j / |R_i-R_j|   (离子排斥)
```

---

## 2. 与其他DFT工具的区别

| 特性 | JaxDFT | 传统DFT (如PySCF/Quantum ESPRESSO) |
|------|--------|-----------------------------------|
| **空间表示** | 实空间网格 | 平面波 / 高斯基组 |
| **边界条件** | 孤立体系 (Dirichlet) | 周期性 (PBC) 或混合 |
| **泊松求解** | Hockney 补零 FFT | FFT (周期性) / 直接积分 |
| **自动微分** | JAX 原生支持 | 有限差分或专门实现 |
| **赝势** | GTH 实空间投影 | 多种赝势格式 |
| **XC 泛函** | LDA-PZ81 | 多种泛函可选 |

### 2.1 实空间 vs 平面波

**JaxDFT (实空间)**:
- ✅ 自然处理孤立体系，无镜像相互作用
- ✅ 非周期性边界，适合分子/团簇
- ✅ 局部势能计算直接、高效
- ❌ 动能计算需要更多网格点保证精度

**平面波 (如 Quantum ESPRESSO)**:
- ✅ 动能计算精确
- ✅ 周期性体系自然适配
- ❌ 孤立体系需要大真空层和校正
- ❌ 非局域势计算复杂

### 2.2 边界条件对比

**JaxDFT (开边界)**:
```
盒子边界: ψ(r) = 0
Hartree 势: Hockney 方法消除镜像
```

**传统 PBC**:
```
ψ(r+L) = ψ(r)
Hartree 势: 包含周期性镜像求和 (Ewald)
```

---

## 3. 使用方法和外部变量说明

### 3.1 基本使用流程

```python
import jax
import jax.numpy as jnp
from JaxDFT.src.hamiltonian import create_grid
from JaxDFT.src.io import load_pseudopotentials
from JaxDFT.src.solver import energy_and_forces

# 1. 创建网格
spacing = 0.5          # 网格间距 (Bohr)
box_size = [10, 10, 10]  # 盒子大小 (Bohr)
grid = create_grid(spacing, box_size)

# 2. 加载赝势
pseudos = load_pseudopotentials(['H', 'C'], 'path/to/potentials')

# 3. 设置原子坐标 (Bohr)
coords = jnp.array([[0.0, 0.0, 0.0],    # H
                    [1.4, 0.0, 0.0]])   # C

# 4. 运行 SCF 计算
key = jax.random.PRNGKey(42)
energy, forces = energy_and_forces(
    grid,
    coords,
    pseudos,
    max_iter=100,    # 最大迭代次数
    mix_alpha=0.4,   # 混合参数
    tolerance=1e-5,  # 收敛阈值
    key=key
)

print(f"总能量: {energy:.6f} Hartree")
```

### 3.2 外部变量说明

#### 网格参数

| 变量 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `spacing` | float | Bohr | 网格间距，决定计算精度 |
| `box_size` | [float, float, float] | Bohr | 模拟盒子尺寸 [Lx, Ly, Lz] |
| `grid.shape` | (int, int, int) | - | 网格点数 [Nx, Ny, Nz] |
| `volume_element` | float | Bohr³ | 单个体素体积 = spacing³ |

#### SCF 参数

| 变量 | 类型 | 典型值 | 说明 |
|------|------|--------|------|
| `max_iter` | int | 100-500 | 最大SCF迭代次数 |
| `mix_alpha` | float | 0.3-0.5 | Anderson混合强度，越大收敛越快但可能不稳定 |
| `tolerance` | float | 1e-5 to 1e-6 | 密度收敛阈值 (max\|Δρ\|) |
| `n_bands` | int | auto | 计算的Kohn-Sham轨道数，自动设为 ceil(N_electrons/2) |

#### 赝势参数 (GTH)

| 变量 | 类型 | 说明 |
|------|------|------|
| `q` | int | 价电子数 |
| `zion` | float | 离子有效电荷 (=q) |
| `rloc` | float | 局域势半径 (Bohr) |
| `c` | [float] | 局域势多项式系数 [c₁, c₂, c₃, c₄] |
| `projectors` | list | 非局域投影器列表，每个包含 `l`, `r`, `h` |

#### 物理量单位

所有物理量采用 **原子单位制 (Hartree Atomic Units)**:

| 量 | 单位 | 换算 |
|----|------|------|
| 长度 | Bohr | 1 Bohr ≈ 0.529 Å |
| 能量 | Hartree | 1 Ha ≈ 27.2114 eV ≈ 627.5 kcal/mol |
| 力 | Hartree/Bohr | 1 Ha/Bohr ≈ 51.4 eV/Å |
| 密度 | Bohr⁻³ | |

### 3.3 配置文件格式 (YAML)

```yaml
grid:
  spacing: 0.5          # 网格间距 (Bohr)
  box_size: [10.0, 10.0, 10.0]  # 盒子大小 (Bohr)

sampling:
  n_samples: 10         # 采样数量
  min_distance: 1.5     # 最小原子间距 (Bohr)

elements: ['H', 'C', 'N', 'O', 'Si']  # 可用元素

scf:
  max_iter: 100         # 最大迭代
  mix_alpha: 0.4        # 混合参数
  tolerance: 1.0e-5     # 收敛阈值
```

---

## 4. 外部变量使用注意事项

### 4.1 网格参数选择

#### spacing (网格间距)

| 值 | 适用场景 | 精度 | 计算成本 |
|----|---------|------|---------|
| 0.3 | 高精度计算 | 高 | 高 |
| 0.5 | 标准精度 | 中 | 中 |
| 0.8 | 快速测试 | 低 | 低 |

**注意事项**:
- 网格越密，动能计算越精确
- 推荐值: **0.18-0.5 Bohr** 用于生产计算
- 必须满足 Nyquist 准则: spacing < π/k_max

#### box_size (盒子大小)

**重要**:
- 盒子必须足够大，使波函数在边界衰减到接近零
- 对于分子，建议至少保留 **5-10 Bohr** 真空层
- 盒子过小会导致边界效应和能量误差

```python
# 检查盒子大小是否合适
# 对于典型的共价键分子，盒子应满足:
min_box = molecular_extent + 10.0  # Bohr
```

### 4.2 SCF 收敛调优

#### mix_alpha (混合参数)

- **值过小 (如 0.1)**: 收敛慢但稳定
- **值过大 (如 0.8)**: 可能振荡或发散
- **推荐**: 从 0.3-0.4 开始，不收敛时减小

#### tolerance (收敛阈值)

- **1e-5**: 标准精度
- **1e-6**: 高精度
- **注意**: 过度收紧 (如 1e-8) 可能难以收敛且收益有限

#### max_iter (最大迭代)

- 简单体系: 50-100 次足够
- 困难体系 (金属性、小分子): 可能需要 200-500 次

### 4.3 赝势注意事项

#### 非局域势投影器

- `h` 矩阵可以是对角矩阵或全矩阵
- 对于轻元素 (H, He)，通常只有 s 通道
- 对于重元素，需要 s 和 p 通道

#### 赝势文件格式

```
<元素> GTH-LDA-q<价电子数>
    <q>
    <rloc> <n_c> <c1> <c2> ...
    <n_channels>
    <l> <rp> <n_proj>
      <h_11> <h_12> ...
      <h_22> ...
    <l> <rp> <n_proj>
      <h_11> ...
```

### 4.4 JAX 特定注意事项

#### 随机数种子

```python
key = jax.random.PRNGKey(seed)  # 设置随机种子保证可复现
```

#### JIT 编译

- 首次运行会有编译开销
- 后续运行使用缓存的编译结果
- 大量小计算可能JIT开销占主导

#### 内存使用

- 网格大小直接影响内存需求
- 对于 N 个网格点，哈密顿矩阵需要 O(N²) 内存
- 示例: N=20,000 时，稠密矩阵约 1.6 GB (float32)

### 4.5 常见问题和解决

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| SCF 不收敛 | mix_alpha 太大 | 减小到 0.2-0.3 |
| 能量为 NaN | 盒子太小/网格太稀 | 增大 box_size，减小 spacing |
| 与参考值偏差大 | XC 泛函不匹配 | 确认使用 LDA-PZ81 |
| 非局域势报错 | h 矩阵格式错误 | 检查赝势文件格式 |
| 计算太慢 | 网格太密 | 适当增大 spacing |

---

## 附录: 核心算法公式汇总

### 交换相关能 (LDA-PZ81)

**交换能密度**:
```
ε_x = -3/4 · (3/π)^(1/3) · ρ^(4/3)
```

**相关能密度**:
```
高密度 (r_s < 1):
  ε_c = A·ln(r_s) + B + C·r_s·ln(r_s) + D·r_s

低密度 (r_s ≥ 1):
  ε_c = γ / (1 + β₁√r_s + β₂·r_s)

r_s = (3/(4πρ))^(1/3)
```

### 动能 - 四阶差分

```
Tψ = -1/2 ∇²ψ

∇²ψ ≈ [-2.5·ψ₀ + (4/3)(ψ₊₁ + ψ₋₁) - (1/12)(ψ₊₂ + ψ₋₂)] / h²
```

### Hartree 能 (Hockney)

```
V_H(r) = ∫ ρ(r')/|r-r'| dr'

FFT 域计算:
  Ṽ_H(k) = ρ̃(k) · 4π/k²
  V_H(r) = IFFT[Ṽ_H]
```

---

*文档版本: 1.0*
*基于 JaxDFT 代码库: RealSpaceDFT-GTH*


---

## 5. Uniform Grid and Adaptive Tensor Grid

JaxDFT now has **two numerical routes** for real-space Kohn-Sham DFT.

### 5.1 Uniform Grid

The uniform route uses a standard Cartesian grid with constant spacing:
- one global spacing
- tensor-product Cartesian coordinates
- simple volume element spacing^3
- FFT-based Poisson treatment in the uniform-grid path

This route is the cleanest baseline for verification because the numerical structure is simple and directly comparable across box/spacing sweeps.

### 5.2 Adaptive Tensor Grid

The adaptive route keeps the tensor-product structure but redistributes each 1D axis independently.

Let
- x_i be the adaptive x-axis nodes
- y_j be the adaptive y-axis nodes
- z_k be the adaptive z-axis nodes

Then the 3D coordinates are still


_(i,j,k) = (x_i, y_j, z_k)

but the local spacings

- h_x(i) = x_(i+1) - x_i
- h_y(j) = y_(j+1) - y_j
- h_z(k) = z_(k+1) - z_k

are no longer constant.

The adaptive grid is therefore:
- structured
- tensor-product
- nonuniform
- still compatible with weighted quadrature and separable axis metadata

This is why the current implementation is best described as an **Adaptive Tensor Grid**, not an unstructured finite-element mesh.

## 6. CG Solver

For the adaptive Hartree route, the Poisson problem is written as a sparse linear system

A u = M f

for

-Laplace(u) = f

on the interior nodes of the adaptive tensor grid, with Dirichlet boundary data entering through the right-hand side.

Conceptually, this is a **CG-solver-compatible SPD system**:
- A is the sparse adaptive stiffness/operator matrix
- M is the adaptive mass / control-volume matrix
- the right-hand side changes with 
ho, but the operator does not change as long as the grid geometry stays fixed

This is important algorithmically because it means:
- the adaptive Poisson operator can be assembled once per grid
- repeated SCF iterations reuse the same sparse geometry-dependent operator
- the adaptive Hartree path is naturally expressed as a sparse linear solve rather than an FFT convolution on a globally uniform mesh

In other words, the uniform and adaptive routes differ not only in the grid, but also in the **linear algebra used for Poisson**.

## 7. Multipole Boundaries

Because the adaptive Hartree solve is performed on a finite box, the boundary model matters.
The codebase now supports several finite-box boundary prescriptions:

- zero_dirichlet
- monopole_dirichlet
- multipole_dirichlet
- uniform_exterior

### 7.1 Monopole Boundaries

The monopole boundary model uses the total charge

Q = integral rho(r) dr

and sets boundary values using a charge-center or box-center reference point.

### 7.2 Multipole Boundaries

The multipole boundary model extends the monopole approximation by including low-order moments of the charge distribution. This gives more isolated-like face data than a pure Q/r boundary model and is a better approximation when the box is finite but the density is localized.

### 7.3 Exterior-Assisted Boundaries

uniform_exterior keeps the adaptive interior solve but obtains boundary values from a larger, coarse auxiliary uniform-grid exterior calculation. This provides a practical bridge between:
- the efficiency of the adaptive interior grid
- the more free-space-like behavior of a larger exterior domain

So the modern adaptive Hartree stack should be understood as:
- **Adaptive Tensor Grid** in the interior
- **sparse / CG-style Poisson solve** on that interior grid
- **multipole or exterior-assisted boundary data** on the finite box

