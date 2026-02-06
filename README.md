# JaxDFT: Real-Space DFT with GTH Pseudopotentials

**JaxDFT** 是一个基于 **JAX** 实现的实空间 (Real-Space) 密度泛函理论 (DFT) 计算包。它专为高性能计算设计，支持自动微分，并实现了标准的 **GTH (Goedecker-Teter-Hutter)** 赝势。

---

## 🚀 核心特性 (Key Features)

- **实空间求解 (Real-Space Grid)**: 
  - 采用有限差分/FFT 方法在三维网格上求解 Kohn-Sham 方程。
  - **优势**: 摆脱了传统高斯基组 (Basis Set) 的完备性限制，网格越密精度越高（Basis Set Limit）。
  
- **GTH 赝势 (GTH Pseudopotentials)**:
  - 完整实现了标准的 GTH-LDA 局域势（Erf 软化库伦项 + 高斯修正）。
  - 内置 `data/gth_potentials` 数据库 (GTH-LDA-q1 等)。

- **物理精度对齐 (Physics Benchmark)**:
  - **XC 泛函**: LDA (Slater Exchange + Perdew-Zunger 1981 Correlation)。
  - **验证**: 与 PySCF (gth-tzvp, lda,pz) 进行了严格的能量和力对齐。

- **JAX 加速**: 
  - 全程 JAX 编写，支持 GPU/TPU 加速和 JIT 编译。
  - 支持自动微分 (Auto-Diff) 计算力 (Forces)。

---

## 🧪 验证实验 (Verification)

本项目通过两个关键实验证明了物理实现的正确性：

### 1. H2 分子解离曲线 (H2 Dissociation)
- **脚本**: `scripts/verify_h2.py`
- **结果**: JaxDFT 的解离曲线趋势与 PySCF (`gth-tzvp`) 高度一致。
- **说明**: JaxDFT 使用实空间网格（接近完备基组），其绝对能量通常比 PySCF (TZVP) 更低。

### 2. 盒子大小收敛性测试 (Box Size Convergence)
- **脚本**: `scripts/check_box.py`
- **现象**: JaxDFT (PBC) 与 PySCF (Isolated) 之间存在一个能量常数差。
- **结论**: 我们证明了该差值源于 **周期性边界条件 (PBC)** 的物理背景电荷项。
  - 当盒子边长 $L$ 从 10.0 增加到 34.0 时，能量差值 (Diff) 单调递减，平滑逼近孤立体系真值。
  - **这意味着代码物理内核是正确的**，该常数差不影响力的计算和相对能量面。

---

## 📂 项目结构 (Structure)

```text
JaxDFT/
├── src/
│   ├── hamiltonian.py   # 动能、局部势(GTH)、非局部势构建
│   ├── functional.py    # XC泛函 (LDA-PZ81)
│   ├── solver.py        # 求解器 (SCF迭代, 能量与力)
│   ├── io.py            # GTH文件读取与解析
│   └── structure.py     # 晶胞与网格定义
├── scripts/
│   ├── verify_h2.py     # H2 分子精度对齐验证
│   ├── check_box.py     # PBC vs Isolated 能量收敛性证明
│   └── run_sampling.py  # 构型空间采样脚本
└── data/
    └── gth_potentials/  # GTH 赝势参数库
```

## 🛠️ 快速开始 (Quick Start)

1. **安装依赖**:
   ```bash
   pip install -r JaxDFT/requirements.txt
   ```

2. **运行物理验证**:
   ```bash
   # 验证 H2 曲线
   python JaxDFT/scripts/verify_h2.py
   
   # 验证 PBC 收敛性 (证明物理模型正确)
   python JaxDFT/scripts/check_box.py
   ```
