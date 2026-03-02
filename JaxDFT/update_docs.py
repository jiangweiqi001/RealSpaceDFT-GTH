import os

readme_content = """# JaxDFT: Real-Space DFT with GTH Pseudopotentials

**JaxDFT** 是一个基于 **JAX** 实现的实空间 (Real-Space) 密度泛函理论 (DFT) 计算包。它专为高性能计算设计，支持自动微分，并实现了标准的 **GTH (Goedecker-Teter-Hutter)** 赝势。
**当前版本专为孤立体系 (Isolated Systems / Open Boundary Conditions) 设计。**

---

## 🚀 核心特性 (Key Features)

- **实空间求解 (Real-Space Grid)**: 
  - 采用**非周期性**有限差分 (Dirichlet 边界) 计算动能。
  - 采用 **Hockney 补零 FFT 法**求解泊松方程，消除镜像相互作用，实现纯正的孤立体系。
- **GTH 赝势 (GTH Pseudopotentials)**:
  - 完整实现了标准的 GTH-LDA 局域势。
- **物理精度对齐 (Physics Benchmark)**:
  - **XC 泛函**: LDA (Slater Exchange + Perdew-Zunger 1981 Correlation)。
  - **验证**: 与 PySCF 孤立体系高精度基组 (`gth-tzvp`) 进行了严格对齐。

---

## 🧪 验证实验 (Verification)

### 1. H2 分子解离曲线
- **脚本**: `scripts/verify_h2.py`
- **结果**: JaxDFT 的解离曲线与 PySCF 高度平行，最低点完美重合，仅存在极小的固有常数差。

### 2. 盒子大小独立性测试
- **脚本**: `scripts/check_box.py`
- **结论**: 无论盒子多大，能量均保持恒定。内核已彻底转为**孤立体系**，消除了周期性边界 (PBC) 带来的镜像排斥问题。
"""

# 将 README 写入上级根目录
with open('../README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("✅ README.md 更新完成！")

# 更新 check_box.py
cb_path = 'scripts/check_box.py'
with open(cb_path, 'r', encoding='utf-8') as f:
    cb = f.read()
cb = cb.replace('JaxDFT (PBC)', 'JaxDFT (Isolated)')
with open(cb_path, 'w', encoding='utf-8') as f:
    f.write(cb)
print("✅ scripts/check_box.py 打印文案更新完成！")

# 更新 solver.py
sol_path = 'src/solver.py'
with open(sol_path, 'r', encoding='utf-8') as f:
    sol = f.read()
old_sol = "The FFT implies periodic boundary conditions on the simulation cell."
new_sol = "Uses Hockney's zero-padding method to enforce open (isolated) boundary conditions."
sol = sol.replace(old_sol, new_sol)
with open(sol_path, 'w', encoding='utf-8') as f:
    f.write(sol)
print("✅ src/solver.py 注释更新完成！")

# 更新 hamiltonian.py
ham_path = 'src/hamiltonian.py'
with open(ham_path, 'r', encoding='utf-8') as f:
    ham = f.read()
old_ham = "The stencil uses periodic shifts via jnp.roll."
new_ham = "The stencil uses non-periodic shifts via masking to enforce Dirichlet bounds."
ham = ham.replace(old_ham, new_ham)
with open(ham_path, 'w', encoding='utf-8') as f:
    f.write(ham)
print("✅ src/hamiltonian.py 注释更新完成！")

