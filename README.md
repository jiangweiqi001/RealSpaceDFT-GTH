# JaxDFT

JaxDFT 是一个基于 JAX 的可微分实空间 DFT 求解器，面向构型采样与数据集生成。项目以原子单位制为准：长度为 Bohr，能量为 Hartree，力为 Hartree/Bohr。

## 项目概览
- 目标：通过实空间 Kohn-Sham DFT 生成可用于机器学习势能的结构-能量-力数据集
- 特点：JAX 自动微分、可批量采样、可在 GPU/CPU 上运行

## 架构

| 模块 | 作用 |
|---|---|
| src/solver.py | SCF 主循环、密度混合、能量分解与总能量计算 |
| src/functional.py | LDA 交换-相关能与势（含 PZ81 相关项） |
| src/hamiltonian.py | 实空间网格、局域赝势与 4 阶拉普拉斯算子 |
| src/io.py | 赝势解析、配置读取与 HDF5 数据输出 |
| src/structure.py | 随机团簇生成与最小距离检查 |
| src/__init__.py | 包初始化 |

## 验证

运行 H2 解离曲线验证：

~~~bash
python JaxDFT/scripts/verify_h2.py
~~~

解释：
- 脚本会输出 JaxDFT 的能量曲线与 PySCF 参考值
- Soft Atom（较大的 rloc）会得到更深的束缚能，曲线会偏离真实物理
- Hard Atom（较小的 rloc）需要更细网格，通常计算更贵但更接近物理
- 绘图输出为 h2_verification.png，横轴为 Bohr，纵轴为 Hartree

## 采样流程

运行采样脚本生成数据集：

~~~bash
python JaxDFT/scripts/run_sampling.py
~~~

说明：
- 默认读取 config/default.yaml 中的采样与 SCF 参数
- 输出文件为 dataset.h5（HDF5 格式）
- 数据集字段与单位：
  - R: 原子坐标，Bohr
  - Z: 原子序数，无量纲
  - E: 总能量，Hartree
  - F: 原子力，Hartree/Bohr

## 依赖

依赖列表来自 requirements.txt：
- jax
- jaxlib
- h5py
- pyyaml
- numpy
- scipy
