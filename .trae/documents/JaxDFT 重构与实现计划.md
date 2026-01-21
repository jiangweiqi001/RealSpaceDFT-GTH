# JaxDFT 重构与实现计划

## 1. 项目概述
构建一个基于 JAX 的高性能实空间 DFT 引擎，专用于高通量随机原子团簇采样。项目将严格遵循模块化设计，利用 JAX 的 `jit` 编译和 `vmap`/`grad` 自动微分特性实现高效计算。

## 2. 架构设计 (Architecture)

### 2.1 目录结构
```text
JaxDFT/
├── config/
│   └── default.yaml        # 配置：增加 simple_mixing_steps 参数
├── data/
│   └── gth_potentials/     # GTH 伪势文件存储
├── src/
│   ├── __init__.py
│   ├── structure.py        # 结构生成与检查
│   ├── hamiltonian.py      # 动能算符与势能积分
│   ├── functional.py       # LDA XC 泛函
│   ├── solver.py           # SCF 求解器 (含混合策略优化)
│   └── io.py               # IO 与 伪势管理
├── scripts/
│   ├── run_sampling.py     # [新建] 主采样程序
│   └── verify_h2.py        # 现有验证脚本
└── requirements.txt
```

## 3. 详细实现方案

### 3.1 混合策略优化 (Solver)
针对随机结构不易收敛的问题，在 `src/solver.py` 中实现**分阶段混合策略**：
- **第一阶段 (预热)**：前 `simple_mixing_steps` (如 10 步) 使用简单的线性混合 (Linear Mixing)，通过 `rho = (1-alpha)*rho + alpha*rho_new` 快速稳定电荷密度。
- **第二阶段 (加速)**：之后切换到 Anderson Mixing，利用历史迭代信息加速收敛。
- **实现方式**：使用 `jax.lax.cond` 在 SCF 循环中动态选择混合算法。

### 3.2 伪势管理 (IO)
- 在 `src/io.py` 中增强 `initialize_gth_potentials`。
- **优先策略**：尝试从 CP2K 官方仓库下载标准的 `GTH_POTENTIALS` 文件。
- **兜底策略**：如果下载失败，使用内置的参数字典（包含 H, C, N, O, Si 等常见元素）或程序化生成器作为 fallback，确保离线环境可用。

### 3.3 主采样程序 (Script)
新建 `scripts/run_sampling.py`，流程如下：
1. **初始化**：加载 YAML 配置，初始化伪势。
2. **采样循环**：
   - 调用 `structure.generate_random_cluster` 生成结构。
   - 调用 `structure.check_min_distance` 严格检查原子间距。
   - 构建网格与哈密顿量。
   - 运行 SCF (带自动重试/跳过机制)。
   - 计算 Force (Auto-grad)。
3. **数据存储**：实时或分批将 `(R, Z, E, F)` 写入 HDF5 文件。

## 4. 执行步骤
1. **文档**：生成中文设计文档（即本计划）。
2. **重构代码**：
   - 修改 `src/solver.py` 引入混合策略。
   - 完善 `src/io.py` 的伪势加载逻辑。
   - 更新 `config/default.yaml`。
3. **新建脚本**：编写 `scripts/run_sampling.py`。
4. **验证**：运行 `scripts/verify_h2.py` 确保物理结果正确。
