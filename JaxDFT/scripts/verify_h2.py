# -*- coding: utf-8 -*-
import os
import sys

# 禁止 JAX 预分配所有显存，并强制使用 CPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import numpy as np

# 尝试导入可选依赖
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from pyscf import gto, dft
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

# 启用 JAX 的 64 位精度
jax.config.update("jax_enable_x64", True)

# 添加项目根目录到 Python 路径
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from src.hamiltonian import create_grid, prepare_system
from src.io import load_pseudopotentials
from src.solver import energy_and_forces

def run_jaxdft(distances, spacing=0.3, box_size=[6.0, 6.0, 6.0]):
    print(f"\n正在运行 JaxDFT (格点间距={spacing}, 盒子大小={box_size})...")
    data_path = os.path.join(root, "data", "gth_potentials")
    # 加载 GTH 赝势
    pseudos = load_pseudopotentials(["H", "H"], data_path)
    
    energies = []
    forces = []
    
    for idx, d in enumerate(distances):
        # 将两个氢原子沿 Z 轴对称放置
        offset = float(d) / 2.0
        coords = np.array([[0.0, 0.0, -offset], [0.0, 0.0, offset]], dtype=np.float64)
        
        # 创建格点和准备系统
        grid = create_grid(spacing, box_size)
        grid = prepare_system(grid, jax.numpy.asarray(coords), pseudos)
        
        key = jax.random.PRNGKey(idx)
        # 计算能量和力
        e, f = energy_and_forces(
            grid, 
            jax.numpy.asarray(coords), 
            pseudos, 
            max_iter=150, 
            mix_alpha=0.1, 
            tolerance=1e-5, 
            key=key
        )
        
        e_val = float(e)
        # 取第二个原子在 Z 轴方向的受力
        f_val = float(f[1][2]) 
        
        energies.append(e_val)
        forces.append(f_val)
        print(f"  距离={d:.2f} Bohr: 能量={e_val:.6f} Ha, 力(Z)={f_val:.6f} Ha/Bohr")
        
    return np.array(energies), np.array(forces)

def run_pyscf(distances):
    if not HAS_PYSCF:
        return None, None
        
    print(f"\n正在运行 PySCF (基组: gth-szv, 赝势: gth-lda)...")
    energies = []
    forces = []
    
    for d in distances:
        # 构建分子
        mol = gto.M(
            atom=f'H 0 0 {-d/2}; H 0 0 {d/2}',
            basis='gth-szv',
            pseudo='gth-lda',
            verbose=0
        )
        # 运行 RKS (DFT)
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn' # GTH-LDA 通常配合 LDA 泛函使用
        e_val = mf.kernel()
        
        # 计算梯度 (力 = -梯度)
        grad = mf.nuc_grad_method().kernel()
        f_val = -grad[1][2]
        
        energies.append(e_val)
        forces.append(f_val)
        print(f"  距离={d:.2f} Bohr: 能量={e_val:.6f} Ha, 力(Z)={f_val:.6f} Ha/Bohr")
        
    return np.array(energies), np.array(forces)

def main():
    # 1. 定义扫描范围 (键长 0.8 到 3.0 Bohr)
    distances = np.linspace(0.8, 3.0, 12)
    
    # 2. 运行 JaxDFT 计算
    jax_e, jax_f = run_jaxdft(distances)
    
    # 3. 运行 PySCF 计算 (如果已安装)
    pyscf_e, pyscf_f = run_pyscf(distances)
    
    # 4. 打印对比表格
    print("\n" + "="*95)
    # 表头使用中文
    header = f"{'距离 (Bohr)':<12} | {'JaxDFT 能量':<12} | {'PySCF 能量':<12} | {'能量差':<10} | {'JaxDFT 力':<10} | {'PySCF 力':<10}"
    print(header)
    print("-" * 95)
    
    for i, d in enumerate(distances):
        j_e = jax_e[i]
        j_f = jax_f[i]
        
        if pyscf_e is not None:
            p_e = pyscf_e[i]
            p_f = pyscf_f[i]
            diff_e = j_e - p_e
            row = f"{d:<12.2f} | {j_e:<12.6f} | {p_e:<12.6f} | {diff_e:<10.6f} | {j_f:<10.6f} | {p_f:<10.6f}"
        else:
            row = f"{d:<12.2f} | {j_e:<12.6f} | {'N/A':<12} | {'N/A':<10} | {j_f:<10.6f} | {'N/A':<10}"
        print(row)
    print("="*95)

    # 5. 绘制曲线图
    if HAS_MATPLOTLIB:
        # 为了兼容性，图表标签仍使用英文，避免在无中文字体的环境下显示乱码
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 能量曲线
        ax1.plot(distances, jax_e, 'o-', label='JaxDFT (RealSpace)')
        if pyscf_e is not None:
            ax1.plot(distances, pyscf_e, 'x--', label='PySCF (Basis)')
        ax1.set_xlabel('Distance (Bohr)')
        ax1.set_ylabel('Energy (Ha)')
        ax1.set_title('H2 Dissociation Curve (Energy)')
        ax1.legend()
        ax1.grid(True)
        
        # 力曲线
        ax2.plot(distances, jax_f, 'o-', label='JaxDFT (RealSpace)')
        if pyscf_f is not None:
            ax2.plot(distances, pyscf_f, 'x--', label='PySCF (Basis)')
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Distance (Bohr)')
        ax2.set_ylabel('Force on H (Ha/Bohr)')
        ax2.set_title('H2 Force Curve')
        ax2.legend()
        ax2.grid(True)
        
        out_path = os.path.join(os.path.dirname(__file__), "h2_verification.png")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"\n图表已保存至: {out_path}")
    else:
        print("\n未检测到 Matplotlib，跳过绘图。")

if __name__ == "__main__":
    main()
