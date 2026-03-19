import sys
import os
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyscf import gto, dft

# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path: sys.path.insert(0, project_root)

try:
    import src.solver as solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src import solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid

# 2. PySCF 设置
def run_pyscf(coords_list):
    try:
        atom_str = f"O {coords_list[0][0]} {coords_list[0][1]} {coords_list[0][2]}; " \
                   f"H {coords_list[1][0]} {coords_list[1][1]} {coords_list[1][2]}; " \
                   f"H {coords_list[2][0]} {coords_list[2][1]} {coords_list[2][2]}"
        mol = gto.M(atom=atom_str, unit='Bohr', basis='gth-tzvp', pseudo='gth-lda', verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'lda,pz'
        return mf.kernel()
    except Exception:
        return float('nan')

print(f"\n{'='*20} H2O 分子对称拉伸验证 (Ultra High Precision) {'='*20}")


target_spacing = 0.12  
L = 18.0  
N_grid = int(round(L / target_spacing))
spacing = L / N_grid
box_size = [L, L, L]

# H2O 平衡键长约 1.81 Bohr, 键角 104.5 度
angle_deg = 104.5
theta = angle_deg * math.pi / 180.0
# 【修改点3】: 在平衡点 1.8 附近增加扫描密度
distances = [1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.8, 3.2]

# 4. 加载赝势
pseudos = load_pseudopotentials(["O", "H"], os.path.join(project_root, "data/gth_potentials"))
# H2O 的原子顺序是 O, H, H
pseudos_for_calc = [pseudos[0], pseudos[1], pseudos[1]]

print(f"设置: Grid Spacing={spacing:.4f}, Box Size={L}")
print(f"O 赝势 rloc: {pseudos[0]['rloc']}, H 赝势 rloc: {pseudos[1]['rloc']}")

# 5. 执行扫描
grid = create_grid(spacing, box_size)
jax_energies = []
pyscf_energies = []
key = jax.random.PRNGKey(42)

print("-" * 75)
print(f"{'O-H Dist':<8} | {'JaxDFT':<15} | {'PySCF (TZVP)':<15} | {'Diff'}")
print("-" * 75)

for d in distances:
    # 构造 H2O 坐标系：O 在原点，H 位于 x-z 平面
    hx = d * math.sin(theta / 2.0)
    hz = d * math.cos(theta / 2.0)
    
    # 转换为普通列表传给 PySCF，同时转为 jnp.array 传给 JaxDFT
    coords_list = [
        [0.0, 0.0, 0.0],   # O
        [hx, 0.0, hz],     # H1
        [-hx, 0.0, hz]     # H2
    ]
    coords_jax = jnp.array(coords_list)
    
    # 【修改点4】: 更稳健的自适应 SCF 策略
    if d >= 2.8:
        alpha = 0.05
        max_iter = 1200
    elif d >= 2.4:
        alpha = 0.1
        max_iter = 800
    else:
        alpha = 0.3
        max_iter = 400
    
    try:
        e_jax, _ = solver.energy_and_forces(
            grid, coords_jax, pseudos_for_calc, max_iter, alpha, 1e-5, key
        )
    except Exception as e:
        print(f"Error at d={d}: {e}")
        e_jax = float("nan")

    e_pyscf = run_pyscf(coords_list)
    
    jax_energies.append(e_jax)
    pyscf_energies.append(e_pyscf)
    
    diff = e_jax - e_pyscf
    print(f"{d:<8.2f} | {e_jax:<15.6f} | {e_pyscf:<15.6f} | {diff:.4f}")

# 6. 绘图
plt.figure(figsize=(10, 6))
plt.plot(distances, jax_energies, 'o-', label=f'JaxDFT (spacing={spacing:.4f})')
plt.plot(distances, pyscf_energies, 'x--', label='PySCF (TZVP)')
plt.xlabel('O-H Symmetric Bond Length (Bohr)')
plt.ylabel('Total Energy (Ha)')
plt.title(f'H2O Symmetric Stretch Curve (Angle={angle_deg}°)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('h2o_verification.png', dpi=150)

print("-" * 75)
print("✅ 验证完成。结果已保存至 h2o_verification.png")