import sys
import os
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
    # 兼容处理
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src import solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid

# 2. PySCF 设置 (参考基准)
def run_pyscf(dist):
    try:
        mol = gto.M(atom=f'H 0 0 0; Cl 0 0 {dist}', unit='Bohr', 
                    basis='gth-tzvp', pseudo='gth-lda', verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'lda,pz' # 对齐 JaxDFT 的 PZ81 实现
        return mf.kernel()
    except Exception:
        return float('nan')

print(f"\n{'='*20} HCl 分子验证 (High Precision) {'='*20}")

# 3. 计算参数
target_spacing = 0.18
L = 20.0  # 增大盒子以容纳 Cl 原子
N = int(round(L / target_spacing))
spacing = L / N
box_size = [L, L, L]
distances = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0]

# 4. 加载赝势
# 使用刚刚更新过的 Cl.txt
pseudos = load_pseudopotentials(["H", "Cl"], os.path.join(project_root, "data/gth_potentials"))
pseudos_for_calc = [pseudos[0], pseudos[1]]

print(f"设置: Grid Spacing={spacing:.4f}, Box Size={L}")
print(f"H 赝势 rloc: {pseudos[0]['rloc']}, Cl 赝势 rloc: {pseudos[1]['rloc']}")

# 5. 执行扫描
grid = create_grid(spacing, box_size)
jax_energies = []
pyscf_energies = []
key = jax.random.PRNGKey(42)

print("-" * 75)
print(f"{'Dist':<6} | {'JaxDFT':<15} | {'PySCF (TZVP)':<15} | {'Diff'}")
print("-" * 75)

for d in distances:
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]])
    
    try:
        e_jax, _ = solver.energy_and_forces(
            grid, coords, pseudos_for_calc, 500, 0.3, 1e-5, key
        )
    except Exception as e:
        print(f"Error at d={d}: {e}")
        e_jax = float("nan")

    e_pyscf = run_pyscf(d)
    
    jax_energies.append(e_jax)
    pyscf_energies.append(e_pyscf)
    
    diff = e_jax - e_pyscf
    print(f"{d:<6.2f} | {e_jax:<15.6f} | {e_pyscf:<15.6f} | {diff:.4f}")

# 6. 绘图
plt.figure(figsize=(10, 6))
plt.plot(distances, jax_energies, 'o-', label='JaxDFT (RealSpace)')
plt.plot(distances, pyscf_energies, 'x--', label='PySCF (TZVP)')
plt.xlabel('Bond Length (Bohr)')
plt.ylabel('Total Energy (Ha)')
plt.title('HCl Dissociation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hcl_verification.png')

print("-" * 75)
print("✅ 验证完成。结果已保存至 hcl_verification.png")
