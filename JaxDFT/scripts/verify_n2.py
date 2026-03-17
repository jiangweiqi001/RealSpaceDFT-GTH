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
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    from src import solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid

# 2. PySCF 设置 (参考基准)
def run_pyscf(dist):
    try:
        # 居中放置原子，以获得更好的盒子对称性
        mol = gto.M(atom=f'N 0 0 {-dist/2}; N 0 0 {dist/2}', unit='Bohr', 
                    basis='gth-tzvp', pseudo='gth-lda', verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'lda,pz' 
        return mf.kernel()
    except Exception:
        return float('nan')

print(f"\n{'='*20} N2 分子解离验证 (High Precision) {'='*20}")

# 3. 计算参数
target_spacing = 0.15  # 可根据显存调整，0.15更精确但更耗内存
L = 20.0
N_grid = int(round(L / target_spacing))
spacing = L / N_grid
box_size = [L, L, L]

# N2 平衡键长约 2.07 Bohr
distances = [1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.6, 2.8, 3.2]

# 4. 加载赝势
pseudos = load_pseudopotentials(["N"], os.path.join(project_root, "data/gth_potentials"))
pseudos_for_calc = [pseudos[0], pseudos[0]]

print(f"设置: Grid Spacing={spacing:.4f}, Box Size={L}")
print(f"N 赝势 rloc: {pseudos[0]['rloc']}")

# 5. 执行扫描
grid = create_grid(spacing, box_size)
jax_energies = []
pyscf_energies = []
key = jax.random.PRNGKey(42)

print("-" * 75)
print(f"{'Dist':<6} | {'JaxDFT':<15} | {'PySCF (TZVP)':<15} | {'Diff'}")
print("-" * 75)

for d in distances:
    # 居中放置
    coords = jnp.array([[0.0, 0.0, -d/2.0], [0.0, 0.0, d/2.0]])
    
    # 针对 N2 三键解离的极端电荷震荡，定制自适应收敛策略
    if d >= 3.0:
        current_alpha = 0.05
        current_max_iter = 700
    elif d >= 2.4:
        current_alpha = 0.1
        current_max_iter = 600
    else:
        current_alpha = 0.3
        current_max_iter = 400
    
    try:
        e_jax, _ = solver.energy_and_forces(
            grid, coords, pseudos_for_calc, current_max_iter, current_alpha, 1e-5, key
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
plt.xlabel('N-N Bond Length (Bohr)')
plt.ylabel('Total Energy (Ha)')
plt.title('N2 Dissociation Curve (Triple Bond)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('n2_verification.png', dpi=150)

print("-" * 75)
print("✅ 验证完成。结果已保存至 n2_verification.png")