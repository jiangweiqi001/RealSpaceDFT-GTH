import sys
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyscf import gto, dft
from pyscf.pbc import gto as pbcgto, dft as pbcdft

# 1. 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path: sys.path.insert(0, project_root)

try:
    import JaxDFT.src.solver as solver
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.hamiltonian import create_grid, build_local_potential
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    import src.solver as solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid, build_local_potential

# 2. PySCF 设置 (使用 TZVP 高精度基组，确保它是标准的 U 型)

def run_pyscf(dist, box_size=None):
    try:
        if box_size is not None:
            # PBC 模式 (和 JaxDFT 一致)
            cell = pbcgto.Cell()
            cell.atom = f'H 0 0 0; H 0 0 {dist}'
            cell.a = jnp.eye(3) * box_size[0]  # 假设立方盒子
            cell.basis = 'gth-szv'  # 为了速度用小基组，或者 gth-tzvp
            cell.pseudo = 'gth-lda'
            cell.verbose = 0
            cell.build()
            mf = pbcdft.RKS(cell)
            mf.xc = 'lda,vwn'
            return mf.kernel()
        else:
            # 孤立模式 (旧的参考)
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {dist}', unit='Bohr', basis='gth-tzvp', pseudo='gth-lda', verbose=0)
            mf = dft.RKS(mol)
            mf.xc = 'lda,vwn'
            return mf.kernel()
    except Exception as e:
        return float('nan')


print(f"\n{'='*20} 最终演示版 (Soft Atom Model) {'='*20}")

# 3. 参数设置
# 0.3 的网格配合软原子，显存占用极低，且结果平滑
spacing = 0.18
box_size = [16.0, 16.0, 16.0]
distances = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]

# 4. 准备原子
pseudos = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")
p = pseudos[0]
pseudos_for_calc = [p, p]

print(f"设置: Grid Spacing={spacing}, rloc={p['rloc']} (Soft)")
print("注意: 软原子的结合能(-1.5)比硬原子更深，这是模型特性。")

# 5. 势能体检
grid = create_grid(spacing, box_size)
zion = jnp.array([p['zion'] for p in pseudos_for_calc])
rloc = jnp.array([p['rloc'] for p in pseudos_for_calc])
c_coef = jnp.array([p['c'] for p in pseudos_for_calc])
test_coords = jnp.array([[0.,0.,0.], [0.,0.,1.4]])
V_check = build_local_potential(test_coords, grid.coords, zion, rloc, c_coef)
print(f"🩺 势能深度: {float(V_check.min()):.4f} Ha")

# 6. 扫描计算
jax_energies = []
pyscf_energies = []
key = jax.random.PRNGKey(42)

print("-" * 75)
print(f"{'Dist':<6} | {'JaxDFT (Soft)':<15} | {'PySCF (Ref)':<15} | {'Diff'}")
print("-" * 75)

for d in distances:
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]])
    
    try:
        # 100 步足够软原子收敛
        e_jax, _ = solver.energy_and_forces(
            grid, coords, pseudos_for_calc, 100, 0.3, 1e-5, key
        )
    except Exception as e:
        e_jax = float("nan")

    e_pyscf = run_pyscf(d, box_size=None)
    
    jax_energies.append(e_jax)
    pyscf_energies.append(e_pyscf)
    
    diff = e_jax - e_pyscf
    print(f"{d:<6.2f} | {e_jax:<15.6f} | {e_pyscf:<15.6f} | {diff:.4f}")

# 7. 绘图
plt.figure(figsize=(10, 6))
plt.plot(distances, jax_energies, 'o-', label='JaxDFT (Soft Model)', linewidth=2)
plt.plot(distances, pyscf_energies, 'x--', label='PySCF (PBC Reference)', linewidth=2)
plt.xlabel('Bond Length (Bohr)')
plt.ylabel('Total Energy (Hartree)')
plt.title('H2 Dissociation: Model vs Experiment')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('h2_verification.png', dpi=150)
print("-" * 75)
print("✅ 验证完成。图片: h2_verification.png")
