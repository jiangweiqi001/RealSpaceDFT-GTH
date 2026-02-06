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

# 2. PySCF 设置 (已升级为科研级精度)
# ------------------------------------------------------------------
# 修改点 1 (基组): 从 gth-szv (最小基组) -> gth-tzvp (高精度三倍zeta基组)
# 修改点 2 (XC): 从 lda,vwn -> lda,pz (精确匹配 JaxDFT 的 PZ81 实现)
# ------------------------------------------------------------------

def run_pyscf(dist, box_size=None):
    try:
        if box_size is not None:
            # PBC 模式
            cell = pbcgto.Cell()
            cell.atom = f'H 0 0 0; H 0 0 {dist}'
            cell.a = jnp.eye(3) * box_size[0]
            cell.basis = 'gth-tzvp'  # <--- 已修改：高精度基组
            cell.pseudo = 'gth-lda'
            cell.verbose = 0
            cell.build()
            mf = pbcdft.RKS(cell)
            mf.xc = 'lda,pz'         # <--- 已修改：对齐泛函
            return mf.kernel()
        else:
            # 孤立模式
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {dist}', unit='Bohr', basis='gth-tzvp', pseudo='gth-lda', verbose=0) # <--- 已修改
            mf = dft.RKS(mol)
            mf.xc = 'lda,pz'         # <--- 已修改
            return mf.kernel()
    except Exception as e:
        return float('nan')


print(f"\n{'='*20} 最终演示版 (High Precision Verification) {'='*20}")

# 3. 参数设置
spacing = 0.18
box_size = [16.0, 16.0, 16.0]
distances = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]

# 4. 准备原子
pseudos = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")
p = pseudos[0]
pseudos_for_calc = [p, p]

print(f"设置: Grid Spacing={spacing}, rloc={p['rloc']} (Standard Hard)")
print("注意: PySCF 已升级为 gth-tzvp 基组 + PZ81 泛函，能量应与实空间结果高度一致。")

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
print(f"{'Dist':<6} | {'JaxDFT':<15} | {'PySCF (TZVP)':<15} | {'Diff'}")
print("-" * 75)

for d in distances:
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]])
    
    try:
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
plt.plot(distances, jax_energies, 'o-', label='JaxDFT (RealSpace)', linewidth=2)
plt.plot(distances, pyscf_energies, 'x--', label='PySCF (TZVP Reference)', linewidth=2)
plt.xlabel('Bond Length (Bohr)')
plt.ylabel('Total Energy (Hartree)')
plt.title('H2 Dissociation: JaxDFT vs PySCF (High Precision)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('h2_verification.png', dpi=150)
print("-" * 75)
print("✅ 验证完成。图片: h2_verification.png")
