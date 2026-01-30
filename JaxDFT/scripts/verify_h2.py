import sys
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyscf import gto, dft

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

# 2. PySCF 设置 (使用 DZV 基组以获得更好精度)
def run_pyscf(dist):
    try:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {dist}", basis="gth-dzv", pseudo="gth-lda", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "lda,vwn"
        return mf.kernel()
    except Exception as e:
        return float("nan")

print(f"\n{'='*20} 最终演示 (Soft Atom vs PySCF) {'='*20}")

# 3. 参数设置
spacing = 0.3
box_size = [5.0, 5.0, 5.0]
# 扩展范围以看到完整的 U 型曲线
distances = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]

# 4. 准备原子
pseudos = load_pseudopotentials(["H"], "JaxDFT/data/gth_potentials")
p = pseudos[0]
pseudos_for_calc = [p, p]

print(f"体系: Soft H2 (rloc={p['rloc']}) vs PySCF (Hard H2)")

# 5. 扫描计算
jax_energies = []
pyscf_energies = []
key = jax.random.PRNGKey(42)

print("-" * 75)
print(f"{'Dist':<6} | {'JaxDFT (Soft)':<15} | {'PySCF (Ref)':<15} | {'Diff'}")
print("-" * 75)

grid = create_grid(spacing, box_size)

for d in distances:
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]])
    
    try:
        e_jax, _ = solver.energy_and_forces(
            grid, coords, pseudos_for_calc, 100, 0.3, 1e-5, key
        )
    except Exception as e:
        e_jax = float("nan")

    e_pyscf = run_pyscf(d)
    
    jax_energies.append(e_jax)
    pyscf_energies.append(e_pyscf)
    
    diff = e_jax - e_pyscf
    print(f"{d:<6.2f} | {e_jax:<15.6f} | {e_pyscf:<15.6f} | {diff:.4f}")

# 6. 绘图
plt.figure(figsize=(10, 6))
plt.plot(distances, jax_energies, 'o-', label='JaxDFT (Soft Model)', linewidth=2)
plt.plot(distances, pyscf_energies, 'x--', label='PySCF (Real Physics)', linewidth=2)
plt.xlabel('Bond Length (Bohr)')
plt.ylabel('Total Energy (Hartree)')
plt.title('H2 Dissociation: Soft Model vs Real Physics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('h2_verification.png', dpi=150)
print("-" * 75)
print("✅ 验证完成。请查看 h2_verification.png")
