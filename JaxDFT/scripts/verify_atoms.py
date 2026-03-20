import os
import sys
import jax
import jax.numpy as jnp
from pyscf import gto, dft

# --------------------------------------------------
# 路径设置：兼容你仓库当前目录结构
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 仓库根目录

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import JaxDFT.src.solver as solver
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.hamiltonian import create_grid
except ImportError:
    sys.path.insert(0, os.path.join(project_root, "JaxDFT"))
    import src.solver as solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid


# --------------------------------------------------
# 要测的元素
# --------------------------------------------------
ELEMENTS = ["H", "C", "N", "O", "Cl"]

# PySCF 原子自旋（2S = N_alpha - N_beta）
# 用真实开壳层原子自旋，便于 PySCF 正常收敛
SPIN_MAP = {
    "H": 1,
    "C": 2,
    "N": 3,
    "O": 2,
    "Cl": 1,
}

# 每个原子的网格参数
ATOM_SETTINGS = {
    "H":  {"target_spacing": 0.18, "L": 18.0, "max_iter": 500, "mix_alpha": 0.20},
    "C":  {"target_spacing": 0.14, "L": 20.0, "max_iter": 700, "mix_alpha": 0.12},
    "N":  {"target_spacing": 0.14, "L": 20.0, "max_iter": 700, "mix_alpha": 0.12},
    "O":  {"target_spacing": 0.12, "L": 20.0, "max_iter": 800, "mix_alpha": 0.10},
    "Cl": {"target_spacing": 0.15, "L": 22.0, "max_iter": 800, "mix_alpha": 0.10},
}


def run_jax_atom(symbol):
    """跑 JaxDFT 单原子总能量"""
    setting = ATOM_SETTINGS[symbol]
    target_spacing = setting["target_spacing"]
    L = setting["L"]

    # 跟你现有脚本同样的做法：保留盒子长度，微调实际 spacing
    N = int(round(L / target_spacing))
    spacing = L / N
    box_size = [L, L, L]

    grid = create_grid(spacing, box_size)
    pseudo_dir = os.path.join(project_root, "JaxDFT", "data", "gth_potentials")
    pseudo = load_pseudopotentials([symbol], pseudo_dir)[0]

    coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    energy, _ = solver.energy_and_forces(
        grid,
        coords,
        [pseudo],
        setting["max_iter"],
        setting["mix_alpha"],
        1e-6,
        key,
    )

    return float(energy), float(spacing), float(L), pseudo


def run_pyscf_atom(symbol):
    """跑 PySCF 单原子总能量"""
    spin = SPIN_MAP[symbol]

    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        unit="Bohr",
        basis="gth-tzvp",
        pseudo="gth-lda",
        spin=spin,
        verbose=0,
    )

    mf = dft.UKS(mol)
    mf.xc = "lda,pz"
    mf.conv_tol = 1e-10
    mf.max_cycle = 200

    energy = mf.kernel()
    return float(energy), spin


def main():
    print("\n" + "=" * 26 + " 单原子测试：JaxDFT vs PySCF " + "=" * 26)
    print("说明：这是粗筛测试，重点看哪些元素明显更偏低。")
    print("-" * 110)
    print(f"{'Atom':<4} | {'JaxDFT':<14} | {'PySCF':<14} | {'Diff':<10} | {'spacing':<8} | {'L':<6} | {'q':<3} | {'rloc':<8} | {'spin'}")
    print("-" * 110)

    for symbol in ELEMENTS:
        try:
            e_jax, spacing, L, pseudo = run_jax_atom(symbol)
        except Exception as e:
            print(f"{symbol:<4} | JaxDFT failed: {e}")
            continue

        try:
            e_pyscf, spin = run_pyscf_atom(symbol)
        except Exception as e:
            print(f"{symbol:<4} | PySCF failed: {e}")
            continue

        diff = e_jax - e_pyscf
        print(
            f"{symbol:<4} | "
            f"{e_jax:<14.6f} | "
            f"{e_pyscf:<14.6f} | "
            f"{diff:<10.6f} | "
            f"{spacing:<8.4f} | "
            f"{L:<6.1f} | "
            f"{pseudo['q']:<3d} | "
            f"{pseudo['rloc']:<8.4f} | "
            f"{spin}"
        )

    print("-" * 110)
    print("看结果时，优先比较 H vs C/N/O/Cl：")
    print("如果 H 还行，而 C/N/O/Cl 明显整体更负，就继续查这些元素的赝势/非局域项。")


if __name__ == "__main__":
    main()
