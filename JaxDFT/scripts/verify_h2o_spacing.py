import os
import sys
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyscf import gto, dft

# --------------------------------------------------
# 路径设置：兼容你仓库当前目录结构
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))  # .../RealSpaceDFT-GTH
jaxdft_root = os.path.join(repo_root, "JaxDFT")

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    import JaxDFT.src.solver as solver
    from JaxDFT.src.io import load_pseudopotentials
    from JaxDFT.src.hamiltonian import create_grid
    pseudo_dir = os.path.join(repo_root, "JaxDFT", "data", "gth_potentials")
except ImportError:
    if jaxdft_root not in sys.path:
        sys.path.insert(0, jaxdft_root)
    import src.solver as solver
    from src.io import load_pseudopotentials
    from src.hamiltonian import create_grid
    pseudo_dir = os.path.join(jaxdft_root, "data", "gth_potentials")


# --------------------------------------------------
# PySCF 参考：和 verify_h2o.py 一样
# --------------------------------------------------
def run_pyscf(coords_list):
    atom_str = (
        f"O {coords_list[0][0]} {coords_list[0][1]} {coords_list[0][2]}; "
        f"H {coords_list[1][0]} {coords_list[1][1]} {coords_list[1][2]}; "
        f"H {coords_list[2][0]} {coords_list[2][1]} {coords_list[2][2]}"
    )
    mol = gto.M(
        atom=atom_str,
        unit="Bohr",
        basis="gth-tzvp",
        pseudo="gth-lda",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,pz"
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    return float(mf.kernel())


def build_h2o_coords(d, angle_deg=104.5):
    theta = angle_deg * math.pi / 180.0
    hx = d * math.sin(theta / 2.0)
    hz = d * math.cos(theta / 2.0)
    coords_list = [
        [0.0, 0.0, 0.0],   # O
        [hx, 0.0, hz],     # H1
        [-hx, 0.0, hz],    # H2
    ]
    return coords_list


def scf_settings(d):
    # 直接沿用 verify_h2o.py 的自适应策略
    if d >= 2.8:
        return 0.05, 800
    elif d >= 2.4:
        return 0.10, 600
    else:
        return 0.30, 400


def run_jax_for_spacing(spacing_target, distances, L=18.0):
    N_grid = int(round(L / spacing_target))
    spacing = L / N_grid
    box_size = [L, L, L]
    grid = create_grid(spacing, box_size)

    pseudos = load_pseudopotentials(["O", "H"], pseudo_dir)
    pseudos_for_calc = [pseudos[0], pseudos[1], pseudos[1]]

    key = jax.random.PRNGKey(42)
    energies = []

    for d in distances:
        coords_jax = jnp.array(build_h2o_coords(d), dtype=jnp.float32)
        alpha, max_iter = scf_settings(d)
        e_jax, _ = solver.energy_and_forces(
            grid,
            coords_jax,
            pseudos_for_calc,
            max_iter,
            alpha,
            1e-5,
            key,
        )
        energies.append(float(e_jax))

    return spacing, energies, pseudos


def summarize_errors(jax_energies, pyscf_energies):
    diffs = [ej - ep for ej, ep in zip(jax_energies, pyscf_energies)]
    abs_diffs = [abs(x) for x in diffs]
    mae = sum(abs_diffs) / len(abs_diffs)
    rmse = (sum(x * x for x in diffs) / len(diffs)) ** 0.5
    max_abs = max(abs_diffs)
    return diffs, mae, rmse, max_abs


def main():
    print("\n" + "=" * 20 + " H2O spacing sweep: JaxDFT vs PySCF " + "=" * 20)

    # 和现有 verify_h2o.py 一样的扫描点
    # 只保留 3 个代表点：
# 1.6  压缩区
# 1.9  近平衡区
# 2.8  拉伸区 
    distances = [1.6, 1.9, 2.8]

# 多测几个 spacing，看误差是否单调收敛
    spacings_to_test = [0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10]
    L = 18.0

    # PySCF 只算一次，所有 spacing 共用
    print("先计算 PySCF 参考...")
    pyscf_energies = []
    for d in distances:
        coords_list = build_h2o_coords(d)
        e_pyscf = run_pyscf(coords_list)
        pyscf_energies.append(e_pyscf)

    print("PySCF 参考完成。")
    print("-" * 90)

    all_results = []

    for spacing_target in spacings_to_test:
        print(f"\n>>> 测试 target spacing = {spacing_target:.3f}")
        spacing, jax_energies, pseudos = run_jax_for_spacing(spacing_target, distances, L=L)
        diffs, mae, rmse, max_abs = summarize_errors(jax_energies, pyscf_energies)

        print(f"实际 spacing = {spacing:.6f}, L = {L}, O rloc = {pseudos[0]['rloc']}, H rloc = {pseudos[1]['rloc']}")
        print("-" * 78)
        print(f"{'O-H Dist':<8} | {'JaxDFT':<15} | {'PySCF':<15} | {'Diff'}")
        print("-" * 78)
        for d, ej, ep, df in zip(distances, jax_energies, pyscf_energies, diffs):
            print(f"{d:<8.2f} | {ej:<15.6f} | {ep:<15.6f} | {df:.6f}")
        print("-" * 78)
        print(f"MAE = {mae:.6f} Ha, RMSE = {rmse:.6f} Ha, Max|Diff| = {max_abs:.6f} Ha")

        all_results.append(
            {
                "target_spacing": spacing_target,
                "spacing": spacing,
                "jax_energies": jax_energies,
                "diffs": diffs,
                "mae": mae,
                "rmse": rmse,
                "max_abs": max_abs,
            }
        )

    # --------------------------------------------------
    # 图1：势能曲线对比
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(distances, pyscf_energies, "k--", linewidth=2, label="PySCF (TZVP, gth-lda, lda,pz)")
    for res in all_results:
        plt.plot(
            distances,
            res["jax_energies"],
            "o-",
            label=f"JaxDFT spacing={res['spacing']:.4f}"
        )
    plt.xlabel("O-H Symmetric Bond Length (Bohr)")
    plt.ylabel("Total Energy (Ha)")
    plt.title("H2O symmetric stretch: spacing sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("h2o_spacing_curves.png", dpi=150)

    # --------------------------------------------------
    # 图2：误差曲线
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    for res in all_results:
        plt.plot(
            distances,
            res["diffs"],
            "o-",
            label=f"spacing={res['spacing']:.4f}"
        )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("O-H Symmetric Bond Length (Bohr)")
    plt.ylabel("JaxDFT - PySCF (Ha)")
    plt.title("H2O error vs bond length under spacing refinement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("h2o_spacing_errors.png", dpi=150)

    # --------------------------------------------------
    # 图3：误差指标 vs spacing
    # --------------------------------------------------
    plot_spacings = [res["spacing"] for res in all_results]
    maes = [res["mae"] for res in all_results]
    rmses = [res["rmse"] for res in all_results]
    maxabs = [res["max_abs"] for res in all_results]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_spacings, maes, "o-", label="MAE")
    plt.plot(plot_spacings, rmses, "s-", label="RMSE")
    plt.plot(plot_spacings, maxabs, "^-", label="Max|Diff|")
    plt.xlabel("Actual grid spacing (Bohr)")
    plt.ylabel("Error (Ha)")
    plt.title("H2O error metrics vs spacing")
    plt.gca().invert_xaxis()  # 右边更细，视觉上更直观
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("h2o_spacing_summary.png", dpi=150)

    print("\n" + "=" * 90)
    print("总结（看这三列是否随 spacing 变细而明显下降）")
    print("-" * 90)
    print(f"{'target':<10} | {'actual':<10} | {'MAE':<12} | {'RMSE':<12} | {'Max|Diff|'}")
    print("-" * 90)
    for res in all_results:
        print(
            f"{res['target_spacing']:<10.3f} | "
            f"{res['spacing']:<10.6f} | "
            f"{res['mae']:<12.6f} | "
            f"{res['rmse']:<12.6f} | "
            f"{res['max_abs']:.6f}"
        )
    print("-" * 90)
    print("图片已保存：")
    print("  - h2o_spacing_curves.png")
    print("  - h2o_spacing_errors.png")
    print("  - h2o_spacing_summary.png")


if __name__ == "__main__":
    main()
