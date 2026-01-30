import jax
import jax.numpy as jnp
import numpy as np

# ==================== 1. 定义物理核心 (修正版) ====================
@jax.jit
def lda_exchange_vxc_safe(rho):
    # 纯交换能 (Slater Exchange)
    # 修正: 加上 1e-12 防止 rho=0 时梯度爆炸
    rho = jnp.clip(rho, 1e-12, None)
    const = (3.0 / jnp.pi) ** (1.0 / 3.0)
    vx = -const * jnp.power(rho, 1.0 / 3.0) # V_x = -(3/pi * rho)^(1/3)
    ex = 0.75 * vx * rho                    # E_x density = 3/4 * V_x * rho
    return ex, vx

@jax.jit
def laplacian_4th(psi, spacing):
    # 4阶有限差分拉普拉斯
    h2 = spacing * spacing
    c0 = -2.5 / h2
    c1 = (4.0 / 3.0) / h2
    c2 = (-1.0 / 12.0) / h2
    lap = 3.0 * c0 * psi
    lap += c1 * (jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0))
    lap += c2 * (jnp.roll(psi, 2, axis=0) + jnp.roll(psi, -2, axis=0))
    lap += c1 * (jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1))
    lap += c2 * (jnp.roll(psi, 2, axis=1) + jnp.roll(psi, -2, axis=1))
    lap += c1 * (jnp.roll(psi, 1, axis=2) + jnp.roll(psi, -1, axis=2))
    lap += c2 * (jnp.roll(psi, 2, axis=2) + jnp.roll(psi, -2, axis=2))
    return lap

@jax.jit
def gth_local_potential(r, zion, rloc):
    # 纯软库仑势 (忽略 c 系数)
    # V = -Z * erf(r/(sqrt(2)*rloc)) / r
    root2 = 1.41421356
    t = r / (root2 * rloc)
    # 避免 r=0 除零
    v_coul = -zion * jax.scipy.special.erf(t) / (r + 1e-12)
    return v_coul

# ==================== 2. 求解器 (带调试打印) ====================
def run_diagnostic():
    print(f"{'='*20} 开始独立诊断 (H2, d=1.4) {'='*20}")
    
    # --- 参数设置 ---
    spacing = 0.5
    box_L = 5.0
    N = int(box_L / spacing) + 1
    print(f"网格: {N}x{N}x{N} (Spacing={spacing})")
    
    # 构建坐标
    x = jnp.linspace(-box_L/2, box_L/2, N)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    grid_coords = jnp.stack([X, Y, Z], axis=-1)
    
    # 原子位置 (H2, d=1.4)
    atom1 = jnp.array([0.0, 0.0, -0.7])
    atom2 = jnp.array([0.0, 0.0,  0.7])
    
    # 构建 V_loc
    r1 = jnp.linalg.norm(grid_coords - atom1, axis=-1)
    r2 = jnp.linalg.norm(grid_coords - atom2, axis=-1)
    # 强制使用软原子参数
    rloc_param = 1.0
    V_loc = gth_local_potential(r1, 1.0, rloc_param) + gth_local_potential(r2, 1.0, rloc_param)
    
    print(f"V_loc 统计: Min={V_loc.min():.4f}, Max={V_loc.max():.4f}")
    
    # 初始猜想 (高斯叠加)
    rho = jnp.exp(-r1**2) + jnp.exp(-r2**2)
    # 归一化到 2.0 (两个电子)
    current_charge = jnp.sum(rho) * (spacing**3)
    rho = rho / current_charge * 2.0
    
    # SCF 循环 (简化版，只跑 15 步)
    print("\n--- 进入 SCF ---")
    volume_element = spacing**3
    occ = jnp.array([2.0]) # 1个轨道，占据2个电子
    
    # 预编译 JIT 函数以加速循环
    @jax.jit
    def step(rho):
        # 1. Poisson
        nx, ny, nz = rho.shape
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=spacing)
        KX, KY, KZ = jnp.meshgrid(kx, kx, kx, indexing="ij")
        k2 = KX**2 + KY**2 + KZ**2
        rho_k = jnp.fft.fftn(rho)
        v_k = 4.0 * jnp.pi * rho_k / jnp.where(k2 > 0, k2, 1.0)
        v_k = v_k.at[0, 0, 0].set(0.0)
        V_H = jnp.fft.ifftn(v_k).real
        
        # 2. XC (纯交换)
        eps_xc, V_xc = lda_exchange_vxc_safe(rho) # eps_xc 是能量密度
        
        # 3. Hamiltonian
        V_eff = V_loc + V_H + V_xc
        
        # 4. 对角化 (Dense)
        n_grid = rho.size
        eye = jnp.eye(n_grid)
        
        # 定义局部 apply_h
        def apply_h_local(psi_flat):
            psi = psi_flat.reshape((N, N, N))
            lap = laplacian_4th(psi, spacing)
            hpsi = -0.5 * lap + V_eff * psi
            return hpsi.flatten()

        h_dense = jax.vmap(apply_h_local, in_axes=1, out_axes=1)(eye)
        # 确保对称
        h_dense = 0.5 * (h_dense + h_dense.T)
        
        eigvals, eigvecs = jnp.linalg.eigh(h_dense)
        psi = eigvecs[:, 0].reshape((N, N, N))
        
        # 5. 更新密度
        # 归一化
        norm = jnp.sqrt(jnp.sum(psi**2) * volume_element)
        psi = psi / norm
        rho_new = (psi**2) * 2.0
        
        return rho_new, eigvals, V_H, eps_xc, V_xc

    # 开始循环
    for i in range(15):
        rho_new, eigvals, V_H, eps_xc, V_xc = step(rho)
        # 简单混合
        rho = 0.7 * rho + 0.3 * rho_new
        print(f"Iter {i+1}: Eval[0]={eigvals[0]:.5f}")

    # ==================== 3. 最终能量分析 ====================
    print("\n" + "="*20 + " 最终能量账单 " + "="*20)
    E_band = eigvals[0] * 2.0
    
    # 能量项积分
    E_H_integral = 0.5 * jnp.sum(rho * V_H) * volume_element
    # eps_xc 是能量密度 (ex * rho)，直接积分即可
    E_XC_integral = jnp.sum(eps_xc) * volume_element 
    E_Vxc_integral = jnp.sum(rho * V_xc) * volume_element
    E_Ion = 1.0 * 1.0 / 1.4 # H-H 排斥
    
    E_total = E_band - E_H_integral + E_XC_integral - E_Vxc_integral + E_Ion
    
    print(f"E_Band (2*epsilon) : {E_band:.5f}")
    print(f"E_Hartree          : {E_H_integral:.5f}")
    print(f"E_XC (Exchange)    : {E_XC_integral:.5f}")
    print(f"E_Vxc (Correction) : {E_Vxc_integral:.5f}")
    print(f"E_Ion              : {E_Ion:.5f}")
    print("-" * 30)
    print(f"Calculated Total   : {E_total:.5f} Ha")
    print("=" * 55)

    # 完整性检查
    print(f"检查: Sum(rho)*dV = {jnp.sum(rho)*volume_element:.4f}")
    print(f"检查: V_xc range = [{V_xc.min():.4f}, {V_xc.max():.4f}]")

if __name__ == "__main__":
    run_diagnostic()
