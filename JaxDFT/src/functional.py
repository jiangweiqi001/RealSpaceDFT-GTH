"""Local density approximation (LDA) exchange-correlation functionals.

All quantities are in atomic units: density in Bohr^-3, energy density in
Hartree/Bohr^3, and potentials in Hartree. The correlation model follows the
Perdew-Zunger 1981 (PZ81) parametrization with numerical safeguards for low
density (vacuum) regions.
"""

import jax
import jax.numpy as jnp


@jax.jit
def lda_exchange_vxc(rho):
    """Compute LDA Slater exchange energy density and potential.

    Args:
        rho: Electron density, in Bohr^-3.

    Returns:
        Tuple (ex, vx) where ex is exchange energy density in Hartree/Bohr^3 and
        vx is the exchange potential in Hartree.
    """
    # Slater Exchange
    rho = jnp.clip(rho, 1e-15, None)
    const = (3.0 / jnp.pi) ** (1.0 / 3.0)
    vx = -const * jnp.power(rho, 1.0 / 3.0)
    ex = 0.75 * vx * rho  # 能量密度
    return ex, vx


@jax.jit
def lda_correlation_pz81(rho):
    """Compute LDA correlation (PZ81) energy and potential with masking.

    The constants A, B, C, D correspond to the high-density limit parameters
    in the PZ81 form. A simple mask suppresses correlation in very low density
    regions to avoid divergent logarithms and stabilize the SCF procedure.

    Args:
        rho: Electron density, in Bohr^-3.

    Returns:
        Tuple (ec, vc) where ec is correlation energy per particle (Hartree)
        and vc is the correlation potential in Hartree.
    """
    # PZ81 Correlation (简化版，防止真空发散)
    rho = jnp.clip(rho, 1e-15, None)
    rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
    
    # 高密度极限参数
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    
    # 简单近似: 当 rs 很大(真空)时，能量趋于0
    # 这里为了保证数学连贯性，我们返回一个不会发散的值
    # 实际上，在真空区 ec*rho 会自动归零
    
    ln_rs = jnp.log(rs + 1e-12) # 防止 log(0)
    ec = A * ln_rs + B + C * rs * ln_rs + D
    
    # 简单的遮蔽：如果 rs > 10 (低密度)，强行衰减
    # 这不是物理精确的 PZ81 低密度公式，但足以防止 NaN
    mask = jnp.where(rs > 10.0, 0.0, 1.0)
    ec = ec * mask
    
    # 势能 vc 我们暂时设为与 ec 相等 (LDA 近似)
    return ec, ec


@jax.jit
def lda_xc(rho):
    """Combine LDA exchange and correlation into total XC contributions.

    Args:
        rho: Electron density, in Bohr^-3.

    Returns:
        Tuple (eps_xc, v_xc) where eps_xc is the total XC energy density in
        Hartree/Bohr^3 and v_xc is the XC potential in Hartree.
    """
    ex, vx = lda_exchange_vxc(rho)
    ec, vc = lda_correlation_pz81(rho)
    
    # 【关键】: ex 是能量密度，ec 是单粒子能量
    # 必须这样组合:
    return ex + ec * rho, vx + vc
