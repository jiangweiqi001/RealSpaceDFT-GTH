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
    """实现完整的 PZ81 相关泛函及其对应的势能。"""
    rho = jnp.clip(rho, 1e-15, None)
    rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)

    # --- 情况 1: 高密度极限 (rs < 1) ---
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    ec_high = A * jnp.log(rs) + B + C * rs * jnp.log(rs) + D * rs
    # 对应的势能项 rho * d(ec)/d(rho) = - (rs/3) * d(ec)/d(rs)
    vc_high = ec_high - (A/3.0 + (C/3.0)*rs*jnp.log(rs) + (C/3.0)*rs + (D/3.0)*rs)

    # --- 情况 2: 低密度区域 (rs >= 1) ---
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    denom = 1.0 + beta1 * jnp.sqrt(rs) + beta2 * rs
    ec_low = gamma / denom
    # 对应的势能项
    vc_low = ec_low * (1.0 + (7.0/6.0)*beta1*jnp.sqrt(rs) + (4.0/3.0)*beta2*rs) / denom

    # 根据 rs 的值进行平滑选择
    ec = jnp.where(rs < 1.0, ec_high, ec_low)
    vc = jnp.where(rs < 1.0, vc_high, vc_low)

    return ec, vc


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
