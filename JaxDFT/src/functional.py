import jax
import jax.numpy as jnp


@jax.jit
def lda_exchange_vxc(rho):
    const = (3.0 / jnp.pi) ** (1.0 / 3.0)
    vx = -const * jnp.power(rho + 1e-20, 1.0 / 3.0)
    ex = 0.75 * vx * rho
    return ex, vx


@jax.jit
def lda_correlation_pz81(rho):
    a = 0.0310907
    b = -0.048
    c = 0.0020
    d = -0.0116
    rs = (3.0 / (4.0 * jnp.pi * (rho + 1e-30))) ** (1.0 / 3.0)
    ec = a * (jnp.log(rs) + b * rs + c * rs * jnp.log(rs) + d)
    d_ec_drho = a * (1.0 / rs + b + c * (1.0 + jnp.log(rs))) * (-rs / (3.0 * rho + 1e-30))
    vc = ec + rho * d_ec_drho
    return ec, vc


@jax.jit
def lda_xc(rho):
    ex, vx = lda_exchange_vxc(rho)
    ec, vc = lda_correlation_pz81(rho)
    return ex + ec, vx + vc
