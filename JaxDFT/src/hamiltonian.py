"""Hamiltonian construction utilities for real-space DFT.

Provides grid creation, GTH local pseudopotential evaluation, and a 4th-order
finite-difference Laplacian. All quantities are in atomic units (Bohr, Hartree).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gamma
from jax.scipy.signal import convolve

def create_grid(spacing, box_size):
    """Create a uniform real-space grid for DFT calculations.

    Args:
        spacing: Grid spacing in Bohr.
        box_size: Simulation box lengths [Lx, Ly, Lz] in Bohr.

    Returns:
        Grid object with coordinates, spacing, volume element, and mask.
    """
    box_size = jnp.array(box_size)
    N = (box_size / spacing).astype(int) + 1
    x = jnp.linspace(-box_size[0]/2, box_size[0]/2, N[0])
    y = jnp.linspace(-box_size[1]/2, box_size[1]/2, N[1])
    z = jnp.linspace(-box_size[2]/2, box_size[2]/2, N[2])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    coords = jnp.stack([X, Y, Z], axis=-1)
    
    class Grid: pass
    grid = Grid()
    grid.coords = coords
    grid.shape = coords.shape[:-1]
    grid.spacing = spacing
    grid.box_size = box_size
    grid.volume_element = spacing ** 3
    grid.mask = jnp.ones(grid.shape, dtype=jnp.float32)
    grid.projectors = [] 
    return grid


@jax.jit
def gth_local_potential_value(r, zion, rloc, c):
    """Evaluate GTH local pseudopotential at radius r.

    Uses an error-function softened Coulomb term and a Gaussian-polynomial
    local correction. A small r shift avoids division-by-zero singularities.

    Args:
        r: Radial distance(s) in Bohr.
        zion: Ionic charge.
        rloc: GTH local radius parameter in Bohr.
        c: GTH local polynomial coefficients.

    Returns:
        Local potential value(s) in Hartree.
    """
    # 【关键】防爆除法 + 软化势
    root2 = 1.41421356
    r_safe = r + 1e-12
    t = r_safe / (root2 * rloc)
    v_coul = -zion * erf(t) / r_safe
    
    val = r_safe / rloc
    gauss = jnp.exp(-0.5 * val * val)
    poly = c[0] + c[1]*(val**2) + c[2]*(val**4) + c[3]*(val**6)
    return v_coul + gauss * poly

@jax.jit
def get_gth_projector(r, l, i, rp):
    """
    计算 GTH 径向投影函数 p_i^l(r)。
    i 是投影器索引 (1, 2, 3)，rp 是投影半径。
    """
    # GTH 标准归一化系数
    t = l + (4.0 * i - 1.0) / 2.0
    norm = jnp.sqrt(2.0) / (rp**t * jnp.sqrt(gamma(t)))
    return norm * (r**(l + 2*i - 2)) * jnp.exp(-0.5 * (r/rp)**2)

@jax.jit
def build_local_potential(atom_coords, grid_coords, zion, rloc, c):
    """Assemble the total local ionic potential on the grid.

    Args:
        atom_coords: Ion coordinates, shape (n_atoms, 3), in Bohr.
        grid_coords: Grid coordinates, shape (nx, ny, nz, 3), in Bohr.
        zion: Ionic charges per atom.
        rloc: Local radius parameters per atom, in Bohr.
        c: Local polynomial coefficients per atom.

    Returns:
        Total local potential on the grid, in Hartree.
    """
    V_total = jnp.zeros(grid_coords.shape[:-1], dtype=jnp.float32)
    for i in range(len(zion)):
        diff = grid_coords - atom_coords[i]
        r = jnp.linalg.norm(diff, axis=-1)
        v = gth_local_potential_value(r, zion[i], rloc[i], c[i])
        V_total = V_total + v
    return V_total



@jax.jit
def laplacian_4th(psi, spacing, mask=None):
    """
    使用 3D 卷积计算 4阶有限差分拉普拉斯算符。
    利用 mode='same' 自动实现零填充（严格的 Dirichlet 边界条件）。
    """
    h2 = spacing * spacing
    c0 = -2.5 / h2
    c1 = (4.0/3.0) / h2
    c2 = (-1.0/12.0) / h2
    
    # 1. 构造 5x5x5 的拉普拉斯卷积核
    # 绝大部分权重为 0，只有中心十字架上有值
    kernel = jnp.zeros((5, 5, 5), dtype=psi.dtype)
    
    # 中心点 (x, y, z)
    kernel = kernel.at[2, 2, 2].set(3.0 * c0)
    
    # 第一层近邻 (距离 1，系数 c1)
    kernel = kernel.at[1, 2, 2].set(c1)
    kernel = kernel.at[3, 2, 2].set(c1)
    kernel = kernel.at[2, 1, 2].set(c1)
    kernel = kernel.at[2, 3, 2].set(c1)
    kernel = kernel.at[2, 2, 1].set(c1)
    kernel = kernel.at[2, 2, 3].set(c1)
    
    # 第二层近邻 (距离 2，系数 c2)
    kernel = kernel.at[0, 2, 2].set(c2)
    kernel = kernel.at[4, 2, 2].set(c2)
    kernel = kernel.at[2, 0, 2].set(c2)
    kernel = kernel.at[2, 4, 2].set(c2)
    kernel = kernel.at[2, 2, 0].set(c2)
    kernel = kernel.at[2, 2, 4].set(c2)
    
    # 2. 执行 3D 卷积
    # mode='same' 会在 psi 边缘自动补零，计算后保持原 shape
    lap = convolve(psi, kernel, mode='same')
    
    # 3. 施加掩膜 (如有)
    if mask is not None: 
        lap = lap * mask
        
    return lap



def precompute_projectors(grid, atom_coords, pseudos):
    """
    在 SCF 外预先计算所有的非局域势投影器在实空间网格上的值。
    将其打包成 4D 张量，供 JAX 进行极速张量批处理。
    """
    p_i_list = []
    p_j_list = []
    coeff_list = []
    
    for i_at in range(len(pseudos)):
        p_at = pseudos[i_at]
        if not p_at['projectors']: continue
        
        diff = grid.coords - atom_coords[i_at]  # (nx, ny, nz, 3)
        r = jnp.linalg.norm(diff, axis=-1)      # (nx, ny, nz)
        r_safe = r + 1e-12
        
        for ch in p_at['projectors']:
            l = ch['l']
            rp = ch['r']
            h_mat = jnp.array(ch['h']) 
            if h_mat.ndim == 1:
                h_mat = jnp.diag(h_mat)
            n_proj = h_mat.shape[0]
            
            for i in range(1, n_proj + 1):
                p_i_rad = get_gth_projector(r, l, i, rp)
                for j in range(1, n_proj + 1):
                    h_ij = h_mat[i-1, j-1]
                    if jnp.abs(h_ij) < 1e-10: 
                        continue # 忽略为 0 的通道
                    
                    p_j_rad = get_gth_projector(r, l, j, rp)
                    
                    if l == 0:
                        p_i_list.append(p_i_rad)
                        p_j_list.append(p_j_rad)
                        coeff_list.append(h_ij / (4.0 * jnp.pi))
                    elif l == 1:
                        for axis in range(3): # p型势包含三个空间方向
                            p_i_full = p_i_rad * (diff[..., axis] / r_safe)
                            p_j_full = p_j_rad * (diff[..., axis] / r_safe)
                            p_i_list.append(p_i_full)
                            p_j_list.append(p_j_full)
                            coeff_list.append(3.0 * h_ij / (4.0 * jnp.pi))
                            
    if not p_i_list:
        return None
        
    P_i = jnp.stack(p_i_list, axis=0) # 形状: (通道数, nx, ny, nz)
    P_j = jnp.stack(p_j_list, axis=0)
    coeffs = jnp.array(coeff_list)    # 形状: (通道数,)
    return P_i, P_j, coeffs

@jax.jit
def apply_nonlocal_precomputed(psi, P_i, P_j, coeffs, dv):
    """
    利用预计算的投影器张量，进行极速批处理点乘，彻底去除高斯指数计算。
    """
    # 计算所有通道的 overlap，并做体积分 (axis 1,2,3 是空间维度)
    overlap = jnp.sum(P_j * psi[None, ...], axis=(1, 2, 3)) * dv
    
    weight = coeffs * overlap
    # 结果累加
    return jnp.sum(weight[:, None, None, None] * P_i, axis=0)