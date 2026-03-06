"""Hamiltonian construction utilities for real-space DFT.

Provides grid creation, GTH local pseudopotential evaluation, and a 4th-order
finite-difference Laplacian. All quantities are in atomic units (Bohr, Hartree).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gamma


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


def shift_array(arr, shift, axis):
    rolled = jnp.roll(arr, shift, axis=axis)
    if axis == 0:
        rolled = jnp.where(jnp.arange(arr.shape[0])[:, None, None] < shift if shift > 0 else jnp.arange(arr.shape[0])[:, None, None] >= arr.shape[0] + shift, 0.0, rolled)
    elif axis == 1:
        rolled = jnp.where(jnp.arange(arr.shape[1])[None, :, None] < shift if shift > 0 else jnp.arange(arr.shape[1])[None, :, None] >= arr.shape[1] + shift, 0.0, rolled)
    elif axis == 2:
        rolled = jnp.where(jnp.arange(arr.shape[2])[None, None, :] < shift if shift > 0 else jnp.arange(arr.shape[2])[None, None, :] >= arr.shape[2] + shift, 0.0, rolled)
    return rolled

@jax.jit
def laplacian_4th(psi, spacing, mask=None):
    h2 = spacing * spacing
    c0 = -2.5 / h2
    c1 = (4.0/3.0) / h2
    c2 = (-1.0/12.0) / h2
    lap = 3.0 * c0 * psi
    lap += c1 * (shift_array(psi, 1, 0) + shift_array(psi, -1, 0))
    lap += c2 * (shift_array(psi, 2, 0) + shift_array(psi, -2, 0))
    lap += c1 * (shift_array(psi, 1, 1) + shift_array(psi, -1, 1))
    lap += c2 * (shift_array(psi, 2, 1) + shift_array(psi, -2, 1))
    lap += c1 * (shift_array(psi, 1, 2) + shift_array(psi, -1, 2))
    lap += c2 * (shift_array(psi, 2, 2) + shift_array(psi, -2, 2))
    if mask is not None: lap = lap * mask
    return lap



def apply_nonlocal(grid, psi, atom_coords, pseudos):
    """
    将非局域势投影器应用到波函数 psi 上，支持 l=0 (s) 和 l=1 (p) 通道。
    """
    res = jnp.zeros_like(psi)
    dv = grid.volume_element
    
    for i_at in range(len(pseudos)):
        p_at = pseudos[i_at]
        if not p_at['projectors']: continue
        
        # 计算相对于原子中心的坐标和距离
        diff = grid.coords - atom_coords[i_at]  # 形状: (nx, ny, nz, 3)
        r = jnp.linalg.norm(diff, axis=-1)      # 形状: (nx, ny, nz)
        r_safe = r + 1e-12                      # 防止除以零
        
        for ch in p_at['projectors']:
            l = ch['l']
            rp = ch['r']
            h_mat = ch['h']  # 系数矩阵
            
            # 获取径向投影函数 p_i^l(r)
            p_rad = get_gth_projector(r, l, 1, rp)
            
            if l == 0:
                # l=0 (s-type): 角度部分是 1/sqrt(4*pi)
                # 贡献因子 = h * <p|psi> * p / (4*pi)
                overlap = jnp.sum(p_rad * psi) * dv
                res = res + (h_mat[0] / (4.0 * jnp.pi)) * overlap * p_rad
                
            elif l == 1:
                # l=1 (p-type): 包含 x, y, z 三个方向投影
                # 贡献因子 = 3 * h * <p_i|psi> * p_i / (4*pi)
                for axis in range(3):
                    # 角度部分: r_i / r
                    p_ang = diff[..., axis] / r_safe
                    p_full = p_rad * p_ang
                    overlap = jnp.sum(p_full * psi) * dv
                    res = res + (3.0 * h_mat[0] / (4.0 * jnp.pi)) * overlap * p_full
                    
    return res