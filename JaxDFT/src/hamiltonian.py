"""Hamiltonian construction utilities for real-space DFT.

Provides grid creation, GTH local pseudopotential evaluation, finite-difference
Laplacians, and nonlocal projector helpers. All quantities are in atomic units
(Bohr, Hartree).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gamma

def create_grid(spacing, box_size, policy="preserve_box", tol=1e-8, dtype=jnp.float32, phase=0.0):
    """
    policy:
        - "strict": require box_size / spacing to be integer in each direction
        - "preserve_box": keep box_size exact, adjust actual spacing
        - "preserve_spacing": keep spacing exact, adjust effective box_size
    """
    dtype = jnp.dtype(dtype)
    spacing = float(spacing)
    box_size = jnp.asarray(box_size, dtype=dtype)
    phase = float(phase)

    ratios = box_size / spacing

    if policy == "strict":
        n_intervals = jnp.rint(ratios).astype(jnp.int32)
        err = jnp.max(jnp.abs(ratios - n_intervals))
        if float(err) > tol:
            raise ValueError(
                f"box_size must be an integer multiple of spacing. "
                f"box_size={box_size.tolist()}, spacing={spacing}, "
                f"box_size/spacing={ratios.tolist()}"
            )
        actual_box_size = box_size
        actual_spacing_vec = box_size / n_intervals

    elif policy == "preserve_box":
        # 保留 box_size，允许真实 spacing 轻微偏离输入 spacing
        n_intervals = jnp.maximum(1, jnp.rint(ratios).astype(jnp.int32))
        actual_box_size = box_size
        actual_spacing_vec = actual_box_size / n_intervals

    elif policy == "preserve_spacing":
        # 保留 spacing，允许 box_size 轻微偏离输入 box_size
        n_intervals = jnp.maximum(1, jnp.rint(ratios).astype(jnp.int32))
        actual_spacing_vec = jnp.array([spacing, spacing, spacing], dtype=dtype)
        actual_box_size = actual_spacing_vec * n_intervals

    else:
        raise ValueError("policy must be 'strict', 'preserve_box', or 'preserve_spacing'")

    # 你当前下游代码假设是各向同性标量 spacing
    if not jnp.allclose(actual_spacing_vec, actual_spacing_vec[0], atol=tol, rtol=0.0):
        raise ValueError(
            f"Current code assumes isotropic scalar spacing, but got actual_spacing={actual_spacing_vec.tolist()}"
        )

    actual_spacing = float(actual_spacing_vec[0])

    nx, ny, nz = map(int, (n_intervals + 1).tolist())

    phase_shift = phase * actual_spacing
    x = jnp.linspace(-actual_box_size[0] / 2, actual_box_size[0] / 2, nx, dtype=dtype) + phase_shift
    y = jnp.linspace(-actual_box_size[1] / 2, actual_box_size[1] / 2, ny, dtype=dtype) + phase_shift
    z = jnp.linspace(-actual_box_size[2] / 2, actual_box_size[2] / 2, nz, dtype=dtype) + phase_shift

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    coords = jnp.stack([X, Y, Z], axis=-1)

    class Grid:
        pass

    grid = Grid()
    grid.coords = coords
    grid.shape = coords.shape[:-1]
    grid.spacing = jnp.asarray(actual_spacing, dtype=dtype)
    grid.box_size = actual_box_size
    grid.volume_element = grid.spacing ** 3
    grid.mask = jnp.ones(grid.shape, dtype=dtype)
    grid.projectors = []

    # 调试信息，方便你检查输入和实际采用值
    grid.requested_spacing = jnp.asarray(spacing, dtype=dtype)
    grid.requested_box_size = box_size
    grid.actual_spacing_vec = actual_spacing_vec
    grid.n_intervals = n_intervals
    grid.grid_phase = jnp.asarray(phase, dtype=dtype)
    grid.dtype = dtype

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
    """
    Assemble total local ionic potential on the grid.
    Uses lax.fori_loop to avoid Python-loop unrolling inside jit.
    """
    dtype = grid_coords.dtype
    atom_coords = jnp.asarray(atom_coords, dtype=dtype)
    zion = jnp.asarray(zion, dtype=dtype)
    rloc = jnp.asarray(rloc, dtype=dtype)
    c = jnp.asarray(c, dtype=dtype)

    init = jnp.zeros(grid_coords.shape[:-1], dtype=dtype)

    def body(i, V_total):
        diff = grid_coords - atom_coords[i]
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
        v = gth_local_potential_value(r, zion[i], rloc[i], c[i])
        return V_total + v

    return jax.lax.fori_loop(0, atom_coords.shape[0], body, init)



@jax.jit
def laplacian_4th(psi, spacing, mask=None):
    """
    4th-order finite-difference Laplacian with strict Dirichlet BC
    implemented by explicit zero-padding + stencil slices.
    """
    inv_h2 = 1.0 / (spacing * spacing)
    c0 = -2.5 * inv_h2
    c1 = (4.0 / 3.0) * inv_h2
    c2 = (-1.0 / 12.0) * inv_h2

    p = jnp.pad(psi, ((2, 2), (2, 2), (2, 2)), mode="constant")

    center = p[2:-2, 2:-2, 2:-2]

    lap = (
        3.0 * c0 * center
        + c1 * (
            p[1:-3, 2:-2, 2:-2] + p[3:-1, 2:-2, 2:-2]
            + p[2:-2, 1:-3, 2:-2] + p[2:-2, 3:-1, 2:-2]
            + p[2:-2, 2:-2, 1:-3] + p[2:-2, 2:-2, 3:-1]
        )
        + c2 * (
            p[0:-4, 2:-2, 2:-2] + p[4:   , 2:-2, 2:-2]
            + p[2:-2, 0:-4, 2:-2] + p[2:-2, 4:   , 2:-2]
            + p[2:-2, 2:-2, 0:-4] + p[2:-2, 2:-2, 4:   ]
        )
    )

    if mask is not None:
        lap = lap * mask
    return lap

@jax.jit
def laplacian_6th(psi, spacing, mask=None):
    """
    6th-order finite-difference Laplacian with strict Dirichlet BC.
    Provides significantly higher accuracy for hard pseudopotentials and 
    steep density gradients without increasing grid resolution.
    """
    inv_h2 = 1.0 / (spacing * spacing)
    # 六阶中心差分系数
    c0 = -49.0 / 18.0 * inv_h2
    c1 = 1.5 * inv_h2
    c2 = -0.15 * inv_h2
    c3 = (1.0 / 90.0) * inv_h2

    # 边界需要向外 padding 3 层 0 (严格的开边界条件)
    p = jnp.pad(psi, ((3, 3), (3, 3), (3, 3)), mode="constant")

    # 中心点
    center = p[3:-3, 3:-3, 3:-3]

    # 六阶差分模板 (Stencil)
    lap = (
        3.0 * c0 * center
        + c1 * (
            p[2:-4, 3:-3, 3:-3] + p[4:-2, 3:-3, 3:-3]
            + p[3:-3, 2:-4, 3:-3] + p[3:-3, 4:-2, 3:-3]
            + p[3:-3, 3:-3, 2:-4] + p[3:-3, 3:-3, 4:-2]
        )
        + c2 * (
            p[1:-5, 3:-3, 3:-3] + p[5:-1, 3:-3, 3:-3]
            + p[3:-3, 1:-5, 3:-3] + p[3:-3, 5:-1, 3:-3]
            + p[3:-3, 3:-3, 1:-5] + p[3:-3, 3:-3, 5:-1]
        )
        + c3 * (
            p[0:-6, 3:-3, 3:-3] + p[6:  , 3:-3, 3:-3]
            + p[3:-3, 0:-6, 3:-3] + p[3:-3, 6:  , 3:-3]
            + p[3:-3, 3:-3, 0:-6] + p[3:-3, 3:-3, 6:  ]
        )
    )

    if mask is not None:
        lap = lap * mask
    return lap

@jax.jit
def laplacian_8th(psi, spacing, mask=None):
    """
    8th-order finite-difference Laplacian with strict Dirichlet BC.
    The ultimate accuracy for real-space grids.
    """
    inv_h2 = 1.0 / (spacing * spacing)
    # 8阶中心差分系数
    c0 = -205.0 / 72.0 * inv_h2
    c1 = 1.6 * inv_h2           # 8/5
    c2 = -0.2 * inv_h2          # -1/5
    c3 = (8.0 / 315.0) * inv_h2
    c4 = (-1.0 / 560.0) * inv_h2

    # 8阶需要向外 pad 4 层
    p = jnp.pad(psi, ((4, 4), (4, 4), (4, 4)), mode="constant")
    
    # 中心点
    center = p[4:-4, 4:-4, 4:-4]

    lap = (
        3.0 * c0 * center
        + c1 * (
            p[3:-5, 4:-4, 4:-4] + p[5:-3, 4:-4, 4:-4]
            + p[4:-4, 3:-5, 4:-4] + p[4:-4, 5:-3, 4:-4]
            + p[4:-4, 4:-4, 3:-5] + p[4:-4, 4:-4, 5:-3]
        )
        + c2 * (
            p[2:-6, 4:-4, 4:-4] + p[6:-2, 4:-4, 4:-4]
            + p[4:-4, 2:-6, 4:-4] + p[4:-4, 6:-2, 4:-4]
            + p[4:-4, 4:-4, 2:-6] + p[4:-4, 4:-4, 6:-2]
        )
        + c3 * (
            p[1:-7, 4:-4, 4:-4] + p[7:-1, 4:-4, 4:-4]
            + p[4:-4, 1:-7, 4:-4] + p[4:-4, 7:-1, 4:-4]
            + p[4:-4, 4:-4, 1:-7] + p[4:-4, 4:-4, 7:-1]
        )
        + c4 * (
            p[0:-8, 4:-4, 4:-4] + p[8:  , 4:-4, 4:-4]
            + p[4:-4, 0:-8, 4:-4] + p[4:-4, 8:  , 4:-4]
            + p[4:-4, 4:-4, 0:-8] + p[4:-4, 4:-4, 8:  ]
        )
    )

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

def projector_overlap_diagnostics(grid, atom_coords, pseudos):
    """Compute angular-normalized projector overlap matrices on the current grid."""
    diagnostics = []
    atom_coords = jnp.asarray(atom_coords, dtype=grid.coords.dtype)
    for i_at, p_at in enumerate(pseudos):
        if not p_at["projectors"]:
            continue

        diff = grid.coords - atom_coords[i_at]
        r = jnp.linalg.norm(diff, axis=-1)
        r_safe = r + 1e-12

        for ch in p_at["projectors"]:
            l = ch["l"]
            rp = ch["r"]
            h_mat = jnp.asarray(ch["h"], dtype=grid.coords.dtype)
            if h_mat.ndim == 1:
                h_mat = jnp.diag(h_mat)
            n_proj = h_mat.shape[0]
            overlap = jnp.zeros((n_proj, n_proj), dtype=grid.coords.dtype)

            for i in range(1, n_proj + 1):
                p_i = get_gth_projector(r, l, i, rp)
                for j in range(1, n_proj + 1):
                    p_j = get_gth_projector(r, l, j, rp)
                    if l == 0:
                        val = jnp.sum(p_i * p_j) * grid.volume_element / (4.0 * jnp.pi)
                    elif l == 1:
                        direction_norm = jnp.sum((diff / r_safe[..., None]) ** 2, axis=-1)
                        val = (
                            3.0
                            * jnp.sum(p_i * p_j * direction_norm)
                            * grid.volume_element
                            / (4.0 * jnp.pi)
                        )
                    else:
                        val = jnp.array(jnp.nan, dtype=grid.coords.dtype)
                    overlap = overlap.at[i - 1, j - 1].set(val)

            diagnostics.append(
                {
                    "atom_index": i_at,
                    "symbol": p_at.get("symbol", ""),
                    "l": l,
                    "r": rp,
                    "n_proj": n_proj,
                    "h": h_mat,
                    "overlap": overlap,
                    "overlap_error": overlap - jnp.eye(n_proj, dtype=grid.coords.dtype),
                }
            )
    return diagnostics


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
