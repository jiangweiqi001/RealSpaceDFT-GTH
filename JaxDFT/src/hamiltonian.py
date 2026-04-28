"""Hamiltonian construction utilities for real-space DFT.

Provides grid creation, GTH local pseudopotential evaluation, and a 4th-order
finite-difference Laplacian. All quantities are in atomic units (Bohr, Hartree).
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gamma

def create_grid(spacing, box_size, policy="preserve_box", tol=1e-8):
    """
    policy:
        - "strict": require box_size / spacing to be integer in each direction
        - "preserve_box": keep box_size exact, adjust actual spacing
        - "preserve_spacing": keep spacing exact, adjust effective box_size
    """
    spacing = float(spacing)
    box_size = jnp.asarray(box_size, dtype=jnp.float32)

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
        actual_spacing_vec = jnp.array([spacing, spacing, spacing], dtype=jnp.float32)
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

    x = jnp.linspace(-actual_box_size[0] / 2, actual_box_size[0] / 2, nx, dtype=jnp.float32)
    y = jnp.linspace(-actual_box_size[1] / 2, actual_box_size[1] / 2, ny, dtype=jnp.float32)
    z = jnp.linspace(-actual_box_size[2] / 2, actual_box_size[2] / 2, nz, dtype=jnp.float32)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    coords = jnp.stack([X, Y, Z], axis=-1)

    class Grid:
        pass

    grid = Grid()
    grid.coords = coords
    grid.shape = coords.shape[:-1]
    grid.spacing = jnp.asarray(actual_spacing, dtype=jnp.float32)
    grid.box_size = actual_box_size
    grid.volume_element = grid.spacing ** 3
    grid.mask = jnp.ones(grid.shape, dtype=jnp.float32)
    grid.projectors = []

    # 调试信息，方便你检查输入和实际采用值
    grid.requested_spacing = jnp.asarray(spacing, dtype=jnp.float32)
    grid.requested_box_size = box_size
    grid.actual_spacing_vec = actual_spacing_vec
    grid.n_intervals = n_intervals

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


@partial(jax.jit, static_argnames=("local_subgrid", "local_mode", "local_patch_radius_factor"))
def build_local_potential(
    atom_coords,
    grid_coords,
    zion,
    rloc,
    c,
    spacing=None,
    local_subgrid=1,
    local_mode="cell_average",
    local_patch_radius_factor=6.0,
):
    """
    Assemble total local ionic potential on the grid.
    Uses lax.fori_loop to avoid Python-loop unrolling inside jit.
    """
    if local_subgrid < 1:
        raise ValueError("local_subgrid must be >= 1")
    if local_mode not in ("cell_average", "patch"):
        raise ValueError("local_mode must be 'cell_average' or 'patch'")
    if (local_subgrid > 1 or local_mode == "patch") and spacing is None:
        raise ValueError("spacing is required when using local subgrid or patch mode")

    atom_coords = jnp.asarray(atom_coords, dtype=jnp.float32)
    zion = jnp.asarray(zion, dtype=jnp.float32)
    rloc = jnp.asarray(rloc, dtype=jnp.float32)
    c = jnp.asarray(c, dtype=jnp.float32)

    if local_subgrid == 1:
        offsets = jnp.zeros((1, 3), dtype=jnp.float32)
    else:
        spacing = jnp.asarray(spacing, dtype=jnp.float32)
        axis = (jnp.arange(local_subgrid, dtype=jnp.float32) + 0.5) / local_subgrid - 0.5
        axis = axis * spacing
        ox, oy, oz = jnp.meshgrid(axis, axis, axis, indexing="ij")
        offsets = jnp.stack([ox, oy, oz], axis=-1).reshape(-1, 3)

    init = jnp.zeros(grid_coords.shape[:-1], dtype=jnp.float32)

    def body(i, V_total):
        def offset_body(j, v_acc):
            diff = grid_coords + offsets[j] - atom_coords[i]
            r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
            return v_acc + gth_local_potential_value(r, zion[i], rloc[i], c[i])

        diff0 = grid_coords - atom_coords[i]
        r0 = jnp.sqrt(jnp.sum(diff0 * diff0, axis=-1))
        v_point = gth_local_potential_value(r0, zion[i], rloc[i], c[i])
        v_sum = jax.lax.fori_loop(0, offsets.shape[0], offset_body, init)
        v_average = v_sum / offsets.shape[0]

        if local_mode == "patch":
            spacing_arr = jnp.asarray(spacing, dtype=jnp.float32)
            half_cell_diag = 0.8660254 * spacing_arr
            radius = local_patch_radius_factor * rloc[i] + half_cell_diag
            in_patch = r0 <= radius
            v = jnp.where(in_patch, v_average, v_point)
        else:
            v = v_average

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

def _make_atom_patch_points(atom_coord, radius, fine_spacing):
    n_side = max(0, int(math.ceil(float(radius) / float(fine_spacing))))
    axis = jnp.arange(-n_side, n_side + 1, dtype=jnp.float32) * fine_spacing
    ox, oy, oz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    offsets = jnp.stack([ox, oy, oz], axis=-1).reshape(-1, 3)
    mask = jnp.linalg.norm(offsets, axis=-1) <= radius + 1e-12
    offsets = offsets[mask]
    positions = atom_coord + offsets
    fine_dv = fine_spacing ** 3
    return offsets, positions, fine_dv


def _scatter_fine_values_to_grid(grid, positions, values, fine_dv):
    nx, ny, nz = grid.coords.shape[:-1]
    spacing = jnp.asarray(grid.spacing, dtype=jnp.float32)
    coarse_dv = spacing ** 3
    origin = grid.coords[0, 0, 0]
    scaled = (positions - origin) / spacing
    base = jnp.floor(scaled).astype(jnp.int32)
    frac = scaled - base.astype(jnp.float32)

    valid = jnp.all(base >= 0, axis=1)
    valid = jnp.logical_and(valid, base[:, 0] + 1 < nx)
    valid = jnp.logical_and(valid, base[:, 1] + 1 < ny)
    valid = jnp.logical_and(valid, base[:, 2] + 1 < nz)

    flat = jnp.zeros((nx * ny * nz,), dtype=jnp.float32)
    scale = fine_dv / coarse_dv
    for cx in (0, 1):
        wx = frac[:, 0] if cx else 1.0 - frac[:, 0]
        for cy in (0, 1):
            wy = frac[:, 1] if cy else 1.0 - frac[:, 1]
            for cz in (0, 1):
                wz = frac[:, 2] if cz else 1.0 - frac[:, 2]
                idx = base + jnp.array([cx, cy, cz], dtype=jnp.int32)
                safe_idx = jnp.where(valid[:, None], idx, 0)
                flat_idx = safe_idx[:, 0] * (ny * nz) + safe_idx[:, 1] * nz + safe_idx[:, 2]
                contrib = values * wx * wy * wz * scale
                contrib = jnp.where(valid, contrib, 0.0)
                flat = flat.at[flat_idx].add(contrib)

    return flat.reshape((nx, ny, nz))
def _lagrange_weights_1d(t):
    x = 1.0 + t
    w0 = ((x - 1.0) * (x - 2.0) * (x - 3.0)) / -6.0
    w1 = (x * (x - 2.0) * (x - 3.0)) / 2.0
    w2 = (x * (x - 1.0) * (x - 3.0)) / -2.0
    w3 = (x * (x - 1.0) * (x - 2.0)) / 6.0
    return jnp.stack([w0, w1, w2, w3], axis=-1)


def _poly2_basis(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return jnp.stack(
        [
            jnp.ones_like(x),
            x,
            y,
            z,
            x * x,
            y * y,
            z * z,
            x * y,
            x * z,
            y * z,
        ],
        axis=1,
    )


def build_patch_polynomial_reconstruction_data(grid, atom_coord, patch_positions, stencil_half_width=2, reg=1e-10):
    nx, ny, nz = grid.coords.shape[:-1]
    spacing = float(grid.spacing)
    origin = grid.coords[0, 0, 0]
    scaled = (jnp.asarray(atom_coord, dtype=jnp.float32) - origin) / spacing
    center = jnp.rint(scaled).astype(jnp.int32)

    def clipped_axis(center_axis, n_axis):
        start = max(0, int(center_axis) - stencil_half_width)
        stop = min(n_axis, int(center_axis) + stencil_half_width + 1)
        return jnp.arange(start, stop, dtype=jnp.int32)

    ix = clipped_axis(center[0], nx)
    iy = clipped_axis(center[1], ny)
    iz = clipped_axis(center[2], nz)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing="ij")
    sample_indices_3d = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    sample_positions = grid.coords[sample_indices_3d[:, 0], sample_indices_3d[:, 1], sample_indices_3d[:, 2]]
    sample_rel = sample_positions - atom_coord
    patch_rel = jnp.asarray(patch_positions, dtype=jnp.float32) - atom_coord

    a = _poly2_basis(sample_rel)
    b = _poly2_basis(patch_rel)
    ata = a.T @ a + reg * jnp.eye(a.shape[1], dtype=jnp.float32)
    pinv = jnp.linalg.solve(ata, a.T)
    eval_matrix = b @ pinv

    flat_indices = (
        sample_indices_3d[:, 0] * (ny * nz)
        + sample_indices_3d[:, 1] * nz
        + sample_indices_3d[:, 2]
    )
    return flat_indices, eval_matrix


def build_fine_interpolation_data(grid, positions):
    nx, ny, nz = grid.coords.shape[:-1]
    spacing = jnp.asarray(grid.spacing, dtype=jnp.float32)
    origin = grid.coords[0, 0, 0]
    scaled = (positions - origin) / spacing
    cell = jnp.floor(scaled).astype(jnp.int32)
    frac = scaled - cell.astype(jnp.float32)

    def axis_stencil(cell_axis, frac_axis, n_axis):
        base_axis = cell_axis - 1
        cubic_valid = jnp.logical_and(base_axis >= 0, base_axis + 3 < n_axis)
        linear_valid = jnp.logical_and(cell_axis >= 0, cell_axis + 1 < n_axis)

        cubic_idx = base_axis[:, None] + jnp.arange(4, dtype=jnp.int32)[None, :]
        linear_idx = jnp.stack(
            [
                cell_axis,
                cell_axis + 1,
                jnp.zeros_like(cell_axis),
                jnp.zeros_like(cell_axis),
            ],
            axis=1,
        )
        linear_weights = jnp.stack(
            [
                1.0 - frac_axis,
                frac_axis,
                jnp.zeros_like(frac_axis),
                jnp.zeros_like(frac_axis),
            ],
            axis=1,
        )
        idx = jnp.where(cubic_valid[:, None], cubic_idx, linear_idx)
        weights = jnp.where(cubic_valid[:, None], _lagrange_weights_1d(frac_axis), linear_weights)
        valid = jnp.logical_or(cubic_valid, linear_valid)
        return idx, weights, valid

    x_idx, wx, x_valid = axis_stencil(cell[:, 0], frac[:, 0], nx)
    y_idx, wy, y_valid = axis_stencil(cell[:, 1], frac[:, 1], ny)
    z_idx, wz, z_valid = axis_stencil(cell[:, 2], frac[:, 2], nz)
    valid = jnp.logical_and(jnp.logical_and(x_valid, y_valid), z_valid)

    idx_list = []
    weight_list = []
    for cx in range(4):
        for cy in range(4):
            for cz in range(4):
                idx = jnp.stack([x_idx[:, cx], y_idx[:, cy], z_idx[:, cz]], axis=1)
                safe_idx = jnp.where(valid[:, None], idx, 0)
                flat_idx = safe_idx[:, 0] * (ny * nz) + safe_idx[:, 1] * nz + safe_idx[:, 2]
                weight = wx[:, cx] * wy[:, cy] * wz[:, cz]
                weight = jnp.where(valid, weight, 0.0)
                idx_list.append(flat_idx)
                weight_list.append(weight)

    flat_indices = jnp.stack(idx_list, axis=1)
    weights = jnp.stack(weight_list, axis=1)
    return flat_indices, weights, valid


@jax.jit
def gather_fine_values(psi, flat_indices, weights):
    psi_flat = psi.reshape(-1)
    gathered = psi_flat[flat_indices]
    return jnp.sum(weights * gathered, axis=-1)


@jax.jit
def reconstruct_fine_wavefunction(psi, flat_indices, weights, fine_dv):
    psi_fine = gather_fine_values(psi, flat_indices, weights)
    rho_fine = gather_fine_values(psi * psi, flat_indices, weights)
    current_mass = fine_dv * jnp.sum(psi_fine * psi_fine, axis=-1)
    target_mass = fine_dv * jnp.sum(rho_fine, axis=-1)
    scale = jnp.sqrt(jnp.where(current_mass > 1e-20, target_mass / current_mass, 1.0))
    return psi_fine * scale[..., None]


@jax.jit
def reconstruct_patch_wavefunction(psi, patch_sample_indices, patch_eval_matrix):
    psi_flat = psi.reshape(-1)
    coarse_samples = psi_flat[patch_sample_indices]
    return jnp.einsum("cfs,cs->cf", patch_eval_matrix, coarse_samples)


@partial(jax.jit, static_argnames=("output_shape",))
def scatter_patch_wavefunction_adjoint(fine_values, output_shape, patch_sample_indices, patch_eval_matrix, fine_dv, coarse_dv):
    flat_out = jnp.zeros((math.prod(output_shape),), dtype=jnp.asarray(fine_values).dtype)
    sample_contrib = jnp.einsum("cfs,cf->cs", patch_eval_matrix, fine_values) * (fine_dv / coarse_dv)
    flat_out = flat_out.at[patch_sample_indices.reshape(-1)].add(sample_contrib.reshape(-1))
    return flat_out.reshape(output_shape)


@partial(jax.jit, static_argnames=("output_shape",))
def scatter_fine_values_adjoint(fine_values, output_shape, flat_indices, weights, fine_dv, coarse_dv):
    flat_out = jnp.zeros((math.prod(output_shape),), dtype=jnp.asarray(fine_values).dtype)
    contrib = fine_values[..., None] * weights * (fine_dv / coarse_dv)
    flat_out = flat_out.at[flat_indices.reshape(-1)].add(contrib.reshape(-1))
    return flat_out.reshape(output_shape)


def _precompute_projectors_patch(
    grid,
    atom_coords,
    pseudos,
    projector_subgrid,
    projector_patch_radius_factor,
):
    p_i_fine_list = []
    p_j_fine_list = []
    coeff_list = []
    flat_index_list = []
    weight_list = []
    patch_sample_index_list = []
    patch_eval_matrix_list = []
    fine_spacing = jnp.asarray(grid.spacing, dtype=jnp.float32) / projector_subgrid
    fine_dv = fine_spacing ** 3
    coarse_dv = jnp.asarray(grid.spacing, dtype=jnp.float32) ** 3

    for i_at in range(len(pseudos)):
        p_at = pseudos[i_at]
        if not p_at["projectors"]:
            continue

        for ch in p_at["projectors"]:
            l = ch["l"]
            rp = ch["r"]
            radius = float(projector_patch_radius_factor) * float(rp)
            offsets, positions, fine_dv = _make_atom_patch_points(
                atom_coords[i_at], radius, fine_spacing
            )
            flat_indices, weights, _ = build_fine_interpolation_data(grid, positions)
            patch_sample_indices, patch_eval_matrix = build_patch_polynomial_reconstruction_data(
                grid,
                atom_coords[i_at],
                positions,
            )
            r = jnp.linalg.norm(offsets, axis=-1)
            h_mat = jnp.array(ch["h"])
            if h_mat.ndim == 1:
                h_mat = jnp.diag(h_mat)
            n_proj = h_mat.shape[0]
            projector_cache = {}

            def patch_projector_values(proj_idx, axis_idx=None):
                key = (proj_idx, axis_idx)
                if key in projector_cache:
                    return projector_cache[key]

                p_rad = get_gth_projector(r, l, proj_idx, rp)
                if axis_idx is None:
                    values = p_rad
                else:
                    values = p_rad * (offsets[:, axis_idx] / (r + 1e-12))
                projector_cache[key] = values
                return values

            for i in range(1, n_proj + 1):
                for j in range(1, n_proj + 1):
                    h_ij = h_mat[i - 1, j - 1]
                    if float(jnp.abs(h_ij)) < 1e-10:
                        continue

                    if l == 0:
                        p_i_fine_list.append(patch_projector_values(i))
                        p_j_fine_list.append(patch_projector_values(j))
                        flat_index_list.append(flat_indices)
                        weight_list.append(weights)
                        patch_sample_index_list.append(patch_sample_indices)
                        patch_eval_matrix_list.append(patch_eval_matrix)
                        coeff_list.append(h_ij / (4.0 * jnp.pi))
                    elif l == 1:
                        for axis in range(3):
                            p_i_fine_list.append(patch_projector_values(i, axis))
                            p_j_fine_list.append(patch_projector_values(j, axis))
                            flat_index_list.append(flat_indices)
                            weight_list.append(weights)
                            patch_sample_index_list.append(patch_sample_indices)
                            patch_eval_matrix_list.append(patch_eval_matrix)
                            coeff_list.append(3.0 * h_ij / (4.0 * jnp.pi))

    if not p_i_fine_list:
        return None

    flat_indices = jnp.stack(flat_index_list, axis=0)
    weights = jnp.stack(weight_list, axis=0)
    P_i = jnp.stack(p_i_fine_list, axis=0)
    P_j = jnp.stack(p_j_fine_list, axis=0)
    patch_sample_indices = jnp.stack(patch_sample_index_list, axis=0)
    patch_eval_matrix = jnp.stack(patch_eval_matrix_list, axis=0)
    coeffs = jnp.array(coeff_list)
    return (
        "fine_integral",
        flat_indices,
        weights,
        P_i,
        P_j,
        coeffs,
        fine_dv,
        coarse_dv,
        patch_sample_indices,
        patch_eval_matrix,
    )


def precompute_projectors(
    grid,
    atom_coords,
    pseudos,
    projector_subgrid=1,
    projector_mode="cell_average",
    projector_patch_radius_factor=6.0,
):
    """
    在 SCF 外预先计算所有的非局域势投影器在实空间网格上的值。
    将其打包成 4D 张量，供 JAX 进行极速张量批处理。
    """
    if projector_subgrid < 1:
        raise ValueError("projector_subgrid must be >= 1")
    if projector_mode not in ("cell_average", "patch"):
        raise ValueError("projector_mode must be 'cell_average' or 'patch'")
    if projector_mode == "patch":
        return _precompute_projectors_patch(
            grid,
            atom_coords,
            pseudos,
            projector_subgrid,
            projector_patch_radius_factor,
        )

    if projector_subgrid == 1:
        offsets = jnp.zeros((1, 3), dtype=jnp.float32)
    else:
        axis = (
            (jnp.arange(projector_subgrid, dtype=jnp.float32) + 0.5)
            / projector_subgrid
            - 0.5
        )
        axis = axis * jnp.asarray(grid.spacing, dtype=jnp.float32)
        ox, oy, oz = jnp.meshgrid(axis, axis, axis, indexing="ij")
        offsets = jnp.stack([ox, oy, oz], axis=-1).reshape(-1, 3)

    p_i_list = []
    p_j_list = []
    coeff_list = []
    
    for i_at in range(len(pseudos)):
        p_at = pseudos[i_at]
        if not p_at['projectors']: continue
        
        for ch in p_at['projectors']:
            l = ch['l']
            rp = ch['r']
            h_mat = jnp.array(ch['h']) 
            if h_mat.ndim == 1:
                h_mat = jnp.diag(h_mat)
            n_proj = h_mat.shape[0]

            projector_cache = {}

            def averaged_projector(proj_idx, axis_idx=None):
                key = (proj_idx, axis_idx)
                if key in projector_cache:
                    return projector_cache[key]

                acc = jnp.zeros(grid.coords.shape[:-1], dtype=jnp.float32)
                for offset in offsets:
                    diff = grid.coords + offset - atom_coords[i_at]
                    r = jnp.linalg.norm(diff, axis=-1)
                    p_rad = get_gth_projector(r, l, proj_idx, rp)
                    if axis_idx is None:
                        value = p_rad
                    else:
                        value = p_rad * (diff[..., axis_idx] / (r + 1e-12))
                    acc = acc + value

                averaged = acc / offsets.shape[0]
                projector_cache[key] = averaged
                return averaged
            
            for i in range(1, n_proj + 1):
                for j in range(1, n_proj + 1):
                    h_ij = h_mat[i-1, j-1]
                    if float(jnp.abs(h_ij)) < 1e-10: 
                        continue # 忽略为 0 的通道
                    
                    
                    if l == 0:
                        p_i_list.append(averaged_projector(i))
                        p_j_list.append(averaged_projector(j))
                        coeff_list.append(h_ij / (4.0 * jnp.pi))
                    elif l == 1:
                        for axis in range(3): # p型势包含三个空间方向
                            p_i_list.append(averaged_projector(i, axis))
                            p_j_list.append(averaged_projector(j, axis))
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


@jax.jit
def apply_nonlocal_fine_integral(
    psi,
    flat_indices,
    weights,
    P_i,
    P_j,
    coeffs,
    fine_dv,
    coarse_dv,
    patch_sample_indices=None,
    patch_eval_matrix=None,
):
    """Apply nonlocal GTH projectors using fine-grid overlap + adjoint scatter."""
    if patch_sample_indices is not None and patch_eval_matrix is not None:
        psi_fine = reconstruct_patch_wavefunction(psi, patch_sample_indices, patch_eval_matrix)
        overlap = jnp.sum(P_j * psi_fine, axis=1) * fine_dv
        values_fine = P_i * (coeffs * overlap)[:, None]
        return scatter_patch_wavefunction_adjoint(
            values_fine,
            psi.shape,
            patch_sample_indices,
            patch_eval_matrix,
            fine_dv,
            coarse_dv,
        )
    else:
        psi_fine = gather_fine_values(psi, flat_indices, weights)
        overlap = jnp.sum(P_j * psi_fine, axis=1) * fine_dv
        values_fine = P_i * (coeffs * overlap)[:, None]
        return scatter_fine_values_adjoint(values_fine, psi.shape, flat_indices, weights, fine_dv, coarse_dv)
