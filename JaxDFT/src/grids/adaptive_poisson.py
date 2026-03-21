"""Prototype Poisson solvers for adaptive tensor-product grids.

This module implements a first finite-volume / flux-form Poisson prototype for
node-centered tensor-product nonuniform grids with Dirichlet boundaries. The
interior operator remains unchanged; nonzero boundary data enters only through
the right-hand-side load vector. The emphasis is correctness and verifiability
rather than performance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import jax.scipy.sparse.linalg as jax_cg
from scipy.sparse import csr_matrix, diags, kron
from scipy.sparse.linalg import spsolve


Array = Any


def _as_numpy_1d(axis: Array, name: str) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least three nodes")
    diffs = np.diff(arr)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _as_numpy_weights(weights: Array, n_expected: int, name: str) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size != n_expected:
        raise ValueError(f"{name} length {arr.size} does not match expected size {n_expected}")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _coerce_boundary_faces_3d(grid, boundary_faces: dict[str, Array] | None) -> dict[str, jnp.ndarray]:
    nx, ny, nz = (int(n) for n in grid.shape)
    expected_shapes = {
        "x_lo": (ny - 2, nz - 2),
        "x_hi": (ny - 2, nz - 2),
        "y_lo": (nx - 2, nz - 2),
        "y_hi": (nx - 2, nz - 2),
        "z_lo": (nx - 2, ny - 2),
        "z_hi": (nx - 2, ny - 2),
    }
    if boundary_faces is None:
        return {
            key: jnp.zeros(shape, dtype=jnp.float32)
            for key, shape in expected_shapes.items()
        }
    unknown = set(boundary_faces) - set(expected_shapes)
    if unknown:
        raise ValueError(f"unknown boundary face keys: {sorted(unknown)}")

    faces = {}
    for key, shape in expected_shapes.items():
        if key in boundary_faces:
            arr = jnp.asarray(boundary_faces[key], dtype=jnp.float32)
            if arr.shape != shape:
                raise ValueError(f"{key} shape {arr.shape} does not match expected shape {shape}")
            if not bool(jnp.all(jnp.isfinite(arr))):
                raise ValueError(f"{key} must contain only finite values")
            faces[key] = arr
        else:
            faces[key] = jnp.zeros(shape, dtype=jnp.float32)
    return faces


def build_poisson_1d_stiffness(axis: Array) -> csr_matrix:
    """Build the 1D finite-volume stiffness matrix for -d^2/dx^2 with zero Dirichlet BC."""
    axis = _as_numpy_1d(axis, "axis")
    hm = axis[1:-1] - axis[:-2]
    hp = axis[2:] - axis[1:-1]
    main = 1.0 / hm + 1.0 / hp
    lower = -1.0 / hm[1:]
    upper = -1.0 / hp[:-1]
    n_int = axis.size - 2
    return diags([lower, main, upper], offsets=[-1, 0, 1], shape=(n_int, n_int), format="csr")


def build_poisson_1d_mass(weights: Array) -> csr_matrix:
    """Build the 1D diagonal mass matrix for interior nodal control volumes."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size < 3:
        raise ValueError("weights must contain at least three nodes")
    if np.any(weights <= 0.0):
        raise ValueError("weights must be strictly positive")
    return diags(weights[1:-1], offsets=0, format="csr")


def solve_poisson_dirichlet_1d(axis: Array, weights: Array, rhs: Array) -> tuple[jnp.ndarray, dict[str, float | int | str]]:
    """Solve -u'' = rhs on a nonuniform 1D grid with zero Dirichlet boundaries."""
    axis_np = _as_numpy_1d(axis, "axis")
    weights_np = _as_numpy_weights(weights, axis_np.size, "weights")
    rhs_np = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if rhs_np.size != axis_np.size:
        raise ValueError(f"rhs length {rhs_np.size} does not match axis size {axis_np.size}")

    K = build_poisson_1d_stiffness(axis_np)
    M = build_poisson_1d_mass(weights_np)
    b = M @ rhs_np[1:-1]
    u_int = spsolve(K, b)
    residual = K @ u_int - b
    rhs_norm = np.linalg.norm(b)
    rel_res = np.linalg.norm(residual) / max(rhs_norm, 1e-30)

    u = np.zeros_like(rhs_np)
    u[1:-1] = u_int
    diagnostics = {
        "method": "spsolve",
        "n_unknowns": int(u_int.size),
        "nnz": int(K.nnz),
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_residual": float(rel_res),
    }
    return jnp.asarray(u, dtype=jnp.float32), diagnostics


def flatten_interior_3d(field: Array) -> jnp.ndarray:
    """Flatten the interior of a 3D tensor field in C order."""
    arr = jnp.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {arr.shape}")
    if min(arr.shape) < 3:
        raise ValueError(f"field must have at least 3 nodes along each axis, got {arr.shape}")
    return jnp.reshape(arr[1:-1, 1:-1, 1:-1], (-1,))


def unflatten_interior_3d(grid, vec: Array, boundary_faces: dict[str, Array] | None = None) -> jnp.ndarray:
    """Restore a flattened interior vector into a full 3D field with Dirichlet faces.

    The face data is stored only on interior-aligned face nodes:
      - x faces: shape ``(ny-2, nz-2)``
      - y faces: shape ``(nx-2, nz-2)``
      - z faces: shape ``(nx-2, ny-2)``
    Edge and corner values are left at zero because they are not needed by the
    current interior finite-volume stencil.
    """
    interior_shape = tuple(int(n) - 2 for n in grid.shape)
    full_shape = tuple(int(n) for n in grid.shape)
    vec_arr = jnp.asarray(vec)
    arr = jnp.zeros(full_shape, dtype=vec_arr.dtype)
    arr = arr.at[1:-1, 1:-1, 1:-1].set(jnp.reshape(vec_arr, interior_shape))

    faces = _coerce_boundary_faces_3d(grid, boundary_faces)
    arr = arr.at[0, 1:-1, 1:-1].set(faces["x_lo"].astype(arr.dtype))
    arr = arr.at[-1, 1:-1, 1:-1].set(faces["x_hi"].astype(arr.dtype))
    arr = arr.at[1:-1, 0, 1:-1].set(faces["y_lo"].astype(arr.dtype))
    arr = arr.at[1:-1, -1, 1:-1].set(faces["y_hi"].astype(arr.dtype))
    arr = arr.at[1:-1, 1:-1, 0].set(faces["z_lo"].astype(arr.dtype))
    arr = arr.at[1:-1, 1:-1, -1].set(faces["z_hi"].astype(arr.dtype))
    return arr


def assemble_poisson_operator_3d(grid) -> tuple[csr_matrix, csr_matrix]:
    """Assemble the 3D finite-volume Poisson operator and diagonal mass matrix.

    The assembled system is
        A u = M f
    for -Laplace(u) = f with Dirichlet boundary values.
    """
    x = _as_numpy_1d(grid.x, "grid.x")
    y = _as_numpy_1d(grid.y, "grid.y")
    z = _as_numpy_1d(grid.z, "grid.z")
    wx = _as_numpy_weights(grid.wx, x.size, "grid.wx")
    wy = _as_numpy_weights(grid.wy, y.size, "grid.wy")
    wz = _as_numpy_weights(grid.wz, z.size, "grid.wz")

    Kx = build_poisson_1d_stiffness(x)
    Ky = build_poisson_1d_stiffness(y)
    Kz = build_poisson_1d_stiffness(z)
    Mx = build_poisson_1d_mass(wx)
    My = build_poisson_1d_mass(wy)
    Mz = build_poisson_1d_mass(wz)

    A = (
        kron(kron(Kx, My, format="csr"), Mz, format="csr")
        + kron(kron(Mx, Ky, format="csr"), Mz, format="csr")
        + kron(kron(Mx, My, format="csr"), Kz, format="csr")
    ).tocsr()
    M = kron(kron(Mx, My, format="csr"), Mz, format="csr").tocsr()
    return A, M


def build_dirichlet_boundary_load_3d(grid, boundary_faces: dict[str, Array]) -> jnp.ndarray:
    """Assemble the flattened RHS load induced by nonzero Dirichlet face values.

    The face data must use interior-aligned face arrays:
      - ``x_lo`` / ``x_hi``: ``(ny-2, nz-2)``
      - ``y_lo`` / ``y_hi``: ``(nx-2, nz-2)``
      - ``z_lo`` / ``z_hi``: ``(nx-2, ny-2)``

    Missing faces may be omitted and will be treated as zero.
    """
    x = jnp.asarray(grid.x)
    y = jnp.asarray(grid.y)
    z = jnp.asarray(grid.z)
    wx = jnp.asarray(grid.wx)
    wy = jnp.asarray(grid.wy)
    wz = jnp.asarray(grid.wz)
    faces = _coerce_boundary_faces_3d(grid, boundary_faces)

    wx_int = wx[1:-1]
    wy_int = wy[1:-1]
    wz_int = wz[1:-1]
    load = jnp.zeros((x.size - 2, y.size - 2, z.size - 2), dtype=wx.dtype)

    hx_lo = x[1] - x[0]
    hx_hi = x[-1] - x[-2]
    hy_lo = y[1] - y[0]
    hy_hi = y[-1] - y[-2]
    hz_lo = z[1] - z[0]
    hz_hi = z[-1] - z[-2]

    load = load.at[0, :, :].add((wy_int[:, None] * wz_int[None, :]) * (faces["x_lo"].astype(load.dtype) / hx_lo))
    load = load.at[-1, :, :].add((wy_int[:, None] * wz_int[None, :]) * (faces["x_hi"].astype(load.dtype) / hx_hi))
    load = load.at[:, 0, :].add((wx_int[:, None] * wz_int[None, :]) * (faces["y_lo"].astype(load.dtype) / hy_lo))
    load = load.at[:, -1, :].add((wx_int[:, None] * wz_int[None, :]) * (faces["y_hi"].astype(load.dtype) / hy_hi))
    load = load.at[:, :, 0].add((wx_int[:, None] * wy_int[None, :]) * (faces["z_lo"].astype(load.dtype) / hz_lo))
    load = load.at[:, :, -1].add((wx_int[:, None] * wy_int[None, :]) * (faces["z_hi"].astype(load.dtype) / hz_hi))

    return jnp.reshape(load, (-1,))


def compute_total_charge(grid, rho: Array) -> jnp.ndarray:
    """Compute total charge Q = integral rho dV using adaptive volume weights."""
    rho_arr = jnp.asarray(rho)
    volume_weights = jnp.asarray(grid.volume_weights)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rho_arr.shape != expected_shape:
        raise ValueError(f"rho shape {rho_arr.shape} does not match grid shape {expected_shape}")
    if volume_weights.shape != expected_shape:
        raise ValueError(
            f"grid.volume_weights shape {volume_weights.shape} does not match grid shape {expected_shape}"
        )
    return jnp.sum(rho_arr * volume_weights)


def _compute_box_center(grid) -> jnp.ndarray:
    return jnp.asarray(
        [
            0.5 * (grid.x[0] + grid.x[-1]),
            0.5 * (grid.y[0] + grid.y[-1]),
            0.5 * (grid.z[0] + grid.z[-1]),
        ],
        dtype=jnp.float32,
    )


def compute_charge_center(
    grid,
    rho: Array,
    *,
    total_charge: Array | None = None,
    charge_tol: float = 1.0e-8,
) -> jnp.ndarray:
    """Compute the weighted density center r_c = (integral r rho dV) / Q.

    If |Q| is too small, this safely falls back to the box center.
    """
    if charge_tol <= 0.0:
        raise ValueError("charge_tol must be positive")

    rho_arr = jnp.asarray(rho)
    coords = jnp.asarray(grid.coords)
    volume_weights = jnp.asarray(grid.volume_weights)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rho_arr.shape != expected_shape:
        raise ValueError(f"rho shape {rho_arr.shape} does not match grid shape {expected_shape}")
    if coords.shape != expected_shape + (3,):
        raise ValueError(f"grid.coords shape {coords.shape} does not match expected shape {expected_shape + (3,)}")

    Q = compute_total_charge(grid, rho_arr) if total_charge is None else jnp.asarray(total_charge, dtype=volume_weights.dtype)
    charge_tol_arr = jnp.asarray(charge_tol, dtype=volume_weights.dtype)
    box_center = _compute_box_center(grid).astype(coords.dtype)
    weighted_position = jnp.sum(coords * rho_arr[..., None] * volume_weights[..., None], axis=(0, 1, 2))
    safe_Q = jnp.where(jnp.abs(Q) <= charge_tol_arr, jnp.ones_like(Q), Q)
    raw_center = weighted_position / safe_Q
    use_fallback = jnp.abs(Q) <= charge_tol_arr
    return jnp.where(use_fallback, box_center, raw_center)


def compute_dipole_moment(grid, rho: Array, r0: Array) -> jnp.ndarray:
    """Compute the dipole moment p_i = integral rho (x_i - r0_i) dV."""
    rho_arr = jnp.asarray(rho)
    coords = jnp.asarray(grid.coords)
    volume_weights = jnp.asarray(grid.volume_weights)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rho_arr.shape != expected_shape:
        raise ValueError(f"rho shape {rho_arr.shape} does not match grid shape {expected_shape}")
    if coords.shape != expected_shape + (3,):
        raise ValueError(f"grid.coords shape {coords.shape} does not match expected shape {expected_shape + (3,)}")
    r0_arr = jnp.asarray(r0, dtype=coords.dtype).reshape(3)
    rel = coords - r0_arr
    return jnp.sum(rho_arr[..., None] * rel * volume_weights[..., None], axis=(0, 1, 2))


def compute_quadrupole_tensor(grid, rho: Array, r0: Array) -> jnp.ndarray:
    """Compute the traceless quadrupole tensor.

    Q_ij = integral rho * [3 (x_i-r0_i)(x_j-r0_j) - |r-r0|^2 delta_ij] dV
    """
    rho_arr = jnp.asarray(rho)
    coords = jnp.asarray(grid.coords)
    volume_weights = jnp.asarray(grid.volume_weights)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rho_arr.shape != expected_shape:
        raise ValueError(f"rho shape {rho_arr.shape} does not match grid shape {expected_shape}")
    if coords.shape != expected_shape + (3,):
        raise ValueError(f"grid.coords shape {coords.shape} does not match expected shape {expected_shape + (3,)}")
    r0_arr = jnp.asarray(r0, dtype=coords.dtype).reshape(3)
    rel = coords - r0_arr
    r2 = jnp.sum(rel * rel, axis=-1)
    rel_outer = rel[..., :, None] * rel[..., None, :]
    identity = jnp.eye(3, dtype=coords.dtype)
    tensor_field = 3.0 * rel_outer - r2[..., None, None] * identity
    return jnp.sum(
        rho_arr[..., None, None] * tensor_field * volume_weights[..., None, None],
        axis=(0, 1, 2),
    )


def get_boundary_reference_center(
    grid,
    rho: Array | None = None,
    center_mode: str = "box_center",
    *,
    charge_tol: float = 1.0e-8,
    return_effective_mode: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, str]:
    """Return the reference center used for asymptotic boundary data.

    Supported modes:
      - ``box_center``
      - ``charge_center`` using weighted density

    If ``charge_center`` is requested but |Q| is too small, the center safely
    falls back to the box center.
    """
    if charge_tol <= 0.0:
        raise ValueError("charge_tol must be positive")

    box_center = _compute_box_center(grid)
    if center_mode == "box_center":
        center = box_center
        effective_mode = "box_center"
    elif center_mode == "charge_center":
        if rho is None:
            raise ValueError("rho is required when center_mode='charge_center'")
        total_charge = compute_total_charge(grid, rho)
        center = compute_charge_center(grid, rho, total_charge=total_charge, charge_tol=charge_tol)
        effective_mode = "box_center" if float(jnp.abs(total_charge)) <= charge_tol else "charge_center"
    else:
        raise ValueError(
            f"unsupported center_mode {center_mode!r}; expected one of ['box_center', 'charge_center']"
        )

    if return_effective_mode:
        return center, effective_mode
    return center


def build_multipole_dirichlet_faces(
    grid,
    rho: Array,
    *,
    r0: Array | None = None,
    center_mode: str = "box_center",
    min_radius: float = 1.0e-8,
) -> tuple[dict[str, jnp.ndarray], dict[str, Array | str]]:
    """Build multipole asymptotic Dirichlet face data for the adaptive Hartree solve.

    The boundary potential uses the first three multipole terms around ``r0``:
        V(R) = Q / R + (p . R) / R^3 + 0.5 * sum_ij Q_ij R_i R_j / R^5
    where Q is total charge, p is the dipole moment, and Q_ij is the traceless
    quadrupole tensor.
    """
    if min_radius <= 0.0:
        raise ValueError("min_radius must be positive")

    if r0 is None:
        r0_arr = get_boundary_reference_center(grid, rho=rho, center_mode=center_mode)
    else:
        r0_arr = jnp.asarray(r0, dtype=jnp.float32).reshape(-1)
        if r0_arr.size != 3:
            raise ValueError("r0 must contain exactly three coordinates")
        if not bool(jnp.all(jnp.isfinite(r0_arr))):
            raise ValueError("r0 must contain only finite values")

    Q = compute_total_charge(grid, rho)
    dipole = compute_dipole_moment(grid, rho, r0_arr)
    quadrupole = compute_quadrupole_tensor(grid, rho, r0_arr)

    x = jnp.asarray(grid.x)
    y = jnp.asarray(grid.y)
    z = jnp.asarray(grid.z)
    x_int = x[1:-1]
    y_int = y[1:-1]
    z_int = z[1:-1]
    min_radius_arr = jnp.asarray(min_radius, dtype=x.dtype)

    def multipole_values(x_face, y_face, z_face):
        R = jnp.stack(
            [
                x_face - r0_arr[0],
                y_face - r0_arr[1],
                z_face - r0_arr[2],
            ],
            axis=-1,
        )
        radius = jnp.sqrt(jnp.sum(R * R, axis=-1))
        radius_safe = jnp.maximum(radius, min_radius_arr)
        monopole = Q / radius_safe
        dipole_term = jnp.sum(dipole * R, axis=-1) / (radius_safe ** 3)
        quadrupole_projection = jnp.einsum('...i,ij,...j->...', R, quadrupole, R)
        quadrupole_term = 0.5 * quadrupole_projection / (radius_safe ** 5)
        return monopole + dipole_term + quadrupole_term

    y_grid_x, z_grid_x = jnp.meshgrid(y_int, z_int, indexing="ij")
    x_grid_y, z_grid_y = jnp.meshgrid(x_int, z_int, indexing="ij")
    x_grid_z, y_grid_z = jnp.meshgrid(x_int, y_int, indexing="ij")

    faces = {
        "x_lo": multipole_values(jnp.full_like(y_grid_x, x[0]), y_grid_x, z_grid_x),
        "x_hi": multipole_values(jnp.full_like(y_grid_x, x[-1]), y_grid_x, z_grid_x),
        "y_lo": multipole_values(x_grid_y, jnp.full_like(x_grid_y, y[0]), z_grid_y),
        "y_hi": multipole_values(x_grid_y, jnp.full_like(x_grid_y, y[-1]), z_grid_y),
        "z_lo": multipole_values(x_grid_z, y_grid_z, jnp.full_like(x_grid_z, z[0])),
        "z_hi": multipole_values(x_grid_z, y_grid_z, jnp.full_like(x_grid_z, z[-1])),
    }
    diagnostics = {
        "boundary_model": "multipole_dirichlet",
        "center_mode": center_mode,
        "total_charge": Q,
        "dipole_moment": dipole,
        "quadrupole_tensor": quadrupole,
        "reference_center": r0_arr,
        "min_radius": min_radius_arr,
    }
    return faces, diagnostics


def build_monopole_dirichlet_faces(
    grid,
    rho: Array,
    *,
    r0: Array | None = None,
    center_mode: str = "box_center",
    min_radius: float = 1.0e-8,
    charge_tol: float = 1.0e-8,
) -> tuple[dict[str, jnp.ndarray], dict[str, Array | str]]:
    """Backward-compatible monopole face builder retained for regression.

    Supported center choices:
      - ``box_center``
      - ``charge_center`` using weighted density
    """
    if min_radius <= 0.0:
        raise ValueError("min_radius must be positive")
    if charge_tol <= 0.0:
        raise ValueError("charge_tol must be positive")

    Q = compute_total_charge(grid, rho)
    if r0 is None:
        r0_arr, effective_center_mode = get_boundary_reference_center(
            grid,
            rho=rho,
            center_mode=center_mode,
            charge_tol=charge_tol,
            return_effective_mode=True,
        )
    else:
        r0_arr = jnp.asarray(r0, dtype=jnp.float32).reshape(-1)
        if r0_arr.size != 3:
            raise ValueError("r0 must contain exactly three coordinates")
        if not bool(jnp.all(jnp.isfinite(r0_arr))):
            raise ValueError("r0 must contain only finite values")
        effective_center_mode = "explicit_r0"

    x = jnp.asarray(grid.x)
    y = jnp.asarray(grid.y)
    z = jnp.asarray(grid.z)
    x_int = x[1:-1]
    y_int = y[1:-1]
    z_int = z[1:-1]
    min_radius_arr = jnp.asarray(min_radius, dtype=x.dtype)
    charge_tol_arr = jnp.asarray(charge_tol, dtype=x.dtype)

    def monopole_values(x_face, y_face, z_face):
        radius = jnp.sqrt((x_face - r0_arr[0]) ** 2 + (y_face - r0_arr[1]) ** 2 + (z_face - r0_arr[2]) ** 2)
        return Q / jnp.maximum(radius, min_radius_arr)

    y_grid_x, z_grid_x = jnp.meshgrid(y_int, z_int, indexing="ij")
    x_grid_y, z_grid_y = jnp.meshgrid(x_int, z_int, indexing="ij")
    x_grid_z, y_grid_z = jnp.meshgrid(x_int, y_int, indexing="ij")

    faces = {
        "x_lo": monopole_values(jnp.full_like(y_grid_x, x[0]), y_grid_x, z_grid_x),
        "x_hi": monopole_values(jnp.full_like(y_grid_x, x[-1]), y_grid_x, z_grid_x),
        "y_lo": monopole_values(x_grid_y, jnp.full_like(x_grid_y, y[0]), z_grid_y),
        "y_hi": monopole_values(x_grid_y, jnp.full_like(x_grid_y, y[-1]), z_grid_y),
        "z_lo": monopole_values(x_grid_z, y_grid_z, jnp.full_like(x_grid_z, z[0])),
        "z_hi": monopole_values(x_grid_z, y_grid_z, jnp.full_like(x_grid_z, z[-1])),
    }
    diagnostics = {
        "boundary_model": "monopole_dirichlet",
        "center_mode": center_mode,
        "effective_center_mode": effective_center_mode,
        "total_charge": Q,
        "reference_center": r0_arr,
        "min_radius": min_radius_arr,
        "charge_tolerance": charge_tol_arr,
    }
    return faces, diagnostics


def _precompute_uniform_exterior_poisson_kernel(grid_shape: tuple[int, int, int], spacing: float) -> jnp.ndarray:
    """Precompute the zero-padded FFT Poisson kernel used for the auxiliary uniform exterior grid."""
    nx, ny, nz = (int(n) for n in grid_shape)
    x = jnp.fft.fftfreq(2 * nx, d=1.0 / (2 * nx)) * spacing
    y = jnp.fft.fftfreq(2 * ny, d=1.0 / (2 * ny)) * spacing
    z = jnp.fft.fftfreq(2 * nz, d=1.0 / (2 * nz)) * spacing
    kx, ky, kz = jnp.meshgrid(x, y, z, indexing="ij")
    r = jnp.sqrt(kx**2 + ky**2 + kz**2)
    kernel = jnp.where(r > 1.0e-8, 1.0 / r, 2.38 / spacing)
    return jnp.fft.fftn(kernel)


def _solve_uniform_exterior_poisson(rho: Array, spacing: float) -> jnp.ndarray:
    """Solve the auxiliary uniform-grid free-space-like Hartree problem via zero-padded FFT."""
    rho_arr = jnp.asarray(rho)
    nx, ny, nz = (int(n) for n in rho_arr.shape)
    kernel_k = _precompute_uniform_exterior_poisson_kernel((nx, ny, nz), float(spacing))
    rho_pad = jnp.pad(rho_arr, ((0, nx), (0, ny), (0, nz)), mode="constant")
    rho_k = jnp.fft.fftn(rho_pad)
    v_pad = jnp.fft.ifftn(rho_k * kernel_k).real * (spacing**3)
    return v_pad[:nx, :ny, :nz]


def _build_uniform_axis_with_padding(vmin: float, vmax: float, spacing: float) -> np.ndarray:
    """Build a uniform axis that covers [vmin, vmax] with exact spacing and symmetric overhang."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if vmax <= vmin:
        raise ValueError("axis upper bound must exceed lower bound")
    length = float(vmax - vmin)
    n_intervals = max(int(np.ceil(length / spacing)), 1)
    total_length = n_intervals * spacing
    overhang = 0.5 * (total_length - length)
    start = float(vmin) - overhang
    return start + spacing * np.arange(n_intervals + 1, dtype=np.float64)


def _build_uniform_exterior_axes(grid, *, padding: float = 8.0, dx_ext: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct the padded auxiliary uniform grid used to provide adaptive Dirichlet face values."""
    if padding <= 0.0:
        raise ValueError("padding must be positive")
    if dx_ext <= 0.0:
        raise ValueError("dx_ext must be positive")

    x = _build_uniform_axis_with_padding(float(grid.x[0]) - padding, float(grid.x[-1]) + padding, dx_ext)
    y = _build_uniform_axis_with_padding(float(grid.y[0]) - padding, float(grid.y[-1]) + padding, dx_ext)
    z = _build_uniform_axis_with_padding(float(grid.z[0]) - padding, float(grid.z[-1]) + padding, dx_ext)
    diagnostics = {
        "padding": float(padding),
        "dx_ext": float(dx_ext),
        "shape_ext": (int(x.size), int(y.size), int(z.size)),
        "x_bounds": (float(x[0]), float(x[-1])),
        "y_bounds": (float(y[0]), float(y[-1])),
        "z_bounds": (float(z[0]), float(z[-1])),
    }
    return x, y, z, diagnostics


def _uniform_axis_cell_indices(points: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lower-cell indices and fractional coordinates for a uniform 1D axis."""
    origin = float(axis[0])
    spacing = float(axis[1] - axis[0])
    n_nodes = int(axis.size)
    raw = np.floor((points - origin) / spacing).astype(np.int64)
    idx0 = np.clip(raw, 0, n_nodes - 2)
    base = origin + spacing * idx0
    frac = np.clip((points - base) / spacing, 0.0, 1.0)
    return idx0, frac


def _deposit_trilinear_to_uniform(points: np.ndarray, charges: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, dict[str, float | tuple[int, int, int]]]:
    """Deposit point charges onto a node-centered uniform grid using trilinear weights."""
    rho = np.zeros((x.size, y.size, z.size), dtype=np.float64)
    ix, tx = _uniform_axis_cell_indices(points[:, 0], x)
    iy, ty = _uniform_axis_cell_indices(points[:, 1], y)
    iz, tz = _uniform_axis_cell_indices(points[:, 2], z)

    wx = (1.0 - tx, tx)
    wy = (1.0 - ty, ty)
    wz = (1.0 - tz, tz)
    dv = float((x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0]))
    density_scale = charges / dv

    for ox, wx_part in enumerate(wx):
        for oy, wy_part in enumerate(wy):
            for oz, wz_part in enumerate(wz):
                np.add.at(
                    rho,
                    (ix + ox, iy + oy, iz + oz),
                    density_scale * wx_part * wy_part * wz_part,
                )

    charge_in = float(np.sum(charges))
    charge_out = float(np.sum(rho) * dv)
    rel_err = abs(charge_out - charge_in) / max(abs(charge_in), 1.0e-12)
    diagnostics = {
        "charge_in": charge_in,
        "charge_deposited": charge_out,
        "relative_charge_error": rel_err,
        "shape_ext": (int(x.size), int(y.size), int(z.size)),
    }
    return rho, diagnostics


def _interpolate_trilinear_from_uniform(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate a node-centered uniform scalar field at arbitrary points."""
    ix, tx = _uniform_axis_cell_indices(points[:, 0], x)
    iy, ty = _uniform_axis_cell_indices(points[:, 1], y)
    iz, tz = _uniform_axis_cell_indices(points[:, 2], z)

    wx = (1.0 - tx, tx)
    wy = (1.0 - ty, ty)
    wz = (1.0 - tz, tz)
    values = np.zeros(points.shape[0], dtype=np.float64)

    for ox, wx_part in enumerate(wx):
        for oy, wy_part in enumerate(wy):
            for oz, wz_part in enumerate(wz):
                values += field[ix + ox, iy + oy, iz + oz] * wx_part * wy_part * wz_part
    return values


def build_uniform_exterior_dirichlet_faces(
    grid,
    rho: Array,
    *,
    padding: float = 8.0,
    dx_ext: float = 0.8,
) -> tuple[dict[str, jnp.ndarray], dict[str, Array | float | tuple[int, int, int] | str]]:
    """Build Dirichlet face values from a padded coarse uniform exterior free-space solve.

    This keeps the adaptive interior operator unchanged. The adaptive density is
    first deposited onto a larger, coarser auxiliary uniform grid via trilinear
    deposition, a free-space-like Hartree potential is computed on that uniform
    grid, and the resulting potential is trilinearly interpolated back to the
    six adaptive box faces.
    """
    rho_arr = jnp.asarray(rho)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rho_arr.shape != expected_shape:
        raise ValueError(f"rho shape {rho_arr.shape} does not match grid shape {expected_shape}")

    x_ext, y_ext, z_ext, ext_diag = _build_uniform_exterior_axes(grid, padding=padding, dx_ext=dx_ext)
    coords = np.asarray(grid.coords, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(jnp.asarray(grid.volume_weights) * rho_arr, dtype=np.float64).reshape(-1)
    rho_ext, deposit_diag = _deposit_trilinear_to_uniform(coords, charges, x_ext, y_ext, z_ext)
    v_ext = np.asarray(_solve_uniform_exterior_poisson(jnp.asarray(rho_ext, dtype=jnp.float32), float(dx_ext)), dtype=np.float64)

    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    z = np.asarray(grid.z, dtype=np.float64)
    x_int = x[1:-1]
    y_int = y[1:-1]
    z_int = z[1:-1]

    y_grid_x, z_grid_x = np.meshgrid(y_int, z_int, indexing="ij")
    x_grid_y, z_grid_y = np.meshgrid(x_int, z_int, indexing="ij")
    x_grid_z, y_grid_z = np.meshgrid(x_int, y_int, indexing="ij")

    def face_values(x_face, y_face, z_face):
        pts = np.stack([x_face.reshape(-1), y_face.reshape(-1), z_face.reshape(-1)], axis=-1)
        vals = _interpolate_trilinear_from_uniform(v_ext, x_ext, y_ext, z_ext, pts)
        return vals.reshape(x_face.shape)

    faces_np = {
        "x_lo": face_values(np.full_like(y_grid_x, x[0]), y_grid_x, z_grid_x),
        "x_hi": face_values(np.full_like(y_grid_x, x[-1]), y_grid_x, z_grid_x),
        "y_lo": face_values(x_grid_y, np.full_like(x_grid_y, y[0]), z_grid_y),
        "y_hi": face_values(x_grid_y, np.full_like(x_grid_y, y[-1]), z_grid_y),
        "z_lo": face_values(x_grid_z, y_grid_z, np.full_like(x_grid_z, z[0])),
        "z_hi": face_values(x_grid_z, y_grid_z, np.full_like(x_grid_z, z[-1])),
    }
    faces = {key: jnp.asarray(val, dtype=jnp.float32) for key, val in faces_np.items()}

    diagnostics = {
        "boundary_model": "uniform_exterior_dirichlet",
        "padding": float(padding),
        "dx_ext": float(dx_ext),
        "shape_ext": ext_diag["shape_ext"],
        "x_bounds_ext": ext_diag["x_bounds"],
        "y_bounds_ext": ext_diag["y_bounds"],
        "z_bounds_ext": ext_diag["z_bounds"],
        "total_charge": compute_total_charge(grid, rho_arr),
        "charge_in": deposit_diag["charge_in"],
        "charge_deposited": deposit_diag["charge_deposited"],
        "relative_charge_error": deposit_diag["relative_charge_error"],
        "ext_potential_min": jnp.asarray(float(np.min(v_ext)), dtype=jnp.float32),
        "ext_potential_max": jnp.asarray(float(np.max(v_ext)), dtype=jnp.float32),
    }
    return faces, diagnostics


def solve_hartree_uniform_exterior_dirichlet_3d(
    grid,
    rho: Array,
    *,
    padding: float = 8.0,
    dx_ext: float = 0.8,
) -> tuple[jnp.ndarray, dict[str, Array | int | str]]:
    """Solve the prototype Hartree problem using an auxiliary uniform exterior boundary provider."""
    rhs = (4.0 * jnp.pi) * jnp.asarray(rho)
    faces, face_diagnostics = build_uniform_exterior_dirichlet_faces(
        grid,
        rho,
        padding=padding,
        dx_ext=dx_ext,
    )
    V, diagnostics = solve_poisson_dirichlet_3d(grid, rhs, boundary_faces=faces)
    diagnostics = dict(diagnostics)
    diagnostics["equation"] = "-Laplace(V_H) = 4*pi*rho"
    diagnostics.update(face_diagnostics)
    return V, diagnostics


def solve_poisson_dirichlet_3d(
    grid,
    rhs: Array,
    boundary_faces: dict[str, Array] | None = None,
) -> tuple[jnp.ndarray, dict[str, Array | int | str]]:
    """Solve -Laplace(u) = rhs on an adaptive tensor grid with Dirichlet boundaries.

    When ``boundary_faces`` is ``None``, this strictly reduces to the existing
    zero-Dirichlet behavior. Nonzero boundary data is injected only through an
    additive RHS load; the interior operator is unchanged.
    """
    rhs_arr = jnp.asarray(rhs)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rhs_arr.shape != expected_shape:
        raise ValueError(f"rhs shape {rhs_arr.shape} does not match grid shape {expected_shape}")

    A_bcoo = getattr(grid, "A_bcoo", None)
    M_bcoo = getattr(grid, "M_bcoo", None)
    if A_bcoo is None or M_bcoo is None:
        raise ValueError(
            "adaptive grid state is missing precomputed Poisson operators; "
            "construct it through AdaptiveBackend.create_grid/create_adaptive_grid"
        )

    b = M_bcoo @ flatten_interior_3d(rhs_arr)
    boundary_load = None
    boundary_mode = "zero_dirichlet"
    if boundary_faces is not None:
        boundary_load = build_dirichlet_boundary_load_3d(grid, boundary_faces).astype(b.dtype)
        b = b + boundary_load
        boundary_mode = "inhomogeneous_dirichlet"

    def apply_A(x):
        return A_bcoo @ x

    u_int, info = jax_cg.cg(apply_A, b, maxiter=800, tol=1e-6)
    for _ in range(3):
        correction, _ = jax_cg.cg(apply_A, -(apply_A(u_int) - b), maxiter=800, tol=1e-6)
        u_int = u_int + correction
    residual = apply_A(u_int) - b
    rhs_norm = jnp.linalg.norm(b)
    residual_norm = jnp.linalg.norm(residual)
    rel_res = residual_norm / jnp.maximum(rhs_norm, jnp.asarray(1.0e-30, dtype=rhs_norm.dtype))

    diagnostics = {
        "method": "jax_cg",
        "n_unknowns": int(u_int.size),
        "nnz": int(getattr(grid, "A_nnz", A_bcoo.nse)),
        "residual_norm": residual_norm,
        "relative_residual": rel_res,
        "boundary_mode": boundary_mode,
        "boundary_load_norm": jnp.asarray(0.0, dtype=b.dtype) if boundary_load is None else jnp.linalg.norm(boundary_load),
        "cg_info": info,
    }
    return unflatten_interior_3d(grid, u_int, boundary_faces=boundary_faces), diagnostics


def solve_hartree_dirichlet_3d(grid, rho: Array) -> tuple[jnp.ndarray, dict[str, Array | int | str]]:
    """Solve the prototype Hartree problem -Laplace(V_H) = 4*pi*rho."""
    rhs = (4.0 * jnp.pi) * jnp.asarray(rho)
    V, diagnostics = solve_poisson_dirichlet_3d(grid, rhs)
    diagnostics = dict(diagnostics)
    diagnostics["equation"] = "-Laplace(V_H) = 4*pi*rho"
    return V, diagnostics


def solve_hartree_multipole_dirichlet_3d(
    grid,
    rho: Array,
    *,
    r0: Array | None = None,
    center_mode: str = "box_center",
    min_radius: float = 1.0e-8,
) -> tuple[jnp.ndarray, dict[str, Array | int | str]]:
    """Solve the prototype Hartree problem using multipole Dirichlet boundary data."""
    rhs = (4.0 * jnp.pi) * jnp.asarray(rho)
    faces, face_diagnostics = build_multipole_dirichlet_faces(
        grid,
        rho,
        r0=r0,
        center_mode=center_mode,
        min_radius=min_radius,
    )
    V, diagnostics = solve_poisson_dirichlet_3d(grid, rhs, boundary_faces=faces)
    diagnostics = dict(diagnostics)
    diagnostics["equation"] = "-Laplace(V_H) = 4*pi*rho"
    diagnostics.update(face_diagnostics)
    return V, diagnostics


def solve_hartree_monopole_dirichlet_3d(
    grid,
    rho: Array,
    *,
    r0: Array | None = None,
    center_mode: str = "box_center",
    min_radius: float = 1.0e-8,
) -> tuple[jnp.ndarray, dict[str, Array | int | str]]:
    """Backward-compatible monopole Hartree wrapper retained for regression."""
    rhs = (4.0 * jnp.pi) * jnp.asarray(rho)
    faces, face_diagnostics = build_monopole_dirichlet_faces(
        grid,
        rho,
        r0=r0,
        center_mode=center_mode,
        min_radius=min_radius,
    )
    V, diagnostics = solve_poisson_dirichlet_3d(grid, rhs, boundary_faces=faces)
    diagnostics = dict(diagnostics)
    diagnostics["equation"] = "-Laplace(V_H) = 4*pi*rho"
    diagnostics.update(face_diagnostics)
    return V, diagnostics


__all__ = [
    "build_poisson_1d_stiffness",
    "build_poisson_1d_mass",
    "flatten_interior_3d",
    "unflatten_interior_3d",
    "assemble_poisson_operator_3d",
    "build_dirichlet_boundary_load_3d",
    "compute_total_charge",
    "compute_charge_center",
    "compute_dipole_moment",
    "compute_quadrupole_tensor",
    "get_boundary_reference_center",
    "build_multipole_dirichlet_faces",
    "build_monopole_dirichlet_faces",
    "build_uniform_exterior_dirichlet_faces",
    "solve_poisson_dirichlet_1d",
    "solve_poisson_dirichlet_3d",
    "solve_hartree_dirichlet_3d",
    "solve_hartree_multipole_dirichlet_3d",
    "solve_hartree_monopole_dirichlet_3d",
    "solve_hartree_uniform_exterior_dirichlet_3d",
]
