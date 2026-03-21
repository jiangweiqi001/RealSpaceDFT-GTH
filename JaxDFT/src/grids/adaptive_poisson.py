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


def _as_numpy_face(face: Array, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(face, dtype=np.float64)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape {arr.shape} does not match expected shape {expected_shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _coerce_boundary_faces_3d(grid, boundary_faces: dict[str, Array] | None) -> dict[str, np.ndarray]:
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
            key: np.zeros(shape, dtype=np.float64)
            for key, shape in expected_shapes.items()
        }
    unknown = set(boundary_faces) - set(expected_shapes)
    if unknown:
        raise ValueError(f"unknown boundary face keys: {sorted(unknown)}")
    faces_np = {}
    for key, shape in expected_shapes.items():
        if key in boundary_faces:
            faces_np[key] = _as_numpy_face(boundary_faces[key], shape, key)
        else:
            faces_np[key] = np.zeros(shape, dtype=np.float64)
    return faces_np


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


def flatten_interior_3d(field: Array) -> np.ndarray:
    """Flatten the interior of a 3D tensor field in C order."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {arr.shape}")
    if min(arr.shape) < 3:
        raise ValueError(f"field must have at least 3 nodes along each axis, got {arr.shape}")
    return np.ascontiguousarray(arr[1:-1, 1:-1, 1:-1]).reshape(-1, order="C")


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
    arr = np.zeros(tuple(int(n) for n in grid.shape), dtype=np.float64)
    arr[1:-1, 1:-1, 1:-1] = np.asarray(vec, dtype=np.float64).reshape(interior_shape, order="C")

    faces_np = _coerce_boundary_faces_3d(grid, boundary_faces)
    arr[0, 1:-1, 1:-1] = faces_np["x_lo"]
    arr[-1, 1:-1, 1:-1] = faces_np["x_hi"]
    arr[1:-1, 0, 1:-1] = faces_np["y_lo"]
    arr[1:-1, -1, 1:-1] = faces_np["y_hi"]
    arr[1:-1, 1:-1, 0] = faces_np["z_lo"]
    arr[1:-1, 1:-1, -1] = faces_np["z_hi"]
    return jnp.asarray(arr, dtype=jnp.float32)


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


def build_dirichlet_boundary_load_3d(grid, boundary_faces: dict[str, Array]) -> np.ndarray:
    """Assemble the flattened RHS load induced by nonzero Dirichlet face values.

    The face data must use interior-aligned face arrays:
      - ``x_lo`` / ``x_hi``: ``(ny-2, nz-2)``
      - ``y_lo`` / ``y_hi``: ``(nx-2, nz-2)``
      - ``z_lo`` / ``z_hi``: ``(nx-2, ny-2)``

    Missing faces may be omitted and will be treated as zero.
    """
    x = _as_numpy_1d(grid.x, "grid.x")
    y = _as_numpy_1d(grid.y, "grid.y")
    z = _as_numpy_1d(grid.z, "grid.z")
    wx = _as_numpy_weights(grid.wx, x.size, "grid.wx")
    wy = _as_numpy_weights(grid.wy, y.size, "grid.wy")
    wz = _as_numpy_weights(grid.wz, z.size, "grid.wz")
    faces_np = _coerce_boundary_faces_3d(grid, boundary_faces)

    wx_int = wx[1:-1]
    wy_int = wy[1:-1]
    wz_int = wz[1:-1]
    load = np.zeros((x.size - 2, y.size - 2, z.size - 2), dtype=np.float64)

    hx_lo = x[1] - x[0]
    hx_hi = x[-1] - x[-2]
    hy_lo = y[1] - y[0]
    hy_hi = y[-1] - y[-2]
    hz_lo = z[1] - z[0]
    hz_hi = z[-1] - z[-2]

    load[0, :, :] += (wy_int[:, None] * wz_int[None, :]) * (faces_np["x_lo"] / hx_lo)
    load[-1, :, :] += (wy_int[:, None] * wz_int[None, :]) * (faces_np["x_hi"] / hx_hi)
    load[:, 0, :] += (wx_int[:, None] * wz_int[None, :]) * (faces_np["y_lo"] / hy_lo)
    load[:, -1, :] += (wx_int[:, None] * wz_int[None, :]) * (faces_np["y_hi"] / hy_hi)
    load[:, :, 0] += (wx_int[:, None] * wy_int[None, :]) * (faces_np["z_lo"] / hz_lo)
    load[:, :, -1] += (wx_int[:, None] * wy_int[None, :]) * (faces_np["z_hi"] / hz_hi)

    return np.ascontiguousarray(load).reshape(-1, order="C")


def solve_poisson_dirichlet_3d(
    grid,
    rhs: Array,
    boundary_faces: dict[str, Array] | None = None,
) -> tuple[jnp.ndarray, dict[str, float | int | str]]:
    """Solve -Laplace(u) = rhs on an adaptive tensor grid with Dirichlet boundaries.

    When ``boundary_faces`` is ``None``, this strictly reduces to the existing
    zero-Dirichlet behavior. Nonzero boundary data is injected only through an
    additive RHS load; the interior operator is unchanged.
    """
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    expected_shape = tuple(int(n) for n in grid.shape)
    if rhs_arr.shape != expected_shape:
        raise ValueError(f"rhs shape {rhs_arr.shape} does not match grid shape {expected_shape}")

    A, M = assemble_poisson_operator_3d(grid)
    b = M @ flatten_interior_3d(rhs_arr)
    boundary_load = None
    boundary_mode = "zero_dirichlet"
    if boundary_faces is not None:
        boundary_load = build_dirichlet_boundary_load_3d(grid, boundary_faces)
        b = b + boundary_load
        boundary_mode = "inhomogeneous_dirichlet"
    u_int = spsolve(A, b)
    residual = A @ u_int - b
    rhs_norm = np.linalg.norm(b)
    rel_res = np.linalg.norm(residual) / max(rhs_norm, 1e-30)

    diagnostics = {
        "method": "spsolve",
        "n_unknowns": int(u_int.size),
        "nnz": int(A.nnz),
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_residual": float(rel_res),
        "boundary_mode": boundary_mode,
        "boundary_load_norm": float(0.0 if boundary_load is None else np.linalg.norm(boundary_load)),
    }
    return unflatten_interior_3d(grid, u_int, boundary_faces=boundary_faces), diagnostics


def solve_hartree_dirichlet_3d(grid, rho: Array) -> tuple[jnp.ndarray, dict[str, float | int | str]]:
    """Solve the prototype Hartree problem -Laplace(V_H) = 4*pi*rho."""
    rhs = 4.0 * np.pi * np.asarray(rho, dtype=np.float64)
    V, diagnostics = solve_poisson_dirichlet_3d(grid, rhs)
    diagnostics = dict(diagnostics)
    diagnostics["equation"] = "-Laplace(V_H) = 4*pi*rho"
    return V, diagnostics


__all__ = [
    "build_poisson_1d_stiffness",
    "build_poisson_1d_mass",
    "flatten_interior_3d",
    "unflatten_interior_3d",
    "assemble_poisson_operator_3d",
    "build_dirichlet_boundary_load_3d",
    "solve_poisson_dirichlet_1d",
    "solve_poisson_dirichlet_3d",
    "solve_hartree_dirichlet_3d",
]
