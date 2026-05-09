"""Density continuation: interpolate electron density between uniform Cartesian grids.

Used for multigrid-style workflows: converge (or partially converge) on a coarse
grid, map ``rho`` onto a finer ``create_grid`` mesh, then warm-start SCF on the
fine grid via ``energy_and_forces(..., initial_rho=...)``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def grid_xyz_axes(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract monotone 1D coordinate axes from a ``create_grid`` grid object."""
    c = np.asarray(grid.coords)
    x = c[:, 0, 0, 0]
    y = c[0, :, 0, 1]
    z = c[0, 0, :, 2]
    return x, y, z


def interpolate_rho_trilinear(
    grid_src,
    rho_src,
    grid_dst,
    n_electrons: float,
):
    """Trilinear interpolation of ``rho_src`` onto ``grid_dst`` nodes, then renormalize charge.

    Args:
        grid_src: Source grid (same physical box as destination).
        rho_src: Density on ``grid_src``, Bohr^-3 (JAX or NumPy array).
        grid_dst: Destination grid.
        n_electrons: Target integrated electron count after renormalization.

    Returns:
        ``rho_dst`` as a JAX array with ``grid_dst.coords.dtype`` shape matching ``grid_dst``.
    """
    import jax.numpy as jnp

    rho_np = np.asarray(rho_src, dtype=np.float64)
    if rho_np.shape != tuple(grid_src.shape):
        raise ValueError(
            f"rho_src shape {rho_np.shape} does not match grid_src.shape {tuple(grid_src.shape)}"
        )
    x, y, z = grid_xyz_axes(grid_src)
    interp = RegularGridInterpolator(
        (x, y, z),
        rho_np,
        bounds_error=False,
        fill_value=0.0,
        method="linear",
    )
    pts = np.asarray(grid_dst.coords.reshape(-1, 3), dtype=np.float64)
    vals = interp(pts).reshape(grid_dst.shape)
    vals = np.maximum(vals, 0.0)
    d_v = float(grid_dst.volume_element)
    charge = float(np.sum(vals) * d_v)
    if charge > 1e-20:
        vals *= float(n_electrons) / charge
    return jnp.asarray(vals, dtype=grid_dst.coords.dtype)
