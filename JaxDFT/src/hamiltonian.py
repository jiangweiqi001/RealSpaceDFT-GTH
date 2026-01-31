"""Hamiltonian construction utilities for real-space DFT.

Provides grid creation, GTH local pseudopotential evaluation, and a 4th-order
finite-difference Laplacian. All quantities are in atomic units (Bohr, Hartree).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


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
    """Compute the 4th-order finite-difference Laplacian in 3D.

    The stencil uses periodic shifts via jnp.roll. An optional mask can zero out
    regions (e.g., outside a physical domain).

    Args:
        psi: Scalar field on the grid.
        spacing: Grid spacing in Bohr.
        mask: Optional multiplicative mask, same shape as psi.

    Returns:
        Laplacian of psi, in Bohr^-2 times psi units.
    """
    h2 = spacing * spacing
    c0 = -2.5 / h2
    c1 = (4.0/3.0) / h2
    c2 = (-1.0/12.0) / h2
    lap = 3.0 * c0 * psi
    lap += c1 * (jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0))
    lap += c2 * (jnp.roll(psi, 2, axis=0) + jnp.roll(psi, -2, axis=0))
    lap += c1 * (jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1))
    lap += c2 * (jnp.roll(psi, 2, axis=1) + jnp.roll(psi, -2, axis=1))
    lap += c1 * (jnp.roll(psi, 1, axis=2) + jnp.roll(psi, -1, axis=2))
    lap += c2 * (jnp.roll(psi, 2, axis=2) + jnp.roll(psi, -2, axis=2))
    if mask is not None: lap = lap * mask
    return lap


def apply_nonlocal(projectors, psi_flat, volume_element):
    """Apply nonlocal pseudopotential projectors (placeholder).

    Args:
        projectors: Projector data structure.
        psi_flat: Flattened wavefunction.
        volume_element: Grid cell volume in Bohr^3.

    Returns:
        Nonlocal potential contribution, currently zero.
    """
    return 0.0
