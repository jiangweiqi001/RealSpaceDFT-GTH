"""Uniform-grid backend wrapper for the existing real-space implementation.

Patch 2 intentionally keeps this backend as a thin adapter over the current
uniform-grid kernels. Patch 3 wires the solver to this backend while keeping
all external APIs unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import ArrayLike, BackendState, NonlocalCache
from ..hamiltonian import (
    apply_nonlocal_precomputed,
    build_local_potential as build_local_potential_uniform,
    create_grid as create_uniform_grid,
    laplacian_8th,
    precompute_projectors,
)


def precompute_uniform_poisson_kernel(grid_shape, spacing):
    """Precompute the current uniform-grid Poisson kernel."""
    nx, ny, nz = grid_shape
    x = jnp.fft.fftfreq(2 * nx, d=1.0 / (2 * nx)) * spacing
    y = jnp.fft.fftfreq(2 * ny, d=1.0 / (2 * ny)) * spacing
    z = jnp.fft.fftfreq(2 * nz, d=1.0 / (2 * nz)) * spacing
    kx, ky, kz = jnp.meshgrid(x, y, z, indexing='ij')
    r = jnp.sqrt(kx**2 + ky**2 + kz**2)

    kernel = jnp.where(r > 1e-8, 1.0 / r, 2.38 / spacing)
    return jnp.fft.fftn(kernel)


@jax.jit
def solve_uniform_poisson(rho, kernel_k, spacing):
    """Solve the current uniform-grid Poisson problem via zero-padded FFT."""
    nx, ny, nz = rho.shape
    rho_pad = jnp.pad(rho, ((0, nx), (0, ny), (0, nz)), mode='constant')

    rho_k = jnp.fft.fftn(rho_pad)
    v_pad = jnp.fft.ifftn(rho_k * kernel_k).real * (spacing**3)
    return v_pad[:nx, :ny, :nz]


class UniformBackend:
    """Thin wrapper over the current uniform-grid numerical kernels."""

    name = 'uniform'

    def __init__(self):
        self._hartree_kernel_cache = {}

    def create_grid(self, spacing: float, box_size: ArrayLike, **kwargs) -> BackendState:
        """Create the existing uniform-grid state."""
        return create_uniform_grid(spacing, box_size, **kwargs)

    def integrate(self, state: BackendState, field: ArrayLike) -> ArrayLike:
        """Integrate a scalar field using the uniform cell volume."""
        return jnp.sum(field) * state.volume_element

    def inner_product(self, state: BackendState, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Return the current uniform-grid inner product."""
        return self.integrate(state, x * y)

    def build_local_potential(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> ArrayLike:
        """Assemble the local ionic potential on the existing uniform grid."""
        zion = jnp.asarray([p['zion'] for p in pseudos], dtype=jnp.float32)
        rloc = jnp.asarray([p['rloc'] for p in pseudos], dtype=jnp.float32)
        c = jnp.asarray([p['c'] for p in pseudos], dtype=jnp.float32)
        return build_local_potential_uniform(atom_coords, state.coords, zion, rloc, c)

    def apply_kinetic(self, state: BackendState, psi: ArrayLike) -> ArrayLike:
        """Apply the current uniform-grid kinetic operator."""
        return -0.5 * laplacian_8th(psi, state.spacing, state.mask)

    def solve_hartree(self, state: BackendState, rho: ArrayLike, v_init: ArrayLike | None = None) -> ArrayLike:
        """Solve Hartree using the current uniform-grid FFT Poisson path."""
        key = (tuple(int(n) for n in state.shape), float(state.spacing))
        kernel_k = self._hartree_kernel_cache.get(key)
        if kernel_k is None:
            kernel_k = precompute_uniform_poisson_kernel(state.shape, state.spacing)
            self._hartree_kernel_cache[key] = kernel_k
        return solve_uniform_poisson(rho, kernel_k, state.spacing)

    def precompute_nonlocal(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> NonlocalCache:
        """Precompute the existing uniform-grid nonlocal projector tensors."""
        return precompute_projectors(state, atom_coords, pseudos)

    def apply_nonlocal(
        self,
        state: BackendState,
        psi: ArrayLike,
        cache: NonlocalCache,
    ) -> ArrayLike:
        """Apply the existing uniform-grid nonlocal operator."""
        if cache is None:
            return jnp.zeros_like(psi)
        p_i, p_j, coeffs = cache
        return apply_nonlocal_precomputed(psi, p_i, p_j, coeffs, state.volume_element)
