"""Uniform-grid backend wrapper for the existing real-space implementation.

Patch 2 intentionally keeps this backend as a thin adapter over the current
uniform-grid kernels. It is not wired into the solver runtime path yet.
"""

from __future__ import annotations

import jax.numpy as jnp

from .base import ArrayLike, BackendState, NonlocalCache
from ..hamiltonian import (
    apply_nonlocal_precomputed,
    build_local_potential as build_local_potential_uniform,
    create_grid as create_uniform_grid,
    laplacian_8th,
    precompute_projectors,
)


class UniformBackend:
    """Thin wrapper over the current uniform-grid numerical kernels."""

    name = "uniform"

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
        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
        c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
        return build_local_potential_uniform(atom_coords, state.coords, zion, rloc, c)

    def apply_kinetic(self, state: BackendState, psi: ArrayLike) -> ArrayLike:
        """Apply the current uniform-grid kinetic operator."""
        return -0.5 * laplacian_8th(psi, state.spacing, state.mask)

    def solve_hartree(self, state: BackendState, rho: ArrayLike) -> ArrayLike:
        """Placeholder until the solver is explicitly wired to backend hooks."""
        raise NotImplementedError(
            "UniformBackend.solve_hartree is intentionally left unwired in Patch 2. "
            "Patch 3 will connect the existing uniform Hartree path through the backend interface."
        )

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
