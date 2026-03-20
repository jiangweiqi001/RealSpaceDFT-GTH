"""Adaptive tensor-grid backend wrapper for the prototype nonuniform discretization.

This backend intentionally exposes only the adaptive-grid capabilities that are
already implemented below the SCF layer. Hartree and nonlocal overlap support
remain explicitly unsupported in this milestone.
"""

from __future__ import annotations

import jax.numpy as jnp

from .base import ArrayLike, BackendState, NonlocalCache
from ..grids.adaptive_tensor import create_adaptive_grid as create_adaptive_grid_state
from ..hamiltonian import build_local_potential as build_local_potential_pointwise


class AdaptiveBackend:
    """Thin wrapper over the prototype adaptive tensor-grid numerics."""

    name = "adaptive_tensor"

    def create_grid(self, spacing: float, box_size: ArrayLike, **kwargs) -> BackendState:
        """Create an adaptive tensor grid using spacing as the minimum spacing."""
        atom_coords = kwargs.pop("atom_coords", None)
        if atom_coords is None:
            raise ValueError("AdaptiveBackend.create_grid requires atom_coords")

        h_min = float(kwargs.pop("h_min", spacing))
        h_max = float(kwargs.pop("h_max", h_min))
        r_core = kwargs.pop("r_core", None)
        stretch_beta = float(kwargs.pop("stretch_beta", 0.0))
        if r_core is None:
            raise ValueError("AdaptiveBackend.create_grid requires r_core")
        return self.create_adaptive_grid(
            box_size,
            atom_coords,
            h_min,
            h_max,
            float(r_core),
            stretch_beta,
            **kwargs,
        )

    def create_adaptive_grid(
        self,
        box_size: ArrayLike,
        atom_coords: ArrayLike,
        h_min: float,
        h_max: float,
        r_core: float,
        stretch_beta: float,
        **kwargs,
    ) -> BackendState:
        """Create the adaptive tensor-grid state explicitly."""
        return create_adaptive_grid_state(
            box_size,
            atom_coords,
            h_min,
            h_max,
            r_core,
            stretch_beta,
            **kwargs,
        )

    def integrate(self, state: BackendState, field: ArrayLike) -> ArrayLike:
        """Integrate a scalar field on the adaptive tensor grid."""
        return state.integrate(field)

    def inner_product(self, state: BackendState, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Return the weighted adaptive-grid inner product."""
        return state.inner_product(x, y)

    def build_local_potential(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> ArrayLike:
        """Assemble the local ionic potential on adaptive-grid coordinates."""
        zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
        rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=jnp.float32)
        c = jnp.asarray([p["c"] for p in pseudos], dtype=jnp.float32)
        return build_local_potential_pointwise(atom_coords, state.coords, zion, rloc, c)

    def apply_kinetic(self, state: BackendState, psi: ArrayLike) -> ArrayLike:
        """Apply the prototype adaptive kinetic operator -0.5 * Laplacian."""
        return -0.5 * state.laplacian(psi)

    def solve_hartree(self, state: BackendState, rho: ArrayLike) -> ArrayLike:
        """Adaptive Hartree support is not implemented in this milestone."""
        raise NotImplementedError("Adaptive Hartree/Poisson support is not implemented yet")

    def precompute_nonlocal(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> NonlocalCache:
        """Adaptive nonlocal projector overlap support is not implemented yet."""
        raise NotImplementedError("Adaptive nonlocal projector support is not implemented yet")

    def apply_nonlocal(
        self,
        state: BackendState,
        psi: ArrayLike,
        cache: NonlocalCache,
    ) -> ArrayLike:
        """Adaptive nonlocal projector overlap support is not implemented yet."""
        raise NotImplementedError("Adaptive nonlocal projector support is not implemented yet")
