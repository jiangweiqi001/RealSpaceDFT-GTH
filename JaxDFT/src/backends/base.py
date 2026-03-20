"""Lightweight backend protocol for future multi-backend support.

This module intentionally defines only interfaces. It does not alter any
existing runtime code paths. Backend-specific grid/state objects are kept
opaque on purpose so the current uniform grid object can later be used as-is
for the first backend implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# Backend state is intentionally left unconstrained in Phase 1.
# The current uniform-grid object is expected to satisfy this role directly.
BackendState = Any
NonlocalCache = Any
ArrayLike = Any


@runtime_checkable
class Backend(Protocol):
    """Minimal backend interface for real-space DFT backends.

    The interface is deliberately lightweight:
    - `state` may be any backend-owned object
    - array-like values are left untyped to avoid coupling to a specific
      numerical array library at the protocol layer
    - `inner_product` is included now to reserve a clean extension point for
      future weighted/adaptive discretizations without changing the interface
      again later
    """

    name: str

    def create_grid(self, spacing: float, box_size: ArrayLike, **kwargs: Any) -> BackendState:
        """Create and return a backend-specific discretization state."""

    def integrate(self, state: BackendState, field: ArrayLike) -> ArrayLike:
        """Integrate a scalar field over the backend discretization."""

    def inner_product(self, state: BackendState, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Return the backend-aware inner product of two fields."""

    def build_local_potential(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> ArrayLike:
        """Assemble the local ionic potential on the backend discretization."""

    def apply_kinetic(self, state: BackendState, psi: ArrayLike) -> ArrayLike:
        """Apply the kinetic operator to a wavefunction field."""

    def solve_hartree(self, state: BackendState, rho: ArrayLike) -> ArrayLike:
        """Solve for the Hartree potential associated with a density field."""

    def precompute_nonlocal(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> NonlocalCache:
        """Precompute backend-specific nonlocal projector data."""

    def apply_nonlocal(
        self,
        state: BackendState,
        psi: ArrayLike,
        cache: NonlocalCache,
    ) -> ArrayLike:
        """Apply the nonlocal pseudopotential operator to a wavefunction."""
