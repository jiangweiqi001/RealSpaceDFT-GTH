"""Adaptive tensor-grid backend wrapper for the prototype nonuniform discretization.

This backend intentionally exposes only the adaptive-grid capabilities that are
already implemented below the SCF layer. The current Hartree path defaults to a
monopole-Dirichlet box Poisson solve, which is still not an exact
isolated/open-boundary treatment but is a first upgrade over the previous
zero-Dirichlet prototype. An explicit zero-Dirichlet fallback remains available
for regression and debugging. The current nonlocal path reuses the existing
pointwise projector tabulation but evaluates overlaps with adaptive volume
weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from .base import ArrayLike, BackendState, NonlocalCache
from ..grids.adaptive_poisson import (
    assemble_poisson_operator_3d,
    solve_hartree_dirichlet_3d,
    solve_hartree_monopole_dirichlet_3d,
)
from ..grids.adaptive_tensor import create_adaptive_grid as create_adaptive_grid_state
from ..hamiltonian import (
    build_local_potential as build_local_potential_pointwise,
    precompute_projectors,
)


@jax.jit
def apply_nonlocal_weighted(
    psi: ArrayLike,
    p_i: ArrayLike,
    p_j: ArrayLike,
    coeffs: ArrayLike,
    volume_weights: ArrayLike,
) -> ArrayLike:
    """Apply the nonlocal operator using adaptive weighted overlaps."""
    overlap = jnp.sum(p_j * psi[None, ...] * volume_weights[None, ...], axis=(1, 2, 3))
    weight = coeffs * overlap
    return jnp.sum(weight[:, None, None, None] * p_i, axis=0)


class AdaptiveBackend:
    """Thin wrapper over the prototype adaptive tensor-grid numerics."""

    name = "adaptive_tensor"

    def __init__(self, *, hartree_boundary_mode: str = "monopole_dirichlet"):
        supported = {"monopole_dirichlet", "zero_dirichlet"}
        if hartree_boundary_mode not in supported:
            raise ValueError(
                f"unsupported hartree_boundary_mode {hartree_boundary_mode!r}; "
                f"expected one of {sorted(supported)}"
            )
        self.hartree_boundary_mode = hartree_boundary_mode

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
        """Create the adaptive tensor-grid state explicitly.

        The Poisson operator depends only on the grid geometry, so we assemble it
        once here and cache JAX BCOO versions on the returned state. This keeps
        the SCF loop from rebuilding large SciPy sparse matrices on every Hartree
        solve.
        """
        state = create_adaptive_grid_state(
            box_size,
            atom_coords,
            h_min,
            h_max,
            r_core,
            stretch_beta,
            **kwargs,
        )
        A, M = assemble_poisson_operator_3d(state)
        state.A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        state.M_bcoo = jsparse.BCOO.from_scipy_sparse(M)
        state.A_nnz = int(A.nnz)
        state.M_nnz = int(M.nnz)
        return state

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
        """Solve Hartree with the current adaptive box-Poisson prototype.

        The default path uses monopole Dirichlet boundary data, which is more
        isolated-like than zero Dirichlet but still not an exact
        isolated/open-boundary Hartree treatment. The previous zero-Dirichlet
        prototype remains available via ``hartree_boundary_mode='zero_dirichlet'``.
        """
        if self.hartree_boundary_mode == "monopole_dirichlet":
            V_h, _ = solve_hartree_monopole_dirichlet_3d(state, rho)
            return V_h
        if self.hartree_boundary_mode == "zero_dirichlet":
            V_h, _ = solve_hartree_dirichlet_3d(state, rho)
            return V_h
        raise ValueError(f"unsupported hartree_boundary_mode {self.hartree_boundary_mode!r}")

    def precompute_nonlocal(
        self,
        state: BackendState,
        atom_coords: ArrayLike,
        pseudos: ArrayLike,
    ) -> NonlocalCache:
        """Precompute pointwise projector tensors on adaptive-grid coordinates."""
        return precompute_projectors(state, atom_coords, pseudos)

    def apply_nonlocal(
        self,
        state: BackendState,
        psi: ArrayLike,
        cache: NonlocalCache,
    ) -> ArrayLike:
        """Apply the nonlocal operator using weighted adaptive-grid overlaps."""
        if cache is None:
            return jnp.zeros_like(psi)
        p_i, p_j, coeffs = cache
        return apply_nonlocal_weighted(psi, p_i, p_j, coeffs, state.volume_weights)
