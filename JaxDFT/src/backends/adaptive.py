"""Adaptive tensor-grid backend wrapper for the prototype nonuniform discretization.

This backend intentionally exposes only the adaptive-grid capabilities that are
already implemented below the SCF layer. The current Hartree path defaults to a
multipole-Dirichlet box Poisson solve, which is still not an exact
isolated/open-boundary treatment but is a further upgrade over the previous
monopole and zero-Dirichlet prototypes. Explicit monopole and zero-Dirichlet
fallbacks remain available for regression and debugging. The monopole fallback
now also supports choosing the reference center via ``hartree_center_mode``.
The current nonlocal path reuses the existing pointwise projector tabulation but
evaluates overlaps with adaptive volume weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from .base import ArrayLike, BackendState, NonlocalCache
from ..grids.adaptive_poisson import (
    assemble_poisson_operator_3d,
    solve_hartree_dirichlet_3d,
    solve_hartree_multipole_dirichlet_3d,
    solve_hartree_monopole_dirichlet_3d,
    solve_hartree_uniform_exterior_dirichlet_3d,
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

    def __init__(
        self,
        *,
        hartree_boundary_mode: str = "multipole_dirichlet",
        hartree_center_mode: str = "box_center",
        kinetic_mode: str = "prototype_fd2",
    ):
        supported = {"multipole_dirichlet", "monopole_dirichlet", "zero_dirichlet", "uniform_exterior"}
        supported_center_modes = {"box_center", "charge_center"}
        supported_kinetic_modes = {"prototype_fd2", "symmetric_fv"}
        if hartree_boundary_mode not in supported:
            raise ValueError(
                f"unsupported hartree_boundary_mode {hartree_boundary_mode!r}; "
                f"expected one of {sorted(supported)}"
            )
        if hartree_center_mode not in supported_center_modes:
            raise ValueError(
                f"unsupported hartree_center_mode {hartree_center_mode!r}; "
                f"expected one of {sorted(supported_center_modes)}"
            )
        if kinetic_mode not in supported_kinetic_modes:
            raise ValueError(
                f"unsupported kinetic_mode {kinetic_mode!r}; "
                f"expected one of {sorted(supported_kinetic_modes)}"
            )
        self.hartree_boundary_mode = hartree_boundary_mode
        self.hartree_center_mode = hartree_center_mode
        self.kinetic_mode = kinetic_mode

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
        diag = np.asarray(A.diagonal(), dtype=np.float32)
        diag = np.where(np.abs(diag) > 1.0e-12, diag, 1.0)
        state.A_inv_diag = jnp.asarray(1.0 / diag, dtype=jnp.float32)
        state.x_host = np.asarray(state.x, dtype=np.float64)
        state.y_host = np.asarray(state.y, dtype=np.float64)
        state.z_host = np.asarray(state.z, dtype=np.float64)
        state.coords_host = np.asarray(state.coords, dtype=np.float64)
        state.uniform_exterior_cache = {}
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
        """Apply the selected adaptive kinetic operator -0.5 * Laplacian."""
        if self.kinetic_mode == "prototype_fd2":
            return -0.5 * state.laplacian(psi)
        if self.kinetic_mode == "symmetric_fv":
            return -0.5 * state.laplacian_symmetric(psi)
        raise ValueError(f"unsupported kinetic_mode {self.kinetic_mode!r}")

    def solve_hartree(self, state: BackendState, rho: ArrayLike, v_init: ArrayLike | None = None) -> ArrayLike:
        """Solve Hartree with the current adaptive box-Poisson prototype.

        The default path uses multipole Dirichlet boundary data, which is more
        isolated-like than the older monopole and zero-Dirichlet box Poisson
        prototypes but is still not an exact isolated/open-boundary Hartree
        treatment. An experimental ``uniform_exterior`` fallback is also
        available; it preserves the adaptive interior operator and obtains face
        data from a larger coarse uniform auxiliary free-space-like solve.
        Explicit monopole and zero-Dirichlet fallbacks remain available for
        regression and debugging. ``hartree_center_mode`` is currently routed
        only to the monopole fallback so that the default multipole behavior
        remains unchanged.
        """
        if self.hartree_boundary_mode == "multipole_dirichlet":
            V_h, _ = solve_hartree_multipole_dirichlet_3d(state, rho, v_init=v_init)
            return V_h
        if self.hartree_boundary_mode == "monopole_dirichlet":
            V_h, _ = solve_hartree_monopole_dirichlet_3d(
                state,
                rho,
                center_mode=self.hartree_center_mode,
                v_init=v_init,
            )
            return V_h
        if self.hartree_boundary_mode == "zero_dirichlet":
            V_h, _ = solve_hartree_dirichlet_3d(state, rho, v_init=v_init)
            return V_h
        if self.hartree_boundary_mode == "uniform_exterior":
            V_h, _ = solve_hartree_uniform_exterior_dirichlet_3d(state, rho, v_init=v_init)
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
