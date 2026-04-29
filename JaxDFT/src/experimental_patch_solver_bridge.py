import jax
import jax.numpy as jnp

from .functional import lda_xc
from .hamiltonian import laplacian_8th
from .mixed_orbital import MixedOrbital
from .patch_builder import build_atom_patch_specs
from .patch_maps import build_patch_maps, coarse_to_patch
from .patch_projector_operator import apply_patch_projector, build_patch_projector_data
from .solver import (
    anderson_mixing,
    precompute_poisson_kernel,
    solve_orbitals_dense,
    solve_orbitals_subspace,
    solve_poisson,
)


def build_mixed_orbital_from_coarse(psi, patch_maps):
    patch_values = {
        patch_map.atom_index: coarse_to_patch(psi, patch_map)
        for patch_map in patch_maps
    }
    return MixedOrbital(coarse=psi, patch_values=patch_values)


def build_experimental_patch_nonlocal_operator(
    grid,
    coords,
    pseudos,
    patch_subgrid=2,
    patch_radius_factor=4.0,
):
    patch_specs = build_atom_patch_specs(
        grid,
        coords,
        pseudos,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
    )
    patch_maps = build_patch_maps(grid, patch_specs)
    projector_data = build_patch_projector_data(patch_specs, patch_maps, pseudos)

    def apply_nonlocal_patch(psi):
        mixed = build_mixed_orbital_from_coarse(psi, patch_maps)
        applied = apply_patch_projector(mixed, projector_data, grid.shape)
        return applied.coarse

    return apply_nonlocal_patch


def build_experimental_patch_apply_h(
    grid,
    coords,
    pseudos,
    v_eff,
    patch_subgrid=2,
    patch_radius_factor=4.0,
):
    v_eff = jnp.asarray(v_eff, dtype=jnp.float32)
    apply_nonlocal_patch = build_experimental_patch_nonlocal_operator(
        grid,
        coords,
        pseudos,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
    )

    def apply_h(psi_flat):
        psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
        lap = laplacian_8th(psi, grid.spacing, grid.mask)
        v_nonlocal = apply_nonlocal_patch(psi)
        hpsi = -0.5 * lap + v_eff * psi + v_nonlocal
        return hpsi.reshape(-1)

    return apply_h


def solve_experimental_patch_orbitals(
    grid,
    coords,
    pseudos,
    v_eff,
    n_bands,
    x_init=None,
    max_iter=30,
    tol=1e-5,
    key=None,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    solver_mode="subspace",
):
    if solver_mode not in ("subspace", "dense"):
        raise ValueError("solver_mode must be 'subspace' or 'dense'")

    apply_h = build_experimental_patch_apply_h(
        grid,
        coords,
        pseudos,
        v_eff,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
    )
    n_grid = int(jnp.prod(jnp.array(grid.shape)))
    if solver_mode == "dense":
        return solve_orbitals_dense(
            apply_h,
            n_grid,
            n_bands,
        )

    return solve_orbitals_subspace(
        apply_h,
        n_grid,
        n_bands,
        x_init=x_init,
        max_iter=max_iter,
        tol=tol,
        key=key,
    )


def experimental_patch_scf(
    grid,
    coords,
    n_bands,
    occ,
    V_loc,
    projectors,
    max_iter,
    mix_alpha,
    tolerance,
    key,
    patch_subgrid=2,
    patch_radius_factor=4.0,
    solver_mode="subspace",
):
    coords = jnp.asarray(coords, dtype=jnp.float32)
    if key is None:
        key = jax.random.PRNGKey(42)

    volume_element = grid.volume_element
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for atom_index in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[atom_index], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = rho / (jnp.sum(rho) * volume_element) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    n_grid = rho.size
    kernel_k = precompute_poisson_kernel(grid.shape, grid.spacing)
    apply_nonlocal_patch = build_experimental_patch_nonlocal_operator(
        grid,
        coords,
        projectors,
        patch_subgrid=patch_subgrid,
        patch_radius_factor=patch_radius_factor,
    )

    eigvals = jnp.zeros((n_bands,), dtype=jnp.float32)
    eigvecs = jnp.zeros((n_grid, n_bands), dtype=jnp.float32)
    V_H = jnp.zeros(grid.shape, dtype=jnp.float32)
    eps_xc = jnp.zeros(grid.shape, dtype=jnp.float32)
    v_xc = jnp.zeros(grid.shape, dtype=jnp.float32)

    for iteration in range(max_iter):
        rho = jnp.clip(rho, 1e-12, None)
        V_H = solve_poisson(rho, kernel_k, grid.spacing)
        eps_xc, v_xc = lda_xc(rho)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = jnp.asarray(psi_flat, dtype=jnp.float32).reshape(grid.shape)
            lap = laplacian_8th(psi, grid.spacing, grid.mask)
            v_nonlocal = apply_nonlocal_patch(psi)
            hpsi = -0.5 * lap + V_eff * psi + v_nonlocal
            return hpsi.reshape(-1)

        iter_key = jax.random.fold_in(key, iteration)
        if solver_mode == "dense":
            eigvals, eigvecs = solve_orbitals_dense(
                apply_h,
                n_grid,
                n_bands,
            )
        else:
            eigvals, eigvecs = solve_orbitals_subspace(
                apply_h,
                n_grid,
                n_bands,
                x_init=eigvecs,
                max_iter=30,
                tol=1e-5,
                key=iter_key,
            )
        norm = jnp.sqrt(jnp.sum(eigvecs**2, axis=0) * volume_element)
        eigvecs = eigvecs / norm

        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        diff = jnp.max(jnp.abs(rho_new - rho))
        rho_flat, f_hist = anderson_mixing(
            rho.reshape(-1),
            rho_new.reshape(-1),
            f_hist,
            mix_alpha,
            jnp.asarray(iteration, dtype=jnp.int32),
        )
        rho = rho_flat.reshape(grid.shape)
        if float(diff) <= tolerance:
            break

    return rho, eigvals, eigvecs, V_H, eps_xc, v_xc
