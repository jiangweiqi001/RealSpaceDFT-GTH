from dataclasses import dataclass
import math

import jax.numpy as jnp

from .hamiltonian import build_patch_polynomial_reconstruction_data


@dataclass(frozen=True)
class PatchMap:
    atom_index: int
    sample_indices: jnp.ndarray
    eval_matrix: jnp.ndarray
    fine_dv: jnp.ndarray
    coarse_dv: jnp.ndarray


def build_patch_maps(grid, patch_specs, stencil_half_width=2):
    coarse_dv = jnp.asarray(grid.spacing, dtype=jnp.float32) ** 3
    patch_maps = []

    for spec in patch_specs:
        sample_indices, eval_matrix = build_patch_polynomial_reconstruction_data(
            grid,
            spec.center,
            spec.positions,
            stencil_half_width=stencil_half_width,
        )
        patch_maps.append(
            PatchMap(
                atom_index=spec.atom_index,
                sample_indices=sample_indices,
                eval_matrix=eval_matrix,
                fine_dv=spec.fine_dv,
                coarse_dv=coarse_dv,
            )
        )

    return patch_maps


def coarse_to_patch(psi, patch_map):
    psi_flat = jnp.asarray(psi).reshape(-1)
    coarse_samples = psi_flat[patch_map.sample_indices]
    return patch_map.eval_matrix @ coarse_samples


def patch_to_coarse_adjoint(values_patch, patch_map, output_shape):
    flat_out = jnp.zeros((math.prod(output_shape),), dtype=jnp.asarray(values_patch).dtype)
    sample_contrib = (
        patch_map.eval_matrix.T @ jnp.asarray(values_patch)
    ) * (patch_map.fine_dv / patch_map.coarse_dv)
    flat_out = flat_out.at[patch_map.sample_indices].add(sample_contrib)
    return flat_out.reshape(output_shape)
