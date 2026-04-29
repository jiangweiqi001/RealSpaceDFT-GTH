from dataclasses import dataclass

import jax.numpy as jnp

from .patch_maps import coarse_to_patch, patch_to_coarse_adjoint


@dataclass(frozen=True)
class IndependentPatchOrbital:
    coarse: jnp.ndarray
    patch_corrections: dict


def physical_patch_values(state, patch_maps, patch_bases=None):
    values = {}
    for patch_map in patch_maps:
        correction = state.patch_corrections.get(
            patch_map.atom_index,
            jnp.zeros(
                (
                    patch_bases[patch_map.atom_index].shape[1]
                    if patch_bases is not None and patch_map.atom_index in patch_bases
                    else patch_map.eval_matrix.shape[0]
                ,),
                dtype=jnp.asarray(state.coarse).dtype,
            ),
        )
        if patch_bases is not None and patch_map.atom_index in patch_bases:
            correction_values = patch_bases[patch_map.atom_index] @ correction
        else:
            correction_values = correction
        values[patch_map.atom_index] = coarse_to_patch(state.coarse, patch_map) + correction_values
    return values


def apply_independent_patch_projector(state, projector_data, patch_maps, output_shape, patch_bases=None):
    patch_map_by_atom = {patch_map.atom_index: patch_map for patch_map in patch_maps}
    patch_values = physical_patch_values(state, patch_maps, patch_bases=patch_bases)
    coarse_result = jnp.zeros(output_shape, dtype=jnp.asarray(state.coarse).dtype)
    patch_result = {
        atom_index: jnp.zeros(
            (
                patch_bases[atom_index].shape[1]
                if patch_bases is not None and atom_index in patch_bases
                else values.shape[0]
            ,),
            dtype=values.dtype,
        )
        for atom_index, values in patch_values.items()
    }

    for channel in projector_data.channels:
        atom_index = channel.atom_index
        patch_map = patch_map_by_atom[atom_index]
        psi_patch = patch_values[atom_index]
        overlap = patch_map.fine_dv * jnp.sum(channel.p_j * psi_patch)
        values_patch = channel.p_i * (channel.coeff * overlap)
        coarse_result = coarse_result + patch_to_coarse_adjoint(
            values_patch,
            patch_map,
            output_shape,
        )
        if patch_bases is not None and atom_index in patch_bases:
            patch_result[atom_index] = patch_result[atom_index] + patch_bases[atom_index].T @ values_patch
        else:
            patch_result[atom_index] = patch_result[atom_index] + values_patch

    return IndependentPatchOrbital(
        coarse=coarse_result,
        patch_corrections=patch_result,
    )
