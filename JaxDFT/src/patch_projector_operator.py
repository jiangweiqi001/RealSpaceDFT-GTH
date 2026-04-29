from dataclasses import dataclass

import jax.numpy as jnp

from .hamiltonian import get_gth_projector
from .mixed_orbital import MixedOrbital
from .patch_maps import patch_to_coarse_adjoint


@dataclass(frozen=True)
class PatchProjectorChannel:
    atom_index: int
    p_i: jnp.ndarray
    p_j: jnp.ndarray
    coeff: jnp.ndarray


@dataclass(frozen=True)
class PatchProjectorData:
    channels: tuple
    patch_maps: dict


def build_patch_projector_data(patch_specs, patch_maps, pseudos):
    patch_map_by_atom = {patch_map.atom_index: patch_map for patch_map in patch_maps}
    channels = []

    for spec in patch_specs:
        pseudo = pseudos[spec.atom_index]
        for channel in pseudo.get("projectors", []):
            h_mat = jnp.asarray(channel["h"], dtype=jnp.float32)
            if h_mat.size == 0:
                continue
            if h_mat.ndim == 1:
                h_mat = jnp.diag(h_mat)
            if h_mat.shape[0] == 0 or h_mat.shape[1] == 0:
                continue

            l = channel["l"]
            if l != 0:
                raise NotImplementedError("v1 experimental patch projector only supports l=0")

            rp = channel["r"]

            r = jnp.linalg.norm(spec.offsets, axis=-1)
            for i in range(1, h_mat.shape[0] + 1):
                p_i = get_gth_projector(r, l, i, rp)
                for j in range(1, h_mat.shape[1] + 1):
                    coeff = h_mat[i - 1, j - 1] / (4.0 * jnp.pi)
                    if float(jnp.abs(coeff)) < 1e-10:
                        continue
                    p_j = get_gth_projector(r, l, j, rp)
                    channels.append(
                        PatchProjectorChannel(
                            atom_index=spec.atom_index,
                            p_i=p_i,
                            p_j=p_j,
                            coeff=coeff,
                        )
                    )

    return PatchProjectorData(channels=tuple(channels), patch_maps=patch_map_by_atom)


def apply_patch_projector(mixed_orbital, projector_data, output_shape):
    coarse_result = jnp.zeros(output_shape, dtype=jnp.asarray(mixed_orbital.coarse).dtype)

    for channel in projector_data.channels:
        patch_values = mixed_orbital.patch_values[channel.atom_index]
        patch_map = projector_data.patch_maps[channel.atom_index]
        overlap = patch_map.fine_dv * jnp.sum(channel.p_j * patch_values)
        values_patch = channel.p_i * (channel.coeff * overlap)
        coarse_result = coarse_result + patch_to_coarse_adjoint(
            values_patch,
            patch_map,
            output_shape,
        )

    return MixedOrbital(coarse=coarse_result, patch_values={})
