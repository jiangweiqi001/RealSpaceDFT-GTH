from dataclasses import dataclass

import jax.numpy as jnp

from .hamiltonian import _make_atom_patch_points


@dataclass(frozen=True)
class AtomPatchSpec:
    atom_index: int
    center: jnp.ndarray
    element: str
    radius: float
    fine_spacing: jnp.ndarray
    positions: jnp.ndarray
    offsets: jnp.ndarray
    fine_dv: jnp.ndarray


def build_atom_patch_specs(
    grid,
    atom_coords,
    pseudos,
    patch_subgrid=2,
    patch_radius_factor=4.0,
):
    if patch_subgrid < 1:
        raise ValueError("patch_subgrid must be >= 1")

    atom_coords = jnp.asarray(atom_coords, dtype=jnp.float32)
    fine_spacing = jnp.asarray(grid.spacing, dtype=jnp.float32) / patch_subgrid
    specs = []

    for atom_index, pseudo in enumerate(pseudos):
        if not pseudo.get("projectors"):
            continue

        element = pseudo.get("symbol", f"atom_{atom_index}")
        radius = float(patch_radius_factor) * float(pseudo["projectors"][0]["r"])
        offsets, positions, fine_dv = _make_atom_patch_points(
            atom_coords[atom_index],
            radius,
            fine_spacing,
        )
        specs.append(
            AtomPatchSpec(
                atom_index=atom_index,
                center=atom_coords[atom_index],
                element=element,
                radius=radius,
                fine_spacing=fine_spacing,
                positions=positions,
                offsets=offsets,
                fine_dv=fine_dv,
            )
        )

    return specs
