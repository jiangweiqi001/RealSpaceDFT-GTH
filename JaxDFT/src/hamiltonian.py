from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Grid3D:
    spacing: float
    box_size: jnp.ndarray
    shape: tuple
    coords: jnp.ndarray
    mask: jnp.ndarray
    volume_element: float
    V_loc: jnp.ndarray
    projectors: list


def create_grid(spacing, box_size):
    box_size = jnp.asarray(box_size, dtype=jnp.float64)
    spacing = float(spacing)
    shape = tuple(int(jnp.round(L / spacing)) + 1 for L in box_size)
    xs = jnp.linspace(-0.5 * box_size[0], 0.5 * box_size[0], shape[0])
    ys = jnp.linspace(-0.5 * box_size[1], 0.5 * box_size[1], shape[1])
    zs = jnp.linspace(-0.5 * box_size[2], 0.5 * box_size[2], shape[2])
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    coords = jnp.stack([X, Y, Z], axis=-1)
    mask = jnp.ones(shape, dtype=jnp.float64)
    mask = mask.at[0, :, :].set(0.0)
    mask = mask.at[-1, :, :].set(0.0)
    mask = mask.at[:, 0, :].set(0.0)
    mask = mask.at[:, -1, :].set(0.0)
    mask = mask.at[:, :, 0].set(0.0)
    mask = mask.at[:, :, -1].set(0.0)
    volume_element = float(spacing ** 3)
    return Grid3D(
        spacing=spacing,
        box_size=box_size,
        shape=shape,
        coords=coords,
        mask=mask,
        volume_element=volume_element,
        V_loc=None,
        projectors=[],
    )


@jax.jit
def laplacian_4th(psi, spacing, mask):
    psi = psi * mask
    h2 = spacing * spacing
    c0 = -2.5 / h2
    c1 = (4.0 / 3.0) / h2
    c2 = (-1.0 / 12.0) / h2
    lap = c0 * psi
    lap += c1 * (jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0))
    lap += c2 * (jnp.roll(psi, 2, axis=0) + jnp.roll(psi, -2, axis=0))
    lap += c1 * (jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1))
    lap += c2 * (jnp.roll(psi, 2, axis=1) + jnp.roll(psi, -2, axis=1))
    lap += c1 * (jnp.roll(psi, 1, axis=2) + jnp.roll(psi, -1, axis=2))
    lap += c2 * (jnp.roll(psi, 2, axis=2) + jnp.roll(psi, -2, axis=2))
    return lap * mask


@jax.jit
def gth_local_potential_value(r, zion, rloc, c):
    t = r / (jnp.sqrt(2.0) * rloc)
    v_coul = -zion * (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-t * t) / (r + 1e-12)
    x = r / rloc
    gauss = jnp.exp(-0.5 * x * x)
    poly = c[..., 0] + c[..., 1] * (x * x) + c[..., 2] * (x ** 4) + c[..., 3] * (x ** 6)
    return v_coul + gauss * poly


@jax.jit
def build_local_potential(coords, grid_coords, zion, rloc, c):
    rvec = grid_coords[None, ...] - coords[:, None, None, None, :]
    r = jnp.sqrt(jnp.sum(rvec * rvec, axis=-1) + 1e-12)
    v = gth_local_potential_value(
        r,
        zion[:, None, None, None],
        rloc[:, None, None, None],
        c[:, None, None, None, :],
    )
    return jnp.sum(v, axis=0)


def build_projectors(coords, grid, pseudos):
    proj_list = []
    flat_coords = grid.coords.reshape(-1, 3)
    for a in range(coords.shape[0]):
        for pj in pseudos[a]["projectors"]:
            if pj["l"] != 0:
                continue
            rvec = flat_coords - coords[a]
            r = jnp.linalg.norm(rvec, axis=-1)
            x = r / pj["r"]
            radial = jnp.exp(-x * x)
            polys = [radial]
            if pj["h"].shape[0] > 1:
                polys.append((x * x) * radial)
            if pj["h"].shape[0] > 2:
                polys.append((x ** 4) * radial)
            if pj["h"].shape[0] > 3:
                polys.append((x ** 6) * radial)
            for k, hk in enumerate(pj["h"]):
                if k < len(polys):
                    proj_list.append(
                        {
                            "h": hk,
                            "vec": polys[k],
                        }
                    )
    return proj_list


def apply_nonlocal(projectors, psi, volume_element):
    if not projectors:
        return jnp.zeros_like(psi)
    out = jnp.zeros_like(psi)
    for p in projectors:
        coeff = volume_element * jnp.dot(p["vec"], psi)
        out = out + p["h"] * coeff * p["vec"]
    return out


def prepare_system(grid, coords, pseudos):
    zion = jnp.asarray([pp["zion"] for pp in pseudos], dtype=jnp.float64)
    rloc = jnp.asarray([pp["rloc"] for pp in pseudos], dtype=jnp.float64)
    c = jnp.asarray([pp["c"] for pp in pseudos], dtype=jnp.float64)
    V_loc = build_local_potential(coords, grid.coords, zion, rloc, c)
    projectors = build_projectors(coords, grid, pseudos)
    return Grid3D(
        spacing=grid.spacing,
        box_size=grid.box_size,
        shape=grid.shape,
        coords=grid.coords,
        mask=grid.mask,
        volume_element=grid.volume_element,
        V_loc=V_loc,
        projectors=projectors,
    )
