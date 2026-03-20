"""Adaptive tensor-product grid helpers for future nonuniform backends.

Patch A/B intentionally stays below the SCF layer. It provides 1D adaptive axes,
nodal integration weights, tensor-product volume weights, and a minimal 3D grid
state without modifying any existing solver execution path.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


Array = jnp.ndarray


class AdaptiveTensorGrid:
    """Lightweight container matching the current uniform-grid object style."""

    def integrate(self, field: Array) -> Array:
        """Integrate a scalar field using tensor-product volume weights."""
        field = jnp.asarray(field)
        if field.shape != self.shape:
            raise ValueError(f"field shape {field.shape} does not match grid shape {self.shape}")
        return jnp.sum(field * self.volume_weights)

    def inner_product(self, x: Array, y: Array) -> Array:
        """Return the weighted inner product on the adaptive tensor grid."""
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if x.shape != self.shape or y.shape != self.shape:
            raise ValueError(
                f"inner_product expects shapes {self.shape}, got x={x.shape}, y={y.shape}"
            )
        return self.integrate(jnp.conjugate(x) * y)


def make_reference_axis(
    box_length: float,
    n_ref: int = 4097,
    dtype: Any = jnp.float32,
) -> Array:
    """Return a dense 1D reference axis spanning the simulation box."""
    box_length = float(box_length)
    n_ref = int(n_ref)
    if box_length <= 0.0:
        raise ValueError(f"box_length must be positive, got {box_length}")
    if n_ref < 2:
        raise ValueError(f"n_ref must be at least 2, got {n_ref}")
    return jnp.linspace(-0.5 * box_length, 0.5 * box_length, n_ref, dtype=dtype)


def build_axis_spacing_profile(
    ref_axis: Array,
    atom_positions_1d: Array,
    h_min: float,
    h_max: float,
    r_core: float,
    stretch_beta: float,
    stretch_rule: str = "gaussian_sum",
) -> Array:
    """Build a target local spacing profile on a dense reference axis."""
    ref_axis = jnp.asarray(ref_axis, dtype=jnp.float32)
    atom_positions_1d = jnp.asarray(atom_positions_1d, dtype=jnp.float32).reshape(-1)
    h_min = float(h_min)
    h_max = float(h_max)
    r_core = float(r_core)
    stretch_beta = float(stretch_beta)

    if ref_axis.ndim != 1:
        raise ValueError("ref_axis must be 1D")
    if h_min <= 0.0:
        raise ValueError(f"h_min must be positive, got {h_min}")
    if h_max < h_min:
        raise ValueError(f"h_max must satisfy h_max >= h_min, got h_min={h_min}, h_max={h_max}")
    if stretch_rule != "gaussian_sum":
        raise ValueError(f"Unsupported stretch_rule: {stretch_rule}")
    if r_core <= 0.0:
        raise ValueError(f"r_core must be positive, got {r_core}")
    if stretch_beta < 0.0:
        raise ValueError(f"stretch_beta must be nonnegative, got {stretch_beta}")

    if h_min == h_max:
        return jnp.full_like(ref_axis, h_min)

    if atom_positions_1d.size == 0 or stretch_beta == 0.0:
        return jnp.full_like(ref_axis, h_max)

    diff = ref_axis[:, None] - atom_positions_1d[None, :]
    gaussian_sum = jnp.sum(jnp.exp(-0.5 * (diff / r_core) ** 2), axis=1)
    raw_spacing = h_max / (1.0 + stretch_beta * gaussian_sum)
    return jnp.clip(raw_spacing, h_min, h_max)


def _uniform_axis_from_spacing(box_length: float, spacing: float) -> tuple[Array, int, float]:
    """Match the current uniform-grid preserve-box node convention."""
    n_intervals = max(1, int(round(box_length / spacing)))
    axis = jnp.linspace(-0.5 * box_length, 0.5 * box_length, n_intervals + 1, dtype=jnp.float32)
    actual_spacing = box_length / n_intervals
    return axis, n_intervals, actual_spacing


def _cumulative_trapezoid(y: Array, x: Array) -> Array:
    """Return cumulative trapezoidal integration values with y[0] anchored at zero."""
    dx = x[1:] - x[:-1]
    y_mid = 0.5 * (y[1:] + y[:-1])
    increments = y_mid * dx
    return jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), jnp.cumsum(increments)])


def _estimate_interval_count(ref_axis: Array, h_profile: Array) -> int:
    """Estimate the number of intervals implied by a spacing profile."""
    monitor = 1.0 / h_profile
    cumulative = _cumulative_trapezoid(monitor, ref_axis)
    return max(1, int(round(float(cumulative[-1]))))


def _redistribute_axis(ref_axis: Array, h_profile: Array, n_intervals: int) -> Array:
    """Redistribute nodes by making the monitor integral uniform between nodes."""
    monitor = 1.0 / h_profile
    cumulative = _cumulative_trapezoid(monitor, ref_axis)
    total = float(cumulative[-1])
    if not jnp.isfinite(cumulative).all():
        raise ValueError("Non-finite values encountered in cumulative monitor integral")
    if total <= 0.0:
        raise ValueError("Adaptive monitor integral must be positive")

    target = jnp.linspace(0.0, total, n_intervals + 1, dtype=ref_axis.dtype)
    axis = jnp.interp(target, cumulative, ref_axis)
    axis = axis.at[0].set(ref_axis[0])
    axis = axis.at[-1].set(ref_axis[-1])
    return axis


def compute_axis_weights(axis: Array) -> Array:
    """Return positive nodal trapezoidal weights for a strictly monotone axis."""
    axis = jnp.asarray(axis, dtype=jnp.float32).reshape(-1)
    if axis.size < 2:
        raise ValueError("axis must contain at least two nodes")

    diffs = axis[1:] - axis[:-1]
    if bool(jnp.any(diffs <= 0.0)):
        raise ValueError("axis must be strictly increasing")

    weights = jnp.zeros_like(axis)
    weights = weights.at[0].set(0.5 * diffs[0])
    weights = weights.at[-1].set(0.5 * diffs[-1])
    if axis.size > 2:
        weights = weights.at[1:-1].set(0.5 * (axis[2:] - axis[:-2]))
    return weights


def build_volume_weights(wx: Array, wy: Array, wz: Array) -> Array:
    """Return tensor-product volume weights from three 1D nodal weight arrays."""
    wx = jnp.asarray(wx, dtype=jnp.float32).reshape(-1)
    wy = jnp.asarray(wy, dtype=jnp.float32).reshape(-1)
    wz = jnp.asarray(wz, dtype=jnp.float32).reshape(-1)
    if wx.size < 2 or wy.size < 2 or wz.size < 2:
        raise ValueError("Each 1D weight array must contain at least two nodes")
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def create_adaptive_axis(
    box_length: float,
    atom_positions_1d: Array,
    h_min: float,
    h_max: float,
    r_core: float,
    stretch_beta: float,
    *,
    n_ref: int = 4097,
    stretch_rule: str = "gaussian_sum",
    uniform_tol: float = 1e-8,
) -> tuple[Array, Array, dict[str, Any]]:
    """Create a nonuniform 1D axis plus nodal weights and metadata.

    When h_min == h_max within uniform_tol, this function takes an explicit
    uniform-degenerate branch that matches the current uniform-grid box-preserve
    node convention used by create_grid(...).
    """
    box_length = float(box_length)
    h_min = float(h_min)
    h_max = float(h_max)
    uniform_tol = float(uniform_tol)
    atom_positions_1d = jnp.asarray(atom_positions_1d, dtype=jnp.float32).reshape(-1)

    if box_length <= 0.0:
        raise ValueError(f"box_length must be positive, got {box_length}")
    if h_min <= 0.0:
        raise ValueError(f"h_min must be positive, got {h_min}")
    if h_max < h_min:
        raise ValueError(f"h_max must satisfy h_max >= h_min, got h_min={h_min}, h_max={h_max}")

    is_uniform = abs(h_max - h_min) <= uniform_tol
    if is_uniform:
        axis, n_intervals, actual_spacing = _uniform_axis_from_spacing(box_length, h_min)
        weights = compute_axis_weights(axis)
        diffs = axis[1:] - axis[:-1]
        meta = {
            "box_length": box_length,
            "h_min": h_min,
            "h_max": h_max,
            "r_core": float(r_core),
            "stretch_beta": float(stretch_beta),
            "stretch_rule": stretch_rule,
            "n_ref": None,
            "n_intervals": n_intervals,
            "n_nodes": int(axis.size),
            "is_uniform_degenerate": True,
            "actual_min_spacing": float(jnp.min(diffs)),
            "actual_max_spacing": float(jnp.max(diffs)),
            "actual_spacing": actual_spacing,
        }
        return axis, weights, meta

    ref_axis = make_reference_axis(box_length, n_ref=n_ref)
    h_profile = build_axis_spacing_profile(
        ref_axis,
        atom_positions_1d,
        h_min,
        h_max,
        r_core,
        stretch_beta,
        stretch_rule=stretch_rule,
    )
    n_intervals = _estimate_interval_count(ref_axis, h_profile)
    axis = _redistribute_axis(ref_axis, h_profile, n_intervals)
    weights = compute_axis_weights(axis)
    diffs = axis[1:] - axis[:-1]

    meta = {
        "box_length": box_length,
        "h_min": h_min,
        "h_max": h_max,
        "r_core": float(r_core),
        "stretch_beta": float(stretch_beta),
        "stretch_rule": stretch_rule,
        "n_ref": int(n_ref),
        "n_intervals": n_intervals,
        "n_nodes": int(axis.size),
        "is_uniform_degenerate": False,
        "actual_min_spacing": float(jnp.min(diffs)),
        "actual_max_spacing": float(jnp.max(diffs)),
        "actual_spacing": None,
    }
    return axis, weights, meta


def create_adaptive_grid(
    box_size: Array,
    atom_coords: Array,
    h_min: float,
    h_max: float,
    r_core: float,
    stretch_beta: float,
    *,
    n_ref: int = 4097,
    stretch_rule: str = "gaussian_sum",
    uniform_tol: float = 1e-8,
) -> AdaptiveTensorGrid:
    """Assemble a minimal 3D adaptive tensor grid state from 1D axes."""
    box_size = jnp.asarray(box_size, dtype=jnp.float32).reshape(-1)
    atom_coords = jnp.asarray(atom_coords, dtype=jnp.float32)

    if box_size.size != 3:
        raise ValueError(f"box_size must contain three entries, got shape {box_size.shape}")
    if bool(jnp.any(box_size <= 0.0)):
        raise ValueError(f"box_size entries must be positive, got {box_size}")
    if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
        raise ValueError(f"atom_coords must have shape (n_atoms, 3), got {atom_coords.shape}")

    x, wx, x_meta = create_adaptive_axis(
        float(box_size[0]),
        atom_coords[:, 0],
        h_min,
        h_max,
        r_core,
        stretch_beta,
        n_ref=n_ref,
        stretch_rule=stretch_rule,
        uniform_tol=uniform_tol,
    )
    y, wy, y_meta = create_adaptive_axis(
        float(box_size[1]),
        atom_coords[:, 1],
        h_min,
        h_max,
        r_core,
        stretch_beta,
        n_ref=n_ref,
        stretch_rule=stretch_rule,
        uniform_tol=uniform_tol,
    )
    z, wz, z_meta = create_adaptive_axis(
        float(box_size[2]),
        atom_coords[:, 2],
        h_min,
        h_max,
        r_core,
        stretch_beta,
        n_ref=n_ref,
        stretch_rule=stretch_rule,
        uniform_tol=uniform_tol,
    )

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    coords = jnp.stack([X, Y, Z], axis=-1)
    volume_weights = build_volume_weights(wx, wy, wz)

    grid = AdaptiveTensorGrid()
    grid.x = x
    grid.y = y
    grid.z = z
    grid.wx = wx
    grid.wy = wy
    grid.wz = wz
    grid.volume_weights = volume_weights
    grid.coords = coords
    grid.shape = coords.shape[:-1]
    grid.mask = jnp.ones(grid.shape, dtype=jnp.float32)
    grid.box_size = box_size.reshape(3)
    grid.backend_name = "adaptive_tensor"
    grid.projectors = []
    grid.h_min = float(h_min)
    grid.h_max = float(h_max)
    grid.r_core = float(r_core)
    grid.stretch_beta = float(stretch_beta)
    grid.stretch_rule = stretch_rule
    grid.requested_box_size = grid.box_size
    grid.n_ref = int(n_ref)
    grid.n_intervals = jnp.asarray([x.size - 1, y.size - 1, z.size - 1], dtype=jnp.int32)
    grid.x_meta = x_meta
    grid.y_meta = y_meta
    grid.z_meta = z_meta
    return grid


__all__ = [
    "AdaptiveTensorGrid",
    "make_reference_axis",
    "build_axis_spacing_profile",
    "create_adaptive_axis",
    "create_adaptive_grid",
    "compute_axis_weights",
    "build_volume_weights",
]
