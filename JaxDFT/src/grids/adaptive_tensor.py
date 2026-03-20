"""Adaptive tensor-product axis helpers for future nonuniform grids.

Patch A intentionally implements only 1D axis generation, nodal integration
weights, and tensor-product volume weights. It does not modify any SCF,
backend, or solver execution path.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


Array = jnp.ndarray


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


__all__ = [
    "make_reference_axis",
    "build_axis_spacing_profile",
    "create_adaptive_axis",
    "compute_axis_weights",
    "build_volume_weights",
]
