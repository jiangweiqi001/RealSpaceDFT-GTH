"""Adaptive grid helpers for future tensor-product backends."""

from .adaptive_tensor import (
    build_axis_spacing_profile,
    build_volume_weights,
    compute_axis_weights,
    create_adaptive_axis,
    make_reference_axis,
)

__all__ = [
    "make_reference_axis",
    "build_axis_spacing_profile",
    "create_adaptive_axis",
    "compute_axis_weights",
    "build_volume_weights",
]
