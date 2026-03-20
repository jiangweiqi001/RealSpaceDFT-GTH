"""Adaptive grid helpers for future tensor-product backends."""

from .adaptive_tensor import (
    AdaptiveTensorGrid,
    build_axis_spacing_profile,
    build_volume_weights,
    compute_axis_weights,
    create_adaptive_axis,
    create_adaptive_grid,
    laplacian_nonuniform_3d,
    make_reference_axis,
    second_derivative_nonuniform_1d,
)

__all__ = [
    "AdaptiveTensorGrid",
    "make_reference_axis",
    "build_axis_spacing_profile",
    "create_adaptive_axis",
    "create_adaptive_grid",
    "compute_axis_weights",
    "build_volume_weights",
    "second_derivative_nonuniform_1d",
    "laplacian_nonuniform_3d",
]
