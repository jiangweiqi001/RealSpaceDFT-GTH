"""Backend interfaces for JaxDFT.

This package exports concrete backend wrappers while keeping solver wiring under
explicit user-controlled milestones.
"""

from .adaptive import AdaptiveBackend
from .base import ArrayLike, Backend, BackendState, NonlocalCache
from .uniform import UniformBackend

__all__ = [
    "ArrayLike",
    "Backend",
    "BackendState",
    "NonlocalCache",
    "UniformBackend",
    "AdaptiveBackend",
]
