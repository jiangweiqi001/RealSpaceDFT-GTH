"""Backend interfaces for JaxDFT.

Patch 2 exports a concrete ``UniformBackend`` wrapper, but existing runtime
code paths remain unchanged until a later patch explicitly wires a backend
into the solver.
"""

from .base import ArrayLike, Backend, BackendState, NonlocalCache
from .uniform import UniformBackend

__all__ = [
    "ArrayLike",
    "Backend",
    "BackendState",
    "NonlocalCache",
    "UniformBackend",
]
