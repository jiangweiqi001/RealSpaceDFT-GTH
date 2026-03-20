"""Backend interfaces for JaxDFT.

Phase 1 only adds scaffolding here. Existing runtime code paths remain
unchanged until a later patch explicitly wires a concrete backend into the
solver.
"""

from .base import ArrayLike, Backend, BackendState, NonlocalCache

__all__ = [
    "ArrayLike",
    "Backend",
    "BackendState",
    "NonlocalCache",
]
