"""Sparse-dictionary building blocks: the candidate library and the SINDy model.

These are the shared representation used for *both* the identified dynamics model
and the sparse feedback policy (Sec. 3 of the paper). The classes are a faithful copy
of ``SparseDPC.vectorized.sindy`` so the framework is self-contained; checkpoints saved
by either package are interchangeable.
"""
from .library import CompiledFunctionLibrary, LibraryPlan
from .model import SINDyVectorized
from .io import save_model, load_model, library_cfg

__all__ = [
    "CompiledFunctionLibrary",
    "LibraryPlan",
    "SINDyVectorized",
    "save_model",
    "load_model",
    "library_cfg",
]
