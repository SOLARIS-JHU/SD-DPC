"""Sparse-dictionary building blocks: the candidate library and the SINDy model.

These are the shared representation used for *both* the identified dynamics model
and the sparse feedback policy. The implementation is self-contained and supports
the checkpoints included with this repository.
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
