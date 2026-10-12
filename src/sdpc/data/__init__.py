"""Dataset generation for system-ID and policy training, plus reference-signal builders."""
from .generation import (
    get_data,
    get_box_policy_data,
    get_obstacle_policy_data,
    get_data_discrete,
)
from .reference import (
    make_constant_reference,
    make_piecewise_reference,
    make_equal_piecewise_reference,
    make_signal_reference,
    build_reference,
)

__all__ = [
    "get_data",
    "get_box_policy_data",
    "get_obstacle_policy_data",
    "get_data_discrete",
    "make_constant_reference",
    "make_piecewise_reference",
    "make_equal_piecewise_reference",
    "make_signal_reference",
    "build_reference",
]
