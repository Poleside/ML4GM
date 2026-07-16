from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


def positive_integer(value: Any, model: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{model} {name} must be a positive integer")
    return int(value)


def finite_number(value: Any, model: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{model} {name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{model} {name} must be a finite number")
    return normalized
