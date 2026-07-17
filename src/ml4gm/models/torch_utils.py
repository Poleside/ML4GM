from __future__ import annotations

import random
from typing import Any

import numpy as np


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch support requires: pip install 'ml4gm[torch]'") from exc
    except OSError as exc:
        raise RuntimeError(f"PyTorch native runtime failed to load: {exc}") from exc
    return torch


def set_torch_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    use_deterministic_algorithms = getattr(torch, "use_deterministic_algorithms", None)
    if use_deterministic_algorithms is not None:
        use_deterministic_algorithms(True)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
