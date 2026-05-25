import os
import warnings


def resolve_mode() -> str:
    """RG_MODE=A|B (default A). B인데 GPU 없으면 A로 폴백."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    mode = os.environ.get("RG_MODE", "A").upper()
    if mode == "B" and not has_gpu:
        warnings.warn("RG_MODE=B 요청됐으나 GPU 없음 → Medium-A로 폴백", stacklevel=2)
        return "A"
    return mode


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
