"""Colab environment utilities."""
from __future__ import annotations

import os
from pathlib import Path


DRIVE_ROOT_ENV_VAR = "DRIVE_ROOT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/vla-risk-wrapper"


def get_drive_root() -> str:
    """Return DRIVE_ROOT from environment, or default."""
    return os.environ.get(DRIVE_ROOT_ENV_VAR, DEFAULT_DRIVE_ROOT)


def mount_drive() -> bool:
    """Mount Google Drive if running in Colab. Returns True if successful."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return True
    except ImportError:
        return False


def ensure_drive_dirs() -> None:
    """Create the standard Drive directory layout if it doesn't exist."""
    root = Path(get_drive_root())
    for sub in ("data/steps", "data/meta", "checkpoints", "results", "figures"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
