from __future__ import annotations

import numpy as np

from src.policy.base import BasePolicy, PolicyOutput


class OpenVLAPolicy(BasePolicy):
    """OpenVLA-7B policy — activated after Phase 2 stop criteria are met.

    Phase 1 & 2 use OctoPolicy. Switch to this by passing policy=openvla in the config.
    Requires: A100 GPU (≥24 GB VRAM), Hugging Face token with openvla/openvla-7b access.
    """

    HIDDEN_DIM = 4096  # LLaMA-style 7B model

    def __init__(self, cfg, device: str = "cuda"):
        raise NotImplementedError(
            "OpenVLA is reserved for after Phase 2 stop criteria are met. "
            "Use OctoPolicy (policy=octo) for Phase 1 and Phase 2. "
            "When ready, remove this guard and implement _load_model()."
        )

    def reset(self, instruction: str) -> None:  # pragma: no cover
        ...

    def predict(self, obs: dict) -> PolicyOutput:  # pragma: no cover
        ...
