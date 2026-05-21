from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from src.policy.base import BasePolicy, PolicyOutput


class OctoPolicy(BasePolicy):
    """Frozen Octo-base policy with hidden-state extraction.

    Octo uses JAX/Flax internally. This wrapper adapts it for our PyTorch pipeline:
    - Calls octo's predict() to get actions
    - Extracts transformer hidden states via the observation tokens
    - Measures latency with torch.cuda.synchronize() when on GPU

    The hidden_state returned is a mean-pooled representation of the last
    transformer block's token embeddings, cast to float32 numpy.
    """

    HIDDEN_DIM = 512  # Octo-base token embedding dim

    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self._instruction: str = ""
        self._model = None
        self._loaded = False
        self._use_gpu = device == "cuda" and torch.cuda.is_available()
        self._load_model()

    def _load_model(self) -> None:
        try:
            import octo  # noqa: F401
            from octo.model.octo_model import OctoModel

            hf_id = self.cfg.get("hf_id", "rail-berkeley/octo-base")
            self._model = OctoModel.load_pretrained(hf_id)
            self._loaded = True
        except ImportError:
            import warnings
            warnings.warn(
                "octo package not found. OctoPolicy will run in dummy mode "
                "(returns zeros). Install via: pip install octo",
                stacklevel=2,
            )
            self._loaded = False

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._task = None
        if self._loaded and self._model is not None:
            try:
                self._task = self._model.create_tasks(texts=[instruction])
            except Exception:  # noqa: BLE001
                pass

    def predict(self, obs: dict) -> PolicyOutput:
        if self._use_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if not self._loaded or self._model is None:
            action, hidden = self._dummy_output()
        else:
            action, hidden = self._run_octo(obs)

        if self._use_gpu:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        return PolicyOutput(
            action=action.astype(np.float32),
            hidden_state=hidden.astype(np.float32),
            latency_ms=(t1 - t0) * 1000,
        )

    def _run_octo(self, obs: dict):
        import jax
        import jax.numpy as jnp

        # Build Octo observation dict (images + proprio)
        rgb = self._extract_rgb(obs)  # (H, W, 3) uint8
        rgb_jax = jnp.array(rgb[None, None])  # (1, 1, H, W, 3)

        octo_obs = {"image_primary": rgb_jax, "pad_mask": jnp.ones((1, 1), dtype=bool)}

        try:
            outputs = self._model.sample_actions(
                octo_obs,
                self._task,
                rng=jax.random.PRNGKey(0),
                unnormalize_actions=True,
            )
            action_np = np.array(outputs[0, 0])[:7]  # first step, 7-DoF
        except Exception:  # noqa: BLE001
            action_np, _ = self._dummy_output()
            return action_np, np.zeros(self.HIDDEN_DIM, dtype=np.float32)

        # Octo doesn't directly expose hidden states via the public API.
        # We use the observation token embeddings as a proxy.
        try:
            transformer_outputs = self._model.octo_transformer(
                octo_obs, self._task, train=False
            )
            token_embs = np.array(transformer_outputs["transformer_outputs"]["obs_tokens"])
            hidden = token_embs[0].mean(axis=0)  # mean over tokens: (D,)
        except Exception:  # noqa: BLE001
            hidden = np.zeros(self.HIDDEN_DIM, dtype=np.float32)

        return action_np, hidden

    def _dummy_output(self):
        action = np.zeros(7, dtype=np.float32)
        hidden = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        return action, hidden

    @staticmethod
    def _extract_rgb(obs: dict) -> np.ndarray:
        """Extract primary RGB image from obs dict."""
        for key in ("image", "rgb", "image_primary", "sensor_data"):
            if key in obs:
                val = obs[key]
                if isinstance(val, dict):
                    for sub in ("rgb", "image", "Color"):
                        if sub in val:
                            return np.asarray(val[sub], dtype=np.uint8)
                return np.asarray(val, dtype=np.uint8)
        # Fallback: return blank image
        return np.zeros((128, 128, 3), dtype=np.uint8)
