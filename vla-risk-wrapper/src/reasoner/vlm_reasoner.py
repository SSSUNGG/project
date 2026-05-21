from __future__ import annotations

import json
import re
import time
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from src.reasoner.base import BaseReasoner, ReasonerOutput, VALID_PRIMITIVES
from src.reasoner.prompts import SYSTEM_PROMPT, build_user_message


class VLMReasoner(BaseReasoner):
    """Vision-Language Model reasoner using Qwen2.5-VL or LLaVA-NeXT.

    Parse failures silently fall back to 'continue'. A per-session counter
    tracks how many parse failures occurred.

    Requires:
        cfg.hf_id  — Hugging Face model ID
        cfg.max_new_tokens
        cfg.temperature
        cfg.do_sample
    """

    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self._parse_fail_count = 0
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        import warnings
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            hf_id = self.cfg.get("hf_id", "Qwen/Qwen2.5-VL-3B-Instruct")
            device_map = "auto"
            self._processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                hf_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"VLMReasoner could not load model ({exc}). "
                "Running in fallback-only mode (always returns 'continue').",
                stacklevel=2,
            )
            self._loaded = False

    def diagnose(
        self,
        rgb: np.ndarray,
        instruction: str,
        recent_actions: np.ndarray,
        risk_score: float,
    ) -> ReasonerOutput:
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if not self._loaded:
            result = "continue"
            rationale = "vlm_not_loaded"
        else:
            result, rationale = self._call_vlm(rgb, instruction, recent_actions, risk_score)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        return ReasonerOutput(
            primitive_id=result,
            rationale=rationale,
            latency_ms=latency_ms,
        )

    def _call_vlm(self, rgb, instruction, recent_actions, risk_score) -> tuple[str, str]:
        from PIL import Image
        import torch

        user_msg = build_user_message(instruction, recent_actions, risk_score)
        pil_image = Image.fromarray(rgb.astype(np.uint8))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_msg},
                ],
            },
        ]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg.get("max_new_tokens", 128)),
                    do_sample=bool(self.cfg.get("do_sample", False)),
                    temperature=float(self.cfg.get("temperature", 0.0)) or None,
                )

            generated = self._processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            return self._parse_output(generated)

        except Exception as exc:  # noqa: BLE001
            self._parse_fail_count += 1
            import logging
            logging.getLogger(__name__).warning(
                "VLM call failed: %s. Falling back to 'continue'. "
                "Total failures: %d", exc, self._parse_fail_count
            )
            return "continue", f"exception: {type(exc).__name__}"

    def _parse_output(self, text: str) -> tuple[str, str]:
        """Extract primitive_id and rationale from VLM JSON output."""
        # Try to find first JSON object in the output
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                pid = str(obj.get("primitive", "continue")).strip().lower()
                rationale = str(obj.get("rationale", ""))
                if pid in VALID_PRIMITIVES:
                    return pid, rationale
            except json.JSONDecodeError:
                pass

        self._parse_fail_count += 1
        import logging
        logging.getLogger(__name__).warning(
            "VLM output parse failed (count=%d). Output: %r. Falling back to 'continue'.",
            self._parse_fail_count, text[:200],
        )
        return "continue", "parse_fail"

    @property
    def parse_fail_count(self) -> int:
        return self._parse_fail_count


class LLaVANextReasoner(VLMReasoner):
    """LLaVA-NeXT 7B reasoner — for ablation only. NOT used in main experiments."""

    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        raise NotImplementedError(
            "LLaVANextReasoner is reserved for ablation experiments only. "
            "Use VLMReasoner with qwen25_vl_3b config for main experiments."
        )
