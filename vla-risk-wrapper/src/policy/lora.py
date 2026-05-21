from __future__ import annotations

from omegaconf import DictConfig


def apply_lora(model, cfg: DictConfig):
    """Wrap a transformer model with LoRA via peft.

    If the model is not PEFT-compatible (e.g., Octo uses JAX/Flax),
    this function is a no-op and logs a warning.

    Args:
        model: A PyTorch nn.Module.
        cfg:   The lora sub-config with keys: r, alpha, dropout, target_modules.

    Returns:
        The wrapped model (or the original if LoRA is not applicable).
    """
    if not cfg.get("enabled", False):
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            r=cfg.get("r", 16),
            lora_alpha=cfg.get("alpha", 32),
            lora_dropout=cfg.get("dropout", 0.05),
            target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        return get_peft_model(model, lora_cfg)
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"LoRA wrapping failed ({exc}). Continuing without LoRA.",
            stacklevel=2,
        )
        return model
