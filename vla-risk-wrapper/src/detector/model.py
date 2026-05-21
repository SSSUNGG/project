from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.detector.features import DetectorConfig


class RiskDetector(nn.Module):
    """MLP-based risk detector.

    Accepts VLA hidden states, action history, and proprioception as input.
    Outputs a scalar risk logit (sigmoid gives probability).
    Temperature scaling is used for calibration (stored as a non-learnable buffer).
    """

    def __init__(self, cfg: DetectorConfig, model_cfg=None):
        super().__init__()
        self.cfg = cfg

        # Optional projection for VLA hidden states (high-dim → proj_dim)
        if cfg.use_vla_hidden:
            self.vla_proj = nn.Linear(cfg.vla_hidden_dim, cfg.vla_proj_dim)
        else:
            self.vla_proj = None

        # Compute input dimension after projection
        in_dim = 0
        if cfg.use_vla_hidden:
            in_dim += cfg.vla_proj_dim
        if cfg.use_action_history:
            in_dim += cfg.action_history_K * cfg.action_dim
        if cfg.use_proprio:
            in_dim += cfg.proprio_dim

        hidden_dim = 256 if model_cfg is None else model_cfg.get("hidden_dim", 256)
        num_layers = 3 if model_cfg is None else model_cfg.get("num_layers", 3)
        dropout = 0.1 if model_cfg is None else model_cfg.get("dropout", 0.1)

        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                out_dim = max(hidden_dim // (2 ** i), 64)
            else:
                out_dim = hidden_dim
            layers += [
                nn.Linear(current_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            current_dim = out_dim

        layers.append(nn.Linear(current_dim, 1))
        self.head = nn.Sequential(*layers)

        # Temperature for calibration (1.0 = no scaling, requires_grad=False until calibration)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        vla_h: Optional[torch.Tensor],
        action_hist: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass. Returns raw logits, shape (B,)."""
        feats = []

        if self.vla_proj is not None and vla_h is not None:
            feats.append(self.vla_proj(vla_h))

        if action_hist is not None and self.cfg.use_action_history:
            feats.append(action_hist.flatten(start_dim=1))

        if proprio is not None and self.cfg.use_proprio:
            feats.append(proprio)

        if not feats:
            raise ValueError("All input streams are None / disabled.")

        x = torch.cat(feats, dim=-1)
        return self.head(x).squeeze(-1)

    def predict_proba(
        self,
        vla_h: Optional[torch.Tensor],
        action_hist: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return calibrated risk probabilities in [0, 1], shape (B,)."""
        logits = self.forward(vla_h, action_hist, proprio)
        return torch.sigmoid(logits / self.temperature.clamp(min=1e-3))

    def predict_proba_numpy(
        self,
        vla_h_np: Optional,
        action_hist_np: Optional,
        proprio_np: Optional,
        device: str = "cpu",
    ) -> float:
        """Convenience wrapper for single-step inference from numpy inputs."""
        self.eval()
        with torch.no_grad():
            def _to_tensor(x, dtype=torch.float32):
                if x is None:
                    return None
                return torch.tensor(x, dtype=dtype, device=device).unsqueeze(0)

            vla_h = _to_tensor(vla_h_np)
            ah = _to_tensor(action_hist_np)
            if ah is not None and ah.ndim == 2:
                ah = ah.unsqueeze(0)  # (1, K, 7)
            prop = _to_tensor(proprio_np)

            prob = self.predict_proba(vla_h, ah, prop)
            return float(prob.item())
