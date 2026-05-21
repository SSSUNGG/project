from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detector.features import DetectorConfig
from src.detector.model import RiskDetector


def train_detector(
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_id: str,
    output_dir: str,
    metrics_path: Optional[str] = None,
    device: str = "cuda",
) -> Path:
    """Train a RiskDetector and return the path to the best checkpoint.

    Args:
        cfg:          Hydra detector config (contains input/model/train sub-dicts).
        train_loader: DataLoader for training set.
        val_loader:   DataLoader for validation set.
        task_id:      Used to name the checkpoint file.
        output_dir:   Directory to save checkpoints.
        metrics_path: JSON file to append epoch metrics.
        device:       'cuda' or 'cpu'.

    Returns:
        Path to best checkpoint (.pt).
    """
    from sklearn.metrics import roc_auc_score

    det_cfg = DetectorConfig.from_omegaconf(cfg)
    model_cfg = cfg.model
    train_cfg = cfg.train

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = RiskDetector(det_cfg, model_cfg).to(device)

    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    pos_weight_val = float(train_cfg.get("pos_weight", 4.0))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    best_val_auroc = -1.0
    epochs_no_improve = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"detector_{task_id}_{cfg.name}.pt"

    all_metrics = []

    for epoch in range(epochs):
        model.train()
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, grad_clip, device, train=True)

        model.eval()
        val_loss, val_probs, val_labels = _run_eval(model, val_loader, criterion, device)

        try:
            val_auroc = float(roc_auc_score(val_labels, val_probs)) if len(np.unique(val_labels)) > 1 else 0.5
        except Exception:  # noqa: BLE001
            val_auroc = 0.5

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auroc": round(val_auroc, 6),
        }
        all_metrics.append(epoch_metrics)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_auroc": val_auroc},
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    if metrics_path is not None:
        _append_metrics(metrics_path, task_id, cfg.name, all_metrics, best_val_auroc)

    return ckpt_path


def _run_epoch(model, loader, optimizer, criterion, grad_clip, device, train: bool) -> float:
    total_loss = 0.0
    n_batches = 0
    model.train(train)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            # Split feature vector according to model config
            vla_h, ah, prop = _split_features(feats, model.cfg)
            logits = model.forward(vla_h, ah, prop)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def _run_eval(model, loader, criterion, device):
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            vla_h, ah, prop = _split_features(feats, model.cfg)
            logits = model.forward(vla_h, ah, prop)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    val_probs = np.concatenate(all_probs) if all_probs else np.array([])
    val_labels = np.concatenate(all_labels) if all_labels else np.array([])
    return val_loss, val_probs, val_labels


def _split_features(feats: torch.Tensor, cfg: DetectorConfig):
    """Split concatenated feature tensor into (vla_h, action_hist, proprio)."""
    idx = 0
    vla_h = None
    ah = None
    prop = None

    if cfg.use_vla_hidden:
        vla_h = feats[:, idx : idx + cfg.vla_hidden_dim]
        idx += cfg.vla_hidden_dim

    if cfg.use_action_history:
        ah_flat = feats[:, idx : idx + cfg.action_history_K * cfg.action_dim]
        ah = ah_flat.reshape(-1, cfg.action_history_K, cfg.action_dim)
        idx += cfg.action_history_K * cfg.action_dim

    if cfg.use_proprio:
        prop = feats[:, idx : idx + cfg.proprio_dim]

    return vla_h, ah, prop


def _append_metrics(path: str, task_id: str, model_name: str, epoch_metrics: list, best_auroc: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    key = f"{task_id}_{model_name}"
    existing[key] = {
        "best_val_auroc": best_auroc,
        "epochs": epoch_metrics,
    }

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
