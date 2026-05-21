"""Phase 4 / Final: Generate all publication figures.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --results-dir /path/to/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd


sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")

FIGSIZE = (6, 4)
DPI = 200
FONTSIZE = 12
plt.rcParams.update({
    "font.size": FONTSIZE,
    "font.family": "sans-serif",
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
})


def load_phase3_data(results_dir: Path) -> pd.DataFrame:
    """Load all phase3 meta parquet files into a single DataFrame."""
    parquets = list(results_dir.glob("phase3_meta_*.parquet"))
    if not parquets:
        return pd.DataFrame()
    dfs = []
    for p in parquets:
        try:
            import pyarrow.parquet as pq
            df = pq.read_table(str(p)).to_pandas()
            dfs.append(df)
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def fig01_main_pareto(phase3_df: pd.DataFrame, figures_dir: Path) -> None:
    """Main success–latency Pareto figure (spec §8.2)."""
    tasks = phase3_df["task_id"].unique() if len(phase3_df) > 0 else ["peg_insertion"]
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, max(n_tasks, 1), figsize=(6 * max(n_tasks, 1), 4), squeeze=False)

    for ax_idx, task in enumerate(tasks):
        ax = axes[0][ax_idx]
        task_df = phase3_df[phase3_df["task_id"] == task] if len(phase3_df) > 0 else pd.DataFrame()

        if len(task_df) == 0:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(task)
            continue

        colors = sns.color_palette("colorblind")

        # C1 and C2 as single points
        for cond, color, marker in [("C1_baseline", colors[0], "o"), ("C2_always_reasoning", colors[1], "s")]:
            cond_df = task_df[task_df["condition"] == cond]
            if len(cond_df) == 0:
                continue
            sr = cond_df["success"].mean()
            lat = cond_df["mean_latency_ms_per_step"].mean()
            ax.errorbar(lat, sr, fmt=marker, color=color, label=cond, markersize=10, capsize=4)

        # C3 as a curve (varying tau)
        c3_rows = []
        for cond in task_df["condition"].unique():
            if "C3" in str(cond) and "tau" in str(cond):
                cond_df = task_df[task_df["condition"] == cond]
                sr = cond_df["success"].mean()
                lat = cond_df["mean_latency_ms_per_step"].mean()
                try:
                    tau_val = float(str(cond).split("tau")[-1])
                except Exception:  # noqa: BLE001
                    tau_val = -1.0
                c3_rows.append({"tau": tau_val, "sr": sr, "lat": lat})

        if c3_rows:
            c3_df = pd.DataFrame(c3_rows).sort_values("tau")
            ax.plot(c3_df["lat"], c3_df["sr"], "D--", color=colors[2],
                    label="C3_risk_gated", markersize=7)
            for _, row in c3_df.iterrows():
                ax.annotate(f"τ={row['tau']:.1f}", (row["lat"], row["sr"]),
                            textcoords="offset points", xytext=(5, 3), fontsize=8)

        ax.set_xlabel("Mean wall-clock latency per step (ms)")
        ax.set_ylabel("Success rate")
        ax.set_title(task)
        ax.legend(fontsize=9)
        ax.set_xscale("log")

    fig.suptitle("Success–Latency Pareto Frontier", fontsize=FONTSIZE + 1)
    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig01_main_pareto")


def fig02_detector_roc(results_dir: Path, figures_dir: Path) -> None:
    """Detector ROC curve per task."""
    metrics_path = results_dir / "phase2_metrics.json"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if not metrics_path.exists():
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    else:
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        colors = sns.color_palette("colorblind")
        for i, (key, val) in enumerate(metrics.items()):
            best_auroc = val.get("best_val_auroc", 0.0)
            ax.annotate(f"{key}: AUROC={best_auroc:.3f}", (0.05, 0.9 - i * 0.1),
                        xycoords="axes fraction", fontsize=10)
        ax.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Detector ROC Curve")
        ax.legend()

    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig02_detector_roc")


def fig03_lead_time_dist(results_dir: Path, figures_dir: Path) -> None:
    """Lead time distribution as boxplot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.text(0.5, 0.5,
            "Compute lead times from step-level predictions\n(run after phase2_train_detector.py)",
            ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
    ax.set_title("Lead Time Distribution (steps before failure)")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig03_lead_time_dist")


def fig04_calibration(results_dir: Path, figures_dir: Path) -> None:
    """Detector calibration plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.text(0.5, 0.5,
            "Compute from val predictions\n(run after phase2_train_detector.py)",
            ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Detector Calibration")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig04_calibration")


def fig05_ablation_features(results_dir: Path, figures_dir: Path) -> None:
    """Feature ablation bar chart."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    metrics_path = results_dir / "phase2_metrics.json"

    ablation_names = ["mlp_hybrid", "mlp_hidden_only", "mlp_action_only", "mlp_proprio_only"]
    auroc_vals = []

    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        for name in ablation_names:
            for key, val in metrics.items():
                if name in key:
                    auroc_vals.append(val.get("best_val_auroc", 0.0))
                    break
            else:
                auroc_vals.append(0.0)
    else:
        auroc_vals = [0.0] * len(ablation_names)

    colors = sns.color_palette("colorblind", len(ablation_names))
    ax.bar(ablation_names, auroc_vals, color=colors)
    ax.axhline(0.75, color="red", linestyle="--", label="Target AUROC=0.75")
    ax.set_xlabel("Detector variant")
    ax.set_ylabel("Val AUROC")
    ax.set_title("Detector Feature Ablation")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig05_ablation_features")


def fig06_primitive_history(phase3_df: pd.DataFrame, figures_dir: Path) -> None:
    """Primitive invocation counts per condition."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if len(phase3_df) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    else:
        conds = phase3_df["condition"].unique()
        x = np.arange(len(conds))
        reasoner_calls = [
            phase3_df[phase3_df["condition"] == c]["reasoner_calls"].mean()
            if "reasoner_calls" in phase3_df.columns else 0.0
            for c in conds
        ]
        ax.bar(x, reasoner_calls, color=sns.color_palette("colorblind", len(conds)))
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=20, ha="right")
        ax.set_ylabel("Mean reasoner calls per episode")
        ax.set_title("Reasoner/Primitive Invocations per Condition")

    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig06_primitive_history")


def _save_fig(fig, base_path: Path) -> None:
    for ext in ("pdf", "png"):
        out = base_path.with_suffix(f".{ext}")
        fig.savefig(str(out), dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate all experiment figures.")
    parser.add_argument("--results-dir", default=None,
                        help="Path to results directory (default: $DRIVE_ROOT/results)")
    parser.add_argument("--figures-dir", default=None,
                        help="Path to figures output directory")
    args = parser.parse_args()

    from src.utils.colab import get_drive_root
    drive_root = Path(get_drive_root())

    results_dir = Path(args.results_dir) if args.results_dir else drive_root / "results"
    figures_dir = Path(args.figures_dir) if args.figures_dir else drive_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"Figures dir: {figures_dir}")

    phase3_df = load_phase3_data(results_dir)
    print(f"Loaded {len(phase3_df)} phase3 records.")

    fig01_main_pareto(phase3_df, figures_dir)
    fig02_detector_roc(results_dir, figures_dir)
    fig03_lead_time_dist(results_dir, figures_dir)
    fig04_calibration(results_dir, figures_dir)
    fig05_ablation_features(results_dir, figures_dir)
    fig06_primitive_history(phase3_df, figures_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
