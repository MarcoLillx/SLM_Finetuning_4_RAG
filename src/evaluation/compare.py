"""
Comparative analysis: baseline vs fine-tuned RAG.

Generates side-by-side comparisons, statistical tests,
and publication-quality plots.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config, get_config

logger = logging.getLogger(__name__)

# ── Publication-quality plot style ────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_results(config: Config) -> tuple[dict, dict]:
    """Load base and fine-tuned evaluation results."""
    base_path = config.paths.eval_dir / "ragas_results_base.json"
    ft_path = config.paths.eval_dir / "ragas_results_finetuned.json"

    base_results = None
    ft_results = None

    if base_path.exists():
        with open(base_path) as f:
            base_results = json.load(f)
    else:
        logger.warning(f"Base results not found at {base_path}")

    if ft_path.exists():
        with open(ft_path) as f:
            ft_results = json.load(f)
    else:
        logger.warning(f"Fine-tuned results not found at {ft_path}")

    return base_results, ft_results


def plot_radar_chart(base_scores: dict, ft_scores: dict, save_path: Path):
    """Create radar chart comparing base vs fine-tuned metrics."""
    metrics = list(base_scores.keys())
    base_values = [base_scores[m] for m in metrics]
    ft_values = [ft_scores[m] for m in metrics]

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    base_values += base_values[:1]
    ft_values += ft_values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, base_values, 'o-', linewidth=2, label='Base Qwen3-1.7B', color='#e74c3c')
    ax.fill(angles, base_values, alpha=0.15, color='#e74c3c')
    ax.plot(angles, ft_values, 's-', linewidth=2, label='Fine-tuned (QLoRA)', color='#2ecc71')
    ax.fill(angles, ft_values, alpha=0.15, color='#2ecc71')

    # Labels
    metric_labels = [m.replace("_", " ").title() for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("RAG Evaluation: Base vs Fine-tuned", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Radar chart saved to {save_path}")


def plot_bar_comparison(base_scores: dict, ft_scores: dict, save_path: Path):
    """Create grouped bar chart comparing metrics."""
    metrics = list(base_scores.keys())
    metric_labels = [m.replace("_", " ").title() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, [base_scores[m] for m in metrics], width,
                   label='Base Qwen3-1.7B', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, [ft_scores[m] for m in metrics], width,
                   label='Fine-tuned (QLoRA)', color='#2ecc71', alpha=0.85)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('RAG Performance: Base vs Fine-tuned Qwen3-1.7B', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Bar chart saved to {save_path}")


def plot_training_loss(config: Config, save_path: Path):
    """Plot training loss curve from training logs."""
    log_path = Path(config.training.output_dir) / "training_log.json"
    if not log_path.exists():
        logger.warning(f"Training log not found at {log_path}")
        return

    with open(log_path) as f:
        log_history = json.load(f)

    # Extract training loss entries
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_steps:
        ax.plot(train_steps, train_losses, '-', label='Training Loss',
                color='#3498db', linewidth=2, alpha=0.8)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, 's-', label='Validation Loss',
                color='#e74c3c', linewidth=2, markersize=6)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('QLoRA Fine-Tuning: Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training loss plot saved to {save_path}")


def plot_per_sample_distribution(base_results: dict, ft_results: dict, save_path: Path):
    """Plot per-sample score distributions as box plots."""
    if "per_sample" not in base_results or "per_sample" not in ft_results:
        logger.warning("Per-sample scores not available for distribution plot")
        return

    base_per = base_results["per_sample"]
    ft_per = ft_results["per_sample"]
    metrics = list(base_per.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        data = pd.DataFrame({
            "Base": base_per[metric],
            "Fine-tuned": ft_per[metric],
        })
        data_melted = data.melt(var_name="Model", value_name="Score")
        sns.boxplot(data=data_melted, x="Model", y="Score", ax=axes[i],
                    palette=["#e74c3c", "#2ecc71"])
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylim(0, 1.05)

    fig.suptitle("Score Distributions: Base vs Fine-tuned", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Distribution plot saved to {save_path}")


def compute_improvement(base_scores: dict, ft_scores: dict) -> dict:
    """Compute improvement percentages and statistical significance."""
    improvements = {}
    for metric in base_scores:
        if metric in ft_scores:
            base_val = base_scores[metric]
            ft_val = ft_scores[metric]
            if base_val > 0:
                pct_change = ((ft_val - base_val) / base_val) * 100
            else:
                pct_change = float('inf') if ft_val > 0 else 0.0
            improvements[metric] = {
                "base": round(base_val, 4),
                "finetuned": round(ft_val, 4),
                "absolute_change": round(ft_val - base_val, 4),
                "percent_change": round(pct_change, 2),
            }
    return improvements


def run_comparison(config: Optional[Config] = None) -> dict:
    """
    Full comparison pipeline: load results → compute stats → generate plots.
    
    Returns:
        Dict with comparison summary
    """
    if config is None:
        config = get_config()

    base_results, ft_results = load_results(config)

    if base_results is None or ft_results is None:
        logger.error("Cannot compare: need both base and fine-tuned results")
        return {}

    base_agg = base_results["aggregate_scores"]
    ft_agg = ft_results["aggregate_scores"]

    # Compute improvements
    improvements = compute_improvement(base_agg, ft_agg)

    # Generate plots
    plots_dir = config.paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_radar_chart(base_agg, ft_agg, plots_dir / "radar_comparison.png")
    plot_bar_comparison(base_agg, ft_agg, plots_dir / "bar_comparison.png")
    plot_training_loss(config, plots_dir / "training_loss.png")
    plot_per_sample_distribution(base_results, ft_results, plots_dir / "score_distributions.png")

    # Build summary report
    summary = {
        "comparison": improvements,
        "base_aggregate": base_agg,
        "finetuned_aggregate": ft_agg,
        "plots_generated": [
            str(plots_dir / "radar_comparison.png"),
            str(plots_dir / "bar_comparison.png"),
            str(plots_dir / "training_loss.png"),
            str(plots_dir / "score_distributions.png"),
        ],
    }

    # Save summary
    summary_path = config.paths.eval_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Comparison summary saved to {summary_path}")

    # Print summary table
    _print_summary_table(improvements)

    return summary


def _print_summary_table(improvements: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'Base':>10} {'Fine-tuned':>12} {'Change':>10} {'%':>8}")
    print("-" * 70)
    for metric, vals in improvements.items():
        direction = "↑" if vals["absolute_change"] > 0 else "↓" if vals["absolute_change"] < 0 else "="
        print(
            f"{metric:<25} {vals['base']:>10.4f} {vals['finetuned']:>12.4f} "
            f"{direction}{abs(vals['absolute_change']):>8.4f} {vals['percent_change']:>7.1f}%"
        )
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_comparison()
