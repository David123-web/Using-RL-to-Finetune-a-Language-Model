import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    if window <= 1:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_stats(stats_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if not stats:
        print("No training stats found.")
        return

    updates = list(range(1, len(stats) + 1))

    def series(key: str, default: float = 0.0):
        return [float(item.get(key, default)) for item in stats]

    reward_centered = series("reward")
    reward_raw = series("raw_reward")
    policy_loss = series("policy_loss")
    value_loss = series("value_loss")
    entropy = series("entropy")
    kl_div = series("kl_div")
    kl_coef = series("kl_coef")
    quality_anchor = series("quality_anchor")
    sentiment = series("sentiment")
    completion_len = series("avg_completion_length_words")
    eos_rate = series("eos_rate")
    reward_quality_gap = series("reward_quality_gap")

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("PPO Training Diagnostics", fontsize=16)

    ax = axes[0, 0]
    ax.plot(updates, reward_centered, color="#2ca02c", linewidth=1.8, label="Centered Reward")
    ax.plot(updates, reward_raw, color="#006d2c", linewidth=1.6, alpha=0.8, label="Raw Reward")
    ax.set_title("Reward")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(updates, sentiment, color="#1f77b4", linewidth=1.8, label="Sentiment")
    ax.plot(updates, quality_anchor, color="#17becf", linewidth=1.8, label="Quality Anchor")
    ax.set_title("Reward Components")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(updates, completion_len, color="#ff7f0e", linewidth=1.8, label="Completion Length")
    ax.plot(updates, eos_rate, color="#d62728", linewidth=1.8, label="EOS Rate")
    ax.set_title("Collapse Signals")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(updates, policy_loss, color="#1f77b4", linewidth=1.8)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(updates, value_loss, color="#ff7f0e", linewidth=1.8)
    ax.set_title("Value Loss")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(updates, entropy, color="#9467bd", linewidth=1.8)
    ax.set_title("Entropy")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(updates, kl_div, color="#d62728", linewidth=1.8)
    ax.set_title("Approx KL")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(updates, kl_coef, color="#8c564b", linewidth=1.8)
    ax.set_title("KL Coefficient")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    window = min(30, max(2, len(reward_quality_gap) // 8))
    if len(reward_quality_gap) >= window:
        smoothed = _moving_average(reward_quality_gap, window)
        smoothed_updates = updates[window - 1 :]
        ax.plot(updates, reward_quality_gap, color="#9edae5", alpha=0.4, label="Raw")
        ax.plot(smoothed_updates, smoothed, color="#17becf", linewidth=2.0, label="Smoothed")
        ax.legend()
    else:
        ax.plot(updates, reward_quality_gap, color="#17becf", linewidth=1.8)
    ax.set_title("Reward-Quality Gap")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_stats.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training plot to {output_path}")


def _extract_mode_summary(comparison: Dict[str, Any], mode: str) -> Dict[str, Any]:
    summary = comparison.get("summary", {})
    if mode in summary:
        return summary[mode]

    # Backward compatibility: old schema was summary -> model
    if summary and isinstance(next(iter(summary.values())), dict) and "mean_reward" in next(iter(summary.values())):
        return summary

    return {}


def plot_comparison(comparison_path: str, output_dir: str = "plots", mode: str = "sampling"):
    os.makedirs(output_dir, exist_ok=True)

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison = json.load(f)

    summary = _extract_mode_summary(comparison, mode)
    if not summary:
        print(f"No summary found for mode '{mode}' in {comparison_path}")
        return

    models = list(summary.keys())
    metrics = {
        "mean_raw_reward": "Mean Raw Reward",
        "mean_quality_anchor": "Quality Anchor",
        "mean_length": "Mean Completion Length",
        "eos_rate": "EOS Rate",
        "diversity_ratio": "Diversity Ratio",
        "composite_score": "Composite Score",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle(f"Model Comparison ({mode})", fontsize=16)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        values = [float(summary[m].get(metric_key, 0.0)) for m in models]
        bars = axes[idx].bar(models, values, color=colors[: len(models)])
        axes[idx].set_title(metric_name)
        axes[idx].grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def _reward_distribution_from_results(results: Dict[str, Any], mode: str) -> List[float]:
    if mode in results:
        return list(results[mode].get("distributions", {}).get("reward", []))

    # Backward compatibility: old schema stored examples only.
    if "examples" in results:
        return [float(example.get("reward", 0.0)) for example in results["examples"]]

    return []


def plot_reward_distribution(
    results_paths: Dict[str, str],
    output_dir: str = "plots",
    mode: str = "sampling",
):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results_paths), figsize=(6 * len(results_paths), 5))
    if len(results_paths) == 1:
        axes = [axes]

    fig.suptitle(f"Reward Distributions ({mode})", fontsize=16)

    for idx, (model_name, path) in enumerate(results_paths.items()):
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)

        rewards = _reward_distribution_from_results(results, mode)
        if not rewards:
            axes[idx].set_title(f"{model_name.upper()} (no data)")
            axes[idx].axis("off")
            continue

        axes[idx].hist(rewards, bins=30, alpha=0.75, color="#2ca02c", edgecolor="black")
        axes[idx].set_xlabel("Reward")
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{model_name.upper()} Model")
        mean_val = float(np.mean(rewards))
        axes[idx].axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "reward_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved reward distribution plot to {output_path}")


def generate_all_plots(
    training_stats_path: str = None,
    comparison_path: str = None,
    results_paths: Dict[str, str] = None,
    output_dir: str = "plots",
    mode: str = "sampling",
):
    print("Generating plots...")

    if training_stats_path and os.path.exists(training_stats_path):
        print("Plotting training statistics...")
        plot_training_stats(training_stats_path, output_dir)

    if comparison_path and os.path.exists(comparison_path):
        print("Plotting model comparison...")
        plot_comparison(comparison_path, output_dir, mode=mode)

    if results_paths:
        print("Plotting reward distributions...")
        existing_paths = {k: v for k, v in results_paths.items() if os.path.exists(v)}
        if existing_paths:
            plot_reward_distribution(existing_paths, output_dir, mode=mode)

    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument("--training_stats", type=str, help="Path to training stats JSON")
    parser.add_argument("--comparison", type=str, help="Path to comparison JSON")
    parser.add_argument("--base_results", type=str, help="Path to base model results")
    parser.add_argument("--sft_results", type=str, help="Path to SFT model results")
    parser.add_argument("--ppo_results", type=str, help="Path to PPO model results")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    parser.add_argument("--mode", type=str, default="sampling", help="Evaluation mode to plot")

    args = parser.parse_args()

    results_paths = {}
    if args.base_results:
        results_paths["base"] = args.base_results
    if args.sft_results:
        results_paths["sft"] = args.sft_results
    if args.ppo_results:
        results_paths["ppo"] = args.ppo_results

    generate_all_plots(
        training_stats_path=args.training_stats,
        comparison_path=args.comparison,
        results_paths=results_paths if results_paths else None,
        output_dir=args.output,
        mode=args.mode,
    )
