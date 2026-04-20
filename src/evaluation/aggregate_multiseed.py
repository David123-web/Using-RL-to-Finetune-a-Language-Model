import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95": 0.0,
        }

    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
    else:
        var = 0.0
    std = math.sqrt(max(var, 0.0))
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0

    return {
        "n": n,
        "mean": float(mean),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95": float(ci95),
    }


def extract_model_mode_metrics(model_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    # New schema: model_result[mode] -> mode_result
    if model_result and all(isinstance(v, dict) and "reward" in v for v in model_result.values()):
        mode_metrics = {}
        for mode_name, mode_res in model_result.items():
            mode_metrics[mode_name] = {
                "mean_reward": float(mode_res["reward"]["mean"]),
                "mean_raw_reward": float(mode_res["raw_reward"]["mean"]),
                "mean_sentiment": float(mode_res["components"]["sentiment"]["mean"]),
                "mean_quality_anchor": float(mode_res["components"]["quality_anchor"]["mean"]),
                "mean_repetition_token": float(mode_res["components"]["repetition_token"]["mean"]),
                "mean_length": float(mode_res["length"]["mean"]),
                "eos_rate": float(mode_res["eos_rate"]),
                "diversity_ratio": float(mode_res["diversity"]["diversity_ratio"]),
                "unique_bigram_ratio": float(mode_res["unique_ngram"]["bigram_ratio"]),
            }
        return mode_metrics

    # Legacy fallback not used by default but kept for safety.
    legacy = {
        "sampling": {
            "mean_reward": float(model_result.get("reward", {}).get("mean", 0.0)),
            "mean_raw_reward": float(model_result.get("reward", {}).get("mean", 0.0)),
            "mean_sentiment": float(model_result.get("sentiment", {}).get("mean", 0.0)),
            "mean_quality_anchor": 0.0,
            "mean_repetition_token": float(model_result.get("repetition", {}).get("mean_repetition", 0.0)),
            "mean_length": float(model_result.get("length", {}).get("mean_length", 0.0)),
            "eos_rate": 0.0,
            "diversity_ratio": float(model_result.get("diversity", {}).get("diversity_ratio", 0.0)),
            "unique_bigram_ratio": 0.0,
        }
    }
    return legacy


def collect_seed_directories(input_root: str) -> List[Tuple[str, str]]:
    seed_dirs: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name)
        if os.path.isdir(path) and name.startswith("seed_"):
            seed_dirs.append((name, path))
    return seed_dirs


def aggregate(input_root: str) -> Dict[str, Any]:
    seed_dirs = collect_seed_directories(input_root)
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {input_root}. Expected folders like seed_42")

    # mode -> model -> metric -> list[values]
    storage: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    per_seed: Dict[str, Any] = {}
    models = ["base", "sft", "ppo"]

    for seed_name, seed_path in seed_dirs:
        per_seed[seed_name] = {}
        for model_name in models:
            model_file = os.path.join(seed_path, f"{model_name}_results.json")
            if not os.path.exists(model_file):
                continue

            model_result = load_json(model_file)
            mode_metrics = extract_model_mode_metrics(model_result)
            per_seed[seed_name][model_name] = mode_metrics

            for mode_name, metrics in mode_metrics.items():
                for metric_name, metric_value in metrics.items():
                    storage[mode_name][model_name][metric_name].append(float(metric_value))

    aggregate_summary: Dict[str, Any] = {}
    for mode_name, model_map in storage.items():
        aggregate_summary[mode_name] = {}
        for model_name, metric_map in model_map.items():
            aggregate_summary[mode_name][model_name] = {
                metric_name: summarize(values) for metric_name, values in metric_map.items()
            }

    return {
        "input_root": input_root,
        "seed_runs": [name for name, _ in seed_dirs],
        "aggregate": aggregate_summary,
        "per_seed": per_seed,
    }


def main(input_root: str, output_path: str):
    results = aggregate(input_root)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Multi-seed aggregation saved to {output_path}")
    print("\nAvailable modes and models:")
    for mode_name, mode_data in results["aggregate"].items():
        print(f"- {mode_name}: {', '.join(mode_data.keys())}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate multi-seed evaluation outputs")
    parser.add_argument("--input_root", type=str, required=True, help="Root with seed_xx subdirectories")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    main(args.input_root, args.output)
