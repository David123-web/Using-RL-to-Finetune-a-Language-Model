import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.models.policy_lm import PolicyConfig, PolicyLM
from src.models.reward_model import RewardAggregationConfig, RewardModel
from src.utils.device import get_device


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tqdm_disabled() -> bool:
    raw = os.environ.get("RLHF_DISABLE_TQDM", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _tqdm_mininterval() -> float:
    raw = os.environ.get("RLHF_TQDM_MININTERVAL", "1.0")
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    return max(0.1, value)


def _clean_prompt_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"<br\\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\\s\\.,!?;:'\"()\\-]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def prepare_prompts(dataset_name: str, split: str, n_samples: int) -> List[str]:
    ds = load_dataset(dataset_name, split=split)
    prompts = []
    for i, example in enumerate(ds):
        if i >= n_samples:
            break
        cleaned = _clean_prompt_text(str(example.get("text", "")))
        if not cleaned:
            continue
        words = cleaned.split()[:20]
        prompt = " ".join(words)
        if any(ch.isalnum() for ch in prompt):
            prompts.append(prompt)
    return prompts


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def tokenize_for_stats(text: str) -> List[str]:
    return [tok for tok in text.lower().strip().split() if tok]


def _sanitize_for_log_text(text: str, max_len: int = 140) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    text = text.encode("ascii", "ignore").decode("ascii")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def compute_diversity(texts: List[str]) -> Dict[str, float]:
    all_tokens: List[str] = []
    for text in texts:
        all_tokens.extend(tokenize_for_stats(text))

    if not all_tokens:
        return {"unique_tokens": 0, "total_tokens": 0, "diversity_ratio": 0.0}

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    return {
        "unique_tokens": unique_tokens,
        "total_tokens": total_tokens,
        "diversity_ratio": unique_tokens / total_tokens,
    }


def unique_ngram_ratio(texts: List[str], n: int) -> float:
    ngrams: List[tuple] = []
    for text in texts:
        tokens = tokenize_for_stats(text)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))

    if not ngrams:
        return 0.0
    return float(len(set(ngrams)) / len(ngrams))


def load_policy(
    model_path: str,
    model_name: str,
    tokenizer_name: str,
    max_length: int,
) -> PolicyLM:
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return PolicyLM(
            PolicyConfig(
                model_name=model_path,
                tokenizer_name=model_path,
                max_length=max_length,
            )
        )

    print(f"Loading base model {model_name}")
    return PolicyLM(
        PolicyConfig(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )
    )


def evaluate_mode(
    policy: PolicyLM,
    prompts: List[str],
    reward_model: RewardModel,
    mode_name: str,
    mode_cfg: Dict[str, Any],
    max_new_tokens: int,
    seed: int,
    records_output_path: Optional[str] = None,
    batch_size: int = 8,
    sample_print_count: int = 3,
) -> Dict[str, Any]:
    mode_start_time = time.perf_counter()
    responses: List[str] = []
    completions: List[str] = []

    reward_values: List[float] = []
    raw_reward_values: List[float] = []

    component_values: Dict[str, List[float]] = {
        "sentiment": [],
        "repetition_token": [],
        "repetition_phrase": [],
        "length": [],
        "quality_anchor": [],
    }

    completion_length_tokens: List[int] = []
    completion_length_words: List[int] = []
    eos_triggered: List[bool] = []

    print(f"Generating responses ({mode_name})...")
    for i in tqdm(
        range(0, len(prompts), batch_size),
        ascii=True,
        disable=_tqdm_disabled(),
        mininterval=_tqdm_mininterval(),
        dynamic_ncols=False,
    ):
        batch_prompts = prompts[i : i + batch_size]
        batch_seed = seed + i

        full_texts, batch_completions, metadata = policy.generate(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=bool(mode_cfg.get("do_sample", True)),
            temperature=float(mode_cfg.get("temperature", 1.0)),
            top_k=int(mode_cfg.get("top_k", 50)),
            top_p=float(mode_cfg.get("top_p", 1.0)),
            repetition_penalty=float(mode_cfg.get("repetition_penalty", 1.15)),
            no_repeat_ngram_size=int(mode_cfg.get("no_repeat_ngram_size", 3)),
            seed=batch_seed,
            return_completions=True,
            return_metadata=True,
        )

        rewards, components = reward_model.compute_reward_for_completions(
            completions=batch_completions,
            prompts=batch_prompts,
            return_components=True,
        )

        responses.extend(full_texts)
        completions.extend(batch_completions)
        reward_values.extend(rewards.tolist())
        raw_reward_values.extend(components["raw_total"].tolist())

        for key in component_values:
            component_values[key].extend(components[key].tolist())

        completion_length_tokens.extend(metadata["completion_token_lengths"])
        completion_length_words.extend([len(text.split()) for text in batch_completions])
        eos_triggered.extend(metadata["eos_triggered"])

    eos_numeric = [1 if flag else 0 for flag in eos_triggered]

    if records_output_path:
        with open(records_output_path, "w", encoding="utf-8") as f:
            for idx in range(len(prompts)):
                f.write(
                    json.dumps(
                        {
                            "index": int(idx),
                            "prompt": prompts[idx],
                            "response": responses[idx],
                            "completion": completions[idx],
                            "reward": float(reward_values[idx]),
                            "raw_reward": float(raw_reward_values[idx]),
                            "sentiment": float(component_values["sentiment"][idx]),
                            "repetition_token": float(component_values["repetition_token"][idx]),
                            "repetition_phrase": float(component_values["repetition_phrase"][idx]),
                            "length_score": float(component_values["length"][idx]),
                            "quality_anchor": float(component_values["quality_anchor"][idx]),
                            "completion_length_tokens": int(completion_length_tokens[idx]),
                            "completion_length_words": int(completion_length_words[idx]),
                            "eos_triggered": bool(eos_triggered[idx]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    runtime_sec = float(time.perf_counter() - mode_start_time)
    results = {
        "mode": mode_name,
        "mode_config": mode_cfg,
        "runtime_sec": runtime_sec,
        "reward": summarize(reward_values),
        "raw_reward": summarize(raw_reward_values),
        "components": {
            key: summarize(values) for key, values in component_values.items()
        },
        "length": summarize([float(v) for v in completion_length_words]),
        "eos_rate": float(np.mean(eos_numeric) if eos_numeric else 0.0),
        "diversity": compute_diversity(completions),
        "unique_ngram": {
            "bigram_ratio": unique_ngram_ratio(completions, 2),
            "trigram_ratio": unique_ngram_ratio(completions, 3),
        },
        "distributions": {
            "reward": reward_values,
            "raw_reward": raw_reward_values,
            "sentiment": component_values["sentiment"],
            "repetition_token": component_values["repetition_token"],
            "repetition_phrase": component_values["repetition_phrase"],
            "length": component_values["length"],
            "quality_anchor": component_values["quality_anchor"],
            "completion_length_tokens": completion_length_tokens,
            "completion_length_words": completion_length_words,
            "eos_triggered": eos_numeric,
        },
        "examples": [
            {
                "prompt": prompts[idx],
                "response": responses[idx],
                "completion": completions[idx],
                "reward": float(reward_values[idx]),
                "raw_reward": float(raw_reward_values[idx]),
                "sentiment": float(component_values["sentiment"][idx]),
                "quality_anchor": float(component_values["quality_anchor"][idx]),
            }
            for idx in range(min(10, len(prompts)))
        ],
        "num_records": int(len(prompts)),
        "records_file": records_output_path,
    }

    print(
        f"[{mode_name}] raw_reward={results['raw_reward']['mean']:.4f}, "
        f"sentiment={results['components']['sentiment']['mean']:.4f}, "
        f"quality={results['components']['quality_anchor']['mean']:.4f}, "
        f"rep_tok={results['components']['repetition_token']['mean']:.4f}, "
        f"len={results['length']['mean']:.2f}, eos={results['eos_rate']:.3f}, "
        f"runtime={runtime_sec:.1f}s"
    )

    if results["examples"]:
        print(f"[{mode_name}] sample generations:")
        for i, example in enumerate(results["examples"][:sample_print_count], start=1):
            prompt_preview = _sanitize_for_log_text(example["prompt"], max_len=120)
            completion_preview = _sanitize_for_log_text(example["completion"], max_len=160)
            print(f"  [{i}] prompt: {prompt_preview}")
            print(f"      completion: {completion_preview}")
            print(
                f"      reward={example['reward']:.4f} raw={example['raw_reward']:.4f} "
                f"sent={example['sentiment']:.4f} quality={example['quality_anchor']:.4f}"
            )

    return results


def composite_score(
    summary_entry: Dict[str, float],
    min_tokens: int,
    max_tokens: int,
    reward_weight: float = 0.4,
    quality_weight: float = 0.3,
    eos_weight: float = 0.15,
    length_weight: float = 0.15,
) -> float:
    target_center = (min_tokens + max_tokens) / 2.0
    length_score = max(
        0.0,
        1.0 - abs(summary_entry["mean_length"] - target_center) / max(target_center, 1.0)
    )
    eos_score = max(0.0, 1.0 - summary_entry["eos_rate"])

    return float(
        reward_weight * summary_entry["mean_raw_reward"]
        + quality_weight * summary_entry["mean_quality_anchor"]
        + eos_weight * eos_score
        + length_weight * length_score
    )


def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: str,
    min_tokens: int,
    max_tokens: int,
):
    available_modes = set()
    for model_results in results_dict.values():
        available_modes.update(model_results.keys())

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in sorted(available_modes):
        summary[mode] = {}
        for model_name, model_results in results_dict.items():
            if mode not in model_results:
                continue
            res = model_results[mode]
            summary_row = {
                "mean_reward": res["reward"]["mean"],
                "mean_raw_reward": res["raw_reward"]["mean"],
                "mean_sentiment": res["components"]["sentiment"]["mean"],
                "mean_quality_anchor": res["components"]["quality_anchor"]["mean"],
                "mean_repetition_token": res["components"]["repetition_token"]["mean"],
                "mean_length": res["length"]["mean"],
                "eos_rate": res["eos_rate"],
                "diversity_ratio": res["diversity"]["diversity_ratio"],
                "unique_bigram_ratio": res["unique_ngram"]["bigram_ratio"],
            }
            summary_row["composite_score"] = composite_score(
                summary_entry=summary_row,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
            summary[mode][model_name] = summary_row

    comparison = {
        "summary": summary,
        "detailed": results_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {output_path}")
    print("\n" + "=" * 64)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 64)
    for mode, mode_summary in summary.items():
        print(f"\n[{mode.upper()}]")
        for model_name, metrics in mode_summary.items():
            print(
                f"  {model_name}: reward={metrics['mean_reward']:.4f}, "
                f"raw_reward={metrics['mean_raw_reward']:.4f}, "
                f"quality={metrics['mean_quality_anchor']:.4f}, "
                f"len={metrics['mean_length']:.2f}, eos={metrics['eos_rate']:.3f}, "
                f"score={metrics['composite_score']:.4f}"
            )
    print("=" * 64)


def build_reward_model(
    reward_cfg: Dict[str, Any],
    center_total_reward_override: Optional[bool] = None,
) -> RewardModel:
    reward_model_cfg = reward_cfg.get("reward_model", {})
    quality_cfg = reward_cfg.get("quality_anchor", {})
    alignment_gate_cfg = reward_cfg.get("alignment_gate", {})
    repetition_cfg = reward_cfg.get("repetition", {})
    clipping_cfg = reward_cfg.get("clipping", {})
    aggregation_cfg = reward_cfg.get("aggregation", {})

    def _bounds(key: str, default):
        values = clipping_cfg.get(key, default)
        if not isinstance(values, list) or len(values) != 2:
            return default
        return float(values[0]), float(values[1])

    center_total_reward = bool(aggregation_cfg.get("center_total_reward", True))
    if center_total_reward_override is not None:
        center_total_reward = bool(center_total_reward_override)

    return RewardModel(
        model_name=reward_model_cfg["name"],
        completion_only=bool(reward_model_cfg.get("completion_only", True)),
        normalize_text=bool(reward_model_cfg.get("normalize_text", True)),
        w_sentiment=float(reward_cfg["weights"]["sentiment"]),
        w_repetition_token=float(reward_cfg["weights"].get("repetition_token", -0.15)),
        w_repetition_phrase=float(reward_cfg["weights"].get("repetition_phrase", -0.1)),
        w_length=float(reward_cfg["weights"]["length"]),
        w_quality_anchor=float(reward_cfg["weights"].get("quality_anchor", 0.0)),
        min_tokens=int(reward_cfg["length_target"]["min_tokens"]),
        max_tokens=int(reward_cfg["length_target"]["max_tokens"]),
        phrase_ngram=int(repetition_cfg.get("phrase_ngram", 3)),
        quality_mode=str(quality_cfg.get("type", "token_overlap")),
        quality_min_token_chars=int(quality_cfg.get("min_token_chars", 3)),
        alignment_gate_enabled=bool(alignment_gate_cfg.get("enabled", False)),
        alignment_gate_quality_floor=float(alignment_gate_cfg.get("quality_floor", 0.15)),
        alignment_gate_strength=float(alignment_gate_cfg.get("strength", 0.85)),
        clip_sentiment=_bounds("sentiment", (0.0, 1.0)),
        clip_repetition_token=_bounds("repetition_token", (0.0, 1.0)),
        clip_repetition_phrase=_bounds("repetition_phrase", (0.0, 1.0)),
        clip_length=_bounds("length", (-1.0, 1.0)),
        clip_quality_anchor=_bounds("quality_anchor", (0.0, 1.0)),
        clip_total=_bounds("total", (-2.0, 2.0)),
        aggregation=RewardAggregationConfig(
            center_total_reward=center_total_reward,
            normalize_total_reward=bool(aggregation_cfg.get("normalize_total_reward", False)),
        ),
    )


def main(
    base_model: bool = True,
    sft_model: Optional[str] = None,
    ppo_model: Optional[str] = None,
    config_path: str = "config/model_config.yaml",
    reward_config_path: str = "config/reward_config.yaml",
    output_dir: str = "results",
    seed: Optional[int] = None,
    modes: Optional[List[str]] = None,
):
    model_cfg = load_config(config_path)
    reward_cfg = load_config(reward_config_path)

    eval_cfg = model_cfg.get("evaluation", {})
    eval_seed = int(seed if seed is not None else eval_cfg.get("seed", 42))
    n_samples = int(eval_cfg.get("n_samples", 500))
    sample_print_count = int(eval_cfg.get("sample_print_count", 3))
    dataset_name = str(eval_cfg.get("dataset_name", "imdb"))
    split = str(eval_cfg.get("prompt_split", "test[:500]"))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 32))
    use_centered_reward = bool(eval_cfg.get("use_centered_reward", False))

    available_modes = eval_cfg.get(
        "modes",
        {
            "greedy": {"do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
            "sampling": {"do_sample": True, "temperature": 1.0, "top_k": 50, "top_p": 1.0},
        },
    )

    if modes:
        selected_modes = [mode for mode in modes if mode in available_modes]
        if not selected_modes:
            raise ValueError(f"None of requested modes exist: {modes}")
    else:
        selected_modes = list(available_modes.keys())

    set_seed(eval_seed)
    run_start_time = time.perf_counter()

    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading reward model...")
    reward_model = build_reward_model(
        reward_cfg,
        center_total_reward_override=use_centered_reward,
    )

    print("Loading prompts...")
    prompts = prepare_prompts(dataset_name, split, n_samples)
    print(f"Loaded {len(prompts)} prompts")

    model_specs = []
    if base_model:
        model_specs.append(("base", model_cfg["model_name"]))
    if sft_model:
        model_specs.append(("sft", sft_model))
    if ppo_model:
        model_specs.append(("ppo", ppo_model))

    all_results: Dict[str, Dict[str, Any]] = {}

    for model_label, model_path in model_specs:
        print("\n" + "=" * 64)
        print(f"Evaluating {model_label.upper()} model")
        print("=" * 64)

        policy = load_policy(
            model_path=model_path,
            model_name=model_cfg["model_name"],
            tokenizer_name=model_cfg["tokenizer_name"],
            max_length=int(model_cfg["max_length"]),
        )

        mode_results: Dict[str, Any] = {}
        for mode_name in selected_modes:
            records_path = os.path.join(output_dir, f"{model_label}_{mode_name}_records.jsonl")
            mode_results[mode_name] = evaluate_mode(
                policy=policy,
                prompts=prompts,
                reward_model=reward_model,
                mode_name=mode_name,
                mode_cfg=available_modes[mode_name],
                max_new_tokens=max_new_tokens,
                seed=eval_seed,
                records_output_path=records_path,
                sample_print_count=sample_print_count,
            )

        all_results[model_label] = mode_results

        model_result_path = os.path.join(output_dir, f"{model_label}_results.json")
        with open(model_result_path, "w", encoding="utf-8") as f:
            json.dump(mode_results, f, indent=2)
        print(f"Saved {model_label} results to {model_result_path}")

    if len(all_results) > 1:
        compare_models(
            results_dict=all_results,
            output_path=os.path.join(output_dir, "comparison.json"),
            min_tokens=int(reward_cfg["length_target"]["min_tokens"]),
            max_tokens=int(reward_cfg["length_target"]["max_tokens"]),
        )

    total_runtime_sec = float(time.perf_counter() - run_start_time)
    print(f"\nEvaluation complete! Results saved to {output_dir}/")
    print(f"Total evaluation runtime: {total_runtime_sec:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate language models")
    parser.add_argument("--base", action="store_true", help="Evaluate base model")
    parser.add_argument("--sft", type=str, default=None, help="Path to SFT model")
    parser.add_argument("--ppo", type=str, default=None, help="Path to PPO model")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--reward_config", type=str, default="config/reward_config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override evaluation seed")
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated decode modes (e.g., greedy,sampling)",
    )

    args = parser.parse_args()
    parsed_modes = args.modes.split(",") if args.modes is not None else None

    main(
        base_model=args.base,
        sft_model=args.sft,
        ppo_model=args.ppo,
        config_path=args.config,
        reward_config_path=args.reward_config,
        output_dir=args.output,
        seed=args.seed,
        modes=parsed_modes,
    )
