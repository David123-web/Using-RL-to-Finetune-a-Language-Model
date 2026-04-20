import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.models.policy_lm import PolicyConfig, PolicyLM
from src.models.reward_model import RewardAggregationConfig, RewardModel
from src.ppo.ppo_trainer import PPOHyperParams, PPOTrainer
from src.utils.device import get_device


@dataclass
class PPOTrainConfig:
    model_name: str
    tokenizer_name: str
    reward_model_name: str
    batch_size: int
    rollout_batch_size: int
    n_updates: int
    epochs_per_update: int
    gamma: float
    lam: float
    clip_range: float
    value_coef: float
    entropy_coef: float
    kl_coef: float
    adaptive_kl: bool
    target_kl: float
    kl_update_rate: float
    kl_coef_min: float
    kl_coef_max: float
    max_length: int
    max_new_tokens: int
    rollout_temperature: float
    rollout_top_k: int
    rollout_top_p: float
    rollout_repetition_penalty: float
    rollout_no_repeat_ngram_size: int
    completion_only: bool
    normalize_text: bool
    w_sentiment: float
    w_repetition_token: float
    w_repetition_phrase: float
    w_length: float
    w_quality_anchor: float
    min_tokens: int
    max_tokens: int
    phrase_ngram: int
    quality_mode: str
    quality_min_token_chars: int
    alignment_gate_enabled: bool
    alignment_gate_quality_floor: float
    alignment_gate_strength: float
    clip_sentiment: Tuple[float, float]
    clip_repetition_token: Tuple[float, float]
    clip_repetition_phrase: Tuple[float, float]
    clip_length: Tuple[float, float]
    clip_quality_anchor: Tuple[float, float]
    clip_total: Tuple[float, float]
    center_total_reward: bool
    normalize_total_reward: bool
    prompt_dataset: str
    prompt_split: str
    save_dir: str
    log_every: int
    log_examples_every: int
    log_example_count: int
    log_examples_do_sample: bool
    eval_every: int
    save_every: int
    collapse_min_avg_completion_len: float
    collapse_max_eos_rate: float
    collapse_max_reward_quality_gap: float
    collapse_patience: int
    ckpt_reward_weight: float
    ckpt_quality_weight: float
    ckpt_eos_weight: float
    ckpt_length_weight: float
    ckpt_mode: str
    ckpt_quality_signal: str
    ckpt_quality_nonzero_eps: float
    ckpt_harmonic_beta: float
    ckpt_stability_weight: float
    ckpt_reward_scale_low: float
    ckpt_reward_scale_high: float


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


def _as_bounds(values: List[float], default: Tuple[float, float]) -> Tuple[float, float]:
    if not isinstance(values, list) or len(values) != 2:
        return default
    return float(values[0]), float(values[1])


def build_train_config(ppo_cfg_raw: Dict[str, Any], reward_cfg_raw: Dict[str, Any]) -> PPOTrainConfig:
    kl_cfg = ppo_cfg_raw.get("kl_control", {})
    collapse_cfg = ppo_cfg_raw.get("collapse_guard", {})
    ckpt_cfg = ppo_cfg_raw.get("checkpoint_selection", {})
    reward_model_cfg = reward_cfg_raw.get("reward_model", {})
    quality_cfg = reward_cfg_raw.get("quality_anchor", {})
    alignment_gate_cfg = reward_cfg_raw.get("alignment_gate", {})
    repetition_cfg = reward_cfg_raw.get("repetition", {})
    clipping_cfg = reward_cfg_raw.get("clipping", {})
    aggregation_cfg = reward_cfg_raw.get("aggregation", {})

    return PPOTrainConfig(
        model_name=ppo_cfg_raw["model"]["name"],
        tokenizer_name=ppo_cfg_raw["model"]["tokenizer_name"],
        reward_model_name=reward_model_cfg["name"],
        batch_size=int(ppo_cfg_raw["ppo"]["batch_size"]),
        rollout_batch_size=int(ppo_cfg_raw["ppo"]["rollout_batch_size"]),
        n_updates=int(ppo_cfg_raw["ppo"]["n_updates"]),
        epochs_per_update=int(ppo_cfg_raw["ppo"]["epochs_per_update"]),
        gamma=float(ppo_cfg_raw["ppo"]["gamma"]),
        lam=float(ppo_cfg_raw["ppo"]["lam"]),
        clip_range=float(ppo_cfg_raw["ppo"]["clip_range"]),
        value_coef=float(ppo_cfg_raw["ppo"]["value_coef"]),
        entropy_coef=float(ppo_cfg_raw["ppo"]["entropy_coef"]),
        kl_coef=float(ppo_cfg_raw["ppo"]["kl_coef"]),
        adaptive_kl=bool(kl_cfg.get("adaptive", False)),
        target_kl=float(kl_cfg.get("target_kl", 0.1)),
        kl_update_rate=float(kl_cfg.get("update_rate", 0.05)),
        kl_coef_min=float(kl_cfg.get("min_coef", 0.01)),
        kl_coef_max=float(kl_cfg.get("max_coef", 1.0)),
        max_length=int(ppo_cfg_raw["ppo"]["max_length"]),
        max_new_tokens=int(ppo_cfg_raw["ppo"].get("max_new_tokens", 32)),
        rollout_temperature=float(ppo_cfg_raw["ppo"].get("rollout_temperature", 1.0)),
        rollout_top_k=int(ppo_cfg_raw["ppo"].get("rollout_top_k", 50)),
        rollout_top_p=float(ppo_cfg_raw["ppo"].get("rollout_top_p", 1.0)),
        rollout_repetition_penalty=float(ppo_cfg_raw["ppo"].get("rollout_repetition_penalty", 1.15)),
        rollout_no_repeat_ngram_size=int(ppo_cfg_raw["ppo"].get("rollout_no_repeat_ngram_size", 3)),
        completion_only=bool(reward_model_cfg.get("completion_only", True)),
        normalize_text=bool(reward_model_cfg.get("normalize_text", True)),
        w_sentiment=float(reward_cfg_raw["weights"]["sentiment"]),
        w_repetition_token=float(reward_cfg_raw["weights"].get("repetition_token", -0.15)),
        w_repetition_phrase=float(reward_cfg_raw["weights"].get("repetition_phrase", -0.1)),
        w_length=float(reward_cfg_raw["weights"]["length"]),
        w_quality_anchor=float(reward_cfg_raw["weights"].get("quality_anchor", 0.0)),
        min_tokens=int(reward_cfg_raw["length_target"]["min_tokens"]),
        max_tokens=int(reward_cfg_raw["length_target"]["max_tokens"]),
        phrase_ngram=int(repetition_cfg.get("phrase_ngram", 3)),
        quality_mode=str(quality_cfg.get("type", "token_overlap")),
        quality_min_token_chars=int(quality_cfg.get("min_token_chars", 3)),
        alignment_gate_enabled=bool(alignment_gate_cfg.get("enabled", False)),
        alignment_gate_quality_floor=float(alignment_gate_cfg.get("quality_floor", 0.15)),
        alignment_gate_strength=float(alignment_gate_cfg.get("strength", 0.85)),
        clip_sentiment=_as_bounds(clipping_cfg.get("sentiment"), (0.0, 1.0)),
        clip_repetition_token=_as_bounds(clipping_cfg.get("repetition_token"), (0.0, 1.0)),
        clip_repetition_phrase=_as_bounds(clipping_cfg.get("repetition_phrase"), (0.0, 1.0)),
        clip_length=_as_bounds(clipping_cfg.get("length"), (-1.0, 1.0)),
        clip_quality_anchor=_as_bounds(clipping_cfg.get("quality_anchor"), (0.0, 1.0)),
        clip_total=_as_bounds(clipping_cfg.get("total"), (-2.0, 2.0)),
        center_total_reward=bool(aggregation_cfg.get("center_total_reward", True)),
        normalize_total_reward=bool(aggregation_cfg.get("normalize_total_reward", False)),
        prompt_dataset=str(ppo_cfg_raw["data"].get("prompt_dataset", "imdb")),
        prompt_split=str(ppo_cfg_raw["data"]["prompt_split"]),
        save_dir=str(ppo_cfg_raw["logging"]["save_dir"]),
        log_every=int(ppo_cfg_raw["logging"]["log_every"]),
        log_examples_every=int(
            ppo_cfg_raw["logging"].get("log_examples_every", ppo_cfg_raw["logging"]["log_every"])
        ),
        log_example_count=int(ppo_cfg_raw["logging"].get("log_example_count", 3)),
        log_examples_do_sample=bool(ppo_cfg_raw["logging"].get("log_examples_do_sample", False)),
        eval_every=int(ppo_cfg_raw["logging"]["eval_every"]),
        save_every=int(ppo_cfg_raw["logging"].get("save_every", 100)),
        collapse_min_avg_completion_len=float(collapse_cfg.get("min_avg_completion_len", 6)),
        collapse_max_eos_rate=float(collapse_cfg.get("max_eos_rate", 0.98)),
        collapse_max_reward_quality_gap=float(collapse_cfg.get("max_reward_quality_gap", 0.45)),
        collapse_patience=int(collapse_cfg.get("patience", 8)),
        ckpt_reward_weight=float(ckpt_cfg.get("reward_weight", 0.4)),
        ckpt_quality_weight=float(ckpt_cfg.get("quality_weight", 0.3)),
        ckpt_eos_weight=float(ckpt_cfg.get("eos_weight", 0.15)),
        ckpt_length_weight=float(ckpt_cfg.get("length_weight", 0.15)),
        ckpt_mode=str(ckpt_cfg.get("mode", "weighted_sum")).strip().lower(),
        ckpt_quality_signal=str(ckpt_cfg.get("quality_signal", "mean")).strip().lower(),
        ckpt_quality_nonzero_eps=float(ckpt_cfg.get("quality_nonzero_eps", 1e-8)),
        ckpt_harmonic_beta=float(ckpt_cfg.get("harmonic_beta", 1.0)),
        ckpt_stability_weight=float(ckpt_cfg.get("stability_weight", 0.2)),
        ckpt_reward_scale_low=float(ckpt_cfg.get("reward_scale_low", 0.0)),
        ckpt_reward_scale_high=float(ckpt_cfg.get("reward_scale_high", 1.0)),
    )


def prepare_prompts(dataset_name: str, split: str) -> List[str]:
    ds = load_dataset(dataset_name, split=split)
    prompts = []
    for example in ds:
        cleaned = _clean_prompt_text(str(example.get("text", "")))
        if not cleaned:
            continue
        words = cleaned.split()[:20]
        prompt = " ".join(words)
        if any(ch.isalnum() for ch in prompt):
            prompts.append(prompt)
    return prompts


def evaluate_policy(
    policy: PolicyLM,
    reward_model: RewardModel,
    prompts: List[str],
    max_new_tokens: int,
    n_samples: int = 10,
) -> Dict[str, Any]:
    eval_start_time = time.perf_counter()
    sample_prompts = prompts[:n_samples]
    _, completions, metadata = policy.generate(
        sample_prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        return_completions=True,
        return_metadata=True,
    )

    rewards, components = reward_model.compute_reward_for_completions(
        completions=completions,
        prompts=sample_prompts,
        return_components=True,
    )

    examples = []
    for i, prompt in enumerate(sample_prompts):
        completion = completions[i]
        examples.append(
            {
                "prompt": prompt,
                "completion": completion,
                "reward": float(rewards[i].item()),
                "raw_reward": float(components["raw_total"][i].item()),
                "sentiment": float(components["sentiment"][i].item()),
                "quality_anchor": float(components["quality_anchor"][i].item()),
                "eos": bool(metadata["eos_triggered"][i]),
                "length_words": int(len(completion.split())),
            }
        )

    mean_centered_reward = float(rewards.mean().item())
    mean_raw_reward = float(components["raw_total"].mean().item())
    quality_nonzero_rate = float((components["quality_anchor"] > 1e-8).float().mean().item())

    return {
        "mean_reward": mean_raw_reward,
        "mean_raw_reward": mean_raw_reward,
        "mean_centered_reward": mean_centered_reward,
        "mean_quality_anchor": float(components["quality_anchor"].mean().item()),
        "quality_anchor_nonzero_rate": quality_nonzero_rate,
        "mean_completion_len": float(sum(len(c.split()) for c in completions) / max(len(completions), 1)),
        "eos_rate": float(sum(metadata["eos_triggered"]) / max(len(metadata["eos_triggered"]), 1)),
        "examples": examples,
        "runtime_sec": float(time.perf_counter() - eval_start_time),
    }


def log_training_examples(
    policy: PolicyLM,
    reward_model: RewardModel,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[Dict[str, Any]]:
    if len(prompts) == 0:
        return []

    def _sanitize_for_log(text: str, max_len: int) -> str:
        text = text.replace("\n", " ").replace("\r", " ").strip()
        # Keep logs ASCII-only to avoid mojibake in Windows terminal code pages.
        text = text.encode("ascii", "ignore").decode("ascii")
        text = " ".join(text.split())
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return text

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": 1.0,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "return_completions": True,
        "return_metadata": True,
    }
    if do_sample:
        gen_kwargs["top_k"] = 50

    _, completions, metadata = policy.generate(prompts, **gen_kwargs)

    rewards, components = reward_model.compute_reward_for_completions(
        completions=completions,
        prompts=prompts,
        return_components=True,
    )

    examples: List[Dict[str, Any]] = []
    print("  Sample generations:")
    for i, prompt in enumerate(prompts):
        example = {
            "prompt": prompt,
            "completion": completions[i],
            "reward": float(rewards[i].item()),
            "raw_reward": float(components["raw_total"][i].item()),
            "sentiment": float(components["sentiment"][i].item()),
            "repetition_token": float(components["repetition_token"][i].item()),
            "repetition_phrase": float(components["repetition_phrase"][i].item()),
            "length_score": float(components["length"][i].item()),
            "quality_anchor": float(components["quality_anchor"][i].item()),
            "eos": bool(metadata["eos_triggered"][i]),
            "length_words": int(len(completions[i].split())),
        }
        examples.append(example)

        completion = _sanitize_for_log(completions[i], max_len=140)
        prompt_preview = _sanitize_for_log(prompt, max_len=120)

        eos_flag = example["eos"]
        length_words = example["length_words"]
        print(f"    [{i + 1}] prompt: {prompt_preview}")
        print(f"        completion: {completion}")
        print(
            "        reward={:.4f} raw={:.4f} sentiment={:.4f} quality={:.4f} "
            "rep_tok={:.4f} rep_phrase={:.4f} len_score={:.4f} "
            "length_words={} eos={}"
            .format(
                example["reward"],
                example["raw_reward"],
                example["sentiment"],
                example["quality_anchor"],
                example["repetition_token"],
                example["repetition_phrase"],
                example["length_score"],
                length_words,
                eos_flag,
            )
        )

    return examples


def save_checkpoint(policy: PolicyLM, trainer: PPOTrainer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    policy.model.save_pretrained(output_dir)
    policy.tokenizer.save_pretrained(output_dir)
    torch.save(trainer.value_head.state_dict(), os.path.join(output_dir, "value_head.pt"))


def append_jsonl(snapshot_path: str, payload: Dict[str, Any]):
    with open(snapshot_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def checkpoint_score(cfg: PPOTrainConfig, stats: Dict[str, float]) -> float:
    target_center = (cfg.min_tokens + cfg.max_tokens) / 2.0
    length_score = max(
        0.0,
        1.0 - abs(stats["avg_completion_length_words"] - target_center) / max(target_center, 1.0)
    )
    eos_score = max(0.0, 1.0 - stats["eos_rate"])

    quality_signal = stats["quality_anchor"]
    if cfg.ckpt_quality_signal == "nonzero_rate":
        quality_signal = stats.get("quality_anchor_nonzero_rate", quality_signal)
    quality_signal = float(min(max(quality_signal, 0.0), 1.0))

    if cfg.ckpt_mode == "harmonic_balance":
        low = min(cfg.ckpt_reward_scale_low, cfg.ckpt_reward_scale_high)
        high = max(cfg.ckpt_reward_scale_low, cfg.ckpt_reward_scale_high)
        denom = max(high - low, 1e-8)
        reward_norm = (stats["raw_reward"] - low) / denom
        reward_norm = float(min(max(reward_norm, 0.0), 1.0))

        beta = max(cfg.ckpt_harmonic_beta, 1e-6)
        beta2 = beta * beta
        harmonic_score = ((1.0 + beta2) * reward_norm * quality_signal) / (
            beta2 * reward_norm + quality_signal + 1e-8
        )

        stability_weight = float(min(max(cfg.ckpt_stability_weight, 0.0), 1.0))
        stability_score = 0.5 * (eos_score + length_score)
        return (1.0 - stability_weight) * harmonic_score + stability_weight * stability_score

    return (
        cfg.ckpt_reward_weight * stats["raw_reward"]
        + cfg.ckpt_quality_weight * quality_signal
        + cfg.ckpt_eos_weight * eos_score
        + cfg.ckpt_length_weight * length_score
    )


def main(
    config_path: str = "config/ppo_config.yaml",
    reward_config_path: str = "config/reward_config.yaml",
    seed: int = 42,
    save_dir_override: str = None,
):
    ppo_cfg_raw = load_config(config_path)
    reward_cfg_raw = load_config(reward_config_path)
    cfg = build_train_config(ppo_cfg_raw, reward_cfg_raw)

    if save_dir_override:
        cfg.save_dir = save_dir_override

    run_start_time = time.perf_counter()

    set_seed(seed)
    os.environ.setdefault("TQDM_ASCII", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    device = get_device()
    print(f"Using device: {device}")

    print("Loading policy model...")
    policy = PolicyLM(
        PolicyConfig(
            model_name=cfg.model_name,
            tokenizer_name=cfg.tokenizer_name,
            max_length=cfg.max_length,
        )
    )

    print("Loading reference policy...")
    ref_policy = PolicyLM(
        PolicyConfig(
            model_name=cfg.model_name,
            tokenizer_name=cfg.tokenizer_name,
            max_length=cfg.max_length,
        )
    )

    print("Loading reward model...")
    reward_model = RewardModel(
        model_name=cfg.reward_model_name,
        completion_only=cfg.completion_only,
        normalize_text=cfg.normalize_text,
        w_sentiment=cfg.w_sentiment,
        w_repetition_token=cfg.w_repetition_token,
        w_repetition_phrase=cfg.w_repetition_phrase,
        w_length=cfg.w_length,
        w_quality_anchor=cfg.w_quality_anchor,
        min_tokens=cfg.min_tokens,
        max_tokens=cfg.max_tokens,
        phrase_ngram=cfg.phrase_ngram,
        quality_mode=cfg.quality_mode,
        quality_min_token_chars=cfg.quality_min_token_chars,
        alignment_gate_enabled=cfg.alignment_gate_enabled,
        alignment_gate_quality_floor=cfg.alignment_gate_quality_floor,
        alignment_gate_strength=cfg.alignment_gate_strength,
        clip_sentiment=cfg.clip_sentiment,
        clip_repetition_token=cfg.clip_repetition_token,
        clip_repetition_phrase=cfg.clip_repetition_phrase,
        clip_length=cfg.clip_length,
        clip_quality_anchor=cfg.clip_quality_anchor,
        clip_total=cfg.clip_total,
        aggregation=RewardAggregationConfig(
            center_total_reward=cfg.center_total_reward,
            normalize_total_reward=cfg.normalize_total_reward,
        ),
    )

    hparams = PPOHyperParams(
        clip_range=cfg.clip_range,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        gamma=cfg.gamma,
        lam=cfg.lam,
        epochs_per_update=cfg.epochs_per_update,
        batch_size=cfg.batch_size,
        kl_coef=cfg.kl_coef,
        adaptive_kl=cfg.adaptive_kl,
        target_kl=cfg.target_kl,
        kl_update_rate=cfg.kl_update_rate,
        kl_coef_min=cfg.kl_coef_min,
        kl_coef_max=cfg.kl_coef_max,
    )

    trainer = PPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        hparams=hparams,
        max_length=cfg.max_length,
        max_new_tokens=cfg.max_new_tokens,
        rollout_temperature=cfg.rollout_temperature,
        rollout_top_k=cfg.rollout_top_k,
        rollout_top_p=cfg.rollout_top_p,
        rollout_repetition_penalty=cfg.rollout_repetition_penalty,
        rollout_no_repeat_ngram_size=cfg.rollout_no_repeat_ngram_size,
    )

    print("Loading prompts...")
    prompts = prepare_prompts(cfg.prompt_dataset, cfg.prompt_split)
    print(f"Loaded {len(prompts)} prompts")

    os.makedirs(cfg.save_dir, exist_ok=True)
    eval_snapshots_path = os.path.join(cfg.save_dir, "eval_snapshots.jsonl")
    training_examples_path = os.path.join(cfg.save_dir, "training_examples.jsonl")

    print("\nStarting PPO training...")
    training_stats: List[Dict[str, float]] = []
    collapse_counter = 0
    best_score = float("-inf")
    best_step = -1

    for update in tqdm(
        range(cfg.n_updates),
        desc="PPO Updates",
        ascii=True,
        disable=_tqdm_disabled(),
        mininterval=_tqdm_mininterval(),
        dynamic_ncols=False,
    ):
        batch_indices = torch.randint(0, len(prompts), (cfg.rollout_batch_size,))
        batch_prompts = [prompts[i] for i in batch_indices]

        stats = trainer.train_step(batch_prompts)
        stats["update"] = update + 1
        stats["checkpoint_score"] = checkpoint_score(cfg, stats)
        training_stats.append(stats)

        if (update + 1) % cfg.log_every == 0:
            print(f"\nUpdate {update + 1}/{cfg.n_updates}")
            print(f"  Reward(centered): {stats['reward']:.4f}")
            print(f"  Raw Reward: {stats['raw_reward']:.4f}")
            print(f"  Sentiment: {stats['sentiment']:.4f}")
            print(f"  Quality Anchor: {stats['quality_anchor']:.4f}")
            print(f"  Quality Nonzero Rate: {stats['quality_anchor_nonzero_rate']:.3f}")
            print(f"  Completion Length(words): {stats['avg_completion_length_words']:.2f}")
            print(f"  EOS Rate: {stats['eos_rate']:.3f}")
            print(f"  KL Div: {stats['kl_div']:.4f} | KL Coef: {stats['kl_coef']:.4f}")

        if (update + 1) % cfg.log_examples_every == 0:
            sample_count = max(1, min(cfg.log_example_count, len(batch_prompts)))
            training_examples = log_training_examples(
                policy=policy,
                reward_model=reward_model,
                prompts=batch_prompts[:sample_count],
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.log_examples_do_sample,
                repetition_penalty=cfg.rollout_repetition_penalty,
                no_repeat_ngram_size=cfg.rollout_no_repeat_ngram_size,
            )
            append_jsonl(
                snapshot_path=training_examples_path,
                payload={
                    "update": update + 1,
                    "do_sample": bool(cfg.log_examples_do_sample),
                    "examples": training_examples,
                },
            )

        collapse_flag = (
            stats["avg_completion_length_words"] < cfg.collapse_min_avg_completion_len
            or stats["eos_rate"] > cfg.collapse_max_eos_rate
            or stats["reward_quality_gap"] > cfg.collapse_max_reward_quality_gap
        )
        if collapse_flag:
            collapse_counter += 1
        else:
            collapse_counter = 0

        if stats["checkpoint_score"] > best_score:
            best_score = stats["checkpoint_score"]
            best_step = update + 1
            best_dir = os.path.join(cfg.save_dir, "best")
            save_checkpoint(policy, trainer, best_dir)

        if (update + 1) % cfg.eval_every == 0:
            eval_results = evaluate_policy(
                policy=policy,
                reward_model=reward_model,
                prompts=prompts,
                max_new_tokens=cfg.max_new_tokens,
                n_samples=5,
            )
            print("\n" + "=" * 50)
            print("EVALUATION (Greedy)")
            print("=" * 50)
            print(f"Mean Reward (raw): {eval_results['mean_reward']:.4f}")
            print(f"Mean Reward (centered): {eval_results['mean_centered_reward']:.4f}")
            print(f"Mean Quality Anchor: {eval_results['mean_quality_anchor']:.4f}")
            print(f"Quality Nonzero Rate: {eval_results['quality_anchor_nonzero_rate']:.3f}")
            print(f"Mean Completion Length: {eval_results['mean_completion_len']:.2f}")
            print(f"EOS Rate: {eval_results['eos_rate']:.3f}")
            print(f"Eval Runtime: {eval_results['runtime_sec']:.1f}s")
            print("Sample Validation Generations:")
            for i, example in enumerate(eval_results["examples"][:3], start=1):
                prompt_preview = example["prompt"].replace("\n", " ").replace("\r", " ").strip()
                prompt_preview = " ".join(prompt_preview.split())
                completion_preview = example["completion"].replace("\n", " ").replace("\r", " ").strip()
                completion_preview = " ".join(completion_preview.split())
                prompt_preview = prompt_preview.encode("ascii", "ignore").decode("ascii")
                completion_preview = completion_preview.encode("ascii", "ignore").decode("ascii")
                if len(prompt_preview) > 120:
                    prompt_preview = prompt_preview[:117] + "..."
                if len(completion_preview) > 180:
                    completion_preview = completion_preview[:177] + "..."
                print(f"  [{i}] prompt: {prompt_preview}")
                print(f"      completion: {completion_preview}")
                print(
                    "      reward={:.4f} raw={:.4f} sentiment={:.4f} quality={:.4f} len={} eos={}"
                    .format(
                        example["reward"],
                        example["raw_reward"],
                        example["sentiment"],
                        example["quality_anchor"],
                        example["length_words"],
                        example["eos"],
                    )
                )
            print("=" * 50 + "\n")

            append_jsonl(
                snapshot_path=eval_snapshots_path,
                payload={
                    "update": update + 1,
                    "mean_reward": eval_results["mean_reward"],
                    "mean_raw_reward": eval_results["mean_raw_reward"],
                    "mean_centered_reward": eval_results["mean_centered_reward"],
                    "mean_quality_anchor": eval_results["mean_quality_anchor"],
                    "quality_anchor_nonzero_rate": eval_results["quality_anchor_nonzero_rate"],
                    "mean_completion_len": eval_results["mean_completion_len"],
                    "eos_rate": eval_results["eos_rate"],
                    "runtime_sec": eval_results["runtime_sec"],
                    "examples": eval_results["examples"],
                },
            )

        if (update + 1) % cfg.save_every == 0:
            checkpoint_dir = os.path.join(cfg.save_dir, f"checkpoint_{update + 1}")
            save_checkpoint(policy, trainer, checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")

        if collapse_counter >= cfg.collapse_patience:
            print(
                f"Early stop triggered at update {update + 1}: "
                f"collapse guard was hit for {cfg.collapse_patience} consecutive updates."
            )
            break

    final_dir = os.path.join(cfg.save_dir, "final")
    save_checkpoint(policy, trainer, final_dir)

    stats_path = os.path.join(cfg.save_dir, "training_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2)

    summary_path = os.path.join(cfg.save_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "best_step": best_step,
                "best_score": best_score,
                "total_updates": len(training_stats),
                "collapse_guard": {
                    "min_avg_completion_len": cfg.collapse_min_avg_completion_len,
                    "max_eos_rate": cfg.collapse_max_eos_rate,
                    "max_reward_quality_gap": cfg.collapse_max_reward_quality_gap,
                    "patience": cfg.collapse_patience,
                },
            },
            f,
            indent=2,
        )

    print(f"\nTraining complete! Model saved to {final_dir}")
    print(f"Best checkpoint step: {best_step} (score={best_score:.4f})")
    print(f"Training statistics saved to {stats_path}")
    print(f"Training sample generations saved to {training_examples_path}")
    print(f"Validation snapshots saved to {eval_snapshots_path}")
    print(f"Total training runtime: {time.perf_counter() - run_start_time:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_config.yaml")
    parser.add_argument("--reward_config", type=str, default="config/reward_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        config_path=args.config,
        reward_config_path=args.reward_config,
        seed=args.seed,
        save_dir_override=args.save_dir,
    )
