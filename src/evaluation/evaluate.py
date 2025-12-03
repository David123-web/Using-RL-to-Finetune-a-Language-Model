import yaml
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import json
import numpy as np

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.models.policy_lm import PolicyLM, PolicyConfig
from src.models.reward_model import RewardModel
from src.utils.device import get_device


@dataclass
class EvalConfig:
    model_path: str
    model_name: str
    tokenizer_name: str
    reward_model_name: str
    max_length: int
    max_new_tokens: int
    dataset_name: str
    split: str
    n_samples: int
    output_path: str
    w_sentiment: float
    w_repetition: float
    w_length: float
    min_tokens: int
    max_tokens: int


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_prompts(dataset_name: str, split: str, n_samples: int) -> List[str]:
    """Load and prepare prompts from dataset"""
    ds = load_dataset(dataset_name, split=split)
    
    prompts = []
    for i, example in enumerate(ds):
        if i >= n_samples:
            break
        text = example['text']
        # Take first ~20 words as prompt
        words = text.split()[:20]
        prompt = ' '.join(words)
        prompts.append(prompt)
    
    return prompts


def compute_diversity(texts: List[str]) -> Dict[str, float]:
    """Compute diversity metrics for generated texts"""
    all_tokens = []
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
    
    if len(all_tokens) == 0:
        return {
            'unique_tokens': 0,
            'total_tokens': 0,
            'diversity_ratio': 0.0
        }
    
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    return {
        'unique_tokens': unique_tokens,
        'total_tokens': total_tokens,
        'diversity_ratio': unique_tokens / total_tokens
    }


def compute_repetition_stats(texts: List[str]) -> Dict[str, float]:
    """Compute repetition statistics"""
    repetition_scores = []
    for text in texts:
        tokens = text.split()
        if len(tokens) == 0:
            repetition_scores.append(0.0)
            continue
        unique = len(set(tokens))
        total = len(tokens)
        distinct_ratio = unique / total
        repetition_scores.append(1.0 - distinct_ratio)
    
    return {
        'mean_repetition': float(np.mean(repetition_scores)),
        'std_repetition': float(np.std(repetition_scores))
    }


def compute_length_stats(texts: List[str]) -> Dict[str, float]:
    """Compute length statistics"""
    lengths = [len(text.split()) for text in texts]
    return {
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths))
    }


def evaluate_model(
    model_path: str,
    prompts: List[str],
    reward_model: RewardModel,
    max_length: int = 64,
    max_new_tokens: int = 32,
    model_name: str = "distilgpt2",
    tokenizer_name: str = "distilgpt2"
) -> Dict[str, Any]:
    """Evaluate a model on given prompts"""
    
    # Load model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # Try to load config first to check model type, if fails default to distilgpt2
        try:
            policy = PolicyLM(PolicyConfig(
                model_name=model_path,
                tokenizer_name=model_path,
                max_length=max_length,
            ))
        except Exception as e:
            print(f"Standard loading failed, trying with explicit distilgpt2 config: {e}")
            # Force distilgpt2 config if auto-detection fails
            policy = PolicyLM(PolicyConfig(
                model_name=model_path,
                tokenizer_name="distilgpt2", # Use base tokenizer
                max_length=max_length,
            ))
            # Manually override the model loading to force distilgpt2
            policy.model = AutoModelForCausalLM.from_pretrained(model_path, config=AutoConfig.from_pretrained("distilgpt2"))
    else:
        print(f"Loading base model {model_name}")
        policy = PolicyLM(PolicyConfig(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        ))
    
    # Generate responses
    print("Generating responses...")
    responses = []
    batch_size = 8
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = policy.generate(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=1.0
        )
        responses.extend(batch_responses)
    
    # Compute rewards
    print("Computing rewards...")
    all_rewards = []
    all_sentiments = []
    for i in tqdm(range(0, len(responses), batch_size)):
        batch_responses = responses[i:i+batch_size]
        rewards = reward_model.compute_reward(batch_responses)
        sentiments = reward_model.sentiment_score(batch_responses)
        all_rewards.extend(rewards.tolist())
        all_sentiments.extend(sentiments.tolist())
    
    # Compute metrics
    diversity = compute_diversity(responses)
    repetition = compute_repetition_stats(responses)
    length_stats = compute_length_stats(responses)
    
    results = {
        'reward': {
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards)),
            'min': float(np.min(all_rewards)),
            'max': float(np.max(all_rewards))
        },
        'sentiment': {
            'mean': float(np.mean(all_sentiments)),
            'std': float(np.std(all_sentiments)),
            'min': float(np.min(all_sentiments)),
            'max': float(np.max(all_sentiments))
        },
        'diversity': diversity,
        'repetition': repetition,
        'length': length_stats,
        'examples': [
            {
                'prompt': prompts[i],
                'response': responses[i],
                'reward': float(all_rewards[i]),
                'sentiment': float(all_sentiments[i])
            }
            for i in range(min(10, len(prompts)))
        ]
    }
    
    return results


def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: str
):
    """Compare multiple models and save comparison"""
    
    comparison = {
        'summary': {},
        'detailed': results_dict
    }
    
    # Create summary comparison table
    for model_name, results in results_dict.items():
        comparison['summary'][model_name] = {
            'mean_reward': results['reward']['mean'],
            'mean_sentiment': results['sentiment']['mean'],
            'diversity_ratio': results['diversity']['diversity_ratio'],
            'mean_repetition': results['repetition']['mean_repetition'],
            'mean_length': results['length']['mean_length']
        }
    
    # Save comparison
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to {output_path}")
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    for model_name, metrics in comparison['summary'].items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("="*60)


def main(
    base_model: bool = True,
    sft_model: str = None,
    ppo_model: str = None,
    config_path: str = "config/model_config.yaml",
    reward_config_path: str = "config/reward_config.yaml",
    output_dir: str = "results"
):
    """
    Main evaluation function
    
    Args:
        base_model: Evaluate base model (distilgpt2)
        sft_model: Path to SFT model checkpoint
        ppo_model: Path to PPO model checkpoint
        config_path: Path to model config
        reward_config_path: Path to reward config
        output_dir: Directory to save results
    """
    
    # Load configs
    model_cfg = load_config(config_path)
    reward_cfg = load_config(reward_config_path)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize reward model
    print("Loading reward model...")
    reward_model = RewardModel(
        model_name=reward_cfg["reward_model"]["name"],
        w_sentiment=reward_cfg["weights"]["sentiment"],
        w_repetition=reward_cfg["weights"]["repetition"],
        w_length=reward_cfg["weights"]["length"],
        min_tokens=reward_cfg["length_target"]["min_tokens"],
        max_tokens=reward_cfg["length_target"]["max_tokens"],
    )
    
    # Load prompts
    print("Loading prompts...")
    prompts = prepare_prompts("imdb", "test[:500]", 500)
    print(f"Loaded {len(prompts)} prompts")
    
    # Evaluate models
    results = {}
    
    if base_model:
        print("\n" + "="*60)
        print("Evaluating BASE MODEL")
        print("="*60)
        results['base'] = evaluate_model(
            model_path=model_cfg["model_name"],
            prompts=prompts,
            reward_model=reward_model,
            max_length=model_cfg["max_length"],
            model_name=model_cfg["model_name"],
            tokenizer_name=model_cfg["tokenizer_name"]
        )
        
        # Save individual results
        with open(os.path.join(output_dir, 'base_results.json'), 'w') as f:
            json.dump(results['base'], f, indent=2)
    
    if sft_model:
        print("\n" + "="*60)
        print("Evaluating SFT MODEL")
        print("="*60)
        results['sft'] = evaluate_model(
            model_path=sft_model,
            prompts=prompts,
            reward_model=reward_model,
            max_length=model_cfg["max_length"],
            model_name=model_cfg["model_name"],
            tokenizer_name=model_cfg["tokenizer_name"]
        )
        
        # Save individual results
        with open(os.path.join(output_dir, 'sft_results.json'), 'w') as f:
            json.dump(results['sft'], f, indent=2)
    
    if ppo_model:
        print("\n" + "="*60)
        print("Evaluating PPO MODEL")
        print("="*60)
        results['ppo'] = evaluate_model(
            model_path=ppo_model,
            prompts=prompts,
            reward_model=reward_model,
            max_length=model_cfg["max_length"],
            model_name=model_cfg["model_name"],
            tokenizer_name=model_cfg["tokenizer_name"]
        )
        
        # Save individual results
        with open(os.path.join(output_dir, 'ppo_results.json'), 'w') as f:
            json.dump(results['ppo'], f, indent=2)
    
    # Compare models
    if len(results) > 1:
        compare_models(results, os.path.join(output_dir, 'comparison.json'))
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate language models")
    parser.add_argument("--base", action="store_true", help="Evaluate base model")
    parser.add_argument("--sft", type=str, default=None, help="Path to SFT model")
    parser.add_argument("--ppo", type=str, default=None, help="Path to PPO model")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--reward_config", type=str, default="config/reward_config.yaml")
    
    args = parser.parse_args()
    
    main(
        base_model=args.base,
        sft_model=args.sft,
        ppo_model=args.ppo,
        config_path=args.config,
        reward_config_path=args.reward_config,
        output_dir=args.output
    )
