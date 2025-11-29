import yaml
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import json

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.models.policy_lm import PolicyLM, PolicyConfig
from src.models.reward_model import RewardModel
from src.ppo.ppo_trainer import PPOTrainer, PPOHyperParams
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
    max_length: int
    max_new_tokens: int
    w_sentiment: float
    w_repetition: float
    w_length: float
    min_tokens: int
    max_tokens: int
    prompt_dataset: str
    prompt_split: str
    save_dir: str
    log_every: int
    eval_every: int
    save_every: int


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_prompts(dataset_name: str, split: str) -> List[str]:
    """Load and prepare prompts from dataset"""
    ds = load_dataset(dataset_name, split=split)
    
    # For IMDB dataset, extract beginning of text as prompt
    prompts = []
    for example in ds:
        text = example['text']
        # Take first ~20 words as prompt
        words = text.split()[:20]
        prompt = ' '.join(words)
        prompts.append(prompt)
    
    return prompts


def evaluate_policy(policy: PolicyLM, reward_model: RewardModel, 
                   prompts: List[str], n_samples: int = 10) -> Dict[str, Any]:
    """Evaluate policy on sample prompts"""
    sample_prompts = prompts[:n_samples]
    
    # Generate responses
    responses = policy.generate(
        sample_prompts,
        max_new_tokens=32,
        temperature=1.0
    )
    
    # Compute rewards
    rewards = reward_model.compute_reward(responses)
    
    # Compute sentiment scores
    sentiment_scores = reward_model.sentiment_score(responses)
    
    return {
        'mean_reward': rewards.mean().item(),
        'mean_sentiment': sentiment_scores.mean().item(),
        'examples': list(zip(sample_prompts, responses, rewards.tolist()))
    }


def main(config_path: str = "config/ppo_config.yaml",
         reward_config_path: str = "config/reward_config.yaml"):
    
    # Load configs
    ppo_cfg_raw = load_config(config_path)
    reward_cfg_raw = load_config(reward_config_path)
    
    cfg = PPOTrainConfig(
        model_name=ppo_cfg_raw["model"]["name"],
        tokenizer_name=ppo_cfg_raw["model"]["tokenizer_name"],
        reward_model_name=reward_cfg_raw["reward_model"]["name"],
        batch_size=ppo_cfg_raw["ppo"]["batch_size"],
        rollout_batch_size=ppo_cfg_raw["ppo"]["rollout_batch_size"],
        n_updates=ppo_cfg_raw["ppo"]["n_updates"],
        epochs_per_update=ppo_cfg_raw["ppo"]["epochs_per_update"],
        gamma=ppo_cfg_raw["ppo"]["gamma"],
        lam=ppo_cfg_raw["ppo"]["lam"],
        clip_range=ppo_cfg_raw["ppo"]["clip_range"],
        value_coef=ppo_cfg_raw["ppo"]["value_coef"],
        entropy_coef=ppo_cfg_raw["ppo"]["entropy_coef"],
        kl_coef=ppo_cfg_raw["ppo"]["kl_coef"],
        max_length=ppo_cfg_raw["ppo"]["max_length"],
        max_new_tokens=ppo_cfg_raw["ppo"].get("max_new_tokens", 32),
        w_sentiment=reward_cfg_raw["weights"]["sentiment"],
        w_repetition=reward_cfg_raw["weights"]["repetition"],
        w_length=reward_cfg_raw["weights"]["length"],
        min_tokens=reward_cfg_raw["length_target"]["min_tokens"],
        max_tokens=reward_cfg_raw["length_target"]["max_tokens"],
        prompt_dataset="imdb",
        prompt_split=ppo_cfg_raw["data"]["prompt_split"],
        save_dir=ppo_cfg_raw["logging"]["save_dir"],
        log_every=ppo_cfg_raw["logging"]["log_every"],
        eval_every=ppo_cfg_raw["logging"]["eval_every"],
        save_every=ppo_cfg_raw["logging"].get("save_every", 100),
    )
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize policy (current)
    print("Loading policy model...")
    policy = PolicyLM(PolicyConfig(
        model_name=cfg.model_name,
        tokenizer_name=cfg.tokenizer_name,
        max_length=cfg.max_length,
    ))
    
    # Initialize reference policy (frozen copy)
    print("Loading reference policy...")
    ref_policy = PolicyLM(PolicyConfig(
        model_name=cfg.model_name,
        tokenizer_name=cfg.tokenizer_name,
        max_length=cfg.max_length,
    ))
    
    # Initialize reward model
    print("Loading reward model...")
    reward_model = RewardModel(
        model_name=cfg.reward_model_name,
        w_sentiment=cfg.w_sentiment,
        w_repetition=cfg.w_repetition,
        w_length=cfg.w_length,
        min_tokens=cfg.min_tokens,
        max_tokens=cfg.max_tokens,
    )
    
    # Initialize PPO trainer
    hparams = PPOHyperParams(
        clip_range=cfg.clip_range,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        gamma=cfg.gamma,
        lam=cfg.lam,
        epochs_per_update=cfg.epochs_per_update,
        batch_size=cfg.batch_size,
        kl_coef=cfg.kl_coef,
    )
    
    trainer = PPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        hparams=hparams,
        max_length=cfg.max_length,
        max_new_tokens=cfg.max_new_tokens,
    )
    
    # Load prompts
    print("Loading prompts...")
    prompts = prepare_prompts(cfg.prompt_dataset, cfg.prompt_split)
    print(f"Loaded {len(prompts)} prompts")
    
    # Create save directory
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting PPO training...")
    training_stats = []
    
    for update in tqdm(range(cfg.n_updates), desc="PPO Updates"):
        # Sample batch of prompts
        batch_indices = torch.randint(0, len(prompts), (cfg.rollout_batch_size,))
        batch_prompts = [prompts[i] for i in batch_indices]
        
        # Perform PPO update
        stats = trainer.train_step(batch_prompts)
        training_stats.append(stats)
        
        # Logging
        if (update + 1) % cfg.log_every == 0:
            print(f"\nUpdate {update + 1}/{cfg.n_updates}")
            print(f"  Reward: {stats['reward']:.4f}")
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")
            print(f"  Value Loss: {stats['value_loss']:.4f}")
            print(f"  Entropy: {stats['entropy']:.4f}")
            print(f"  KL Div: {stats['kl_div']:.4f}")
        
        # Evaluation
        if (update + 1) % cfg.eval_every == 0:
            print("\n" + "="*50)
            print("EVALUATION")
            print("="*50)
            eval_results = evaluate_policy(policy, reward_model, prompts, n_samples=5)
            print(f"Mean Reward: {eval_results['mean_reward']:.4f}")
            print(f"Mean Sentiment: {eval_results['mean_sentiment']:.4f}")
            print("\nSample Generations:")
            for i, (prompt, response, reward) in enumerate(eval_results['examples'][:3]):
                print(f"\n--- Example {i+1} ---")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Response: {response}")
                print(f"Reward: {reward:.4f}")
            print("="*50 + "\n")
        
        # Save checkpoint
        if (update + 1) % cfg.save_every == 0:
            checkpoint_dir = os.path.join(cfg.save_dir, f"checkpoint_{update+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            policy.model.save_pretrained(checkpoint_dir)
            policy.tokenizer.save_pretrained(checkpoint_dir)
            torch.save(trainer.value_head.state_dict(), 
                      os.path.join(checkpoint_dir, "value_head.pt"))
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_dir = os.path.join(cfg.save_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy.model.save_pretrained(final_dir)
    policy.tokenizer.save_pretrained(final_dir)
    torch.save(trainer.value_head.state_dict(), 
              os.path.join(final_dir, "value_head.pt"))
    
    # Save training stats
    stats_path = os.path.join(cfg.save_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {final_dir}")
    print(f"Training statistics saved to {stats_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_config.yaml")
    parser.add_argument("--reward_config", type=str, default="config/reward_config.yaml")
    args = parser.parse_args()
    main(args.config, args.reward_config)
