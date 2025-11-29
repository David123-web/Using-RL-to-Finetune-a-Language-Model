import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os


def plot_training_stats(stats_path: str, output_dir: str = "plots"):
    """Plot training statistics from saved JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Extract metrics
    updates = list(range(1, len(stats) + 1))
    rewards = [s['reward'] for s in stats]
    policy_losses = [s['policy_loss'] for s in stats]
    value_losses = [s['value_loss'] for s in stats]
    entropies = [s['entropy'] for s in stats]
    kl_divs = [s['kl_div'] for s in stats]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PPO Training Statistics', fontsize=16)
    
    # Rewards
    axes[0, 0].plot(updates, rewards, linewidth=2, color='green')
    axes[0, 0].set_xlabel('Update')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Policy Loss
    axes[0, 1].plot(updates, policy_losses, linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Update')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Value Loss
    axes[0, 2].plot(updates, value_losses, linewidth=2, color='orange')
    axes[0, 2].set_xlabel('Update')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Value Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 0].plot(updates, entropies, linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[1, 1].plot(updates, kl_divs, linewidth=2, color='red')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].set_title('KL vs Reference Policy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Smoothed Reward (moving average)
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_updates = updates[window-1:]
        axes[1, 2].plot(updates, rewards, alpha=0.3, color='green', label='Raw')
        axes[1, 2].plot(smoothed_updates, smoothed_rewards, linewidth=2, 
                       color='darkgreen', label=f'Smoothed (window={window})')
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_title('Reward (Raw vs Smoothed)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].plot(updates, rewards, linewidth=2, color='green')
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_title('Average Reward')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training plot to {output_path}")
    plt.close()


def plot_comparison(comparison_path: str, output_dir: str = "plots"):
    """Plot comparison of different models"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)
    
    summary = comparison['summary']
    models = list(summary.keys())
    
    # Extract metrics
    metrics = {
        'mean_reward': 'Average Reward',
        'mean_sentiment': 'Average Sentiment',
        'diversity_ratio': 'Diversity Ratio',
        'mean_repetition': 'Mean Repetition',
        'mean_length': 'Mean Length'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        if idx >= 6:
            break
        
        values = [summary[model][metric_key] for model in models]
        bars = axes[idx].bar(models, values, color=colors[:len(models)])
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(metric_name)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=10)
    
    # Hide the last empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_reward_distribution(results_paths: Dict[str, str], output_dir: str = "plots"):
    """Plot reward distribution for different models"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results_paths), figsize=(6*len(results_paths), 5))
    if len(results_paths) == 1:
        axes = [axes]
    
    fig.suptitle('Reward Distributions', fontsize=16)
    
    for idx, (model_name, path) in enumerate(results_paths.items()):
        with open(path, 'r') as f:
            results = json.load(f)
        
        # Get reward distribution from examples
        rewards = [ex['reward'] for ex in results['examples']]
        
        axes[idx].hist(rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[idx].set_xlabel('Reward')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{model_name.upper()} Model')
        axes[idx].axvline(np.mean(rewards), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'reward_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reward distribution plot to {output_path}")
    plt.close()


def generate_all_plots(
    training_stats_path: str = None,
    comparison_path: str = None,
    results_paths: Dict[str, str] = None,
    output_dir: str = "plots"
):
    """Generate all available plots"""
    print("Generating plots...")
    
    if training_stats_path and os.path.exists(training_stats_path):
        print("Plotting training statistics...")
        plot_training_stats(training_stats_path, output_dir)
    
    if comparison_path and os.path.exists(comparison_path):
        print("Plotting model comparison...")
        plot_comparison(comparison_path, output_dir)
    
    if results_paths:
        print("Plotting reward distributions...")
        # Filter to only existing paths
        existing_paths = {k: v for k, v in results_paths.items() if os.path.exists(v)}
        if existing_paths:
            plot_reward_distribution(existing_paths, output_dir)
    
    print(f"✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument("--training_stats", type=str, help="Path to training stats JSON")
    parser.add_argument("--comparison", type=str, help="Path to comparison JSON")
    parser.add_argument("--base_results", type=str, help="Path to base model results")
    parser.add_argument("--sft_results", type=str, help="Path to SFT model results")
    parser.add_argument("--ppo_results", type=str, help="Path to PPO model results")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    
    args = parser.parse_args()
    
    results_paths = {}
    if args.base_results:
        results_paths['base'] = args.base_results
    if args.sft_results:
        results_paths['sft'] = args.sft_results
    if args.ppo_results:
        results_paths['ppo'] = args.ppo_results
    
    generate_all_plots(
        training_stats_path=args.training_stats,
        comparison_path=args.comparison,
        results_paths=results_paths if results_paths else None,
        output_dir=args.output
    )
