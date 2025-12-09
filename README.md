# RL-based Language Model Finetuning with PPO

This project implements a complete pipeline for finetuning language models using Reinforcement Learning with Human Feedback (RLHF), specifically using Proximal Policy Optimization (PPO). 

## Overview

The project demonstrates how PPO can be used to steer a language model's output towards desired behaviors (e.g., positive sentiment) using reward signals. Starting from a small pretrained model (DistilGPT2), we implement both supervised finetuning (SFT) as a baseline and PPO-based RL finetuning.

## Features

- ✅ **Complete PPO Implementation**: Full PPO algorithm with value head, advantage estimation (GAE), and clipped objective
- ✅ **Custom Reward Model**: Combines sentiment analysis, repetition penalty, and length constraints
- ✅ **Supervised Finetuning Baseline**: Standard SFT implementation for comparison
- ✅ **Comprehensive Evaluation**: Metrics for reward, sentiment, diversity, and generation quality
- ✅ **Visualization Tools**: Plotting utilities for training progress and model comparison
- ✅ **Demo Notebook**: Interactive Jupyter notebook for experimentation

## Project Structure

```
├── README.md
├── requirements.txt
├── config/
│   ├── model_config.yaml      # Model architecture config
│   ├── sft_config.yaml         # Supervised finetuning config
│   ├── ppo_config.yaml         # PPO training config
│   └── reward_config.yaml      # Reward model config
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed data
│   └── prompts/                # Prompt datasets
├── models/
│   ├── policy_sft/             # SFT checkpoints
│   ├── policy_ppo/             # PPO checkpoints
│   └── reward/                 # Reward model checkpoints
├── src/
│   ├── models/
│   │   ├── policy_lm.py        # Language model wrapper
│   │   └── reward_model.py     # Reward model implementation
│   ├── training/
│   │   ├── train_sft.py        # SFT training script
│   │   └── train_ppo.py        # PPO training script
│   ├── evaluation/
│   │   └── evaluate.py         # Evaluation metrics and comparison
│   ├── ppo/
│   │   └── ppo_trainer.py      # PPO algorithm implementation
│   └── utils/
│       ├── device.py           # Device management
│       └── plotting.py         # Visualization utilities
├── scripts/
│   ├── run_sft.sh              # Run supervised finetuning
│   ├── run_ppo.sh              # Run PPO training
│   └── run_eval.sh             # Run evaluation
├── notebooks/
│   └── demo.ipynb              # Interactive demo notebook
└── results/                    # Evaluation results and plots
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM (16GB recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Using-RL-to-Finetune-a-Language-Model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Supervised Finetuning (Baseline)

Train a baseline model using standard supervised learning:

```bash
bash scripts/run_sft.sh
```

This will:
- Load DistilGPT2 pretrained model
- Finetune on IMDB dataset
- Save checkpoint to `models/policy_sft/`

### 2. PPO Training

Train the model using Reinforcement Learning:

```bash
bash scripts/run_ppo.sh
```

This will:
- Initialize policy and reference policy
- Load sentiment-based reward model
- Perform PPO updates with custom rewards
- Save checkpoints to `models/policy_ppo/`

### 3. Evaluation

Compare all models:

```bash
bash scripts/run_eval.sh --base --sft models/policy_sft --ppo models/policy_ppo/final
```

This generates:
- Individual model evaluations
- Comparative analysis
- Results saved to `results/`

### 4. Visualization

Generate plots from results:

```bash
python -m src.utils.plotting \
  --training_stats models/policy_ppo/training_stats.json \
  --comparison results/comparison.json \
  --output plots
```

**Note**: Training visualizations and results are saved in the `plots/` folder. This includes experiments with `kl_coef` values of 0.1 and 0.3 for comparison.

## Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
model_name: "distilgpt2"
tokenizer_name: "distilgpt2"
max_length: 64
```

### PPO Configuration (`config/ppo_config.yaml`)

Key hyperparameters:
- `clip_range`: 0.2 (PPO clipping parameter)
- `value_coef`: 0.5 (value loss coefficient)
- `entropy_coef`: 0.01 (entropy bonus coefficient)
- `kl_coef`: 0.1 (KL divergence penalty) **[Tested: 0.1 and 0.3]**
- `gamma`: 0.99 (discount factor)
- `lam`: 0.95 (GAE lambda)

**Note**: We have tested `kl_coef` values of 0.1 and 0.3 to explore the tradeoff between policy improvement and maintaining similarity to the reference model. Results and visualizations from these experiments are available in the `plots/` folder.

### Reward Configuration (`config/reward_config.yaml`)

The reward model combines multiple signals:
- **Sentiment**: Uses DistilBERT sentiment classifier (weight: 1.0)
- **Repetition Penalty**: Discourages repetitive text (weight: -0.5)
- **Length Bonus**: Encourages appropriate length (weight: 0.1)

## Methodology

### Reward Model

The reward function combines three components:

```
R(text) = w_sentiment * sentiment_score(text) 
          + w_repetition * repetition_penalty(text)
          + w_length * length_bonus(text)
```

Where:
- `sentiment_score`: Probability of positive sentiment from DistilBERT
- `repetition_penalty`: 1 - (unique_tokens / total_tokens)
- `length_bonus`: 0 if within target range, -1 otherwise

### PPO Algorithm

1. **Rollout Collection**: Generate completions from current policy
2. **Reward Computation**: Score completions using reward model
3. **Advantage Estimation**: Compute advantages using GAE
4. **Policy Update**: Optimize clipped PPO objective
5. **Value Update**: Train value head to predict returns

The PPO objective:

```
L_CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
```

Where r(θ) is the probability ratio between new and old policies.

## Evaluation Metrics

The evaluation pipeline computes:

1. **Reward Metrics**: Mean, std, min, max rewards
2. **Sentiment Scores**: Positive sentiment probability
3. **Diversity Metrics**: Unique token ratio, vocabulary diversity
4. **Repetition Metrics**: Token repetition statistics
5. **Length Statistics**: Mean, std, min, max generation lengths
6. **Qualitative Examples**: Side-by-side comparisons

## Results

Expected improvements from PPO training:
- ✅ Higher average reward (sentiment + diversity + length)
- ✅ More positive sentiment in generations
- ✅ Better control over output characteristics
- ✅ Maintained fluency and coherence

### Viewing Results

- **Training visualizations**: See the `plots/` folder for generated training statistics and comparison plots from experiments with `kl_coef` values of 0.1 and 0.3
- **Quantitative metrics**: See `results/comparison.json` for detailed metrics after training
- **Model checkpoints**: Located in `models/policy_ppo/` and `models/policy_sft/`

The `plots/` folder contains visualizations comparing training results with different KL coefficient values, demonstrating the effect of this hyperparameter on training stability and model performance.

## Interactive Demo

Explore the project interactively:

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook includes:
- Model loading and generation
- Reward computation
- Training visualization
- Model comparison

## Development

### Adding Custom Reward Functions

Modify `src/models/reward_model.py` to add custom reward signals:

```python
def custom_reward(self, texts: List[str]) -> torch.Tensor:
    # Your custom reward logic here
    scores = []
    for text in texts:
        score = compute_your_metric(text)
        scores.append(score)
    return torch.tensor(scores, device=self.device)
```

Then update the `compute_reward` method to include your signal.

### Modifying PPO Hyperparameters

Edit `config/ppo_config.yaml` to tune training:
- Increase `n_updates` for longer training
- Adjust `clip_range` for different policy update sizes
- Tune `entropy_coef` for exploration vs exploitation
- Modify `kl_coef` to control divergence from reference policy

### Using Different Base Models

Change model in `config/model_config.yaml`:
```yaml
model_name: "gpt2"  # or "gpt2-medium", "distilgpt2", etc.
```

## Computing Resources

### Recommended Specifications
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 2080, V100)
- **RAM**: 16GB+ system memory
- **Storage**: 5GB for models and datasets

### Training Time Estimates
- **SFT**: ~30-60 minutes (2000 steps, batch size 8)
- **PPO**: ~1-2 hours (500 updates, batch size 16)
- **Evaluation**: ~10-15 minutes (500 samples)

### Memory Optimization Tips
- Reduce `batch_size` and `rollout_batch_size` in configs
- Use `gradient_accumulation_steps` for effective larger batches
- Decrease `max_length` and `max_new_tokens` for shorter sequences
- Enable mixed precision training (add to training scripts)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch sizes in config files
- Use gradient checkpointing
- Try DistilGPT2 instead of GPT2

### Slow Training
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce evaluation frequency
- Use fewer training steps for initial experiments

### Poor Reward Improvements
- Check reward model is working: test on sample texts
- Adjust reward weights in `reward_config.yaml`
- Increase `kl_coef` if diverging too much from reference
- Try different learning rates

## Academic Integrity

This project uses the following external resources:
- **Transformers Library**: HuggingFace for model loading
- **Datasets Library**: HuggingFace for data processing
- **PyTorch**: Deep learning framework
- **DistilGPT2**: Pretrained base model
- **DistilBERT**: Sentiment classification reward model

All implementations of PPO, reward modeling, and training pipelines are original work for this course project.

## References

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. Ziegler et al. (2019). "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593
3. Ouyang et al. (2022). "Training language models to follow instructions with human feedback." (InstructGPT)
4. Stiennon et al. (2020). "Learning to summarize from human feedback."

## License

This project is for academic purposes as part of ECE1508 course at University of Toronto.

## Contributors

Zuhao（David）Zhang 1005828080
Yiwei (George) Cao 1005556426
