# RLHF Project Report Notes

## 1. Experiment Setup & Motivation
- **Objective**: Demonstrate how PPO shapes LM outputs under a reward signal, specifically focusing on the trade-off between reward maximization and natural language generation.
- **Baseline**: Supervised Fine-Tuning (SFT) on IMDB dataset.
- **RL Method**: PPO with a composite reward function (Sentiment + Length + Repetition Penalty).

## 2. Key Findings & Analysis

### A. The "Reward Hacking" Phenomenon (Low KL Constraint)
- **Experiment**: `kl_coef = 0.1`, `entropy_coef = 0.05`.
- **Observation**: 
    - The model quickly converged (around 300 steps) to a stable reward of ~0.3.
    - **Qualitative Analysis**: The model learned a "lazy" strategy: appending generic positive phrases like **"better and"** to the end of sentences.
    - **Why**: This is a local optimum. It's a low-risk, high-stability way to gain positive sentiment scores without risking the linguistic structure of the sentence (which might incur KL penalties if changed too drastically).
- **Conclusion**: Demonstrates the effectiveness of PPO in optimizing the reward metric, but highlights the danger of "Goodhart's Law" – when a measure becomes a target, it ceases to be a good measure.

### B. The Impact of KL Regularization (High KL Constraint)
- **Experiment**: `kl_coef = 0.3` (Ablation Study).
- **Observation**:
    - **Training Dynamics**: Unlike the smooth curve of KL=0.1, the reward curve for KL=0.3 was much more volatile and slower to rise. It did not fully plateau even after 500 steps, indicating ongoing exploration.
    - **Reward Distribution**: The PPO model showed a **polarized distribution** (bimodal at 0.0 and ~1.0).
        - *Interpretation*: The model engaged in "Selective Optimization". 
        - For "easy" prompts, it successfully optimized to high sentiment (1.0).
        - For "hard" prompts (where changing sentiment would require drastic rewriting), it chose to "play it safe" (stay close to base model) to avoid heavy KL penalties, resulting in neutral rewards (0.0) rather than risking negative rewards.
    - **Negative Reward Elimination**: Unlike Base/SFT models which had a peak at -0.2, PPO successfully eliminated negative rewards, showing it learned to avoid penalties.

### C. Convergence Speed vs. Constraint Strength
- **Low KL (0.1)**: Fast convergence (~300 steps). The model quickly found a loophole.
- **High KL (0.3)**: Slow convergence (>500 steps). The model struggled to find a policy that satisfies both the Reward Model (be positive) and the KL Constraint (be natural/close to original).
- **Implication**: Stricter constraints require longer training times as the solution space becomes more complex to navigate.

### D. Discrepancy between Evaluation and Training Dynamics
- **Observation**: 
    - In the High KL experiment, the **Training Reward** (green curve) continued to trend upwards (with variance) until the end.
    - However, the **Evaluation Reward** (on fixed 5 prompts) plateaued early at ~0.6491 and consistently produced the "better and" suffix.
- **Analysis**:
    - **Sampling Difference**: Training uses stochastic sampling on diverse batches (exploration), while evaluation often uses greedy/low-temp decoding on fixed prompts (exploitation).
    - **Persistent Local Optimum**: Even though the model was improving globally (on the training set), for those specific evaluation prompts, the "better and" strategy remained the most confident (highest probability) output, showing how stubborn reward hacking behaviors can be.
- **Takeaway**: Evaluation on a small, fixed set of prompts can mask the true learning progress and dynamics of the model.

## 3. Future Work / Limitations
- **Reward Model Robustness**: The "better and" hack suggests the Reward Model (DistilBERT) is susceptible to simple keyword spotting. A more robust RM (e.g., trained on human preference pairs) might mitigate this.
- **Training Duration**: The High KL experiment suggests that 500 steps is insufficient for convergence under strict constraints. Extending to 1000+ steps would be necessary to see the final policy performance.

## 4. Visual Evidence (Plots to Include)
1. **Reward Curve Comparison**: Overlay the smoothed reward curves of KL=0.1 vs KL=0.3 to show the difference in stability and convergence speed.
2. **Reward Distribution Histogram**: The "Base vs SFT vs PPO" plot showing the polarization effect in PPO.
3. **Sample Generations**: Table comparing a specific prompt's output under Base, SFT, PPO(Low KL), and PPO(High KL).
