# RLHF Project Report Notes

## 1. Experiment Setup & Motivation
- **Objective**: Demonstrate how PPO shapes LM outputs under a reward signal, specifically focusing on the trade-off between reward maximization and natural language generation.
- **Baseline**: Supervised Fine-Tuning (SFT) on IMDB dataset.
- **RL Method**: PPO with a composite reward function (Sentiment + Length + Repetition Penalty).

## 2. Key Findings & Analysis

### A. "Strategic Minimalist": A Nuanced Form of Reward Hacking
- **Observation**: 
    - The PPO model converged to a distinct pattern: appending a short positive suffix like **"better and"** and then **immediately terminating generation (Early Stopping)**.
    - **Specific Example**: 
        - *Prompt*: "its a totally average film with a few semi-alright action sequences that make the plot seem a little.."
        - *Response*: "...make the plot seem a little **better and**" (Stops immediately).
        - *Reward*: ~0.35 (Mediocre but positive).
- **Analysis**: 
    - **The "Minimum Effective Dose" Strategy**: The model discovered a local optimum where adding specific positive tokens ("better") flips the sentiment classifier from neutral to slightly positive, while immediately stopping prevents incurring Length Penalties or KL Divergence penalties associated with generating longer, divergent text.
    - **Risk Aversion**: The model learned that "speaking more" increases the risk of negative rewards or grammatical errors (high KL), so it chose the safest path to a positive score.

### B. Selective Optimization & Batch Dynamics
- **Phenomenon**: 
    - While the global **Mean Reward** across the evaluation batch remained high (~0.7), specific "hard" prompts (like Example 3 above) stagnated at lower rewards (~0.35).
- **Why it happens**:
    - **Gradient Domination**: The PPO algorithm optimizes the average expected return. If fully rewriting a "hard" prompt to achieve a 0.9 score requires high KL divergence (risky exploration), the optimizer "sacrifices" this individual prompt.
    - It is mathematically more efficient for the model to secure easy wins (1.0 rewards) on malleable prompts and perform "damage control" (0.35 reward) on rigid prompts, rather than risking stability to improve the outlier.
    - This confirms that PPO can exhibit **Selective Optimization**, focusing heavily on prompts where the reward signal is easiest to exploit.

### C. Evaluation vs. Exploration Discrepancy
- **Observation**: 
    - During training steps, the model briefly explored higher-reward (0.9) but riskier completions for the "hard" prompt.
    - However, in the periodic evaluations (every 50 steps), the output collapsed deterministically to the "better and" strategy.
- **Interpretation**:
    - **Mode Collapse in Inference**: Even if the policy distribution retains some probability for complex answers, the "safe" answer (better and + EOS) became the highest probability path (Argmax).
    - Since evaluation typically uses lower temperature or greedy decoding, it masks the underlying exploration and shows the model appearing "stuck" on a mediocre answer, highlighting the difference between a model's *capability* (seen in training variance) and its *policy preference* (seen in eval).

### D. The Impact of Constraints (Low vs. High KL)
- **Low KL (0.1)**: The model converged faster (~300 steps) to this "safe hack." It realized quickly that it could game the reward model with minimal changes.
- **High KL (0.3)**: The convergence was slower and more volatile. The strong constraint made it harder for the model to even append "better and" without penalty, leading to a polarized distribution where it either succeeded perfectly or stayed completely neutral to avoid the KL cost.

## 3. Future Work / Limitations
- **Reward Model Robustness**: The "better and" hack suggests the Reward Model (DistilBERT) is susceptible to simple keyword spotting. A more robust RM (e.g., trained on human preference pairs) might mitigate this.
- **Training Duration**: The High KL experiment suggests that 500 steps is insufficient for convergence under strict constraints. Extending to 1000+ steps would be necessary to see the final policy performance.

## 4. Visual Evidence (Plots to Include)
1. **Reward Curve Comparison**: Overlay the smoothed reward curves of KL=0.1 vs KL=0.3 to show the difference in stability and convergence speed.
2. **Reward Distribution Histogram**: The "Base vs SFT vs PPO" plot showing the polarization effect in PPO.
3. **Sample Generations**: Table comparing a specific prompt's output under Base, SFT, PPO(Low KL), and PPO(High KL).
