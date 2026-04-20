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

## 5. Final Correction Plan (Implemented in Code) （April 18th 2026）

### A. Evaluation Protocol First
- Use fixed seed for reproducible comparisons.
- Evaluate with two decoding paths: **greedy** and **sampling**.
- Export full distributions (not only examples): reward, raw reward, length, EOS trigger, and reward components.
- Track unique n-gram diversity to catch short-template collapse.

### B. Correlated-Proxies Mainline Fixes
- Switch reward target to **completion-only** to avoid prompt contamination.
- Keep prompt as context for a quality anchor score, not for direct reward token counting.
- Add adaptive KL controller (target KL with bounded coefficient).
- Add collapse guards (length drop, EOS spikes, reward-quality divergence).

### C. Reward Shaping Redesign
- Separate repetition into token-level and phrase-level penalties.
- Replace binary length penalty with interval-centered length score.
- Add bounded clipping for each reward component and total reward.
- Keep reward decomposition logs to identify which component drives optimization.

### D. Checkpoint Selection Rule
- Do not select checkpoints using reward only.
- Use multi-objective score: raw reward + quality anchor + EOS stability + length stability.

### E. Post-fix Research Upgrades (Deferred)
- Token-level credit assignment refinement.
- Learned reward model pipeline + InfoRM/information bottleneck.

## 6. Rigorous Protocol Files (Added for Report-grade Results)

### A. Protocol Config
- **File**: `config/rigorous_protocol.yaml`
- **Purpose**: Defines the recommended experiment contract for report results.
- **What it contains**:
    - Training seed list (default: 11, 22, 33).
    - Evaluation seed list (default: 42, 43, 44).
    - Decode modes (greedy + sampling).
    - Canonical output roots for multi-seed training/evaluation and aggregate output.
- **Why this matters**: Keeps all collaborators on the same protocol and prevents ad-hoc comparisons.

### B. Multi-seed PPO Training Entry
- **File**: `scripts/run_multiseed_ppo.sh`
- **Purpose**: Automates running PPO training across a seed list.
- **Key behavior**:
    - Loops over `--seeds`.
    - Calls `src.training.train_ppo` with `--seed` and per-seed `--save_dir`.
    - Stores checkpoints under `models/policy_ppo_multiseed/seed_<seed>/`.
- **Why this matters**: Final model claims are no longer tied to one training random seed.

### C. Multi-seed Evaluation Entry
- **File**: `scripts/run_multiseed_eval.sh`
- **Purpose**: Runs evaluation repeatedly over multiple evaluation seeds.
- **Key behavior**:
    - Loops over evaluation seeds.
    - Runs base/SFT/PPO evaluation into `results/multiseed_eval/seed_<seed>/`.
    - Uses both decode modes (`--modes greedy,sampling` by default).
    - Calls aggregation script automatically after all seeds finish.
- **Why this matters**: Separates policy quality from one-shot decoding luck.

### D. Cross-seed Aggregation
- **File**: `src/evaluation/aggregate_multiseed.py`
- **Purpose**: Aggregates per-seed evaluation outputs into report-ready statistics.
- **Output statistics**:
    - `n`, `mean`, `std`, `min`, `max`, `ci95` for each metric.
    - Organized by mode (`greedy` / `sampling`) and model (`base` / `sft` / `ppo`).
    - Preserves per-seed raw records for auditability.
- **Why this matters**: Enables rigorous reporting with mean ± std and confidence intervals.

## 7. Detailed Change Summary by File (What changed and why)

### A. Core Config and Protocol Surface

1. `config/model_config.yaml`
- Added evaluation block for fixed defaults (seed, dataset split, decode modes).
- Fixed YAML indentation to spaces (tab indentation is parser-fragile).
- Rationale: one canonical config for deterministic comparison and mode split.

2. `config/ppo_config.yaml`
- Added adaptive KL control (`target_kl`, coefficient bounds, update rate).
- Added collapse guard thresholds (`avg_completion_len`, `eos_rate`, reward-quality gap).
- Added checkpoint selection weights for multi-objective model picking.
- Rationale: avoid reward-only optimization and make trust-region behavior explicit.

3. `config/reward_config.yaml`
- Switched to completion-only reward by config.
- Split repetition into token and phrase penalties.
- Added quality-anchor weight and clipping/aggregation controls.
- Rationale: reduce exploitability of naive composite reward and improve interpretability.

4. `config/rigorous_protocol.yaml` (new)
- Stores the recommended seed/mode/output contract for report-grade experiments.
- Rationale: reproducible team protocol.

### B. Reward and Policy Interfaces

5. `src/models/reward_model.py`
- Rebuilt reward into decomposed components:
    - sentiment
    - repetition_token
    - repetition_phrase
    - length (interval-centered)
    - quality_anchor
- Added clipping and centered/normalized total reward options.
- Added `compute_reward_for_completions(..., return_components=True)`.
- Rationale: completion-only objective and per-component logging are necessary to diagnose reward hacking.

6. `src/models/policy_lm.py`
- `generate()` now supports:
    - explicit `do_sample` switch
    - seed-controlled generation
    - completion extraction
    - generation metadata (token lengths, EOS trigger)
- Fixed completion slicing with left-padding-safe logic.
- Rationale: evaluation must support deterministic greedy and stochastic sampling under controlled seeds.

### C. PPO Training Logic

7. `src/ppo/ppo_trainer.py`
- Reward path changed to completion-only.
- Rollout now records decomposition stats and collapse diagnostics.
- Added adaptive KL coefficient update.
- Added outputs for:
    - raw vs centered reward
    - quality anchor
    - avg completion length
    - EOS rate
    - reward-quality gap
- Rationale: training should expose *why* reward is increasing, not just that it increases.

8. `src/training/train_ppo.py`
- Reads expanded config fields (KL control, guards, reward decomposition).
- Added deterministic `set_seed(seed)`.
- Added `--seed` and `--save_dir` CLI args for multi-seed orchestration.
- Added multi-objective checkpoint scoring and best-checkpoint saving.
- Added early-stop based on collapse guard triggers.
- Writes richer `training_stats.json` and a `training_summary.json` with seed and best step.
- Rationale: turn one-off training into report-grade, repeatable training runs.

### D. Evaluation and Plotting

9. `src/evaluation/evaluate.py`
- Added fixed-seed setup.
- Added dual-mode evaluation (`greedy`, `sampling`).
- Added full distribution export for key metrics (not just examples).
- Added completion-only reward decomposition in evaluation output.
- Added uniqueness and collapse diagnostics (n-gram, length, EOS).
- Replaced fragile fallback branch with a cleaner model-loading path.
- Rationale: evaluation now measures both quality and failure modes in a reproducible way.

10. `src/utils/plotting.py`
- Updated to new schema (mode-aware summary + decomposed metrics).
- Reward histogram now reads full distributions.
- Training plot now includes KL coefficient and collapse indicators.
- Rationale: plots should diagnose behavior, not only show headline reward.

11. `src/evaluation/aggregate_multiseed.py` (new)
- Aggregates per-seed results with mean/std/CI95 and per-seed traceability.
- Rationale: this is the statistical layer needed for report claims.

### E. Scripts and UX

12. `scripts/run_eval.sh`
- Added `--seed` and `--modes` support.
- Removed unsafe `eval` string execution.
- Rationale: safer and reproducible CLI evaluation.

13. `scripts/run_ppo.sh`
- Supports passthrough args for advanced options.
- Rationale: easier integration with seed and output overrides.

14. `scripts/run_multiseed_ppo.sh` (new)
- Multi-seed training wrapper.
- Rationale: one command to build robust training cohorts.

15. `scripts/run_multiseed_eval.sh` (new)
- Multi-seed evaluation wrapper + auto aggregation.
- Rationale: one command to produce report-ready aggregate metrics.

### F. Documentation and Notes

16. `README.md`
- Updated methodology and evaluation descriptions to match completion-only + mode-split design.
- Updated command examples with seed and mode controls.
- Rationale: documentation aligned with actual code behavior.

17. `REPORT_NOTES.md`
- Added final correction plan and this detailed file-level rationale.
- Rationale: keeps implementation decisions and report argument coherent.

## 8. Reporting Guidance for Final Experiments

- Keep fixed-seed results as reproducibility baseline.
- Use multi-seed aggregates as primary claims.
- Report both decode modes.
- Use component-level reward trends to explain gains and avoid overclaiming from total reward alone.

## 9. Post-rerun Issue Log (Apr 19th 5am 2026)

### A. Why "reward ~ 0" happened and how it was fixed
- **Observed confusion**: Earlier comparison files showed mean reward close to zero, which looked like PPO was not learning.
- **Root cause**: Training reward uses centered aggregation (`center_total_reward: true`), so batch mean is intentionally shifted toward 0.
- **Fix applied**:
    - Keep centered reward for PPO optimization stability.
    - Switch evaluation reporting to non-centered view (`evaluation.use_centered_reward: false`) and report `mean_raw_reward` for interpretation.
- **Result**: Evaluation numbers are now human-readable and no longer falsely suggest "no learning".

### B. Short-sentence collapse ("minimal completion" strategy)
- **Observed behavior**: Model preferred short, low-risk completions to avoid penalties.
- **Why short-penalty alone was insufficient**: The model could still exploit sentiment keywords while terminating early.
- **Fix bundle applied**:
    - Prompt cleaning before training/evaluation (remove HTML/noise tokens).
    - Interval-centered length score retained, but combined with richer diagnostics.
    - Collapse guard + checkpoint selection now include reward-quality divergence and EOS/length stability.
    - Decoding controls aligned across rollout, logging, and evaluation.
- **Current status**: Mean completion length increased substantially versus the old reward-hacking run.

### C. Repetition-driven reward hacking
- **Observed behavior**: Repeated phrase loops were rewarded by sentiment while harming fluency.
- **Fix applied**:
    - Token-level and phrase-level repetition penalties split and logged separately.
    - `repetition_penalty` and `no_repeat_ngram_size` propagated consistently through policy generation and evaluation modes.
- **Current status**: Repetition metrics dropped sharply compared with the old run; explicit loop patterns are much less frequent.

### D. Windows terminal visibility and UX issues
- **Observed behavior**: Unicode progress bars and mixed encoding caused unreadable logs.
- **Fix applied**:
    - Enforce UTF-8/ASCII progress bar settings.
    - Silence noisy HF hub progress/warnings where possible.
    - Add per-step and total runtime printing in the orchestration script.

## 10. What Improved vs. the Previous Reward-Hacking Run

- PPO now shows stronger advantage over base model in both decode modes.
- Repetition exploitation is significantly reduced.
- Reward reporting is clearer (`raw_reward` visible), reducing misinterpretation.
- Multi-seed full rerun reached all configured updates and produced aggregate outputs cleanly.

## 11. Next Direction Toward the Final Goal (Fluent, Positive, Non-repetitive, On-topic)

### Recommended primary direction: strengthen semantic alignment reward
- **Problem still visible**: reward-quality gap remains large in training traces, meaning the policy can still gain reward without being strongly prompt-aligned.
- **Next change**:
    - Upgrade `quality_anchor` from token overlap to semantic similarity (embedding-based sentence alignment).
    - Keep repetition penalties as-is (already effective), but increase the influence of semantic alignment in checkpoint selection.
- **Why this direction**:
    - It targets the remaining failure mode (off-topic but high-sentiment completions).
    - It does not undo current gains on repetition and stability.

### Suggested minimal ablation plan
1. Keep current PPO/training settings unchanged.
2. Replace quality-anchor metric only (token overlap -> embedding similarity).
3. Run a short controlled sweep on quality weight (e.g., 0.35 / 0.45 / 0.55).
4. Select by multi-objective score plus manual sample inspection.

## 12. Sync Update for Latest Rerun (Apr 19th 2026)

### A. Run status and artifact paths
- Full rerun completed successfully (exit code 0).
- Total pipeline runtime: `2060.2s`.
- Train root: `models/policy_ppo_multiseed_goal_tuned_try_now_rerun`.
- Eval root: `results/multiseed_eval_goal_tuned_try_now_rerun`.
- Aggregate file: `results/multiseed_eval_goal_tuned_try_now_rerun/aggregate.json`.

### B. Key current metrics (multi-seed aggregate)

- **Greedy mode**:
    - PPO `mean_raw_reward = 0.8918` vs Base `0.8466` vs SFT `0.5995`.
    - PPO `mean_repetition_token = 0.0251` (close to Base `0.0244`, much better than SFT `0.1386`).
    - PPO `mean_quality_anchor = 0.0241` (lower than SFT `0.0548`).
    - PPO `mean_length = 27.83`, `eos_rate = 0.002`.

- **Sampling mode**:
    - PPO `mean_raw_reward = 0.7969` vs Base `0.7855` vs SFT `0.6065`.
    - PPO `mean_repetition_token = 0.0154` (good anti-repetition behavior).
    - PPO `mean_quality_anchor = 0.0183` (still low).
    - PPO `mean_length = 27.59`, `eos_rate = 0.008`.

### C. What changed vs the old report's reward-hacking pattern

- **Old report pattern**: reward hacking mainly appeared as *short positive suffix + immediate stop* (risk-averse early stopping).
- **Current pattern**: this short-stop behavior is no longer dominant.
    - Evidence: completion length increased to around `27-28` tokens; EOS rate is near zero in greedy and very low in sampling.
- **New remaining issue**: reward hacking shifted to *high-sentiment but weakly prompt-aligned continuations*.
    - Evidence: PPO reward is high, repetition is controlled, but quality-anchor remains low.

### D. Interpretation to carry into final report

- The mitigation work successfully removed the most obvious collapse mode (early-stop minimalist policy).
- PPO now improves reward while maintaining low repetition and stable output length.
- The unresolved gap is semantic alignment to the prompt, not fluency length control.
- Therefore, the next iteration should prioritize stronger semantic quality reward (embedding-based alignment) rather than further tightening repetition/length penalties.

### E. Reproducibility note

- The latest rerun aggregate is numerically consistent with the previous `rerun_full` aggregate for key reported metrics.
- File hashes differ because `input_root` path strings differ, but metric content matches after path field normalization.

## 13. VS Code Reopen Window Incident (Apr 19th 2026)

### A. What happened

- During evaluation, VS Code became unresponsive and showed a Reopen Window prompt.
- This was an editor/UI-layer issue, not a confirmed deadlock of the Python training/evaluation process.

### B. Log-level evidence

- VS Code main process log recorded repeated `CodeWindow: detected unresponsive` and `UnresponsiveSampleError` entries around the incident time.
- Renderer/exthost logs also recorded temporary extension-host unresponsiveness, then recovery/restart.
- Terminal logs showed PTY heartbeat/deadline warnings (intermittent UI communication lag), which can amplify freeze symptoms under heavy terminal redraw pressure.

### C. Practical mitigation applied

- Updated orchestration runner to support a more stable default output path:
    - Default mode now favors file logging per step (low terminal redraw pressure).
    - Optional live streaming is still available when needed.
- Kept low-redraw defaults for progress display (`RLHF_DISABLE_TQDM=1` unless explicitly enabled).
- Added per-step log file outputs and failure-tail printing to keep debuggability without flooding the integrated terminal.

### D. Recommended run mode for long experiments on Windows VS Code

- Use the stable default mode (no `-LiveOutput`) for long train/eval jobs.
- Only enable `-LiveOutput` for short debug runs where immediate token-level output is required.

## 14. Unicode Display Artifacts vs. Real Model Output (Apr 19th 2026)

### A. What looked like "garbled text"

- Some terminal previews showed characters like `鈥?` during sample inspection.
- This was primarily a Windows terminal code-page display issue, not necessarily corrupted JSON output.

### B. What was verified from saved records

- Raw records in `results/*/seed_44/*_records.jsonl` were valid UTF-8 and mostly contained normal punctuation variants (for example smart quotes `’`, `“` and symbol `£`).
- Therefore, the screenshot-level garble was mostly render/encoding mismatch in terminal display.

### C. Fixes applied

- Runner now explicitly enforces UTF-8 console settings in `scripts/run_goal_tuned_experiment.ps1` before execution.
- Reward text normalization was strengthened in `src/models/reward_model.py`:
    - Unicode NFKC normalization.
    - Canonical mapping of smart quotes/dashes/ellipsis/zero-width chars to stable forms.
- This reduces quality/repetition metric instability caused by punctuation variants while preserving the original completion text for audit files.

### D. Practical interpretation rule

- Always treat JSONL records as source of truth for content analysis.
- Treat terminal previews as convenience only (subject to local rendering/encoding differences).

## 15. Fair Recheck Under Unified Metric (Apr 19th 2026, seed=44)

### A. Why this recheck was necessary

- Previous short-run comparisons mixed different reward configs (`token_overlap` vs `token_f1_clean`), so quality numbers were not strictly apples-to-apples.
- To make a fair decision, all candidate PPO checkpoints were re-evaluated under one shared setup:
    - Eval config: `config/model_config_qualityfix_short.yaml`
    - Reward config: `config/reward_config_qualityfix_short_v3.yaml`
    - Decode modes: greedy + sampling
    - Samples: 300 prompts

### B. Fair comparison summary (PPO only)

- `gateB`
    - greedy: raw=0.4426, quality=0.0527, rep_tok=0.0250, q_zero=0.2833, eos=0.0000
    - sampling: raw=0.4163, quality=0.0383, rep_tok=0.0202, q_zero=0.4167, eos=0.0167

- `qfix_v1`
    - greedy: raw=0.4400, quality=0.0546, rep_tok=0.0263, q_zero=0.2567, eos=0.0100
    - sampling: raw=0.4011, quality=0.0385, rep_tok=0.0193, q_zero=0.3900, eos=0.0400

- `qfix_v2`
    - greedy: raw=0.4426, quality=0.0521, rep_tok=0.0250, q_zero=0.2800, eos=0.0033
    - sampling: raw=0.4198, quality=0.0376, rep_tok=0.0181, q_zero=0.4200, eos=0.0200

- `qfix_v3`
    - greedy: raw=0.4384, quality=0.0530, rep_tok=0.0273, q_zero=0.2933, eos=0.0100
    - sampling: raw=0.4030, quality=0.0389, rep_tok=0.0189, q_zero=0.4033, eos=0.0300

### C. Decision for current objective

- Objective: reduce frequent quality=0 while preserving anti-repetition and reward behavior.
- Best quality-fix effect: `qfix_v1` (largest q_zero reduction in both greedy and sampling).
- Most conservative reward/repetition profile: `qfix_v2` (raw/rep closest or slightly better than gateB, but little quality gain).
- Practical recommendation:
    - If priority is quality-zero reduction: choose `qfix_v1`.
    - If priority is minimal behavioral drift: choose `qfix_v2`.

### D. Additional note on Unicode-like artifacts in outputs

- Under the unified recheck, non-ASCII characters still appear in a minority of sampling completions across all variants.
- This is model text variation, not JSON corruption; scoring is now more robust due normalization updates.

## 16. Why Current Numbers Differ From Earlier Terminal Table (Apr 19th 2026)

### A. Root cause of mismatch

- Earlier table compared runs evaluated under different reward metrics:
    - `gateB_recheck` used `token_overlap` quality definition.
    - `qualityfix_v1/v2/v3` used `token_f1_clean` quality definition.
- After re-evaluating all variants under one shared metric (`reward_config_qualityfix_short_v3.yaml`), values changed as expected.
- Therefore, the mismatch is a metric-protocol mismatch (not random bug, not file corruption).

### B. Verified effect on trend interpretation

- Under the mixed old protocol, reward vs quality-nonzero correlation looked strongly negative.
- Under the unified fair protocol:
    - Greedy side correlation becomes weak (near 0).
    - Sampling side still shows a strong trade-off.
- Interpretation: trade-off exists mainly in sampling regime; it is not uniformly a strict inverse law in all settings.

## 17. F1-Style Trade-off Feasibility and Short Trial

### A. Feasibility judgment for this project

- The "F1-like balance" idea is feasible for **checkpoint selection / model ranking**.
- It is less safe as direct PPO reward replacement in one step (can destabilize optimization scale).
- Practical engineering approach:
    - Keep PPO raw reward pipeline stable.
    - Add balanced multi-objective checkpoint score using reward and quality-nonzero rate.
    - Use unified fair eval for final decision.

### B. Code changes added for this

- `src/training/train_ppo.py`
    - Added configurable `checkpoint_selection.mode`:
        - `weighted_sum` (existing)
        - `harmonic_balance` (new)
    - Added `quality_signal` option (`mean` or `nonzero_rate`).
    - Added F-beta style control (`harmonic_beta`) and reward normalization bounds.
    - Added `quality_anchor_nonzero_rate` to logging/eval snapshots.

- `src/ppo/ppo_trainer.py`
    - Added runtime stat `quality_anchor_nonzero_rate` for each update.

### C. New short trial run (harmonic checkpoint selection)

- New configs:
    - `config/ppo_config_qualityfix_short_f1.yaml`
    - `config/reward_config_qualityfix_short_f1.yaml`
- New model:
    - `models/policy_ppo_qualityfix_short_f1/seed_11/best`
- Fair eval output:
    - `results/eval_fair_v3metric/qfix_f1/seed_44/ppo_results.json`

### D. Result summary vs existing candidates (same fair protocol)

- `qfix_f1` achieved:
    - greedy raw: `0.4472` (best among compared variants)
    - greedy quality-nonzero: `0.7367` (close to top)
    - sampling raw: `0.4026` (not top)
    - sampling quality-nonzero: `0.5967` (mid-high)

- By F-beta style average ranking (`beta=1.3`, quality emphasized):
    - Top tier: `qfix_v2` and `qfix_f1` (very close).

- By constrained score (`avg raw * min(q_nonzero_greedy, q_nonzero_sampling)`):
    - Top: `qfix_v1`, second: `qfix_f1`.

### E. Decision guidance

- If target is strict worst-case quality protection: keep `qfix_v1`.
- If target is balanced reward + quality under F-style selection: `qfix_f1` is a valid improved compromise and worth continuing.
- Recommended next step: run multi-seed (`11,22,33`) fair recheck for `qfix_f1` and `qfix_v1` only, then choose final report model.

## 18. Multi-seed Fair Recheck: v1 vs f1 (Apr 19th 2026)

### A. Run setup

- Train seeds: `11, 22, 33`
- Eval seed: `44`
- Shared eval config: `config/model_config_qualityfix_short.yaml`
- Shared reward metric for fair comparison: `config/reward_config_qualityfix_short_v3.yaml`
- Orchestration script: `scripts/run_v1_vs_f1_multiseed_fair.ps1`

### B. Output artifacts

- Summary JSON: `results/eval_fair_v3metric_v1_vs_f1/summary_eval_seed_44.json`
- Run logs: `results/run_logs/v1_vs_f1_multiseed_20260419_205909`
- Per-seed eval outputs:
    - `results/eval_fair_v3metric_v1_vs_f1/v1/train_seed_*/seed_44/`
    - `results/eval_fair_v3metric_v1_vs_f1/f1/train_seed_*/seed_44/`

### C. Aggregate comparison (mean +- std across 3 train seeds)

- `v1`
    - greedy_raw: `0.4541 +- 0.0130`
    - greedy_q_nonzero: `0.8322 +- 0.0853`
    - sampling_raw: `0.4079 +- 0.0059`
    - sampling_q_nonzero: `0.7222 +- 0.0983`
    - avg_fbeta_13: `0.5630 +- 0.0426`
    - constrained_score: `0.3119 +- 0.0487`

- `f1`
    - greedy_raw: `0.4527 +- 0.0048`
    - greedy_q_nonzero: `0.8356 +- 0.0913`
    - sampling_raw: `0.4139 +- 0.0098`
    - sampling_q_nonzero: `0.7333 +- 0.1183`
    - avg_fbeta_13: `0.5681 +- 0.0446`
    - constrained_score: `0.3184 +- 0.0562`

### D. Decision after multi-seed check

- `f1` is slightly better on the two composite objectives (`avg_fbeta_13`, `constrained_score`) and also slightly better on sampling-side quality nonzero.
- `v1` remains a strong baseline and is still competitive, but no longer clearly dominant after multi-seed averaging.
- Recommended final pick for a balanced report narrative: `f1`.
- Recommended backup baseline in report tables: `v1`.

## 19. Deferred Upgrade Roadmap (Literature-backed, Not Implemented Yet)

### A. Current decision boundary

- We are **not** upgrading the training algorithm now.
- Current practical state:
    - Use `f1` as primary balanced result.
    - Keep `v1` as backup baseline.
- The main unresolved issue is still the reward-quality tension, especially in sampling mode.

### B. Candidate methods to revisit later

1. **Constrained DPO (C-DPO)**
- Paper: *Enhancing LLM Safety via Constrained Direct Preference Optimization* (arXiv:2403.02475)
- Link: https://arxiv.org/abs/2403.02475
- Why relevant: directly optimizes preference objective with an explicit constraint, matching our reward vs quality trade-off framing.
- Expected benefit: more controllable trade-off than manual weighted scalarization.

2. **One-shot dualization for constrained alignment**
- Paper: *One-Shot Safety Alignment for Large Language Models via Optimal Dualization* (arXiv:2405.19544)
- Link: https://arxiv.org/abs/2405.19544
- Why relevant: avoids unstable and expensive iterative primal-dual loops by solving a smooth dual shortcut.
- Expected benefit: better stability when enforcing quality-style constraints.

3. **DPO / ORPO family (simpler direct alignment)**
- DPO paper: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (arXiv:2305.18290)
- Link: https://arxiv.org/abs/2305.18290
- ORPO paper: *ORPO: Monolithic Preference Optimization without Reference Model* (arXiv:2403.07691)
- Link: https://arxiv.org/abs/2403.07691
- Why relevant: removes PPO/RM complexity, often improving training simplicity and stability.

4. **KTO (human-aware utility loss)**
- Paper: *KTO: Model Alignment as Prospect Theoretic Optimization* (arXiv:2402.01306)
- Link: https://arxiv.org/abs/2402.01306
- Why relevant: useful when feedback is binary desirable/undesirable, and utility shaping is important.

5. **Offline regularized RL / single-trajectory optimization (DRO)**
- Paper: *Offline Regularised Reinforcement Learning for Large Language Models Alignment* (arXiv:2405.19107)
- Link: https://arxiv.org/abs/2405.19107
- Why relevant: can learn from prompt-response-feedback triplets, reducing dependence on expensive pairwise preferences.

### C. Priority order for future implementation

1. Priority-1: constrained objective route (C-DPO style or dualized constrained PPO).
2. Priority-2: direct preference baseline route (DPO or ORPO).
3. Priority-3: KTO / DRO route if supervision format shifts to binary or single-trajectory feedback.

### D. Minimal future experiment contract (if resumed)

- Keep the same fair comparison protocol used in Section 18.
- Train seeds: `11,22,33`; eval seed: `44` (or extend to `42,43,44`).
- Compare at least: `old_goal`, `v1`, `f1`, and one constrained-method candidate.
- Report both raw reward and quality-nonzero (greedy + sampling), plus composite criteria.

## 20. Full-Scale Re-run Completed (Apr 19th 2026, Fair v3, 5x3)

### A. Protocol and artifacts

- Objective: run the old-scale volume while keeping the unified fair v3 metric.
- Variants compared: `old_goal`, `v1`, `f1`.
- Train seeds: `11,22,33,44,55`.
- Eval seeds: `42,43,44`.
- Each variant has `5 x 3 = 15` eval cells.
- Shared eval reward config: `config/reward_config_qualityfix_short_v3.yaml`.
- Orchestration script: `scripts/run_three_variants_full_fair_v3.ps1`.
- Summary artifact: `results/eval_fair_v3_full_three_variants/summary.json`.
- Run logs: `results/run_logs/three_variants_full_fair_v3_20260419_230609`.

### B. Aggregate results (mean +- std across 15 cells)

- `old_goal`
    - greedy_raw: `0.4542 +- 0.0036`
    - greedy_q_nonzero: `0.7688 +- 0.0324`
    - sampling_raw: `0.4204 +- 0.0053`
    - sampling_q_nonzero: `0.6627 +- 0.0462`
    - avg_fbeta_13: `0.5497 +- 0.0158`
    - constrained_score: `0.2898 +- 0.0208`

- `v1`
    - greedy_raw: `0.4558 +- 0.0132`
    - greedy_q_nonzero: `0.8460 +- 0.0600`
    - sampling_raw: `0.4196 +- 0.0108`
    - sampling_q_nonzero: `0.7757 +- 0.0683`
    - avg_fbeta_13: `0.5833 +- 0.0341`
    - constrained_score: `0.3402 +- 0.0368`

- `f1`
    - greedy_raw: `0.4544 +- 0.0089`
    - greedy_q_nonzero: `0.8372 +- 0.0663`
    - sampling_raw: `0.4215 +- 0.0094`
    - sampling_q_nonzero: `0.7629 +- 0.0779`
    - avg_fbeta_13: `0.5799 +- 0.0343`
    - constrained_score: `0.3347 +- 0.0395`

### C. Decision update under full-scale evidence

- New full-scale winner is `v1` (slightly above `f1` on both composite metrics).
- `v1 - f1` gap:
    - avg_fbeta_13: `+0.0034`
    - constrained_score: `+0.0055`
    - sampling_q_nonzero: `+0.0128`
- Versus `old_goal`, both quality-fix variants are clearly better on quality-nonzero and composite criteria.

### D. Interpretation note

- Earlier section-18 preference for `f1` came from smaller-scale multi-seed evidence.
- After restoring old-scale volume (5 train seeds x 3 eval seeds), `v1` regains a small but consistent edge.
- Final recommendation for report main table (full-scale fair v3):
    - Primary: `v1`
    - Secondary: `f1`
    - Historical baseline: `old_goal`
