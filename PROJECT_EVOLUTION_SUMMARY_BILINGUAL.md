# Project Evolution Summary (Bilingual)

## 0) Scope and Goal | 范围与目标

This document summarizes the project development path up to now, focusing on:
1. Main design logic over time
2. Problems encountered
3. How each problem was addressed
4. What the final evidence says

本文梳理项目迄今为止的演进过程，重点包括：
1. 每个阶段的核心思路
2. 出现的问题
3. 如何克服问题
4. 目前证据支持的结论

---

## 0.1) Corrections to the remembered timeline | 对你回忆流程的纠正

Your 8-step memory is largely accurate. Two important corrections:

你的 8 步回忆整体是准确的，但有两点需要纠正：

- Correction A:
  - CN: KL=0.3 是早期“固定 KL”对照实验中的一个设置，不是最终长期采用的唯一策略。后续主线改为自适应 KL（adaptive KL）+ 多目标检查点选择。
  - EN: KL=0.3 was an early fixed-KL trial setting, not the final long-term strategy. The later mainline used adaptive KL plus multi-objective checkpoint selection.

- Correction B:
  - CN: InfoRM/信息瓶颈是“延期升级路线”的候选，不是已经完整落地并用于最终 old_goal 主结果的模块。
  - EN: InfoRM/information bottleneck remained a deferred upgrade item, not a fully implemented module in the final old_goal main results.

---

## 1) Build PPO pipeline | 创建 PPO 训练主线

### Data snapshot | 数据快照

| Item | Value |
|---|---|
| Baseline | SFT on IMDB |
| RL method | PPO |
| Initial reward components | Sentiment + Length + Repetition penalty |
| Early quantitative sign | Batch mean reward around 0.7 (observed) |

### Analysis | 分析

- CN: 第一阶段完成了从 SFT 到 PPO 的可运行闭环，系统能够在 reward 指标上快速抬升，说明优化链路有效。
- EN: Phase 1 established a working SFT-to-PPO pipeline. Reward rose quickly, confirming the optimization loop was functional.

### Conclusion | 结论

- CN: “能训起来”这件事成立，但此时 reward 上升并不等于语言质量提升，后续风险很快暴露。
- EN: The pipeline was trainable, but reward gain did not necessarily mean quality gain; this risk surfaced immediately afterward.

---

## 2) Reward hacking discovered | 发现 reward hacking

### Data snapshot | 数据快照

| Observation | Evidence |
|---|---|
| Hard prompt reward stagnation | Around 0.35 |
| Global mean reward still high | Around 0.7 |
| Typical hacked behavior | Add short positive suffix (for example better) and stop early |

### Analysis | 分析

- CN: 模型学到了“最小有效剂量”策略：用极少量正向词触发情感奖励，再提前停止以规避长度/偏离风险。这是典型目标错配，而不是训练崩溃。
- EN: The model learned a minimum-effective-dose strategy: append a small positive token and terminate early to avoid length/divergence risk. This is objective misalignment, not optimizer failure.

### Conclusion | 结论

- CN: 仅靠总 reward 无法防止投机策略，必须增加与“真实质量”相关的约束或代理指标。
- EN: Total reward alone cannot prevent exploitative strategies; quality-correlated constraints/proxies are required.

---

## 3) Increase KL (including 0.3) | 调高 KL（包括 0.3）

### Data snapshot | 数据快照

| KL setting | What happened |
|---|---|
| 0.1 (low KL) | Faster convergence (around 300 steps) to safe hack |
| 0.3 (high KL) | Slower, more volatile, but little quality recovery |

### Analysis | 分析

- CN: 提高 KL 确实增加了“偏离原模型”的成本，但并没有直接把优化目标变成“更高质量文本”，因此收益有限。
- EN: Higher KL increased deviation cost but did not directly optimize text quality, so quality gains were limited.

### Conclusion | 结论

- CN: KL 主要是“约束强度旋钮”，不是解决 reward-quality 错配的根本手段。
- EN: KL is mainly a regularization knob, not a fundamental fix for reward-quality misalignment.

---

## 4) Correlated proxies + reward shaping mainline (old_goal) | 引入 correlated proxies 与 reward shaping（old_goal 主线）

### What was changed | 做了什么改动

| Area | Core changes |
|---|---|
| Reward design | Completion-only target, split repetition penalties, quality anchor, clipping |
| PPO stability | Adaptive KL controller, collapse guards |
| Selection logic | Multi-objective checkpoint score (not reward-only) |
| Evaluation protocol | Fixed seed, greedy+sampling, distribution export |

### Key result snapshot (old_goal rerun, previous protocol) | 核心结果快照（old_goal 旧协议重跑）

| Mode | PPO raw reward | Base raw reward | SFT raw reward | PPO quality_anchor | PPO repetition_token |
|---|---:|---:|---:|---:|---:|
| Greedy | 0.8918 | 0.8466 | 0.5995 | 0.0241 | 0.0251 |
| Sampling | 0.7969 | 0.7855 | 0.6065 | 0.0183 | 0.0154 |

### Analysis | 分析

- CN: 这一轮改动显著降低了重复和短句坍缩，reward 指标很强，但 quality_anchor 仍偏低，说明“高分但不够贴题/高质量”的问题仍在。
- EN: This stage significantly reduced repetition and short-collapse behavior; reward became strong, but quality_anchor remained low, indicating residual high-reward/low-quality behavior.

### Conclusion | 结论

- CN: old_goal 证明了工程修复有效，但也暴露出下一阶段核心矛盾：reward 可以高，quality 不一定高。
- EN: old_goal validated engineering fixes, but exposed the next core contradiction: high reward does not guarantee high quality.

---

## 5) High reward but low quality confirmed | 确认“高 reward 低 quality”

### Data snapshot | 数据快照

| Indicator | Greedy | Sampling |
|---|---:|---:|
| old_goal raw reward (previous protocol) | 0.8918 | 0.7969 |
| old_goal quality_anchor (previous protocol) | 0.0241 | 0.0183 |

### Analysis | 分析

- CN: raw reward 与 quality_anchor 数值量级严重不对称，直接体现了“可优化目标”和“真实语义质量”之间的错配。
- EN: The scale gap between raw reward and quality_anchor directly shows misalignment between optimized objective and semantic quality.

### Conclusion | 结论

- CN: 接下来必须把 quality 显式纳入模型选择标准，而不是只看 reward。
- EN: Quality must be explicitly included in selection criteria rather than relying on reward alone.

---

## 6) Three-variant parameter study and reward-quality tradeoff | 三组参数对比与 reward-quality 反向关系

### Fair short recheck (same metric, seed=44) | 公平短跑复核（同一指标，seed=44）

| Variant | Greedy raw | Greedy q_zero | Sampling raw | Sampling q_zero |
|---|---:|---:|---:|---:|
| gateB | 0.4426 | 0.2833 | 0.4163 | 0.4167 |
| qfix_v1 | 0.4400 | 0.2567 | 0.4011 | 0.3900 |
| qfix_v2 | 0.4426 | 0.2800 | 0.4198 | 0.4200 |
| qfix_v3 | 0.4384 | 0.2933 | 0.4030 | 0.4033 |

### Tradeoff quantification | 反向关系量化

| Correlation (across 4 variants) | Value |
|---|---:|
| Greedy: corr(raw, q_nonzero) | 0.0694 |
| Sampling: corr(raw, q_nonzero) | -0.9470 |
| Sampling: corr(raw, q_zero) | 0.9470 |

### Analysis | 分析

- CN: 你提到的“反直觉反比”是成立的，但主要发生在 sampling 侧，不是所有模式都强反比。其本质是：在当前 reward 结构下，模型更容易通过提高情感得分来抬高 reward，而这条路径不必然提升语义贴题质量。
- EN: Your “counter-intuitive inverse relation” is real, but mainly in sampling mode, not universally. Under the current reward structure, sentiment-driven reward gains are easier than improving semantic alignment.

### Conclusion | 结论

- CN: 这个现象非常值得在总结中强调：它是目标函数结构导致的可预期现象，而非偶然噪声。
- EN: This should be highlighted in the final summary: it is a structural objective effect, not random noise.

---

## 7) Introduce F1-like balancing and iterate model selection | 引入 F1 风格平衡并迭代选型

### Stage A: short multi-seed (3x1) | 阶段 A：短规模多种子（3x1）

| Variant | avg_fbeta_13 | constrained_score | sampling_q_nonzero |
|---|---:|---:|---:|
| v1 | 0.5630 | 0.3119 | 0.7222 |
| f1 | 0.5681 | 0.3184 | 0.7333 |

### Stage B: full-scale fair v3 (5x3) | 阶段 B：全量 fair v3（5x3）

| Variant | avg_fbeta_13 | constrained_score | sampling_q_nonzero |
|---|---:|---:|---:|
| v1 | 0.5833 | 0.3402 | 0.7757 |
| f1 | 0.5799 | 0.3347 | 0.7629 |
| old_goal | 0.5497 | 0.2898 | 0.6627 |

### Analysis | 分析

- CN: 小规模阶段 f1 略优，但扩大到 full-scale 后 v1 小幅反超。说明该问题对样本量与种子覆盖敏感，单次小样本判断不够稳健。
- EN: f1 slightly led at small scale, but v1 regained a small lead at full scale. This indicates sensitivity to sample size and seed coverage.

### Conclusion | 结论

- CN: 最终主结果应以 full-scale 为准：v1 作为主模型，f1 作为次优对照。
- EN: Final model choice should follow full-scale evidence: v1 primary, f1 secondary.

---

## 8) Current final status and deferred roadmap | 当前最终状态与延期路线

### Final model recommendation (as of now) | 当前推荐

| Role | Variant |
|---|---|
| Primary model | v1 |
| Secondary model | f1 |
| Historical baseline | old_goal |

### Deferred upgrade roadmap (from Section 19) | 延期升级路线（来自 Section 19）

| Priority | Direction | Status |
|---|---|---|
| P1 | Constrained objective route (for example C-DPO or dualized constrained PPO) | Deferred |
| P2 | Direct preference route (DPO/ORPO) | Deferred |
| P3 | KTO / offline regularized RL route | Deferred |

### Analysis | 分析

- CN: 本阶段最重要成果不是“找到绝对完美模型”，而是建立了一条可复现实验链路，并把 reward-quality 矛盾从“现象”推进到“可量化、可解释、可比较”。
- EN: The major outcome is not a perfect model, but a reproducible experimentation pipeline that turns reward-quality tension into a quantifiable, explainable, and comparable phenomenon.

### Conclusion | 结论

- CN: 项目已经从“能跑 PPO”进化到“能做严谨对比并据证据决策”。下一步若继续，应该优先在约束优化/偏好优化框架上解决结构性 tradeoff。
- EN: The project has evolved from “PPO runs” to “rigorous evidence-based selection.” Next progress should target structural tradeoff with constrained/preference optimization frameworks.

---

## One-paragraph takeaway | 一段话总结

CN: 这个项目的发展逻辑非常清晰：先把 PPO 训练跑通，再识别 reward hacking，接着验证“只调 KL 不够”，随后通过 correlated proxies 与 reward shaping 提升稳定性，发现高 reward 仍可能伴随低 quality，于是进入三组参数的公平对比与 F1 风格平衡选型，最终在 full-scale fair v3 证据下收敛到 v1 主模型、f1 次模型，并形成了明确的后续升级路线。

EN: The project progressed in a coherent arc: build PPO, detect reward hacking, verify that KL-only tuning is insufficient, stabilize with correlated proxies and reward shaping, confirm high-reward/low-quality mismatch, run fair multi-variant comparisons with F1-style balancing, and finally converge on v1 (primary) and f1 (secondary) under full-scale fair-v3 evidence, with a concrete deferred upgrade roadmap.
