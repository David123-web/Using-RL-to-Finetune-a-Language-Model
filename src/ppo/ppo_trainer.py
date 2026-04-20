from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.policy_lm import PolicyLM
from src.models.reward_model import RewardModel
from src.utils.device import get_device


@dataclass
class PPOHyperParams:
    clip_range: float
    value_coef: float
    entropy_coef: float
    gamma: float
    lam: float
    epochs_per_update: int
    batch_size: int
    kl_coef: float = 0.1
    adaptive_kl: bool = False
    target_kl: float = 0.1
    kl_update_rate: float = 0.05
    kl_coef_min: float = 0.01
    kl_coef_max: float = 1.0


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)


class PPOTrainer:
    def __init__(
        self,
        policy: PolicyLM,
        ref_policy: PolicyLM,
        reward_model: RewardModel,
        hparams: PPOHyperParams,
        max_length: int = 64,
        max_new_tokens: int = 32,
        rollout_temperature: float = 1.0,
        rollout_top_k: int = 50,
        rollout_top_p: float = 1.0,
        rollout_repetition_penalty: float = 1.15,
        rollout_no_repeat_ngram_size: int = 3,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.ref_policy.model.eval()
        for param in self.ref_policy.model.parameters():
            param.requires_grad = False

        self.reward_model = reward_model
        self.hparams = hparams
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.rollout_temperature = rollout_temperature
        self.rollout_top_k = rollout_top_k
        self.rollout_top_p = rollout_top_p
        self.rollout_repetition_penalty = rollout_repetition_penalty
        self.rollout_no_repeat_ngram_size = rollout_no_repeat_ngram_size
        self.device = get_device()
        self.current_kl_coef = hparams.kl_coef

        hidden_size = self.policy.model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.model.parameters()) + list(self.value_head.parameters()),
            lr=1e-6,
        )

        self.stats = {
            "rewards": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_div": [],
            "kl_coef": [],
            "completion_length_words": [],
            "eos_rate": [],
            "quality_anchor_nonzero_rate": [],
        }

    def get_logprobs_and_values(self, input_ids, attention_mask):
        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        logprobs = F.log_softmax(logits, dim=-1)
        action_logprobs = torch.gather(
            logprobs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        values = self.value_head(hidden_states)
        return action_logprobs, values

    @torch.no_grad()
    def get_ref_logprobs(self, input_ids, attention_mask):
        outputs = self.ref_policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)

        ref_logprobs = torch.gather(
            logprobs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        return ref_logprobs

    def _extract_completions(
        self, sequences: torch.Tensor, prompt_input_len: int
    ) -> Tuple[List[str], List[int], List[bool]]:
        completions: List[str] = []
        token_lengths: List[int] = []
        eos_triggered: List[bool] = []
        eos_id = self.policy.tokenizer.eos_token_id

        for seq in sequences:
            start = int(prompt_input_len)
            completion_ids = seq[start:]
            completions.append(
                self.policy.tokenizer.decode(completion_ids, skip_special_tokens=True)
            )
            token_lengths.append(int(completion_ids.numel()))
            if eos_id is None:
                eos_triggered.append(False)
            else:
                eos_triggered.append(bool((completion_ids == eos_id).any().item()))

        return completions, token_lengths, eos_triggered

    @torch.no_grad()
    def collect_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        self.policy.model.eval()
        self.value_head.eval()

        prompt_enc = self.policy.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        prompt_input_len = prompt_enc["input_ids"].shape[1]

        output_ids = self.policy.model.generate(
            **prompt_enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.rollout_temperature,
            top_k=self.rollout_top_k,
            top_p=self.rollout_top_p,
            repetition_penalty=self.rollout_repetition_penalty,
            no_repeat_ngram_size=self.rollout_no_repeat_ngram_size,
            pad_token_id=self.policy.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        sequences = output_ids.sequences

        completions, completion_lengths_tokens, eos_triggered = self._extract_completions(
            sequences, prompt_input_len
        )
        completion_lengths_words = [len(text.split()) for text in completions]

        rewards, reward_components = self.reward_model.compute_reward_for_completions(
            completions=completions,
            prompts=prompts,
            return_components=True,
        )

        attention_mask = (sequences != self.policy.tokenizer.pad_token_id).long()
        action_logprobs, values = self.get_logprobs_and_values(sequences, attention_mask)
        ref_logprobs = self.get_ref_logprobs(sequences, attention_mask)

        gen_mask_full = torch.zeros_like(sequences, dtype=torch.bool)
        gen_mask_full[:, int(prompt_input_len) :] = True
        gen_mask = gen_mask_full[:, 1:]

        advantages, returns = self.compute_gae(rewards, values, gen_mask)

        return {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_logprobs": action_logprobs,
            "ref_logprobs": ref_logprobs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
            "gen_mask": gen_mask,
            "reward_components": reward_components,
            "completion_lengths_words": completion_lengths_words,
            "completion_lengths_tokens": completion_lengths_tokens,
            "eos_triggered": eos_triggered,
        }

    def compute_gae(self, rewards, values, gen_mask):
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        if advantages.shape != gen_mask.shape:
            min_len = min(advantages.shape[1], gen_mask.shape[1])
            advantages = advantages[:, :min_len]
            returns = returns[:, :min_len]
            values = values[:, :min_len]
            gen_mask = gen_mask[:, :min_len]

        for i in range(batch_size):
            gen_positions = gen_mask[i].nonzero(as_tuple=True)[0]
            if len(gen_positions) == 0:
                continue

            gae = 0.0
            last_pos = int(gen_positions[-1].item())
            for pos in reversed(gen_positions.tolist()):
                t = int(pos)
                if t == last_pos:
                    delta = rewards[i] - values[i, t]
                elif t + 1 < values.shape[1]:
                    delta = self.hparams.gamma * values[i, t + 1] - values[i, t]
                else:
                    delta = -values[i, t]

                gae = delta + self.hparams.gamma * self.hparams.lam * gae
                advantages[i, t] = gae
                returns[i, t] = gae + values[i, t]

        mask_sum = gen_mask.sum()
        if mask_sum > 0:
            adv_mean = (advantages * gen_mask).sum() / mask_sum
            adv_var = ((advantages - adv_mean) ** 2 * gen_mask).sum() / mask_sum
            advantages = (advantages - adv_mean) / (torch.sqrt(adv_var) + 1e-8)

        return advantages, returns

    def ppo_step(self, batch: Dict[str, torch.Tensor]):
        self.policy.model.train()
        self.value_head.train()

        sequences = batch["sequences"]
        attention_mask = batch["attention_mask"]
        old_logprobs = batch["action_logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        gen_mask = batch["gen_mask"]
        ref_logprobs = batch["ref_logprobs"]

        new_logprobs, new_values = self.get_logprobs_and_values(sequences, attention_mask)

        min_len = min(new_logprobs.shape[1], gen_mask.shape[1], old_logprobs.shape[1])
        new_logprobs = new_logprobs[:, :min_len]
        new_values = new_values[:, :min_len]
        old_logprobs = old_logprobs[:, :min_len]
        advantages = advantages[:, :min_len]
        returns = returns[:, :min_len]
        gen_mask = gen_mask[:, :min_len]
        ref_logprobs = ref_logprobs[:, :min_len]

        logratio = new_logprobs - old_logprobs
        ratio = torch.exp(logratio)

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(
            ratio, 1 - self.hparams.clip_range, 1 + self.hparams.clip_range
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2)
        policy_loss = (policy_loss * gen_mask).sum() / (gen_mask.sum() + 1e-8)

        value_loss = F.mse_loss(
            new_values * gen_mask,
            returns * gen_mask,
            reduction="sum",
        ) / (gen_mask.sum() + 1e-8)

        entropy = -(new_logprobs * gen_mask).sum() / (gen_mask.sum() + 1e-8)

        approx_kl = ((new_logprobs - ref_logprobs) ** 2 * gen_mask).sum() / (
            gen_mask.sum() + 1e-8
        )

        loss = (
            policy_loss
            + self.hparams.value_coef * value_loss
            - self.hparams.entropy_coef * entropy
            + self.current_kl_coef * approx_kl
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.model.parameters()) + list(self.value_head.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl_div": approx_kl.item(),
            "kl_coef": self.current_kl_coef,
        }

    def _update_kl_coef(self, measured_kl: float):
        if not self.hparams.adaptive_kl:
            return

        target = max(self.hparams.target_kl, 1e-8)
        if measured_kl > target * 1.5:
            self.current_kl_coef *= 1.0 + self.hparams.kl_update_rate
        elif measured_kl < target / 1.5:
            self.current_kl_coef /= 1.0 + self.hparams.kl_update_rate

        self.current_kl_coef = float(
            np.clip(
                self.current_kl_coef,
                self.hparams.kl_coef_min,
                self.hparams.kl_coef_max,
            )
        )

    def train_step(self, prompts: List[str]):
        rollout = self.collect_rollout(prompts)

        step_stats = []
        for _ in range(self.hparams.epochs_per_update):
            stats = self.ppo_step(rollout)
            step_stats.append(stats)

        avg_stats = {
            key: float(np.mean([s[key] for s in step_stats])) for key in step_stats[0].keys()
        }

        component_means = {
            key: rollout["reward_components"][key].mean().item()
            for key in [
                "sentiment",
                "repetition_token",
                "repetition_phrase",
                "length",
                "quality_anchor",
                "raw_total",
            ]
        }

        avg_stats["reward"] = rollout["rewards"].mean().item()
        avg_stats["raw_reward"] = component_means["raw_total"]
        avg_stats["sentiment"] = component_means["sentiment"]
        avg_stats["repetition_token"] = component_means["repetition_token"]
        avg_stats["repetition_phrase"] = component_means["repetition_phrase"]
        avg_stats["length"] = component_means["length"]
        avg_stats["quality_anchor"] = component_means["quality_anchor"]
        avg_stats["quality_anchor_nonzero_rate"] = float(
            (rollout["reward_components"]["quality_anchor"] > 1e-8).float().mean().item()
        )

        avg_stats["avg_completion_length_words"] = float(
            np.mean(rollout["completion_lengths_words"])
        )
        avg_stats["avg_completion_length_tokens"] = float(
            np.mean(rollout["completion_lengths_tokens"])
        )
        avg_stats["eos_rate"] = float(np.mean(rollout["eos_triggered"]))
        avg_stats["reward_quality_gap"] = abs(
            avg_stats["raw_reward"] - avg_stats["quality_anchor"]
        )

        self._update_kl_coef(avg_stats["kl_div"])
        avg_stats["kl_coef"] = self.current_kl_coef

        self.stats["rewards"].append(avg_stats["reward"])
        self.stats["policy_loss"].append(avg_stats["policy_loss"])
        self.stats["value_loss"].append(avg_stats["value_loss"])
        self.stats["entropy"].append(avg_stats["entropy"])
        self.stats["kl_div"].append(avg_stats["kl_div"])
        self.stats["kl_coef"].append(avg_stats["kl_coef"])
        self.stats["completion_length_words"].append(avg_stats["avg_completion_length_words"])
        self.stats["eos_rate"].append(avg_stats["eos_rate"])
        self.stats["quality_anchor_nonzero_rate"].append(avg_stats["quality_anchor_nonzero_rate"])

        return avg_stats
