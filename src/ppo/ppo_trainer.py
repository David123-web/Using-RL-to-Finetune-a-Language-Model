from dataclasses import dataclass
from typing import List, Dict
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

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


class ValueHead(nn.Module):
    """Simple value head for PPO critic"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)


class PPOTrainer:
    def __init__(self,
                 policy: PolicyLM,
                 ref_policy: PolicyLM,
                 reward_model: RewardModel,
                 hparams: PPOHyperParams,
                 max_length: int = 64,
                 max_new_tokens: int = 32):
        self.policy = policy
        self.ref_policy = ref_policy
        self.ref_policy.model.eval()   # frozen
        for param in self.ref_policy.model.parameters():
            param.requires_grad = False
            
        self.reward_model = reward_model
        self.hparams = hparams
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = get_device()

        # Add value head
        hidden_size = self.policy.model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)

        # Optimizer for both policy and value head
        self.optimizer = torch.optim.AdamW(
            list(self.policy.model.parameters()) + list(self.value_head.parameters()),
            lr=1e-5
        )
        
        self.stats = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_div': []
        }

    def get_logprobs_and_values(self, input_ids, attention_mask):
        """Get log probabilities for actions and value estimates"""
        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        hidden_states = outputs.hidden_states[-1]  # last layer
        
        # Get log probs for the actual tokens
        logprobs = F.log_softmax(logits, dim=-1)
        # Select logprobs of actual next tokens
        action_logprobs = torch.gather(
            logprobs[:, :-1, :],  # exclude last position
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1)  # actual tokens as indices
        ).squeeze(-1)
        
        # Get values from value head
        values = self.value_head(hidden_states)
        
        return action_logprobs, values

    @torch.no_grad()
    def get_ref_logprobs(self, input_ids, attention_mask):
        """Get log probabilities from reference policy"""
        outputs = self.ref_policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)
        
        ref_logprobs = torch.gather(
            logprobs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return ref_logprobs

    @torch.no_grad()
    def collect_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        1) Generate responses from policy
        2) Compute rewards
        3) Compute logprobs and values for PPO
        Return a dict of tensors ready for PPO update.
        """
        self.policy.model.eval()
        self.value_head.eval()
        
        # Encode prompts
        prompt_enc = self.policy.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        prompt_len = prompt_enc['input_ids'].shape[1]
        
        # Generate completions
        output_ids = self.policy.model.generate(
            **prompt_enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            pad_token_id=self.policy.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        
        sequences = output_ids.sequences  # (batch, prompt_len + gen_len)
        
        # Decode full sequences for reward computation
        full_texts = self.policy.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        # Compute rewards
        rewards = self.reward_model.compute_reward(full_texts)  # (batch,)
        
        # Get logprobs and values for the sequences
        attention_mask = (sequences != self.policy.tokenizer.pad_token_id).long()
        action_logprobs, values = self.get_logprobs_and_values(sequences, attention_mask)
        
        # Get reference logprobs for KL penalty
        ref_logprobs = self.get_ref_logprobs(sequences, attention_mask)
        
        # Only consider generated tokens (not prompt)
        gen_mask = torch.zeros_like(sequences, dtype=torch.bool)
        gen_mask[:, prompt_len:] = True
        gen_mask = gen_mask[:, 1:]  # align with logprobs (shifted by 1)
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(
            rewards, values, gen_mask, prompt_len
        )
        
        return {
            'sequences': sequences,
            'attention_mask': attention_mask,
            'action_logprobs': action_logprobs,
            'ref_logprobs': ref_logprobs,
            'values': values,
            'advantages': advantages,
            'returns': returns,
            'rewards': rewards,
            'gen_mask': gen_mask,
        }

    def compute_gae(self, rewards, values, gen_mask, prompt_len):
        """Compute Generalized Advantage Estimation"""
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # For simplicity, treat the final reward as occurring at the last generated token
        for i in range(batch_size):
            # Find last generated token
            gen_positions = gen_mask[i].nonzero(as_tuple=True)[0]
            if len(gen_positions) == 0:
                continue
                
            last_gen_pos = gen_positions[-1].item()
            
            # Ensure we don't go out of bounds
            last_gen_pos = min(last_gen_pos, seq_len - 1)
            
            # Assign reward to last position
            gae = 0
            start_pos = max(0, prompt_len - 2)
            for t in range(last_gen_pos, start_pos, -1):
                if t == last_gen_pos:
                    delta = rewards[i] - values[i, t]
                else:
                    if t + 1 < seq_len:
                        delta = -values[i, t] + self.hparams.gamma * values[i, t + 1]
                    else:
                        delta = -values[i, t]
                
                gae = delta + self.hparams.gamma * self.hparams.lam * gae
                advantages[i, t] = gae
                returns[i, t] = gae + values[i, t]
        
        # Normalize advantages - ensure same shape as gen_mask
        if advantages.shape != gen_mask.shape:
            # Truncate or pad to match gen_mask shape
            min_len = min(advantages.shape[1], gen_mask.shape[1])
            advantages = advantages[:, :min_len]
            returns = returns[:, :min_len]
            values = values[:, :min_len]
            gen_mask = gen_mask[:, :min_len]
        
        mask_sum = gen_mask.sum()
        if mask_sum > 0:
            adv_mean = (advantages * gen_mask).sum() / mask_sum
            adv_std = torch.sqrt(((advantages - adv_mean) ** 2 * gen_mask).sum() / mask_sum)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return advantages, returns

    def ppo_step(self, batch: Dict[str, torch.Tensor]):
        """
        Implement PPO clipped objective here:
        - policy loss
        - value loss
        - entropy bonus
        """
        self.policy.model.train()
        self.value_head.train()
        
        sequences = batch['sequences']
        attention_mask = batch['attention_mask']
        old_logprobs = batch['action_logprobs']
        old_values = batch['values']
        advantages = batch['advantages']
        returns = batch['returns']
        gen_mask = batch['gen_mask']
        ref_logprobs = batch['ref_logprobs']
        
        # Get current policy logprobs and values
        new_logprobs, new_values = self.get_logprobs_and_values(sequences, attention_mask)
        
        # Ensure all tensors have same shape as gen_mask
        min_len = min(new_logprobs.shape[1], gen_mask.shape[1], old_logprobs.shape[1])
        new_logprobs = new_logprobs[:, :min_len]
        new_values = new_values[:, :min_len]
        old_logprobs = old_logprobs[:, :min_len]
        old_values = old_values[:, :min_len]
        advantages = advantages[:, :min_len]
        returns = returns[:, :min_len]
        gen_mask = gen_mask[:, :min_len]
        ref_logprobs = ref_logprobs[:, :min_len]
        
        # Compute ratios
        logratio = new_logprobs - old_logprobs
        ratio = torch.exp(logratio)
        
        # PPO policy loss
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(
            ratio, 1 - self.hparams.clip_range, 1 + self.hparams.clip_range
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2)
        policy_loss = (policy_loss * gen_mask).sum() / (gen_mask.sum() + 1e-8)
        
        # Value loss
        value_loss = F.mse_loss(
            new_values * gen_mask,
            returns * gen_mask,
            reduction='sum'
        ) / (gen_mask.sum() + 1e-8)
        
        # Entropy bonus (encourage exploration)
        # Approximate entropy from log probs
        entropy = -(new_logprobs * gen_mask).sum() / (gen_mask.sum() + 1e-8)
        
        # KL divergence from reference policy
        kl_div = ((new_logprobs - ref_logprobs) * gen_mask).sum() / (gen_mask.sum() + 1e-8)
        
        # Total loss
        loss = (
            policy_loss +
            self.hparams.value_coef * value_loss -
            self.hparams.entropy_coef * entropy +
            self.hparams.kl_coef * kl_div
        )
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.model.parameters()) + list(self.value_head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Track statistics
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
        }

    def train_step(self, prompts: List[str]):
        """Single training step: collect rollout and perform PPO updates"""
        rollout = self.collect_rollout(prompts)
        
        step_stats = []
        for _ in range(self.hparams.epochs_per_update):
            stats = self.ppo_step(rollout)
            step_stats.append(stats)
        
        # Average stats over epochs
        avg_stats = {
            key: np.mean([s[key] for s in step_stats])
            for key in step_stats[0].keys()
        }
        avg_stats['reward'] = rollout['rewards'].mean().item()
        
        # Update running statistics
        self.stats['rewards'].append(avg_stats['reward'])
        self.stats['policy_loss'].append(avg_stats['policy_loss'])
        self.stats['value_loss'].append(avg_stats['value_loss'])
        self.stats['entropy'].append(avg_stats['entropy'])
        self.stats['kl_div'].append(avg_stats['kl_div'])
        
        return avg_stats
