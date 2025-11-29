from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.device import get_device


class RewardModel(nn.Module):
    def __init__(self, model_name: str,
                 w_sentiment: float = 1.0,
                 w_repetition: float = -0.5,
                 w_length: float = 0.1,
                 min_tokens: int = 10,
                 max_tokens: int = 40):
        super().__init__()
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.w_sentiment = w_sentiment
        self.w_repetition = w_repetition
        self.w_length = w_length
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    @torch.no_grad()
    def sentiment_score(self, texts: List[str]) -> torch.Tensor:
        self.model.eval()
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        logits = self.model(**enc).logits  # (B, 2) for SST-2
        probs = torch.softmax(logits, dim=-1)
        positive_prob = probs[:, 1]
        return positive_prob  # [0, 1]

    def repetition_penalty(self, texts: List[str]) -> torch.Tensor:
        scores = []
        for t in texts:
            tokens = t.split()
            unique = len(set(tokens))
            total = max(len(tokens), 1)
            distinct_ratio = unique / total
            # penalty is higher (more negative) for less distinct text
            scores.append(1.0 - distinct_ratio)
        return torch.tensor(scores, device=self.device)

    def length_bonus(self, texts: List[str]) -> torch.Tensor:
        scores = []
        for t in texts:
            n = len(t.split())
            if n < self.min_tokens or n > self.max_tokens:
                scores.append(-1.0)
            else:
                scores.append(0.0)
        return torch.tensor(scores, device=self.device)

    @torch.no_grad()
    def compute_reward(self, texts: List[str]) -> torch.Tensor:
        s = self.sentiment_score(texts)
        r_rep = self.repetition_penalty(texts)
        r_len = self.length_bonus(texts)

        reward = (
            self.w_sentiment * s +
            self.w_repetition * r_rep +
            self.w_length * r_len
        )
        return reward
