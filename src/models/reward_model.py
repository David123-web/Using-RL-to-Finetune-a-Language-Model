from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import os
import unicodedata

import torch
from torch import nn
from huggingface_hub import logging as hf_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
except Exception:
    hf_disable_progress_bars = None

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
if hf_disable_progress_bars is not None:
    hf_disable_progress_bars()
hf_logging.set_verbosity_error()
if hasattr(transformers_logging, "disable_progress_bar"):
    transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()

from src.utils.device import get_device


UNICODE_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00A0": " ",
        "\u200B": "",
        "\u200C": "",
        "\u200D": "",
        "\uFEFF": "",
        "\uFFFD": " ",
    }
)


@dataclass
class RewardAggregationConfig:
    center_total_reward: bool = True
    normalize_total_reward: bool = False


class RewardModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        completion_only: bool = True,
        normalize_text: bool = True,
        w_sentiment: float = 1.0,
        w_repetition_token: float = -0.15,
        w_repetition_phrase: float = -0.1,
        w_length: float = 0.2,
        w_quality_anchor: float = 0.25,
        min_tokens: int = 12,
        max_tokens: int = 40,
        phrase_ngram: int = 3,
        quality_mode: str = "token_overlap",
        quality_min_token_chars: int = 3,
        alignment_gate_enabled: bool = False,
        alignment_gate_quality_floor: float = 0.15,
        alignment_gate_strength: float = 0.85,
        clip_sentiment: Tuple[float, float] = (0.0, 1.0),
        clip_repetition_token: Tuple[float, float] = (0.0, 1.0),
        clip_repetition_phrase: Tuple[float, float] = (0.0, 1.0),
        clip_length: Tuple[float, float] = (-1.0, 1.0),
        clip_quality_anchor: Tuple[float, float] = (0.0, 1.0),
        clip_total: Tuple[float, float] = (-2.0, 2.0),
        aggregation: Optional[RewardAggregationConfig] = None,
    ):
        super().__init__()
        self.device = get_device()

        # Prefer local cache first to avoid repetitive unauthenticated Hub calls.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

        self.completion_only = completion_only
        self.normalize_text_enabled = normalize_text

        self.w_sentiment = w_sentiment
        self.w_repetition_token = w_repetition_token
        self.w_repetition_phrase = w_repetition_phrase
        self.w_length = w_length
        self.w_quality_anchor = w_quality_anchor

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.phrase_ngram = max(2, phrase_ngram)
        self.quality_mode = quality_mode
        self.quality_min_token_chars = quality_min_token_chars
        self.alignment_gate_enabled = alignment_gate_enabled
        self.alignment_gate_quality_floor = float(
            min(max(alignment_gate_quality_floor, 0.0), 1.0)
        )
        self.alignment_gate_strength = float(min(max(alignment_gate_strength, 0.0), 1.0))

        self.clip_sentiment = clip_sentiment
        self.clip_repetition_token = clip_repetition_token
        self.clip_repetition_phrase = clip_repetition_phrase
        self.clip_length = clip_length
        self.clip_quality_anchor = clip_quality_anchor
        self.clip_total = clip_total

        self.aggregation = aggregation or RewardAggregationConfig()

    def _normalize_text(self, text: str) -> str:
        if not self.normalize_text_enabled:
            return text
        # Canonicalize punctuation so quality/repetition stats are robust to Unicode variants.
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(UNICODE_PUNCT_TRANSLATION)
        text = text.lower()
        text = text.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize_for_stats(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        return [tok for tok in normalized.split(" ") if tok]

    def _stem_token_for_quality(self, token: str) -> str:
        tok = token
        if tok.endswith("'s") and len(tok) > 3:
            tok = tok[:-2]

        for suffix, min_len in (("ingly", 7), ("edly", 6), ("ing", 6), ("ed", 5), ("ly", 5), ("es", 5), ("s", 4)):
            if len(tok) >= min_len and tok.endswith(suffix):
                tok = tok[: -len(suffix)]
                break

        return tok

    def _tokenize_for_quality(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        raw_tokens = re.findall(r"[a-z0-9']+", normalized)
        tokens: List[str] = []
        for token in raw_tokens:
            stemmed = self._stem_token_for_quality(token)
            if len(stemmed) >= self.quality_min_token_chars:
                tokens.append(stemmed)
        return tokens

    def _clip_tensor(self, values: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        return torch.clamp(values, min=bounds[0], max=bounds[1])

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

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]

    def repetition_token_penalty(self, texts: List[str]) -> torch.Tensor:
        scores = []
        for text in texts:
            tokens = self._tokenize_for_stats(text)
            total = max(len(tokens), 1)
            unique = len(set(tokens))
            scores.append(1.0 - (unique / total))
        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def repetition_phrase_penalty(self, texts: List[str]) -> torch.Tensor:
        scores = []
        for text in texts:
            tokens = self._tokenize_for_stats(text)
            if len(tokens) < self.phrase_ngram:
                scores.append(0.0)
                continue

            ngrams = [
                tuple(tokens[i : i + self.phrase_ngram])
                for i in range(len(tokens) - self.phrase_ngram + 1)
            ]
            total = len(ngrams)
            unique = len(set(ngrams))
            scores.append(1.0 - (unique / max(total, 1)))

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def length_score(self, texts: List[str]) -> torch.Tensor:
        scores = []
        for text in texts:
            n_tokens = len(self._tokenize_for_stats(text))
            if self.min_tokens <= n_tokens <= self.max_tokens:
                scores.append(1.0)
            elif n_tokens < self.min_tokens:
                gap = (self.min_tokens - n_tokens) / max(self.min_tokens, 1)
                scores.append(-gap)
            else:
                gap = (n_tokens - self.max_tokens) / max(self.max_tokens, 1)
                scores.append(-gap)

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def quality_anchor_score(
        self,
        prompts: Optional[List[str]],
        completions: List[str],
    ) -> torch.Tensor:
        if prompts is None or self.w_quality_anchor == 0.0:
            return torch.zeros(len(completions), dtype=torch.float32, device=self.device)

        scores = []
        for prompt, completion in zip(prompts, completions):
            if self.quality_mode == "token_overlap":
                prompt_tokens = {
                    tok
                    for tok in self._tokenize_for_stats(prompt)
                    if len(tok) >= self.quality_min_token_chars
                }
                completion_tokens = {
                    tok
                    for tok in self._tokenize_for_stats(completion)
                    if len(tok) >= self.quality_min_token_chars
                }
                if not completion_tokens:
                    scores.append(0.0)
                    continue

                overlap = len(prompt_tokens.intersection(completion_tokens))
                scores.append(overlap / max(len(completion_tokens), 1))
                continue

            if self.quality_mode in {"token_overlap_clean", "token_recall_clean", "token_f1_clean"}:
                prompt_tokens = set(self._tokenize_for_quality(prompt))
                completion_tokens = set(self._tokenize_for_quality(completion))
                if not prompt_tokens or not completion_tokens:
                    scores.append(0.0)
                    continue

                overlap = len(prompt_tokens.intersection(completion_tokens))
                precision = overlap / max(len(completion_tokens), 1)
                recall = overlap / max(len(prompt_tokens), 1)

                if self.quality_mode == "token_overlap_clean":
                    scores.append(precision)
                elif self.quality_mode == "token_recall_clean":
                    scores.append(recall)
                else:
                    if precision + recall <= 0.0:
                        scores.append(0.0)
                    else:
                        scores.append((2.0 * precision * recall) / (precision + recall))
                continue

            scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def compute_reward_components(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        sentiment = self._clip_tensor(self.sentiment_score(completions), self.clip_sentiment)
        repetition_token = self._clip_tensor(
            self.repetition_token_penalty(completions), self.clip_repetition_token
        )
        repetition_phrase = self._clip_tensor(
            self.repetition_phrase_penalty(completions), self.clip_repetition_phrase
        )
        length = self._clip_tensor(self.length_score(completions), self.clip_length)
        quality_anchor = self._clip_tensor(
            self.quality_anchor_score(prompts, completions), self.clip_quality_anchor
        )

        if self.alignment_gate_enabled:
            quality_gate = torch.clamp(quality_anchor, min=0.0, max=1.0)
            sentiment_gate = self.alignment_gate_quality_floor + (
                1.0 - self.alignment_gate_quality_floor
            ) * quality_gate
            sentiment_gate = (1.0 - self.alignment_gate_strength) + (
                self.alignment_gate_strength * sentiment_gate
            )
        else:
            sentiment_gate = torch.ones_like(sentiment)

        weighted_sentiment = self.w_sentiment * sentiment * sentiment_gate
        weighted_repetition_token = self.w_repetition_token * repetition_token
        weighted_repetition_phrase = self.w_repetition_phrase * repetition_phrase
        weighted_length = self.w_length * length
        weighted_quality_anchor = self.w_quality_anchor * quality_anchor

        raw_total = (
            weighted_sentiment
            + weighted_repetition_token
            + weighted_repetition_phrase
            + weighted_length
            + weighted_quality_anchor
        )
        raw_total = self._clip_tensor(raw_total, self.clip_total)

        total = raw_total
        if self.aggregation.center_total_reward:
            total = total - total.mean()

        if self.aggregation.normalize_total_reward:
            total = (total - total.mean()) / (total.std(unbiased=False) + 1e-8)

        return {
            "sentiment": sentiment,
            "repetition_token": repetition_token,
            "repetition_phrase": repetition_phrase,
            "length": length,
            "quality_anchor": quality_anchor,
            "weighted_sentiment": weighted_sentiment,
            "sentiment_gate": sentiment_gate,
            "weighted_repetition_token": weighted_repetition_token,
            "weighted_repetition_phrase": weighted_repetition_phrase,
            "weighted_length": weighted_length,
            "weighted_quality_anchor": weighted_quality_anchor,
            "raw_total": raw_total,
            "total": total,
        }

    @torch.no_grad()
    def compute_reward_for_completions(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        return_components: bool = False,
    ):
        components = self.compute_reward_components(completions=completions, prompts=prompts)
        reward = components["total"]
        if return_components:
            return reward, components
        return reward

    @torch.no_grad()
    def compute_reward(self, texts: List[str], return_components: bool = False):
        return self.compute_reward_for_completions(
            completions=texts,
            prompts=None,
            return_components=return_components,
        )
