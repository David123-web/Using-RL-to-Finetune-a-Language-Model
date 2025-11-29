from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.device import get_device


@dataclass
class PolicyConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    max_length: int = 64


class PolicyLM(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name
        )
        if self.tokenizer.pad_token is None:
            # GPT2-style models need a pad token; reuse eos
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side to left for decoder-only models (for generation)
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.max_length = config.max_length

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0):
        self.model.eval()
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        output_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return texts
