from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

import torch
from torch import nn
from huggingface_hub import logging as hf_logging
from transformers import AutoModelForCausalLM, AutoTokenizer
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


@dataclass
class PolicyConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    max_length: int = 64


class PolicyLM(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.device = get_device()
        model_or_tokenizer_name = config.tokenizer_name or config.model_name

        # Prefer local cache first to avoid HF Hub requests/warnings during repeated runs.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_or_tokenizer_name,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_tokenizer_name)

        if self.tokenizer.pad_token is None:
            # GPT2-style models need a pad token; reuse eos
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side to left for decoder-only models (for generation)
        self.tokenizer.padding_side = 'left'

        try:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    local_files_only=True,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        except ValueError:
            # Fallback for local checkpoints without config.json: assume distilgpt2
            print(f"Warning: Could not load config from {config.model_name}, assuming distilgpt2 architecture.")
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name, config=model_config)
            
        self.model.to(self.device)
        self.max_length = config.max_length

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        seed: Optional[int] = None,
        return_completions: bool = False,
        return_metadata: bool = False,
    ):
        self.model.eval()
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # Keep per-call reproducibility without relying on model.generate(generator=...),
        # which is not supported by some transformers versions.
        if seed is not None:
            devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                output_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        else:
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if not return_completions and not return_metadata:
            return full_texts

        input_len = enc["input_ids"].shape[1]
        completions: List[str] = []
        completion_token_lengths: List[int] = []
        eos_triggered: List[bool] = []

        eos_id = self.tokenizer.eos_token_id
        for seq in output_ids:
            start_idx = int(input_len)
            completion_ids = seq[start_idx:]
            completions.append(
                self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            )
            completion_token_lengths.append(int(completion_ids.numel()))
            if eos_id is None:
                eos_triggered.append(False)
            else:
                eos_triggered.append(bool((completion_ids == eos_id).any().item()))

        if return_completions and not return_metadata:
            return full_texts, completions

        metadata: Dict[str, List] = {
            "completion_token_lengths": completion_token_lengths,
            "eos_triggered": eos_triggered,
        }

        if return_completions and return_metadata:
            return full_texts, completions, metadata
        return full_texts, metadata
