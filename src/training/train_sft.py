import yaml
from dataclasses import dataclass
from typing import Any, Dict
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from src.models.policy_lm import PolicyLM, PolicyConfig
from src.utils.device import get_device


@dataclass
class SFTConfig:
    model_name: str
    tokenizer_name: str
    batch_size: int
    max_steps: int
    lr: float
    max_length: int
    dataset_name: str
    text_field: str
    split: str
    save_dir: str


def build_dataloader(cfg: SFTConfig, tokenizer):
    ds = load_dataset(cfg.dataset_name, split=cfg.split)

    def preprocess(example):
        text = example[cfg.text_field]
        # Tokenize the text
        enc = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,  # Don't pad here, let collator handle it
        )
        return enc

    ds = ds.map(preprocess, batched=False, remove_columns=ds.column_names)
    
    # Use DataCollatorForLanguageModeling for causal LM training
    # This will automatically create labels by copying input_ids
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    return DataLoader(
        ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=data_collator
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config/sft_config.yaml"):
    raw_cfg = load_config(config_path)

    sft_cfg = SFTConfig(
        model_name=raw_cfg["model"]["name"],
        tokenizer_name=raw_cfg["model"]["tokenizer_name"],
        batch_size=int(raw_cfg["training"]["batch_size"]),
        max_steps=int(raw_cfg["training"]["max_steps"]),
        lr=float(raw_cfg["training"]["lr"]),
        max_length=int(raw_cfg["training"]["max_length"]),
        dataset_name=raw_cfg["data"]["dataset_name"],
        text_field=raw_cfg["data"]["text_field"],
        split=raw_cfg["data"]["split"],
        save_dir=raw_cfg["logging"]["save_dir"],
    )

    device = get_device()
    policy = PolicyLM(PolicyConfig(
        model_name=sft_cfg.model_name,
        tokenizer_name=sft_cfg.tokenizer_name,
        max_length=sft_cfg.max_length,
    ))
    tokenizer = policy.tokenizer

    dataloader = build_dataloader(sft_cfg, tokenizer)

    optimizer = torch.optim.AdamW(
        policy.model.parameters(), lr=sft_cfg.lr
    )

    # Create save directory
    os.makedirs(sft_cfg.save_dir, exist_ok=True)
    
    policy.train()
    step = 0
    for batch in tqdm(dataloader):
        if step >= sft_cfg.max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = policy(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Step {step} | loss = {loss.item():.4f}")

        step += 1

    policy.model.save_pretrained(sft_cfg.save_dir)
    tokenizer.save_pretrained(sft_cfg.save_dir)
    print(f"Saved SFT model to {sft_cfg.save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sft_config.yaml")
    args = parser.parse_args()
    main(args.config)
