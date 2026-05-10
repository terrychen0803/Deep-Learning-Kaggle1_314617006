#!/usr/bin/env python3
"""QLoRA answer-only SFT training for Chinese-released LLMs such as Qwen.

The dataset is prompt/completion JSONL. The prompt tokens are masked with -100,
so loss is computed only on the answer letter and EOS tokens.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class PromptCompletionDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row["prompt"]
        completion = row["completion"]
        eos = self.tokenizer.eos_token or ""

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(prompt + completion + eos, add_special_tokens=False).input_ids

        full_ids = full_ids[: self.max_length]
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        # If truncation accidentally removes the answer, keep last token trainable.
        if all(x == -100 for x in labels) and len(labels) > 0:
            labels[-1] = full_ids[-1]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: object

    def __call__(self, features: List[Dict[str, List[int]]]):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--valid_jsonl", required=True)
    parser.add_argument("--output_dir", default="outputs/qwen25_lora")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--wandb_project", default="", help="Set this to enable wandb logging.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = PromptCompletionDataset(args.train_jsonl, tokenizer, args.max_length)
    valid_ds = PromptCompletionDataset(args.valid_jsonl, tokenizer, args.max_length)

    report_to = ["wandb"] if args.wandb_project else []
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to=report_to,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
