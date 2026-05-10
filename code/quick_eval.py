#!/usr/bin/env python3
"""Evaluate validation accuracy by next-token logit scoring over A/B/C/D."""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

CHOICES = ["A", "B", "C", "D"]


def load_model(base_model, adapter_path, no_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = None
    if not no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    scores = {ch: float(logits[tokenizer(ch, add_special_tokens=False).input_ids[0]]) for ch in CHOICES}
    return max(scores, key=scores.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--valid_jsonl", required=True)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.valid_jsonl, "r", encoding="utf-8") if line.strip()]
    model, tokenizer = load_model(args.base_model, args.adapter_path, args.no_4bit)

    correct = 0
    counts = {c: 0 for c in CHOICES}
    for row in tqdm(rows):
        pred = predict(model, tokenizer, row["prompt"])
        counts[pred] += 1
        correct += int(pred == row["answer"])
    acc = correct / len(rows) if rows else 0.0
    print(f"valid accuracy = {acc:.4f} ({correct}/{len(rows)})")
    print("prediction distribution:", counts)


if __name__ == "__main__":
    main()
