#!/usr/bin/env python3
"""Create Kaggle submission using a locally trained LoRA adapter.

Default method is next-token logit scoring over A/B/C/D. This is more stable
than free-form generation for single-choice tasks.
"""

import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

CHOICES = ["A", "B", "C", "D"]


def clean_text(x) -> str:
    return str(x).strip()


def format_prompt(row: Dict[str, str]) -> str:
    return (
        "請根據題目與四個選項，選出唯一正確答案。\n"
        "規則：只輸出 A、B、C 或 D，不要輸出解釋。\n\n"
        f"題目：{clean_text(row['Question'])}\n"
        f"A. {clean_text(row['Option A'])}\n"
        f"B. {clean_text(row['Option B'])}\n"
        f"C. {clean_text(row['Option C'])}\n"
        f"D. {clean_text(row['Option D'])}\n"
        "答案："
    )


def load_model(base_model: str, adapter_path: str, no_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path if os.path.exists(adapter_path) else base_model, trust_remote_code=True)
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def score_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    scores = {}
    for ch in CHOICES:
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        scores[ch] = float(logits[ids[0]])
    return max(scores, key=scores.get)


def generate_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    for ch in text:
        if ch in CHOICES:
            return ch
    return score_one(model, tokenizer, prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--sample_submission", required=True)
    parser.add_argument("--output_csv", default="submission.csv")
    parser.add_argument("--method", choices=["logit", "generate"], default="logit")
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv, encoding="utf-8-sig")
    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    model, tokenizer = load_model(args.base_model, args.adapter_path, args.no_4bit)

    preds: List[str] = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = format_prompt(row.to_dict())
        pred = score_one(model, tokenizer, prompt) if args.method == "logit" else generate_one(model, tokenizer, prompt)
        preds.append(pred)

    answer_col = "Answer" if "Answer" in sample.columns else sample.columns[-1]
    output = sample.copy()
    output[answer_col] = preds
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved submission to {args.output_csv}")
    print(output[answer_col].value_counts().sort_index())


if __name__ == "__main__":
    main()
