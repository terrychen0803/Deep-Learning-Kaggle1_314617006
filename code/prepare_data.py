#!/usr/bin/env python3
"""Prepare answer-only SFT data for NYCU IAL DL 2026 LLM-1.

This script cleans the official training csv and optionally appends HW1_.csv
as extra supervised data. It outputs prompt/completion JSONL files where only
A/B/C/D is used as the completion target.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd

VALID_ANSWERS = {"A", "B", "C", "D"}


def clean_text(x) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip('"').strip()
    return s


def normalize_answer(x):
    s = str(x).strip().upper()
    mapping = {
        "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D",
        "A.": "A", "B.": "B", "C.": "C", "D.": "D",
        "A、": "A", "B、": "B", "C、": "C", "D、": "D",
        "選項A": "A", "選項B": "B", "選項C": "C", "選項D": "D",
    }
    s = mapping.get(s, s)
    return s if s in VALID_ANSWERS else None


def fix_official_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["Question", "Option A", "Option B", "Option C", "Option D", "Answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}; got {df.columns.tolist()}")

    # The uploaded train.csv contains one malformed row caused by an internal comma
    # in "Great Firewall, GFW". Its answer appears as the text of option D.
    for idx, row in df.iterrows():
        raw_answer = str(row["Answer"]).strip()
        if normalize_answer(raw_answer) is None and raw_answer == "以上皆是":
            df.at[idx, "Question"] = f"{clean_text(row['Question'])}, {clean_text(row['Option A'])}"
            df.at[idx, "Option A"] = row["Option B"]
            df.at[idx, "Option B"] = row["Option C"]
            df.at[idx, "Option C"] = row["Option D"]
            df.at[idx, "Option D"] = raw_answer
            df.at[idx, "Answer"] = "D"

    for col in required:
        df[col] = df[col].map(clean_text)
    df["Answer"] = df["Answer"].map(normalize_answer)
    df = df[df["Answer"].isin(VALID_ANSWERS)].copy()
    df["source"] = "official_train"
    return df[["Question", "Option A", "Option B", "Option C", "Option D", "Answer", "source"]]


def load_hw1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={
        "題目": "Question",
        "選項A": "Option A",
        "選項B": "Option B",
        "選項C": "Option C",
        "選項D": "Option D",
        "答案": "Answer",
    })
    required = ["Question", "Option A", "Option B", "Option C", "Option D", "Answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HW1 csv missing columns: {missing}; got {df.columns.tolist()}")

    for col in required:
        df[col] = df[col].map(clean_text)
    df["Answer"] = df["Answer"].map(normalize_answer)
    df = df[df["Answer"].isin(VALID_ANSWERS)].copy()
    df["source"] = "hw1_augmented"
    return df[["Question", "Option A", "Option B", "Option C", "Option D", "Answer", "source"]]


def remove_exact_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Question", "Option A", "Option B", "Option C", "Option D"]
    grouped = df.groupby(key_cols)["Answer"].nunique().reset_index(name="n_answers")
    conflict_keys = grouped[grouped["n_answers"] > 1][key_cols]
    if len(conflict_keys) > 0:
        conflict_keys = set(tuple(x) for x in conflict_keys.to_numpy())
        mask = df[key_cols].apply(lambda r: tuple(r.to_numpy()) in conflict_keys, axis=1)
        df = df[~mask].copy()
    df = df.drop_duplicates(subset=key_cols + ["Answer"]).reset_index(drop=True)
    return df


def format_prompt(row: Dict[str, str]) -> str:
    return (
        "請根據題目與四個選項，選出唯一正確答案。\n"
        "規則：只輸出 A、B、C 或 D，不要輸出解釋。\n\n"
        f"題目：{row['Question']}\n"
        f"A. {row['Option A']}\n"
        f"B. {row['Option B']}\n"
        f"C. {row['Option C']}\n"
        f"D. {row['Option D']}\n"
        "答案："
    )


def to_examples(df: pd.DataFrame) -> List[Dict[str, str]]:
    examples = []
    for _, row in df.iterrows():
        r = row.to_dict()
        examples.append({
            "prompt": format_prompt(r),
            "completion": r["Answer"],
            "answer": r["Answer"],
            "question": r["Question"],
            "source": r["source"],
        })
    return examples


def stratified_split(examples: List[Dict[str, str]], valid_ratio: float, seed: int):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for ex in examples:
        by_label[ex["answer"]].append(ex)
    train, valid = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_valid = max(1, int(round(len(items) * valid_ratio))) if valid_ratio > 0 else 0
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path: str, examples: List[Dict[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--hw1_csv", default=None)
    parser.add_argument("--use_hw1", action="store_true", help="Append cleaned HW1_.csv as extra SFT data.")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--valid_ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    official_raw = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    official = fix_official_train(official_raw)
    frames = [official]

    if args.use_hw1:
        if not args.hw1_csv:
            raise ValueError("--use_hw1 requires --hw1_csv")
        frames.append(load_hw1(args.hw1_csv))

    df = pd.concat(frames, ignore_index=True)
    before_dedup = len(df)
    df = remove_exact_conflicts(df)
    after_dedup = len(df)

    examples = to_examples(df)
    train_examples, valid_examples = stratified_split(examples, args.valid_ratio, args.seed)

    df.to_csv(os.path.join(args.out_dir, "cleaned_sft_data.csv"), index=False, encoding="utf-8-sig")
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_examples)
    write_jsonl(os.path.join(args.out_dir, "valid.jsonl"), valid_examples)

    report = {
        "official_rows_after_cleaning": int(len(official)),
        "use_hw1": bool(args.use_hw1),
        "total_before_dedup": int(before_dedup),
        "total_after_dedup": int(after_dedup),
        "train_examples": int(len(train_examples)),
        "valid_examples": int(len(valid_examples)),
        "answer_distribution_all": dict(Counter(df["Answer"])),
        "source_distribution_all": dict(Counter(df["source"])),
    }
    with open(os.path.join(args.out_dir, "prepare_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
