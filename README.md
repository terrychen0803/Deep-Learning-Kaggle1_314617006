# NYCU IAL DL 2026 — LLM-1 SFT on Answer-Only Data

本專案用於完成 Kaggle 作業「Reasoning LLM - Step1: Supervised Task Finetuning w/o reasoning information」。方法為：使用中國釋出的大語言模型，例如 Qwen，進行 answer-only supervised fine-tuning，讓模型針對單選題只輸出 `A/B/C/D`。


---

## 1. 檔案結構

```text
nycu_llm1_sft_project/
├── scripts/
│   ├── prepare_data.py          # 清理 train.csv / HW1_.csv，產生 answer-only SFT JSONL
│   ├── train_qlora_sft.py       # 使用 Qwen + QLoRA 進行本地端 SFT
│   ├── quick_eval.py            # 用 validation set 估計 A/B/C/D 準確率
│   └── infer_submission.py      # 對 test-check-v2 產生 Kaggle submission.csv
├── configs/
│   └── qwen25_15b_qlora.json    # 建議訓練設定
├── data/                        # prepare_data.py 輸出
├── outputs/                     # 訓練後 LoRA adapter 與 submission.csv
└── requirements.txt
 
```

---

## 2. 資料檢查結果

上傳資料的基本狀況：

| 檔案 | 筆數 | 欄位 | 用途 |
|---|---:|---|---|
| `train.csv` | 2825 | `ID, Question, Option A-D, Answer` | 官方訓練集 |
| `HW1_.csv` | 3500 | `題目, 選項A-D, 答案` | 可選擇作為補充訓練資料 |
| `kaggle_test_set_792.csv` | 792 | `ID, Question, Option A-D` | 正式測試集，應使用此檔 |
| `kaggle_submission_template_792.csv` | 792 | `ID, Answer` | Kaggle 提交格式 |

清理策略：

1. `train.csv` 有一筆因 `Great Firewall, GFW` 中的逗號造成欄位錯位，已在 `prepare_data.py` 中修正，正確答案設定為 `D`。
2. `HW1_.csv` 中答案含空白者會自動 strip，例如 ` A` 轉成 `A`。
3. `HW1_.csv` 中答案為 `X` 的資料會移除。
4. Exact duplicate 會移除；同題同選項但答案衝突者會整組移除。
5. Prompt/completion 格式只把 `A/B/C/D` 作為 completion，避免模型學到推理文字。

---

## 3. 環境安裝

建議使用 Linux / WSL / Ubuntu。Windows 原生也可以，但 `bitsandbytes` 可能較容易遇到相容性問題。

```bash
conda create -n nycu-llm1 python=3.10 -y
conda activate nycu-llm1
pip install -r requirements.txt
wandb login
```

---

## 4. 放置原始資料

請建立 `raw_data/`，並放入四個原始檔案：

```text
raw_data/
├── train.csv
├── HW1_.csv
├── kaggle_test_set_792.csv
└── kaggle_submission_template_792.csv
```

---

## 5. 產生訓練資料

### 使用官方 train.csv + HW1_.csv 補充資料

```bash
python scripts/prepare_data.py \
  --train_csv raw_data/train.csv \
  --hw1_csv raw_data/HW1_.csv \
  --use_hw1 \
  --out_dir data \
  --valid_ratio 0.08
```

輸出檔案：

```text
data/cleaned_sft_data.csv
data/train.jsonl
data/valid.jsonl
data/prepare_report.json
```

---

## 6. 訓練 Qwen + QLoRA

測試使用 `Qwen/Qwen2.5-0.5B-Instruct

### 穩定 baseline

```bash
python scripts/train_qlora_sft.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --output_dir outputs/qwen25_15b_lora \
  --epochs 3 \
  --max_length 768 \
  --lr 2e-4 \
  --batch_size 1 \
  --grad_accum 16 \
  --wandb_project nycu-llm1-sft
```

---

## 7. Validation 評估

```bash
python scripts/quick_eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/qwen25_15b_lora \
  --valid_jsonl data/valid.jsonl
```

---

## 8. 產生 Kaggle submission.csv

```bash
python scripts/infer_submission.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/qwen25_15b_lora \
  --test_csv raw_data/kaggle_test_set_792.csv \
  --sample_submission raw_data/kaggle_submission_template_792.csv \
  --output_csv outputs/submission.csv \
  --method logit
```

`--method logit` 會對 `A/B/C/D` 做 next-token logit scoring，通常比讓模型自由生成更穩定。

輸出：

```text
outputs/submission.csv
```

提交到 Kaggle 時，欄位會維持：

```csv
ID,Answer
1,A
2,B
...
```

---

## 9. 實驗紀錄

| 實驗 | 模型 | 資料 | Epoch | 推論方式 |  |
|---|---|---|---:|---|---|
| Exp-1 | Qwen2.5-0.5B-Instruct | official  | 1 | logit | 
| Exp-2 | Qwen2.5-1.5B-Instruct | official  | 3 | logit | 
| Exp-3 | Qwen2.5-0.5B-Instruct | official  | 3 | logit | 
| Exp-4 | Qwen2.5-0.5B-Instruct | mixed     | 3 | logit | 
| Exp-5 | Qwen2.5-0.5B-Instruct | official  | 4 | logit |

結果紀錄：\
Exp-1:\
validation accuraccy: 0.8462\
score: 0.62127\
Exp-2:\
改用1.5b模型，epoch增加為3，增加訓練量
validation accuraccy: 0.9873
score: 0.61276
本地測試表現良好，但是分數不佳，推測對於資料過擬合
Exp-3:
改回0.5b模型，learning rate調低，增加drop out防止過擬合
validation accuraccy: 0.9223
score:0.66382
分數有提升，繼續調整參數
Exp-4:
使用官方訓練集與補充訓練集混合資料
validation accuraccy: 0.8864
score:0.63404
分數並沒有比較好，推測合併補充訓練集後答案分布偏向 B，而 D 很少，可能使模型產偏差
Exp-5:
改回使用原始資料集，已Exp-2為主調整參數，增加epoch數，降低learning rate
validation accuraccy:valid accuracy = 0.9223
score:0.63829


---


