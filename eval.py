import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from config.path_config import (
    COMPARISON_DETAIL_CSV,
    COMPARISON_SUMMARY_CSV,
    TASK2_BASELINE_CSV,
    TASK2_FINETUNE_CSV,
    ensure_runtime_dirs,
)

BASELINE_CSV = TASK2_BASELINE_CSV
FINETUNE_CSV = TASK2_FINETUNE_CSV
OUTPUT_CSV = COMPARISON_SUMMARY_CSV
DETAIL_CSV = COMPARISON_DETAIL_CSV

smooth_fn = SmoothingFunction().method1
rouge = Rouge()

def score_example(reference, prediction):
    ref = str(reference).strip()
    pred = str(prediction).strip()

    em = 1 if pred == ref else 0
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)

    rouge_scores = rouge.get_scores(pred, ref)[0]
    rouge_1 = rouge_scores["rouge-1"]["f"]
    rouge_2 = rouge_scores["rouge-2"]["f"]
    rouge_l = rouge_scores["rouge-l"]["f"]

    return em, bleu, rouge_1, rouge_2, rouge_l

def evaluate_predictions(file_path, model_name):
    df = pd.read_csv(file_path)
    rows = []

    for _, row in df.iterrows():
        em, bleu, rouge_1, rouge_2, rouge_l = score_example(
            row["reference"], row["generated"]
        )
        rows.append(
            {
                "model": model_name,
                "prompt": row["prompt"],
                "reference": row["reference"],
                "generated": row["generated"],
                "EM": em,
                "BLEU": bleu,
                "ROUGE-1": rouge_1,
                "ROUGE-2": rouge_2,
                "ROUGE-L": rouge_l,
            }
        )

    detail_df = pd.DataFrame(rows)
    summary = {
        "model": model_name,
        "n_samples": len(detail_df),
        "EM": detail_df["EM"].mean(),
        "BLEU": detail_df["BLEU"].mean(),
        "ROUGE-1": detail_df["ROUGE-1"].mean(),
        "ROUGE-2": detail_df["ROUGE-2"].mean(),
        "ROUGE-L": detail_df["ROUGE-L"].mean(),
    }
    return detail_df, summary

baseline_detail, baseline_summary = evaluate_predictions(BASELINE_CSV, "baseline")
finetune_detail, finetune_summary = evaluate_predictions(FINETUNE_CSV, "finetuned")

detail_df = pd.concat([baseline_detail, finetune_detail], ignore_index=True)
summary_df = pd.DataFrame([baseline_summary, finetune_summary])

ensure_runtime_dirs()
detail_df.to_csv(DETAIL_CSV, index=False)
summary_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved test-set summary metrics to {OUTPUT_CSV}")
print(f"Saved per-sample test-set metrics to {DETAIL_CSV}")
