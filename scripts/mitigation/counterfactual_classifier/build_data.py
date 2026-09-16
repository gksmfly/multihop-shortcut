"""B-v3 (counterfactual labeling), reviewed and approved with two known
limitations (see docs/mitigation-experiment.md "향후 과제"):
  1. Stage-1-specific: this labels *this* checkpoint's shortcut behavior,
     not a general notion of bridge dependence.
  2. Label is correctness-agnostic: P1 != P2 only means the bridge changed
     the prediction, not that it changed it to the *correct* answer.

Runs the ORIGINAL (unmitigated) model - models/multihop_shortcut_qa/best,
the same Stage 1 used throughout pipeline/evaluate_conditions.py through fame_bias_analysis.py and in bridge_relatedness_classifier/evaluate.py - on the
train/val split under Answer-hop-only (P1) and Full (P2) context, and
labels each row 1 if bridge changed the prediction (P1 != P2 after answer
normalization) else 0. Also writes the correctness breakdown (both
wrong / bridge helped / bridge hurt / no change) needed to quantify
limitation 2 with real numbers.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast

from multihop_shortcut.inference import run_qa_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.metrics import counterfactual_bucket, exact_match
from multihop_shortcut.paths import MODELS_DIR, PROCESSED_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa" / "best"  # original, unmitigated

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def label_split(rows: list[dict], pool_by_qid: dict[str, dict], model, tokenizer, device) -> list[dict]:
    examples_p1 = [{"question": r["question"], "context": pool_by_qid[r["qid"]]["answer_hop_text"]} for r in rows]
    examples_p2 = [{"question": r["question"], "context": r["context"]} for r in rows]

    preds_p1 = run_qa_inference(model, tokenizer, examples_p1, device, max_length=MAX_LENGTH)
    preds_p2 = run_qa_inference(model, tokenizer, examples_p2, device, max_length=MAX_LENGTH)

    out = []
    for row, p1, p2 in zip(rows, preds_p1, preds_p2):
        pool_row = pool_by_qid[row["qid"]]
        p1_text, p2_text = p1["pred_text"], p2["pred_text"]
        p1_correct = exact_match(p1_text, row["answer"]) == 1.0
        p2_correct = exact_match(p2_text, row["answer"]) == 1.0
        bucket = counterfactual_bucket(p1_text, p2_text, p1_correct, p2_correct)

        out.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "bridge_hop_text": pool_row["bridge_hop_text"],
                "label": int(bucket != "no_change"),
                "bucket": bucket,
                "p1_pred": p1_text,
                "p2_pred": p2_text,
                "p1_correct": p1_correct,
                "p2_correct": p2_correct,
            }
        )
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = BertForQuestionAnswering.from_pretrained(str(MODEL_DIR)).to(device)

    pool_rows = load_jsonl(PROCESSED_DIR / "train_pool.jsonl")
    pool_by_qid = {r["qid"]: r for r in pool_rows}

    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")

    train_out = label_split(train_rows, pool_by_qid, model, tokenizer, device)
    val_out = label_split(val_rows, pool_by_qid, model, tokenizer, device)

    save_jsonl(train_out, SPLITS_DIR / "train_counterfactual.jsonl")
    save_jsonl(val_out, SPLITS_DIR / "val_counterfactual.jsonl")

    for name, rows in [("train", train_out), ("val", val_out)]:
        n = len(rows)
        buckets = {}
        for r in rows:
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
        print(f"\n[{name}] n={n}")
        for b, c in sorted(buckets.items()):
            print(f"  {b}: {c} ({c/n:.1%})")


if __name__ == "__main__":
    main()
