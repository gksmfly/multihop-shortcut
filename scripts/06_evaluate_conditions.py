"""Run the trained model on the test set under all three eval conditions
(Full / Answer-hop only / Bridge-hop only) and record per-sample EM/F1 and
confidence.

Bridge-hop only's EM/F1 is recorded for completeness but is not the metric
hypothesis 2 is judged on (the answer isn't in that context by construction,
so EM/F1 is expected to be near 0 regardless of what the model does - see
README and scripts/07_confidence_bias_analysis.py).
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast

from multihop_shortcut.constants import BASE_MODEL_NAME
from multihop_shortcut.inference import run_qa_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.metrics import exact_match, f1_score
from multihop_shortcut.paths import ERRORS_DIR, MODELS_DIR, PROCESSED_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa" / "best"

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]

CONDITIONS = ["full", "answer_only", "bridge_only"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = BertForQuestionAnswering.from_pretrained(str(MODEL_DIR)).to(device)

    rows = load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")

    predictions_by_condition = {}
    for condition in CONDITIONS:
        context_field = f"{condition}_context"
        examples = [{"question": r["question"], "context": r[context_field]} for r in rows]
        predictions_by_condition[condition] = run_qa_inference(
            model, tokenizer, examples, device, max_length=MAX_LENGTH
        )

    out = []
    for i, row in enumerate(rows):
        record = {
            "qid": row["qid"],
            "question": row["question"],
            "answer": row["answer"],
            "level": row["level"],
            "answer_hop_title": row["answer_hop_title"],
            "bridge_hop_title": row["bridge_hop_title"],
        }
        for condition in CONDITIONS:
            pred = predictions_by_condition[condition][i]
            record[f"{condition}_pred"] = pred["pred_text"]
            record[f"{condition}_confidence"] = pred["confidence"]
            record[f"{condition}_cls_prob"] = pred["cls_prob"]
            record[f"{condition}_em"] = exact_match(pred["pred_text"], row["answer"])
            record[f"{condition}_f1"] = f1_score(pred["pred_text"], row["answer"])
        out.append(record)

    save_jsonl(out, ERRORS_DIR / "test_predictions.jsonl")

    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {
            "em": sum(r[f"{condition}_em"] for r in out) / len(out),
            "f1": sum(r[f"{condition}_f1"] for r in out) / len(out),
            "mean_confidence": sum(r[f"{condition}_confidence"] for r in out) / len(out),
            "mean_cls_prob": sum(r[f"{condition}_cls_prob"] for r in out) / len(out),
        }
    with open(ERRORS_DIR / "condition_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{'condition':<15}{'EM':>8}{'F1':>8}{'mean_conf':>12}{'mean_cls_prob':>15}")
    for condition, s in summary.items():
        print(
            f"{condition:<15}{s['em']:>8.4f}{s['f1']:>8.4f}"
            f"{s['mean_confidence']:>12.4f}{s['mean_cls_prob']:>15.4f}"
        )


if __name__ == "__main__":
    main()
