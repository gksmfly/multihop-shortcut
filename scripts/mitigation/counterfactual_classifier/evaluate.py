"""B-v3 evaluation. Ground-truth counterfactual labels on the TEST set come
for free from pipeline/evaluate_conditions.py's already-saved test_predictions.jsonl (full_pred,
answer_only_pred, full_em, answer_only_em) - no need to re-run Stage 1.

Reports three things:
  1. Classifier accuracy/precision/recall against the ground-truth P1!=P2
     label - the train label is heavily imbalanced (90.8% no_change), so
     raw accuracy alone can hide a classifier that just always predicts 0.
  2. Veto gap - same metric as B-v1 (scripts/mitigation/bridge_relatedness_classifier/evaluate.py): does the predicted score
     differ between Stage 1 correct vs incorrect cases?
  3. Limitation 2, quantified on test: among cases the classifier flags as
     "bridge changes the answer", what fraction are actually bridge_helped
     vs bridge_hurt vs changed_still_wrong - i.e. would trusting this flag
     as a veto signal net help or hurt.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from multihop_shortcut.inference import run_classifier_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.metrics import counterfactual_bucket
from multihop_shortcut.paths import ERRORS_DIR, MODELS_DIR, PROCESSED_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "support_classifier_counterfactual" / "best"

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)

    preds = load_jsonl(ERRORS_DIR / "test_predictions.jsonl")
    conditions = {r["qid"]: r for r in load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")}

    questions = [r["question"] for r in preds]
    bridges = [conditions[r["qid"]]["bridge_only_context"] for r in preds]
    pred_scores = run_classifier_inference(model, tokenizer, device, questions, bridges, max_length=MAX_LENGTH)

    out = []
    for r, s in zip(preds, pred_scores):
        p1_correct = r["answer_only_em"] == 1
        p2_correct = r["full_em"] == 1
        bucket = counterfactual_bucket(r["answer_only_pred"], r["full_pred"], p1_correct, p2_correct)
        out.append(
            {
                "qid": r["qid"],
                "gt_label": int(bucket != "no_change"),
                "bucket": bucket,
                "answer_only_em": r["answer_only_em"],
                "pred_score": s,
                "pred_label": int(s >= 0.5),
            }
        )
    save_jsonl(out, ERRORS_DIR / "counterfactual_classifier_predictions.jsonl")

    n = len(out)
    tp = sum(1 for o in out if o["gt_label"] == 1 and o["pred_label"] == 1)
    fp = sum(1 for o in out if o["gt_label"] == 0 and o["pred_label"] == 1)
    fn = sum(1 for o in out if o["gt_label"] == 1 and o["pred_label"] == 0)
    tn = sum(1 for o in out if o["gt_label"] == 0 and o["pred_label"] == 0)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    majority_baseline = max(sum(1 for o in out if o["gt_label"] == 0), sum(1 for o in out if o["gt_label"] == 1)) / n

    correct = [o for o in out if o["answer_only_em"] == 1]
    wrong = [o for o in out if o["answer_only_em"] == 0]
    mean_score_correct = sum(o["pred_score"] for o in correct) / len(correct)
    mean_score_wrong = sum(o["pred_score"] for o in wrong) / len(wrong)
    veto_gap = mean_score_wrong - mean_score_correct  # want this positive: flagged more on wrong answers

    flagged = [o for o in out if o["pred_label"] == 1]
    flagged_buckets = {}
    for o in flagged:
        flagged_buckets[o["bucket"]] = flagged_buckets.get(o["bucket"], 0) + 1

    lines = [
        "# B-v3 Counterfactual 라벨링 — 평가 결과\n",
        f"n = {n} (test set, 원본 모델 Stage 1 그대로)\n",
        "## 1. 분류기 성능 (ground-truth P1≠P2 라벨 대비)",
        "| | 값 |",
        "|---|---|",
        f"| accuracy | {accuracy:.4f} |",
        f"| precision (label=1) | {precision:.4f} |",
        f"| recall (label=1) | {recall:.4f} |",
        f"| 다수결 baseline(항상 다수 클래스 예측 시 accuracy) | {majority_baseline:.4f} |",
        "",
        (
            "accuracy가 다수결 baseline과 비슷하면 → 분류기가 사실상 '항상 no_change로 "
            "찍는' 수준이라는 뜻(라벨 불균형 90.8:9.2 때문에 흔한 함정). recall이 낮으면 "
            "실제 '영향 있음' 케이스를 거의 못 잡아낸다는 뜻."
        ),
        "",
        "## 2. Veto gap (B-v1과 동일 지표)",
        "| | n | pred_score (mean) |",
        "|---|---|---|",
        f"| Stage 1 정답 | {len(correct)} | {mean_score_correct:.4f} |",
        f"| Stage 1 오답 | {len(wrong)} | {mean_score_wrong:.4f} |",
        f"| 격차(오답 − 정답) | | {veto_gap:+.4f} |",
        "",
        "## 3. 한계 2 실증 — 분류기가 '영향 있음(1)'으로 flag한 케이스의 실제 구성",
        "| bucket | n | flag된 것 중 비율 |",
        "|---|---|---|",
    ]
    n_flagged = len(flagged) if flagged else 1
    for b in ["bridge_helped", "bridge_hurt", "changed_still_wrong", "no_change"]:
        c = flagged_buckets.get(b, 0)
        lines.append(f"| {b} | {c} | {c/n_flagged:.1%} |")
    lines += [
        "",
        (
            "이 표가 veto 시스템의 실제 가치를 보여준다 — flag된 케이스 중 "
            "bridge_helped 비율이 낮고 bridge_hurt/changed_still_wrong 비율이 높으면, "
            "이 flag를 신뢰해서 답을 바꾸는 건 득보다 실이 크다는 뜻."
        ),
    ]

    summary = {
        "n": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "majority_baseline": majority_baseline,
        "veto_gap": veto_gap,
        "mean_score_stage1_correct": mean_score_correct,
        "mean_score_stage1_wrong": mean_score_wrong,
        "flagged_bucket_counts": flagged_buckets,
        "n_flagged": len(flagged),
    }
    with open(ERRORS_DIR / "counterfactual_classifier_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ERRORS_DIR / "counterfactual_classifier_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
