"""v2/B-v1 two-stage design - evaluation.

Stage 1 is NOT retrained: it's the *original* (unmitigated) model's
Answer-hop-only predictions from pipeline/evaluate_conditions.py (data/errors/test_predictions.jsonl)
- exactly the shortcut-prone candidate generator H1-H3 diagnosed.

Stage 2 is the scripts/mitigation/bridge_relatedness_classifier/train.py support classifier, scored two ways on the same
test set:
  1. against the TRUE bridge_hop_text (does it recognize genuine support?)
  2. against a RANDOM other row's bridge_hop_text (the ablation: if the
     score doesn't drop for a random/mismatched bridge, Stage 2 - and by
     extension this whole verification layer - isn't actually using bridge
     content, i.e. the shortcut survives one level up).

Also splits support_score_true by whether Stage 1's shortcut-prone answer
was actually right or wrong (original_answer_only_em) - if wrong answers
get systematically lower support scores, Stage 2 can act as a veto/flag on
shortcut failures even without a full top-k re-ranking pipeline (out of
scope given the time budget - see README-equivalent note in the report).
"""

import json
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from multihop_shortcut.inference import run_classifier_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.paths import ERRORS_DIR, MODELS_DIR, PROCESSED_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "support_classifier" / "best"
SEED = 42

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def main() -> None:
    rng = random.Random(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)

    preds = load_jsonl(ERRORS_DIR / "test_predictions.jsonl")  # original (unmitigated) Stage 1
    conditions = {r["qid"]: r for r in load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")}

    questions = [r["question"] for r in preds]
    true_bridges = [conditions[r["qid"]]["bridge_only_context"] for r in preds]

    random_bridges = []
    for i in range(len(preds)):
        j = i
        while j == i:
            j = rng.randrange(len(preds))
        random_bridges.append(conditions[preds[j]["qid"]]["bridge_only_context"])

    support_true = run_classifier_inference(model, tokenizer, device, questions, true_bridges, max_length=MAX_LENGTH)
    support_random = run_classifier_inference(
        model, tokenizer, device, questions, random_bridges, max_length=MAX_LENGTH
    )

    out = []
    for r, s_true, s_rand in zip(preds, support_true, support_random):
        out.append(
            {
                "qid": r["qid"],
                "answer_only_em": r["answer_only_em"],
                "answer_only_confidence": r["answer_only_confidence"],
                "support_score_true_bridge": s_true,
                "support_score_random_bridge": s_rand,
                "combined_score": r["answer_only_confidence"] * s_true,
            }
        )
    save_jsonl(out, ERRORS_DIR / "two_stage_predictions.jsonl")

    n = len(out)
    mean_true = sum(o["support_score_true_bridge"] for o in out) / n
    mean_random = sum(o["support_score_random_bridge"] for o in out) / n
    correct = [o for o in out if o["answer_only_em"] == 1]
    wrong = [o for o in out if o["answer_only_em"] == 0]
    mean_true_correct = sum(o["support_score_true_bridge"] for o in correct) / len(correct)
    mean_true_wrong = sum(o["support_score_true_bridge"] for o in wrong) / len(wrong)

    ablation_gap = mean_true - mean_random
    veto_gap = mean_true_correct - mean_true_wrong

    lines = [
        "# v2 Two-stage 실험 (Stage 1 그대로 + Stage 2 support classifier)\n",
        f"n = {n}\n",
        "## Ablation — 진짜 bridge vs 랜덤 bridge",
        "| | support score (mean) |",
        "|---|---|",
        f"| 진짜 bridge_hop | {mean_true:.4f} |",
        f"| 랜덤 bridge_hop (다른 질문에서) | {mean_random:.4f} |",
        f"| 격차 | {ablation_gap:+.4f} |",
        "",
        (
            "격차가 크면(양수, 유의미) → Stage 2가 실제로 bridge 내용을 이용해 "
            "지지 여부를 구분한다는 뜻. 격차가 작으면 → Stage 2도 표면적 패턴만 "
            "학습했을 뿐, bridge 정보를 실제로 검증하지 못한다는 뜻."
        ),
        "",
        "## Veto 가능성 — Stage 1이 맞았을 때 vs 틀렸을 때, 진짜 bridge에 대한 support score",
        "| | n | support score (mean) |",
        "|---|---|---|",
        f"| Stage 1 정답(EM=1) | {len(correct)} | {mean_true_correct:.4f} |",
        f"| Stage 1 오답(EM=0) | {len(wrong)} | {mean_true_wrong:.4f} |",
        f"| 격차 | | {veto_gap:+.4f} |",
        "",
        (
            "오답 쪽 support score가 눈에 띄게 낮으면 → Stage 2가 Stage 1의 "
            "shortcut 오답을 사후에 flag/veto할 수 있는 신호를 제공한다는 뜻 "
            "(top-k 재순위화까지는 이번 실험 범위 밖 - 시간 제약)."
        ),
    ]
    summary = {
        "n": n,
        "mean_support_true_bridge": mean_true,
        "mean_support_random_bridge": mean_random,
        "ablation_gap": ablation_gap,
        "mean_support_true_when_stage1_correct": mean_true_correct,
        "mean_support_true_when_stage1_wrong": mean_true_wrong,
        "veto_gap": veto_gap,
    }
    with open(ERRORS_DIR / "two_stage_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ERRORS_DIR / "two_stage_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
