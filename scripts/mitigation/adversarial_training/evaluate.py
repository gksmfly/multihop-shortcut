"""Evaluate the mitigated model (scripts/mitigation/adversarial_training/01-02) on the exact same 3-condition
oracle test set as pipeline/evaluate_conditions.py, and compare against the original model's
condition_comparison.json to check whether the adversarial-training fix
actually closed the H3 shortcut (Answer-hop only should drop noticeably
relative to Full, unlike the original model where it was *higher*).
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from multihop_shortcut.constants import get_max_length
from multihop_shortcut.inference import load_qa_model, run_qa_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.metrics import exact_match, f1_score
from multihop_shortcut.paths import ERRORS_DIR, MODELS_DIR, PROCESSED_DIR

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa_mitigated" / "best"

MAX_LENGTH = get_max_length()

CONDITIONS = ["full", "answer_only", "bridge_only"]


def main() -> None:
    tokenizer, model, device = load_qa_model(MODEL_DIR)

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

    save_jsonl(out, ERRORS_DIR / "mitigated_test_predictions.jsonl")

    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {
            "em": sum(r[f"{condition}_em"] for r in out) / len(out),
            "f1": sum(r[f"{condition}_f1"] for r in out) / len(out),
            "mean_confidence": sum(r[f"{condition}_confidence"] for r in out) / len(out),
            "mean_cls_prob": sum(r[f"{condition}_cls_prob"] for r in out) / len(out),
        }
    with open(ERRORS_DIR / "mitigated_condition_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{'condition':<15}{'EM':>8}{'F1':>8}{'mean_conf':>12}{'mean_cls_prob':>15}")
    for condition, s in summary.items():
        print(
            f"{condition:<15}{s['em']:>8.4f}{s['f1']:>8.4f}"
            f"{s['mean_confidence']:>12.4f}{s['mean_cls_prob']:>15.4f}"
        )

    with open(ERRORS_DIR / "condition_comparison.json", encoding="utf-8") as f:
        original = json.load(f)

    gap_before = original["answer_only"]["em"] - original["full"]["em"]
    gap_after = summary["answer_only"]["em"] - summary["full"]["em"]

    lines = [
        "# 완화(mitigation) 실험 — Before / After\n",
        "H3에서 확인된 원인(질문의 타입 제약 누출)을 겨냥한 adversarial training "
        "(Answer-hop only를 학습 시 unanswerable로 라벨링)이 shortcut 의존도를 "
        "실제로 줄이는지 확인.\n",
        "| 조건 | EM (원본) | EM (완화 후) | F1 (원본) | F1 (완화 후) |",
        "|---|---|---|---|---|",
    ]
    for c in CONDITIONS:
        lines.append(
            f"| {c} | {original[c]['em']:.4f} | {summary[c]['em']:.4f} "
            f"| {original[c]['f1']:.4f} | {summary[c]['f1']:.4f} |"
        )
    lines += [
        "",
        f"**Answer-hop only − Full (EM 격차)**: 원본 {gap_before:+.4f} → "
        f"완화 후 {gap_after:+.4f}",
        "",
        (
            "격차가 양수(+)면 여전히 shortcut이 우세(Answer-hop only가 Full보다 높음), "
            "음수(−)로 뒤집히거나 0에 가까워지면 완화가 통했다는 뜻."
        ),
    ]
    with open(ERRORS_DIR / "mitigation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
