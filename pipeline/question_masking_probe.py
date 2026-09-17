"""Hypothesis 3: does Answer-hop-only performance survive because the
*question itself* (independent of the bridge paragraph) already narrows the
answer? We mask the literal bridge-entity mention in the question with a
placeholder and re-run Answer-hop only - if performance barely drops, the
rest of the question wording (not the bridge entity name) is doing the work.

Only applicable to the subset of test samples whose question contains the
bridge_hop_title as a literal substring (masking only makes sense there).
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from multihop_shortcut.constants import get_max_length
from multihop_shortcut.inference import load_qa_model, run_qa_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.metrics import exact_match, f1_score
from multihop_shortcut.paths import ERRORS_DIR, MODELS_DIR, PROCESSED_DIR

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa" / "best"
MASK_TOKEN = "[ENTITY]"

MAX_LENGTH = get_max_length()


def main() -> None:
    tokenizer, model, device = load_qa_model(MODEL_DIR)

    preds = {r["qid"]: r for r in load_jsonl(ERRORS_DIR / "test_predictions.jsonl")}
    conditions = {r["qid"]: r for r in load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")}

    subset = []
    for qid, pred in preds.items():
        cond = conditions[qid]
        if pred["bridge_hop_title"] in pred["question"]:
            subset.append(
                {
                    "qid": qid,
                    "question": pred["question"],
                    "masked_question": pred["question"].replace(
                        pred["bridge_hop_title"], MASK_TOKEN
                    ),
                    "answer": pred["answer"],
                    "answer_hop_text": cond["answer_only_context"],
                    "original_answer_only_em": pred["answer_only_em"],
                    "original_answer_only_f1": pred["answer_only_f1"],
                    "original_answer_only_pred": pred["answer_only_pred"],
                }
            )

    examples = [{"question": r["masked_question"], "context": r["answer_hop_text"]} for r in subset]
    masked_preds = run_qa_inference(model, tokenizer, examples, device, max_length=MAX_LENGTH)

    out = []
    for r, pred in zip(subset, masked_preds):
        out.append(
            {
                **r,
                "masked_pred": pred["pred_text"],
                "masked_em": exact_match(pred["pred_text"], r["answer"]),
                "masked_f1": f1_score(pred["pred_text"], r["answer"]),
            }
        )

    save_jsonl(out, ERRORS_DIR / "question_masking_predictions.jsonl")

    n = len(out)
    if n == 0:
        print("No samples had a literal bridge-title mention in the question - nothing to probe.")
        return

    orig_em = sum(r["original_answer_only_em"] for r in out) / n
    orig_f1 = sum(r["original_answer_only_f1"] for r in out) / n
    masked_em = sum(r["masked_em"] for r in out) / n
    masked_f1 = sum(r["masked_f1"] for r in out) / n

    summary = {
        "n_subset": n,
        "n_total_test": len(preds),
        "subset_fraction": n / len(preds),
        "answer_only_em_before_masking": orig_em,
        "answer_only_em_after_masking": masked_em,
        "answer_only_f1_before_masking": orig_f1,
        "answer_only_f1_after_masking": masked_f1,
        "em_drop": orig_em - masked_em,
        "f1_drop": orig_f1 - masked_f1,
    }
    with open(ERRORS_DIR / "question_masking_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# 가설 3 — 질문 마스킹 프로브\n",
        f"질문에 bridge 제목이 literal하게 등장하는 서브셋: {n}/{len(preds)} "
        f"({summary['subset_fraction']:.1%})\n",
        "| | EM | F1 |",
        "|---|---|---|",
        f"| 마스킹 전(원래 질문) | {orig_em:.4f} | {orig_f1:.4f} |",
        f"| 마스킹 후([ENTITY] 치환) | {masked_em:.4f} | {masked_f1:.4f} |",
        "",
        f"EM 하락폭: {summary['em_drop']:.4f} / F1 하락폭: {summary['f1_drop']:.4f}",
        "",
        "하락폭이 작으면 → 질문의 나머지 부분만으로 이미 답이 좁혀진다는 뜻"
        "(가설 3 지지). 하락폭이 크면 → bridge 엔티티의 구체적인 이름 자체가"
        " 필요했다는 뜻(가설 3 반증, 가설 4 쪽에 무게 실림).",
    ]
    with open(ERRORS_DIR / "question_masking_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
