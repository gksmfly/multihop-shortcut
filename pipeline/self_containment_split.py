"""Hypothesis 4: does Answer-hop-only performance survive because the
answer_hop paragraph itself re-mentions the bridge entity (i.e. the
paragraph is self-contained), rather than because of anything in the
question (hypothesis 3)? No new model calls needed - this just regroups the
predictions pipeline/evaluate_conditions.py already produced.

Restricted to samples the model gets right in the Full condition (same
denominator as pipeline/error_taxonomy.py's shortcut_success/bridge_needed split), so the
question is "given the model *could* answer this, does self-containment
predict whether it still can without the bridge paragraph?"

Cross-tabulated against hypothesis 3's signal (bridge title literally in the
question) to see which of the two loci - question or paragraph - predicts
the shortcut better; they are not mutually exclusive.
"""

import json

from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import ERRORS_DIR, PROCESSED_DIR


def rate(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def main() -> None:
    preds = {r["qid"]: r for r in load_jsonl(ERRORS_DIR / "test_predictions.jsonl")}
    conditions = {r["qid"]: r for r in load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")}

    rows = []
    for qid, pred in preds.items():
        if pred["full_em"] != 1:
            continue
        cond = conditions[qid]
        rows.append(
            {
                "qid": qid,
                "answer_only_em": pred["answer_only_em"],
                "bridge_in_answer_hop": pred["bridge_hop_title"] in cond["answer_only_context"],
                "bridge_in_question": pred["bridge_hop_title"] in pred["question"],
            }
        )

    n = len(rows)
    mentioned = [r for r in rows if r["bridge_in_answer_hop"]]
    not_mentioned = [r for r in rows if not r["bridge_in_answer_hop"]]

    cells = {}
    for bq in (True, False):
        for ba in (True, False):
            group = [
                r
                for r in rows
                if r["bridge_in_question"] == bq and r["bridge_in_answer_hop"] == ba
            ]
            cells[f"bridge_in_question={bq}, bridge_in_answer_hop={ba}"] = {
                "n": len(group),
                "answer_only_em_rate": rate(group, "answer_only_em"),
            }

    summary = {
        "n_full_correct": n,
        "answer_only_em_rate_bridge_mentioned_in_answer_hop": rate(mentioned, "answer_only_em"),
        "answer_only_em_rate_bridge_not_mentioned_in_answer_hop": rate(
            not_mentioned, "answer_only_em"
        ),
        "n_bridge_mentioned_in_answer_hop": len(mentioned),
        "n_bridge_not_mentioned_in_answer_hop": len(not_mentioned),
        "cross_tab_with_question_mention": cells,
    }

    with open(ERRORS_DIR / "self_containment_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["# 가설 4 — Answer-hop 문단 자기완결성 분석\n"]
    lines.append(f"Full 정답 {n}건 기준\n")
    lines.append("| 그룹 | n | Answer-hop only EM |")
    lines.append("|---|---|---|")
    lines.append(
        f"| bridge 제목이 answer_hop에 언급됨 | {len(mentioned)} | "
        f"{summary['answer_only_em_rate_bridge_mentioned_in_answer_hop']:.4f} |"
    )
    lines.append(
        f"| bridge 제목이 answer_hop에 언급 안 됨 | {len(not_mentioned)} | "
        f"{summary['answer_only_em_rate_bridge_not_mentioned_in_answer_hop']:.4f} |"
    )
    lines.append("")
    lines.append("## 가설 3(질문)과 2x2 교차\n")
    lines.append("| 질문에 bridge 언급 | answer_hop에 bridge 언급 | n | Answer-hop only EM |")
    lines.append("|---|---|---|---|")
    for bq in (True, False):
        for ba in (True, False):
            c = cells[f"bridge_in_question={bq}, bridge_in_answer_hop={ba}"]
            lines.append(f"| {bq} | {ba} | {c['n']} | {c['answer_only_em_rate']:.4f} |")
    lines.append("")
    lines.append(
        "두 마진 비율 차이가 크면 그쪽(질문 vs 문단)이 shortcut의 더 강한 예측 "
        "인자 - pipeline/question_masking_probe.py의 마스킹 결과와 함께 해석할 것."
    )

    with open(ERRORS_DIR / "self_containment_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
