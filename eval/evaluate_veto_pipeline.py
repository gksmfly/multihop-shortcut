"""Closes the loop the user pointed out was missing: does the B-v4 veto
classifier actually move the FINAL EM/F1 when wired into a real decision
rule, not just precision/recall/harm_rate in isolation?

Decision rule per threshold t: if the B-v4 classifier's p(bridge_helped)
>= t, use the original model's Full-context prediction as the final answer;
otherwise keep its Answer-hop-only prediction (the shortcut-prone default).
No retraining needed - both predictions and the veto score are already
saved (test_predictions.jsonl from pipeline/evaluate_conditions.py,
3way_classifier_predictions.jsonl from eval/evaluate_3way_classifier.py).
"""

import json

from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.metrics import exact_match, f1_score
from multihop_shortcut.paths import ERRORS_DIR

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    preds = {r["qid"]: r for r in load_jsonl(ERRORS_DIR / "test_predictions.jsonl")}
    veto = {r["qid"]: r for r in load_jsonl(ERRORS_DIR / "3way_classifier_predictions.jsonl")}

    n = len(preds)
    baseline_answer_only_em = sum(r["answer_only_em"] for r in preds.values()) / n
    baseline_answer_only_f1 = sum(r["answer_only_f1"] for r in preds.values()) / n
    baseline_full_em = sum(r["full_em"] for r in preds.values()) / n
    baseline_full_f1 = sum(r["full_f1"] for r in preds.values()) / n

    rows = []
    for t in THRESHOLDS:
        n_switched = 0
        em_sum = 0.0
        f1_sum = 0.0
        for qid, p in preds.items():
            v = veto[qid]
            if v["p1_bridge_helped"] >= t:
                n_switched += 1
                final_pred = p["full_pred"]
            else:
                final_pred = p["answer_only_pred"]
            em_sum += exact_match(final_pred, p["answer"])
            f1_sum += f1_score(final_pred, p["answer"])
        rows.append(
            {
                "threshold": t,
                "n_switched": n_switched,
                "em": em_sum / n,
                "f1": f1_sum / n,
            }
        )

    lines = [
        "# B-v4 veto를 실제로 적용했을 때 최종 EM/F1\n",
        "결정 규칙: p(bridge_helped) >= threshold 이면 Full 조건 예측으로 교체, "
        "아니면 Answer-hop only 예측(원본 shortcut-prone 기본값) 유지.\n",
        "## Baseline (교체 없음)",
        "| | EM | F1 |",
        "|---|---|---|",
        f"| Answer-hop only (전부 유지) | {baseline_answer_only_em:.4f} | {baseline_answer_only_f1:.4f} |",
        f"| Full (전부 교체했다면) | {baseline_full_em:.4f} | {baseline_full_f1:.4f} |",
        "",
        "## Veto 적용 결과",
        "| threshold | n_switched | EM | F1 | EM 변화(vs Answer-hop only) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        delta = row["em"] - baseline_answer_only_em
        lines.append(
            f"| {row['threshold']:.1f} | {row['n_switched']} | {row['em']:.4f} "
            f"| {row['f1']:.4f} | {delta:+.4f} |"
        )
    lines += [
        "",
        (
            "이 표가 실제 답. EM 변화가 유의미하게 양수(+)인 threshold가 있으면 "
            "veto가 최종 성능을 실제로 올린 것 - '분류기 성능은 좋은데 파이프라인 "
            "성능은 안 오른' 경우라면 여기서 0 근방이거나 음수로 나온다."
        ),
    ]

    summary = {
        "n": n,
        "baseline_answer_only_em": baseline_answer_only_em,
        "baseline_answer_only_f1": baseline_answer_only_f1,
        "baseline_full_em": baseline_full_em,
        "baseline_full_f1": baseline_full_f1,
        "veto_applied": rows,
    }
    with open(ERRORS_DIR / "veto_pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ERRORS_DIR / "veto_pipeline_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
