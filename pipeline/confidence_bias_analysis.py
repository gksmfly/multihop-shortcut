"""Hypothesis 2: Bridge-hop only has no correct answer in context by
construction, so EM/F1 there is uninformative (see README). What matters is
whether the model (a) stays as confident as when it actually has evidence,
and (b) still guesses an answer of the *right coarse type* even though it's
wrong - both would indicate a structural bias toward confidently-wrong
guesses rather than the model "knowing" it lacks evidence.
"""

import json

from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import ERRORS_DIR
from multihop_shortcut.typing_heuristics import classify_answer_type


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    rows = load_jsonl(ERRORS_DIR / "test_predictions.jsonl")
    n = len(rows)

    confidence = {
        c: [r[f"{c}_confidence"] for r in rows] for c in ("full", "answer_only", "bridge_only")
    }
    cls_prob = {
        c: [r[f"{c}_cls_prob"] for r in rows] for c in ("full", "answer_only", "bridge_only")
    }

    confidence_drop_vs_full = [
        r["full_confidence"] - r["bridge_only_confidence"] for r in rows
    ]
    confidence_drop_vs_answer_only = [
        r["answer_only_confidence"] - r["bridge_only_confidence"] for r in rows
    ]

    type_match = 0
    type_match_wrong_only = 0
    n_wrong = 0
    for r in rows:
        gold_type = classify_answer_type(r["answer"])
        pred_type = classify_answer_type(r["bridge_only_pred"])
        if gold_type == pred_type:
            type_match += 1
        if not r["bridge_only_em"]:
            n_wrong += 1
            if gold_type == pred_type:
                type_match_wrong_only += 1

    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    bin_labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins) - 1)]
    histogram = {c: [0] * (len(bins) - 1) for c in ("full", "bridge_only")}
    for c in histogram:
        for v in confidence[c]:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    histogram[c][i] += 1
                    break

    summary = {
        "n": n,
        "mean_confidence": {c: mean(confidence[c]) for c in confidence},
        "mean_cls_prob": {c: mean(cls_prob[c]) for c in cls_prob},
        "mean_confidence_drop_full_minus_bridge_only": mean(confidence_drop_vs_full),
        "mean_confidence_drop_answer_only_minus_bridge_only": mean(
            confidence_drop_vs_answer_only
        ),
        "frac_confidence_drop_le_0.1_vs_full": sum(
            1 for d in confidence_drop_vs_full if d <= 0.1
        )
        / n,
        "bridge_only_type_match_rate_all": type_match / n,
        "bridge_only_type_match_rate_wrong_only": (
            type_match_wrong_only / n_wrong if n_wrong else float("nan")
        ),
        "n_bridge_only_wrong": n_wrong,
        "confidence_histogram_bins": bin_labels,
        "confidence_histogram": histogram,
    }

    with open(ERRORS_DIR / "confidence_bias_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["# 가설 2 — Confidence / Bias 분석\n"]
    lines.append(f"n = {n}\n")
    lines.append("## 조건별 평균 confidence / cls_prob\n")
    lines.append("| 조건 | 평균 confidence | 평균 cls_prob |")
    lines.append("|---|---|---|")
    for c in ("full", "answer_only", "bridge_only"):
        lines.append(f"| {c} | {mean(confidence[c]):.4f} | {mean(cls_prob[c]):.4f} |")
    lines.append("")
    lines.append(
        f"Full 대비 Bridge-hop only confidence 평균 하락폭: "
        f"{summary['mean_confidence_drop_full_minus_bridge_only']:.4f}"
    )
    lines.append(
        f"(하락폭이 0.1 이하인 샘플 비율: "
        f"{summary['frac_confidence_drop_le_0.1_vs_full']:.2%} — 이 비율이 높으면 "
        "'증거가 없어도 확신도가 안 떨어진다'는 뜻)\n"
    )
    lines.append(
        f"Bridge-hop only 오답 중 엔티티 타입이 정답과 일치하는 비율: "
        f"{summary['bridge_only_type_match_rate_wrong_only']:.2%} "
        f"({n_wrong}건 중) — 높으면 무작위 추측이 아니라 '그럴듯한' 오답을 "
        "고르는 편향이 있다는 뜻"
    )
    with open(ERRORS_DIR / "confidence_bias_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
