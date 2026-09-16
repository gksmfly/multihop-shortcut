"""Hypothesis 5: are Bridge-hop-only "plausible guesses" actually BERT's
memorized pretraining knowledge about well-known entities, rather than
random guessing? Proxy for "fame": how many times a given entity's title
appears as a context paragraph across the *entire* HotpotQA corpus (train +
validation, ~98k questions x up to 10 paragraphs each) - an entity that
shows up as supporting/distractor material for many unrelated questions is,
by construction, one Wikipedia tends to reference often (a rough notability
signal), independent of this project's own filtered subset.

The single most direct piece of evidence for hypothesis 5 is samples where
`bridge_only_em == 1`: since bridge_hop_text was filtered in pipeline/load_hotpotqa.py to
*never* literally contain the answer string, an exact match there cannot
come from copying the context - it can only come from the model already
"knowing" the fact.
"""

import json
from collections import Counter

from datasets import load_dataset

from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import ERRORS_DIR
from multihop_shortcut.typing_heuristics import classify_answer_type


def build_title_frequency() -> Counter:
    counter = Counter()
    for split in ("train", "validation"):
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        for ex in ds:
            counter.update(ex["context"]["title"])
    return counter


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def main() -> None:
    title_freq = build_title_frequency()

    preds = load_jsonl(ERRORS_DIR / "test_predictions.jsonl")

    rows = []
    for r in preds:
        fame = title_freq.get(r["bridge_hop_title"], 0)
        type_match = classify_answer_type(r["answer"]) == classify_answer_type(
            r["bridge_only_pred"]
        )
        rows.append(
            {
                "qid": r["qid"],
                "question": r["question"],
                "answer": r["answer"],
                "bridge_hop_title": r["bridge_hop_title"],
                "bridge_only_pred": r["bridge_only_pred"],
                "bridge_only_em": r["bridge_only_em"],
                "bridge_only_confidence": r["bridge_only_confidence"],
                "fame": fame,
                "type_match": type_match,
            }
        )

    exact_correct = [r for r in rows if r["bridge_only_em"] == 1]
    wrong = [r for r in rows if r["bridge_only_em"] == 0]
    type_match_wrong = [r for r in wrong if r["type_match"]]
    type_mismatch_wrong = [r for r in wrong if not r["type_match"]]
    rest = [r for r in rows if r["bridge_only_em"] != 1]

    summary = {
        "n_total": len(rows),
        "n_bridge_only_exact_correct_despite_no_answer_in_context": len(exact_correct),
        "mean_fame_exact_correct": mean([r["fame"] for r in exact_correct]),
        "median_fame_exact_correct": median([r["fame"] for r in exact_correct]),
        "mean_fame_rest": mean([r["fame"] for r in rest]),
        "median_fame_rest": median([r["fame"] for r in rest]),
        "mean_fame_type_match_wrong": mean([r["fame"] for r in type_match_wrong]),
        "mean_fame_type_mismatch_wrong": mean([r["fame"] for r in type_mismatch_wrong]),
    }

    with open(ERRORS_DIR / "fame_bias_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["# 가설 5 — 유명도(사전학습 지식) 편향 분석\n"]
    lines.append(
        f"Bridge-hop only인데도 EM=1(암기 없이는 불가능 - bridge_hop_text에는 "
        f"정답 문자열이 없도록 이미 필터링됨): "
        f"{summary['n_bridge_only_exact_correct_despite_no_answer_in_context']}"
        f"/{summary['n_total']}건\n"
    )
    lines.append("| 그룹 | n | 평균 fame | 중앙값 fame |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| EM=1(사실상 암기) | {len(exact_correct)} | "
        f"{summary['mean_fame_exact_correct']:.2f} | {summary['median_fame_exact_correct']:.1f} |"
    )
    lines.append(
        f"| 나머지 | {len(rest)} | {summary['mean_fame_rest']:.2f} | "
        f"{summary['median_fame_rest']:.1f} |"
    )
    lines.append(
        f"| 오답·타입 일치 | {len(type_match_wrong)} | "
        f"{summary['mean_fame_type_match_wrong']:.2f} | - |"
    )
    lines.append(
        f"| 오답·타입 불일치 | {len(type_mismatch_wrong)} | "
        f"{summary['mean_fame_type_mismatch_wrong']:.2f} | - |"
    )
    lines.append("")
    lines.append(
        "EM=1 그룹의 fame이 나머지보다 뚜렷이 높다면 → 문맥 없이 정답을 맞힌 "
        "사례가 무작위가 아니라 유명 엔티티에 편중된다는 뜻(가설 5 지지)."
    )
    lines.append("")
    lines.append("## 케이스 스터디 — bridge_only EM=1 (암기로 추정되는 사례)\n")
    for r in sorted(exact_correct, key=lambda r: -r["fame"])[:10]:
        lines.append(
            f"- qid={r['qid']} bridge_entity={r['bridge_hop_title']!r} "
            f"(fame={r['fame']}) Q: {r['question']} answer={r['answer']!r} "
            f"conf={r['bridge_only_confidence']:.3f}"
        )

    with open(ERRORS_DIR / "fame_bias_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
