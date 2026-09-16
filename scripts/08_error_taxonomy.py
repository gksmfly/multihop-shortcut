"""Core result: taxonomize *why* Answer-hop only keeps up with Full (or
doesn't), and *what kind of* wrong guess the model makes in Bridge-hop only.
This is the project's main analytical contribution (see README "연구 질문") -
hypothesis 1/2 only establish that shortcut learning happens; this script
looks at when/why.

Two contrastive groups, restricted to samples the model gets right in the
Full condition (so any Answer-hop-only failure isn't just a Full-condition
failure carrying over):
- shortcut_success: Full correct AND Answer-hop-only also correct.
- bridge_needed:    Full correct BUT Answer-hop-only wrong.

For each group we report the rate of two candidate explanations:
- has_type_constraint_cue: the question contains a narrow-answer-type cue
  ("what year", "which country", "how many", ...) - candidate support for
  hypothesis 3 (the question alone narrows the answer).
- qa_word_overlap: unigram overlap between question content words and the
  answer_hop paragraph - a high-overlap question already "points at" the
  right sentence without needing the bridge paragraph.
- bridge_title_in_question / bridge_title_in_answer_hop: literal bridge
  entity mentions in the question vs. in the answer_hop paragraph itself -
  the two candidate loci from hypotheses 3 and 4.

Bridge-hop only wrong answers are split by whether the guessed span's coarse
type matches the gold answer's type (from scripts/07), to see whether wrong
guesses are "plausible-looking" (hypothesis 2) rather than arbitrary.
"""

import json
import re

from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import ERRORS_DIR, PROCESSED_DIR
from multihop_shortcut.typing_heuristics import classify_answer_type

TYPE_CUES = [
    r"\bwhat year\b", r"\bwhich year\b",
    r"\bwhat (country|nationality)\b", r"\bwhich (country|nationality)\b",
    r"\bhow many\b", r"\bhow much\b",
    r"\bwhat city\b", r"\bwhich city\b",
    r"\bwhat state\b", r"\bwhich state\b",
    r"\bwhen\b", r"\bwhat date\b",
    r"\bwhat number\b",
    r"\bwhat (position|title|role|occupation)\b",
    r"\bwhich (position|title|role|occupation)\b",
]
TYPE_CUE_RE = re.compile("|".join(TYPE_CUES), re.IGNORECASE)

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "is", "was", "were", "are",
    "to", "for", "and", "or", "by", "with", "that", "this", "which", "who",
    "whom", "what", "where", "when", "how", "did", "does", "do", "has",
    "have", "had", "as", "also", "it", "its", "from", "be", "been",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def qa_word_overlap(question: str, paragraph: str) -> float:
    q_words = content_words(question)
    if not q_words:
        return 0.0
    p_words = content_words(paragraph)
    return len(q_words & p_words) / len(q_words)


def main() -> None:
    preds = {r["qid"]: r for r in load_jsonl(ERRORS_DIR / "test_predictions.jsonl")}
    conditions = {r["qid"]: r for r in load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")}

    rows = []
    for qid, pred in preds.items():
        cond = conditions[qid]
        rows.append(
            {
                "qid": qid,
                "question": pred["question"],
                "answer": pred["answer"],
                "answer_hop_title": pred["answer_hop_title"],
                "bridge_hop_title": pred["bridge_hop_title"],
                "answer_hop_text": cond["answer_only_context"],
                "bridge_hop_text": cond["bridge_only_context"],
                "full_em": pred["full_em"],
                "answer_only_em": pred["answer_only_em"],
                "answer_only_pred": pred["answer_only_pred"],
                "bridge_only_em": pred["bridge_only_em"],
                "bridge_only_pred": pred["bridge_only_pred"],
                "bridge_only_confidence": pred["bridge_only_confidence"],
                "has_type_constraint_cue": bool(TYPE_CUE_RE.search(pred["question"])),
                "qa_word_overlap": qa_word_overlap(pred["question"], cond["answer_only_context"]),
                "bridge_title_in_question": pred["bridge_hop_title"] in pred["question"],
                "bridge_title_in_answer_hop": pred["bridge_hop_title"]
                in cond["answer_only_context"],
            }
        )

    full_correct = [r for r in rows if r["full_em"] == 1]
    shortcut_success = [r for r in full_correct if r["answer_only_em"] == 1]
    bridge_needed = [r for r in full_correct if r["answer_only_em"] == 0]

    def group_stats(group: list[dict]) -> dict:
        n = len(group)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "has_type_constraint_cue_rate": sum(r["has_type_constraint_cue"] for r in group) / n,
            "mean_qa_word_overlap": sum(r["qa_word_overlap"] for r in group) / n,
            "bridge_title_in_question_rate": sum(r["bridge_title_in_question"] for r in group)
            / n,
            "bridge_title_in_answer_hop_rate": sum(
                r["bridge_title_in_answer_hop"] for r in group
            )
            / n,
        }

    taxonomy = {
        "n_full_correct": len(full_correct),
        "shortcut_success": group_stats(shortcut_success),
        "bridge_needed": group_stats(bridge_needed),
    }

    # Bridge-hop only wrong-answer taxonomy: plausible (type-matching) vs not.
    wrong = [r for r in rows if r["bridge_only_em"] == 0]
    for r in wrong:
        r["bridge_only_type_match"] = classify_answer_type(r["answer"]) == classify_answer_type(
            r["bridge_only_pred"]
        )
    type_match_wrong = [r for r in wrong if r["bridge_only_type_match"]]
    type_mismatch_wrong = [r for r in wrong if not r["bridge_only_type_match"]]
    taxonomy["bridge_only_wrong"] = {
        "n": len(wrong),
        "type_match_rate": len(type_match_wrong) / len(wrong) if wrong else float("nan"),
        "mean_confidence_type_match": (
            sum(r["bridge_only_confidence"] for r in type_match_wrong) / len(type_match_wrong)
            if type_match_wrong
            else float("nan")
        ),
        "mean_confidence_type_mismatch": (
            sum(r["bridge_only_confidence"] for r in type_mismatch_wrong)
            / len(type_mismatch_wrong)
            if type_mismatch_wrong
            else float("nan")
        ),
    }

    with open(ERRORS_DIR / "error_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    def case_study(r: dict) -> str:
        return (
            f"- qid={r['qid']}\n"
            f"  Q: {r['question']}\n"
            f"  answer: {r['answer']}\n"
            f"  answer_hop({r['answer_hop_title']}): {r['answer_hop_text'][:200]}...\n"
            f"  bridge_hop({r['bridge_hop_title']}): {r['bridge_hop_text'][:200]}...\n"
            f"  answer_only_pred: {r['answer_only_pred']!r} | "
            f"bridge_only_pred: {r['bridge_only_pred']!r} "
            f"(conf={r['bridge_only_confidence']:.3f})\n"
            f"  type_constraint_cue={r['has_type_constraint_cue']} "
            f"qa_word_overlap={r['qa_word_overlap']:.2f} "
            f"bridge_in_question={r['bridge_title_in_question']} "
            f"bridge_in_answer_hop={r['bridge_title_in_answer_hop']}"
        )

    lines = ["# 에러 taxonomy(가설 1 핵심 결과)\n"]
    lines.append(f"Full 정답 {len(full_correct)}건 중 shortcut_success "
                 f"{len(shortcut_success)}건 / bridge_needed {len(bridge_needed)}건\n")
    lines.append("## 그룹별 특징 비율\n")
    lines.append("| 그룹 | n | type-constraint cue | qa_word_overlap 평균 | "
                 "bridge 제목이 질문에 | bridge 제목이 answer_hop에 |")
    lines.append("|---|---|---|---|---|---|")
    for name, stats in (("shortcut_success", taxonomy["shortcut_success"]),
                        ("bridge_needed", taxonomy["bridge_needed"])):
        if stats["n"] == 0:
            lines.append(f"| {name} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {stats['n']} | {stats['has_type_constraint_cue_rate']:.2%} | "
            f"{stats['mean_qa_word_overlap']:.3f} | "
            f"{stats['bridge_title_in_question_rate']:.2%} | "
            f"{stats['bridge_title_in_answer_hop_rate']:.2%} |"
        )
    lines.append("")
    lines.append("## Bridge-hop only 오답 taxonomy\n")
    bo = taxonomy["bridge_only_wrong"]
    lines.append(f"오답 {bo['n']}건 중 타입 일치 {bo['type_match_rate']:.2%} "
                 f"(일치 시 평균 confidence {bo['mean_confidence_type_match']:.3f}, "
                 f"불일치 시 {bo['mean_confidence_type_mismatch']:.3f})\n")

    lines.append("## 케이스 스터디 — shortcut_success (질문/문단만으로 충분)\n")
    for r in shortcut_success[:4]:
        lines.append(case_study(r))
        lines.append("")

    lines.append("## 케이스 스터디 — bridge_needed (bridge 없이는 실패)\n")
    for r in bridge_needed[:4]:
        lines.append(case_study(r))
        lines.append("")

    lines.append("## 케이스 스터디 — bridge_only 오답 (타입 일치, 그럴듯한 오답)\n")
    for r in sorted(type_match_wrong, key=lambda r: -r["bridge_only_confidence"])[:4]:
        lines.append(case_study(r))
        lines.append("")

    with open(ERRORS_DIR / "error_taxonomy_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines[:40]))
    print("\n... (전체는 data/errors/error_taxonomy_report.md 참고)")


if __name__ == "__main__":
    main()
