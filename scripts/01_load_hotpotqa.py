"""Load HotpotQA (distractor), keep pure 2-hop bridge-type samples, and tag
each sample's two supporting paragraphs as answer_hop (contains the answer
string) vs bridge_hop (the other one).

Oracle setting: no distractor paragraphs are kept anywhere in this pipeline
(see README) - only the two gold supporting paragraphs are stored here.

Filters applied, in order:
1. type == "bridge" (comparison questions have a different hop structure,
   and their answers are typically "yes"/"no", not extractable spans).
2. answer not in {"yes", "no"} (defensive - see 1).
3. supporting_facts span exactly 2 distinct paragraph titles (pure 2-hop;
   HotpotQA sometimes has 3+ supporting titles even for "bridge" type).
4. the answer string is found (case-sensitive substring) in exactly one of
   the two gold paragraphs, not zero and not both. Zero means the answer
   isn't literally extractable from either paragraph (common for HotpotQA -
   the answer is synthesized, not copied). Both means the answer string
   coincidentally also occurs in the bridge paragraph (e.g. a common word
   or number), which would break the "bridge_hop has no answer" premise
   that hypothesis 2 and the bridge-hop-only condition depend on.

HF "train" split -> data/processed/train_pool.jsonl (further split into our
own train/val in scripts/03_split_dataset.py).
HF "validation" split -> data/processed/test.jsonl (kept aside untouched as
the final held-out test set used for all three eval conditions).
"""

import json

from datasets import load_dataset

from multihop_shortcut.io_utils import save_jsonl
from multihop_shortcut.paths import PROCESSED_DIR

HOTPOT_YES_NO = {"yes", "no"}


def paragraph_text(context: dict, title: str) -> str:
    idx = context["title"].index(title)
    return "".join(context["sentences"][idx])


def process_split(split_name: str) -> tuple[list[dict], dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)

    stats = {
        "total": len(ds),
        "not_bridge_type": 0,
        "yes_no_answer": 0,
        "not_pure_2hop": 0,
        "answer_in_zero_paragraphs": 0,
        "answer_in_both_paragraphs": 0,
        "kept": 0,
    }

    rows = []
    for ex in ds:
        if ex["type"] != "bridge":
            stats["not_bridge_type"] += 1
            continue
        if ex["answer"].strip().lower() in HOTPOT_YES_NO:
            stats["yes_no_answer"] += 1
            continue

        sf_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        if len(sf_titles) != 2:
            stats["not_pure_2hop"] += 1
            continue

        title_a, title_b = sf_titles
        text_a = paragraph_text(ex["context"], title_a)
        text_b = paragraph_text(ex["context"], title_b)
        idx_a = ex["context"]["title"].index(title_a)
        idx_b = ex["context"]["title"].index(title_b)
        answer = ex["answer"]

        pos_a = text_a.find(answer)
        pos_b = text_b.find(answer)
        has_a, has_b = pos_a != -1, pos_b != -1

        if not has_a and not has_b:
            stats["answer_in_zero_paragraphs"] += 1
            continue
        if has_a and has_b:
            stats["answer_in_both_paragraphs"] += 1
            continue

        if has_a:
            answer_hop_title, answer_hop_text, answer_char_start, answer_idx = (
                title_a,
                text_a,
                pos_a,
                idx_a,
            )
            bridge_hop_title, bridge_hop_text, bridge_idx = title_b, text_b, idx_b
        else:
            answer_hop_title, answer_hop_text, answer_char_start, answer_idx = (
                title_b,
                text_b,
                pos_b,
                idx_b,
            )
            bridge_hop_title, bridge_hop_text, bridge_idx = title_a, text_a, idx_a

        rows.append(
            {
                "qid": ex["id"],
                "question": ex["question"],
                "answer": answer,
                "level": ex["level"],
                "answer_hop_title": answer_hop_title,
                "answer_hop_text": answer_hop_text,
                "answer_char_start": answer_char_start,
                "bridge_hop_title": bridge_hop_title,
                "bridge_hop_text": bridge_hop_text,
                # natural document order (as the two paragraphs appeared in
                # the original context list) - used by 02_build_eval_conditions.py
                # to order the Full condition without leaking hop role via
                # a fixed answer-first/bridge-first convention.
                "answer_hop_before_bridge_hop": answer_idx < bridge_idx,
            }
        )
        stats["kept"] += 1

    return rows, stats


def main() -> None:
    all_stats = {}

    train_rows, train_stats = process_split("train")
    save_jsonl(train_rows, PROCESSED_DIR / "train_pool.jsonl")
    all_stats["train_pool"] = train_stats

    test_rows, test_stats = process_split("validation")
    save_jsonl(test_rows, PROCESSED_DIR / "test.jsonl")
    all_stats["test"] = test_stats

    with open(PROCESSED_DIR / "load_filter_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    for name, stats in all_stats.items():
        print(f"\n[{name}]")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
