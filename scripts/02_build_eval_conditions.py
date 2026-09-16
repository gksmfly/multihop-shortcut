"""Build the Full/Answer-hop-only/Bridge-hop-only contexts from the tagged
rows produced by scripts/01_load_hotpotqa.py.

Full-condition context concatenates the two gold paragraphs in their
*natural document order* (as they appeared in the original HotpotQA context
list), not a fixed "answer-first" or "bridge-first" order - fixing the order
by hop role would let the model learn a positional shortcut ("the answer is
always in the second paragraph") that has nothing to do with the phenomenon
under study.

train_pool.jsonl only needs the Full condition (that's the only condition
ever used for training - see README "실험 설계 원칙"), so its output is a
flat SQuAD-like schema ready for scripts/03_split_dataset.py and
scripts/05_train_bert.py.

test.jsonl needs all three conditions kept side by side (same qid), so
scripts/06_evaluate_conditions.py can run the same trained model on all
three and compare per-sample.
"""

from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.paths import PROCESSED_DIR

SEP = " "


def build_full_context(row: dict) -> tuple[str, int]:
    if row["answer_hop_before_bridge_hop"]:
        context = row["answer_hop_text"] + SEP + row["bridge_hop_text"]
        answer_start = row["answer_char_start"]
    else:
        context = row["bridge_hop_text"] + SEP + row["answer_hop_text"]
        answer_start = len(row["bridge_hop_text"]) + len(SEP) + row["answer_char_start"]

    assert context[answer_start : answer_start + len(row["answer"])] == row["answer"]
    return context, answer_start


def build_train_pool() -> None:
    rows = load_jsonl(PROCESSED_DIR / "train_pool.jsonl")
    out = []
    for row in rows:
        full_context, full_answer_start = build_full_context(row)
        out.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "context": full_context,
                "answer": row["answer"],
                "answer_start": full_answer_start,
                "level": row["level"],
            }
        )
    save_jsonl(out, PROCESSED_DIR / "train_full.jsonl")
    print(f"train_full.jsonl: {len(out)} rows")


def build_test() -> None:
    rows = load_jsonl(PROCESSED_DIR / "test.jsonl")
    out = []
    for row in rows:
        full_context, full_answer_start = build_full_context(row)
        assert (
            row["answer_hop_text"][
                row["answer_char_start"] : row["answer_char_start"] + len(row["answer"])
            ]
            == row["answer"]
        )
        out.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "answer": row["answer"],
                "level": row["level"],
                "answer_hop_title": row["answer_hop_title"],
                "bridge_hop_title": row["bridge_hop_title"],
                "full_context": full_context,
                "full_answer_start": full_answer_start,
                "answer_only_context": row["answer_hop_text"],
                "answer_only_answer_start": row["answer_char_start"],
                "bridge_only_context": row["bridge_hop_text"],
            }
        )
    save_jsonl(out, PROCESSED_DIR / "test_conditions.jsonl")
    print(f"test_conditions.jsonl: {len(out)} rows")


def main() -> None:
    build_train_pool()
    build_test()


if __name__ == "__main__":
    main()
