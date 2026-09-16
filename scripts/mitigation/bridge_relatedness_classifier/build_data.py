"""v2 two-stage design - Stage 2 data. Stage 2 is a binary "does this
bridge-hop paragraph genuinely support this question" classifier:
input = Question [SEP] bridge_hop_text, label = 1 for the TRUE bridge_hop
paired with that question, 0 for a random OTHER row's bridge_hop_text.

This avoids the substring-heuristic labeling problem (answer_hop_title in
bridge_hop_text is only 54.7% literal-match coverage, and a miss there
doesn't mean the pairing is wrong - the link may just be paraphrased) by
training on real-pair-vs-random-pair instead: every row contributes one
clean positive and one clean negative, no data loss, and it is exactly the
signal the later ablation test needs (swap in a random bridge-hop and check
whether the support score drops).
"""

import random

from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.paths import PROCESSED_DIR, SPLITS_DIR

SEED = 42


def build(split_rows: list[dict], pool_by_qid: dict[str, dict], rng: random.Random) -> list[dict]:
    qids = list(pool_by_qid.keys())
    out = []
    for row in split_rows:
        pool_row = pool_by_qid[row["qid"]]
        out.append(
            {
                "qid": row["qid"] + "_pos",
                "question": row["question"],
                "bridge_hop_text": pool_row["bridge_hop_text"],
                "label": 1,
            }
        )
        neg_qid = row["qid"]
        while neg_qid == row["qid"]:
            neg_qid = rng.choice(qids)
        out.append(
            {
                "qid": row["qid"] + "_neg",
                "question": row["question"],
                "bridge_hop_text": pool_by_qid[neg_qid]["bridge_hop_text"],
                "label": 0,
            }
        )
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(SEED)
    pool_rows = load_jsonl(PROCESSED_DIR / "train_pool.jsonl")
    pool_by_qid = {r["qid"]: r for r in pool_rows}

    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")

    train_out = build(train_rows, pool_by_qid, rng)
    val_out = build(val_rows, pool_by_qid, rng)

    save_jsonl(train_out, SPLITS_DIR / "train_support.jsonl")
    save_jsonl(val_out, SPLITS_DIR / "val_support.jsonl")

    print(f"train_support: {len(train_out)} rows")
    print(f"val_support: {len(val_out)} rows")


if __name__ == "__main__":
    main()
