"""Mitigation experiment (targets H3 only - the one causal hypothesis that
was strongly supported; see README "가설 3"). H3 showed Answer-hop-only
performance survives even when the literal bridge-entity mention is masked
out of the question - the rest of the question's wording (a type
constraint) is already enough to find the answer in answer_hop_text alone.

Fix under test: adversarial data augmentation (Jiang & Bansal 2019-style).
For every train/val row, add a second "adversarial" row with the SAME
question but context = answer_hop_text ALONE (no bridge_hop_text), labeled
unanswerable (start/end = CLS, token 0) - even though the answer string is
literally present in that paragraph. The label is a deliberate training
signal, not a factual claim: it tells the model "a single paragraph without
the bridge hop should not be trusted", directly targeting the H3 shortcut.

Only the training signal changes; pipeline/evaluate_conditions.py's oracle test conditions are
never touched, so scripts/mitigation/adversarial_training/evaluate.py stays comparable to the
original pipeline/evaluate_conditions.py numbers.
"""

from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.paths import PROCESSED_DIR, SPLITS_DIR


def augment(split_rows: list[dict], pool_by_qid: dict[str, dict]) -> list[dict]:
    out = []
    for row in split_rows:
        # Original Full-condition row, unchanged.
        out.append({**row, "is_unanswerable": False})

        pool_row = pool_by_qid[row["qid"]]
        out.append(
            {
                "qid": row["qid"] + "_adv",
                "question": row["question"],
                "context": pool_row["answer_hop_text"],
                "answer": "",
                "answer_start": 0,
                "level": row["level"],
                "is_unanswerable": True,
            }
        )
    return out


def main() -> None:
    pool_rows = load_jsonl(PROCESSED_DIR / "train_pool.jsonl")
    pool_by_qid = {r["qid"]: r for r in pool_rows}

    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")

    train_out = augment(train_rows, pool_by_qid)
    val_out = augment(val_rows, pool_by_qid)

    save_jsonl(train_out, SPLITS_DIR / "train_mitigated.jsonl")
    save_jsonl(val_out, SPLITS_DIR / "val_mitigated.jsonl")

    print(f"train_mitigated: {len(train_out)} rows ({len(train_rows)} original + {len(train_rows)} adversarial)")
    print(f"val_mitigated: {len(val_out)} rows ({len(val_rows)} original + {len(val_rows)} adversarial)")


if __name__ == "__main__":
    main()
