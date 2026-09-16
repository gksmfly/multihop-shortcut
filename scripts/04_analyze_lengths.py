"""Tokenize (question, context) pairs from the train split to pick
max_length for scripts/05_train_bert.py. Full-condition context is the
longest of the three eval conditions (it's the only one used for training),
so its length distribution is what matters here.
"""

import json

from transformers import BertTokenizerFast

from multihop_shortcut.constants import BASE_MODEL_NAME
from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import SPLITS_DIR

CANDIDATE_MAX_LENGTHS = [256, 320, 384, 448, 512]


def percentile(sorted_vals: list[int], p: float) -> int:
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def main() -> None:
    rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)

    lengths = []
    for row in rows:
        encoded = tokenizer(row["question"], row["context"])
        lengths.append(len(encoded["input_ids"]))

    lengths.sort()
    n = len(lengths)
    stats = {
        "n": n,
        "min": lengths[0],
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": lengths[-1],
    }
    for max_len in CANDIDATE_MAX_LENGTHS:
        truncated_frac = sum(1 for l in lengths if l > max_len) / n
        stats[f"truncated_frac_at_{max_len}"] = round(truncated_frac, 4)

    recommended = next(
        (m for m in CANDIDATE_MAX_LENGTHS if stats[f"truncated_frac_at_{m}"] <= 0.01),
        CANDIDATE_MAX_LENGTHS[-1],
    )
    stats["recommended_max_length"] = recommended

    with open(SPLITS_DIR / "max_length_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
