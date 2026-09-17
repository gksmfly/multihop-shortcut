import json

from multihop_shortcut.paths import SPLITS_DIR

BASE_MODEL_NAME = "bert-base-cased"

ANSWER_HOP = "answer_hop"
BRIDGE_HOP = "bridge_hop"


def get_max_length() -> int:
    """Reads the tokenized-length recommendation written by
    pipeline/analyze_lengths.py - every training/inference entry point uses
    this instead of repeating the same file read.
    """
    with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
        return json.load(f)["recommended_max_length"]
