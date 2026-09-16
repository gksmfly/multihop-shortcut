"""Shared binary sequence-classification training utilities, used by
scripts/mitigation/bridge_relatedness_classifier/train.py (B-v1, real-vs-random bridge
pairing) and scripts/mitigation/counterfactual_classifier/train.py (B-v3, P1!=P2
counterfactual labeling) - same architecture and hyperparameters, only the
label source differs between the two scripts.
"""

import numpy as np


def build_augmented_question(question: str, pred_text: str, confidence: float) -> str:
    """Prefixes Stage 1's Answer-hop-only prediction and confidence onto the
    question text, so a PairClassificationDataset built from rows with this
    as their "question" field lets Stage 2 condition on "how suspicious is
    this specific answer" instead of only (question, bridge_hop_text) -
    the eval/ follow-up experiment requested to address B-v3's limitations.
    """
    return f"{question} [ANSWER] {pred_text} [CONF] {confidence:.2f}"


class PairClassificationDataset:
    """Wraps a list of rows into a Trainer-compatible (question,
    bridge_hop_text) -> label dataset.
    """

    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        enc = tokenizer(
            [r["question"] for r in rows],
            [r["bridge_hop_text"] for r in rows],
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
        )
        self.encodings = enc
        self.labels = [r["label"] for r in rows]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}
