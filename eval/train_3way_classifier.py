"""B-v4 follow-up experiment on B-v3's limitations (see docs/mitigation-experiment.md
and this eval/ dir's own findings): two changes over scripts/mitigation/counterfactual_classifier/ (B-v3).

1. 3-way label instead of binary P1!=P2: {0: no_change or changed_still_wrong
   (no benefit from switching), 1: bridge_helped (wrong->correct - the case
   we actually want to catch), 2: bridge_hurt (correct->wrong - overriding
   here is actively harmful)}. See metrics.BUCKET_TO_3WAY.
2. Richer Stage-2 input: instead of just (question, bridge_hop_text), the
   question is prefixed with Stage 1's own Answer-hop-only prediction text
   and confidence (classifier_training.build_augmented_question) - "how
   suspicious is *this specific answer*", not just "is this bridge related".

Requires eval/add_p1_confidence.py to have been run first (adds
p1_confidence to data/splits/{train,val}_counterfactual.jsonl).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

from multihop_shortcut.classifier_training import (
    PairClassificationDataset,
    build_augmented_question,
    compute_metrics,
)
from multihop_shortcut.constants import BASE_MODEL_NAME
from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.metrics import BUCKET_TO_3WAY
from multihop_shortcut.paths import MODELS_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "support_classifier_3way"

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def build_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "question": build_augmented_question(r["question"], r["p1_pred"], r["p1_confidence"]),
            "bridge_hop_text": r["bridge_hop_text"],
            "label": BUCKET_TO_3WAY[r["bucket"]],
        }
        for r in rows
    ]


def main() -> None:
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=3)

    train_rows = build_rows(load_jsonl(SPLITS_DIR / "train_counterfactual.jsonl"))
    val_rows = build_rows(load_jsonl(SPLITS_DIR / "val_counterfactual.jsonl"))

    train_dataset = PairClassificationDataset(train_rows, tokenizer, MAX_LENGTH)
    val_dataset = PairClassificationDataset(val_rows, tokenizer, MAX_LENGTH)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=200,
        report_to=[],
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    best_dir = MODEL_DIR / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best 3-way classifier to {best_dir}")


if __name__ == "__main__":
    main()
