"""v2/B-v1 two-stage design - train Stage 2: a binary BERT sequence
classifier that scores whether a bridge-hop paragraph genuinely supports a
question (see scripts/mitigation/bridge_relatedness_classifier/build_data.py). Runs on GPU 0 by default here since scripts/mitigation/adversarial_training/train.py
(the adversarial-training experiment) already claims GPU 1 in this session -
override with CUDA_VISIBLE_DEVICES if that changes.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

from multihop_shortcut.classifier_training import PairClassificationDataset, compute_metrics
from multihop_shortcut.constants import BASE_MODEL_NAME, get_max_length
from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import MODELS_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "support_classifier"

MAX_LENGTH = get_max_length()


def main() -> None:
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)

    train_rows = load_jsonl(SPLITS_DIR / "train_support.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val_support.jsonl")

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
    print(f"Saved best support classifier to {best_dir}")


if __name__ == "__main__":
    main()
