"""Fine-tune bert-base-cased as a SQuAD-style extractive QA model on the
Full-condition train split only (see README "실험 설계 원칙" - which hop is
shown is a variable manipulated only at evaluation time, never at training
time).
"""

import os

# Must be set before torch is imported - see README "환경 설정".
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json

from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments

from multihop_shortcut.constants import BASE_MODEL_NAME
from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import MODELS_DIR, SPLITS_DIR
from multihop_shortcut.qa_training import JsonlQADataset

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa"

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def main() -> None:
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = BertForQuestionAnswering.from_pretrained(BASE_MODEL_NAME)

    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")

    train_dataset = JsonlQADataset(train_rows, tokenizer, MAX_LENGTH)
    val_dataset = JsonlQADataset(val_rows, tokenizer, MAX_LENGTH)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    )
    trainer.train()

    best_dir = MODEL_DIR / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best model to {best_dir}")


if __name__ == "__main__":
    main()
