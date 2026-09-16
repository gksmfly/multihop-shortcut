"""Fine-tune bert-base-cased as a SQuAD-style extractive QA model on the
Full-condition train split only (see README "실험 설계 원칙" - which hop is
shown is a variable manipulated only at evaluation time, never at training
time).
"""

import os

# Must be set before torch is imported - see README "환경 설정".
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json

import numpy as np
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from multihop_shortcut.constants import BASE_MODEL_NAME
from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import MODELS_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa"

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def prepare_features(examples: dict, tokenizer) -> dict:
    encodings = tokenizer(
        examples["question"],
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )

    start_positions = []
    end_positions = []
    for i, offsets in enumerate(encodings["offset_mapping"]):
        answer_start_char = examples["answer_start"][i]
        answer_end_char = answer_start_char + len(examples["answer"][i])
        sequence_ids = encodings.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if (
            offsets[context_start][0] > answer_start_char
            or offsets[context_end][1] < answer_end_char
        ):
            # Answer got truncated out of this window - point to CLS (token 0)
            start_positions.append(0)
            end_positions.append(0)
            continue

        tok_start = context_start
        while tok_start <= context_end and offsets[tok_start][0] <= answer_start_char:
            tok_start += 1
        tok_start -= 1

        tok_end = context_end
        while tok_end >= context_start and offsets[tok_end][1] >= answer_end_char:
            tok_end -= 1
        tok_end += 1

        start_positions.append(tok_start)
        end_positions.append(tok_end)

    encodings["start_positions"] = start_positions
    encodings["end_positions"] = end_positions
    encodings.pop("offset_mapping")
    return encodings


class JsonlQADataset:
    def __init__(self, rows: list[dict], tokenizer):
        examples = {
            "question": [r["question"] for r in rows],
            "context": [r["context"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "answer_start": [r["answer_start"] for r in rows],
        }
        self.encodings = prepare_features(examples, tokenizer)
        self.n = len(rows)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()}


def main() -> None:
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = BertForQuestionAnswering.from_pretrained(BASE_MODEL_NAME)

    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")

    train_dataset = JsonlQADataset(train_rows, tokenizer)
    val_dataset = JsonlQADataset(val_rows, tokenizer)

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
