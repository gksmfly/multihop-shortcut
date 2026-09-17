"""Fine-tune bert-base-cased on the mitigation-augmented data from
scripts/mitigation/adversarial_training/build_data.py (Full-condition rows + adversarial Answer-hop-only "unanswerable"
rows). Saved to a separate model dir so the original pipeline/train_bert.py checkpoint
used throughout pipeline/evaluate_conditions.py through fame_bias_analysis.py is never touched - this is a new experiment,
run alongside the original, not a replacement.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments

from multihop_shortcut.constants import BASE_MODEL_NAME, get_max_length
from multihop_shortcut.io_utils import load_jsonl
from multihop_shortcut.paths import MODELS_DIR, SPLITS_DIR
from multihop_shortcut.qa_training import JsonlQADataset

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa_mitigated"

MAX_LENGTH = get_max_length()


def main() -> None:
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = BertForQuestionAnswering.from_pretrained(BASE_MODEL_NAME)

    train_rows = load_jsonl(SPLITS_DIR / "train_mitigated.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val_mitigated.jsonl")

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
    print(f"Saved best mitigated model to {best_dir}")


if __name__ == "__main__":
    main()
