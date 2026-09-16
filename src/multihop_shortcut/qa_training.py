"""Shared SQuAD-style extractive QA training utilities, used by
scripts/05_train_bert.py (original Full-condition training) and
scripts/mitigation/adversarial_training/train.py (adversarial-training mitigation, which
adds unanswerable/CLS-labeled rows on top of the same schema).
"""


def prepare_features(examples: dict, tokenizer, max_length: int) -> dict:
    encodings = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )

    is_unanswerable = examples.get("is_unanswerable")

    start_positions = []
    end_positions = []
    for i, offsets in enumerate(encodings["offset_mapping"]):
        if is_unanswerable is not None and is_unanswerable[i]:
            # Adversarial row: force CLS (token 0), no real answer span to
            # look up - see scripts/mitigation/adversarial_training/build_data.py.
            start_positions.append(0)
            end_positions.append(0)
            continue

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
    """Wraps a list of {"question", "context", "answer", "answer_start"}
    rows (optionally with "is_unanswerable": bool) into a Trainer-compatible
    dataset.
    """

    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        examples = {
            "question": [r["question"] for r in rows],
            "context": [r["context"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "answer_start": [r["answer_start"] for r in rows],
        }
        if rows and "is_unanswerable" in rows[0]:
            examples["is_unanswerable"] = [r["is_unanswerable"] for r in rows]
        self.encodings = prepare_features(examples, tokenizer, max_length)
        self.n = len(rows)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()}
