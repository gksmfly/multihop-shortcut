import torch
import torch.nn.functional as F

CLS_INDEX = 0


def run_multiclass_probs(
    model,
    tokenizer,
    device,
    questions: list[str],
    bridges: list[str],
    max_length: int = 448,
    batch_size: int = 32,
) -> list[list[float]]:
    """Same batching as run_classifier_inference, but returns the full
    softmax distribution per example instead of collapsing to P(label=1) -
    for classifiers with more than two classes (eval/train_3way_classifier.py).
    """
    model.eval()
    all_probs = []
    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        batch_b = bridges[start : start + batch_size]
        enc = tokenizer(
            batch_q,
            batch_b,
            max_length=max_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        all_probs.extend(probs.cpu().tolist())
    return all_probs


def run_classifier_inference(
    model,
    tokenizer,
    device,
    questions: list[str],
    bridges: list[str],
    max_length: int = 448,
    batch_size: int = 32,
) -> list[float]:
    """Batched binary sequence-classification inference for the Stage-2
    support classifiers (scripts/mitigation/bridge_relatedness_classifier/ and scripts/mitigation/counterfactual_classifier/). Returns, per (question,
    bridge_hop_text) pair, the softmax probability of label=1.
    """
    model.eval()
    scores = []
    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        batch_b = bridges[start : start + batch_size]
        enc = tokenizer(
            batch_q,
            batch_b,
            max_length=max_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[:, 1]
        scores.extend(probs.cpu().tolist())
    return scores


def run_qa_inference(
    model,
    tokenizer,
    examples: list[dict],
    device,
    max_length: int = 448,
    batch_size: int = 32,
    max_answer_length: int = 30,
) -> list[dict]:
    """Batched SQuAD-style extractive QA inference.

    `examples` is a list of {"question": str, "context": str}. Returns, per
    example, the predicted answer text/char span plus two probabilities read
    off the *unmasked* softmax over the whole sequence (question + context +
    specials): `confidence` (probability mass on the predicted span) and
    `cls_prob` (probability mass on the [CLS] position, a proxy for the model
    routing probability toward "no good span in this context" - see
    scripts/07_confidence_bias_analysis.py).
    """
    model.eval()
    results = []
    questions = [e["question"] for e in examples]
    contexts = [e["context"] for e in examples]

    for start in range(0, len(examples), batch_size):
        batch_q = questions[start : start + batch_size]
        batch_c = contexts[start : start + batch_size]
        enc = tokenizer(
            batch_q,
            batch_c,
            max_length=max_length,
            truncation="only_second",
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        start_logits = outputs.start_logits.cpu()
        end_logits = outputs.end_logits.cpu()
        seq_len = start_logits.size(1)

        idx = torch.arange(seq_len)
        span_len = idx.unsqueeze(0) - idx.unsqueeze(1) + 1  # [start, end]
        length_ok = (span_len >= 1) & (span_len <= max_answer_length)

        for i in range(len(batch_q)):
            seq_ids = enc.sequence_ids(i)
            offsets = offset_mapping[i].tolist()
            context_mask = torch.tensor([sid == 1 for sid in seq_ids])

            s_logits = start_logits[i]
            e_logits = end_logits[i]
            s_probs = F.softmax(s_logits, dim=-1)
            e_probs = F.softmax(e_logits, dim=-1)

            valid = context_mask.unsqueeze(1) & context_mask.unsqueeze(0) & length_ok
            score_matrix = s_logits.unsqueeze(1) + e_logits.unsqueeze(0)
            score_matrix = score_matrix.masked_fill(~valid, float("-inf"))

            flat_idx = int(torch.argmax(score_matrix))
            s_idx, e_idx = divmod(flat_idx, seq_len)

            char_start = offsets[s_idx][0]
            char_end = offsets[e_idx][1]
            pred_text = batch_c[i][char_start:char_end]

            results.append(
                {
                    "pred_text": pred_text,
                    "confidence": (s_probs[s_idx] * e_probs[e_idx]).item(),
                    "cls_prob": (s_probs[CLS_INDEX] * e_probs[CLS_INDEX]).item(),
                    "start_char": char_start,
                    "end_char": char_end,
                }
            )

    return results
