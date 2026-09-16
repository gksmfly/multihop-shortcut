import re
import string


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


BUCKET_TO_3WAY = {
    "no_change": 0,
    "changed_still_wrong": 0,  # neither answer is right - no benefit from switching
    "bridge_helped": 1,  # wrong -> correct: the case a veto system should catch
    "bridge_hurt": 2,  # correct -> wrong: overriding here is actively harmful
}


def counterfactual_bucket(p1_pred: str, p2_pred: str, p1_correct: bool, p2_correct: bool) -> str:
    """Classifies a (P1=Answer-hop-only pred, P2=Full pred) pair into the
    four buckets used by the B-v3 counterfactual-labeling mitigation
    experiment (scripts/mitigation/counterfactual_classifier/01,03; see docs/mitigation-experiment.md):
    does adding the bridge hop change the prediction, and if so, does it
    change it to the *correct* answer, the *wrong* answer, or another wrong
    answer.
    """
    changed = normalize_answer(p1_pred) != normalize_answer(p2_pred)
    if not changed:
        return "no_change"
    if not p1_correct and p2_correct:
        return "bridge_helped"
    if p1_correct and not p2_correct:
        return "bridge_hurt"
    return "changed_still_wrong"
