import re

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}
_NUMBER_RE = re.compile(r"^-?\d[\d,.]*$")
_YEAR_RE = re.compile(r"\b\d{3,4}\b")


def classify_answer_type(text: str) -> str:
    """Coarse, rule-based answer typing (no NER model dependency) - used as
    a plausibility proxy: does a wrong answer at least "look like" the
    right kind of thing (see README hypothesis 2 / scripts/07)."""
    text = text.strip()
    if not text:
        return "EMPTY"
    if _NUMBER_RE.match(text):
        return "NUMBER"
    tokens = [tok.strip(",.") for tok in text.split()]
    if any(tok.lower() in _MONTHS for tok in tokens) or _YEAR_RE.search(text):
        return "DATE"
    if text[0].isupper():
        return "PROPER_NOUN"
    return "OTHER"
