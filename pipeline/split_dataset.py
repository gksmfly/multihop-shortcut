"""Split train_full.jsonl (Full-condition, from pipeline/build_eval_conditions.py) into our own
train/val, stratified by HotpotQA `level` (easy/medium/hard). The official
HotpotQA validation split is already reserved as the untouched test set
(pipeline/load_hotpotqa.py), so this script never touches it.
"""

from sklearn.model_selection import train_test_split

from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.paths import PROCESSED_DIR, SPLITS_DIR

VAL_FRACTION = 0.1
SEED = 42


def main() -> None:
    rows = load_jsonl(PROCESSED_DIR / "train_full.jsonl")
    levels = [row["level"] for row in rows]

    train_rows, val_rows = train_test_split(
        rows,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=levels,
    )

    save_jsonl(train_rows, SPLITS_DIR / "train.jsonl")
    save_jsonl(val_rows, SPLITS_DIR / "val.jsonl")

    print(f"train: {len(train_rows)} rows")
    print(f"val: {len(val_rows)} rows")


if __name__ == "__main__":
    main()
