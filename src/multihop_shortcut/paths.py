from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ERRORS_DIR = DATA_DIR / "errors"

MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, ERRORS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
