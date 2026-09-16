"""One-off enrichment for the B-v4 (3-way + richer-input) follow-up
experiment: scripts/mitigation/counterfactual_classifier/build_data.py's train_counterfactual.jsonl / val_counterfactual.jsonl
already have Stage 1's Answer-hop-only prediction text (p1_pred) but not its
confidence. Re-runs *only* the Answer-hop-only condition (not Full - that's
not needed here) on the original model to fill it in, rather than re-running
scripts/mitigation/counterfactual_classifier/build_data.py's full P1+P2 pass again.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast

from multihop_shortcut.inference import run_qa_inference
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.paths import MODELS_DIR, PROCESSED_DIR, SPLITS_DIR

MODEL_DIR = MODELS_DIR / "multihop_shortcut_qa" / "best"  # original, unmitigated - same Stage 1 as scripts/mitigation/counterfactual_classifier/build_data.py

with open(SPLITS_DIR / "max_length_recommendation.json", encoding="utf-8") as f:
    MAX_LENGTH = json.load(f)["recommended_max_length"]


def enrich(path, pool_by_qid, model, tokenizer, device):
    rows = load_jsonl(path)
    examples = [{"question": r["question"], "context": pool_by_qid[r["qid"]]["answer_hop_text"]} for r in rows]
    preds = run_qa_inference(model, tokenizer, examples, device, max_length=MAX_LENGTH)
    for r, p in zip(rows, preds):
        assert r["p1_pred"] == p["pred_text"], f"P1 mismatch for {r['qid']} - Stage 1 model must be unchanged"
        r["p1_confidence"] = p["confidence"]
    save_jsonl(rows, path)
    print(f"{path.name}: {len(rows)} rows enriched with p1_confidence")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = BertForQuestionAnswering.from_pretrained(str(MODEL_DIR)).to(device)

    pool_rows = load_jsonl(PROCESSED_DIR / "train_pool.jsonl")
    pool_by_qid = {r["qid"]: r for r in pool_rows}

    for name in ["train_counterfactual.jsonl", "val_counterfactual.jsonl"]:
        enrich(SPLITS_DIR / name, pool_by_qid, model, tokenizer, device)


if __name__ == "__main__":
    main()
