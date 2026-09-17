"""Evaluates the B-v4 3-way classifier (eval/train_3way_classifier.py) on
the test set. Ground truth again comes for free from
data/errors/test_predictions.jsonl (pipeline/evaluate_conditions.py) - no Stage 1 re-inference
needed.

A veto/override decision is: flag this Answer-hop-only prediction as
untrustworthy and prefer the Full-context answer, iff p(label=1,
"bridge_helped") >= threshold. Sweeps the threshold (a text PR curve) to
show the precision/recall trade-off the user asked for, plus harm_rate
(fraction of overrides that were actually label=2 "bridge_hurt" - i.e. the
override would make things worse) at each point, since precision/recall
alone hide that cost.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from multihop_shortcut.classifier_training import build_augmented_question
from multihop_shortcut.constants import get_max_length
from multihop_shortcut.inference import load_classifier_model, run_multiclass_probs
from multihop_shortcut.io_utils import load_jsonl, save_jsonl
from multihop_shortcut.metrics import BUCKET_TO_3WAY, counterfactual_bucket
from multihop_shortcut.paths import ERRORS_DIR, MODELS_DIR, PROCESSED_DIR

MODEL_DIR = MODELS_DIR / "support_classifier_3way" / "best"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MAX_LENGTH = get_max_length()


def main() -> None:
    tokenizer, model, device = load_classifier_model(MODEL_DIR)

    preds = load_jsonl(ERRORS_DIR / "test_predictions.jsonl")
    conditions = {r["qid"]: r for r in load_jsonl(PROCESSED_DIR / "test_conditions.jsonl")}

    aug_questions = [
        build_augmented_question(r["question"], r["answer_only_pred"], r["answer_only_confidence"]) for r in preds
    ]
    bridges = [conditions[r["qid"]]["bridge_only_context"] for r in preds]
    probs = run_multiclass_probs(model, tokenizer, device, aug_questions, bridges, max_length=MAX_LENGTH)

    out = []
    for r, p in zip(preds, probs):
        p1_correct = r["answer_only_em"] == 1
        p2_correct = r["full_em"] == 1
        bucket = counterfactual_bucket(r["answer_only_pred"], r["full_pred"], p1_correct, p2_correct)
        true_label = BUCKET_TO_3WAY[bucket]
        out.append(
            {
                "qid": r["qid"],
                "true_label": true_label,
                "bucket": bucket,
                "p0_no_benefit": p[0],
                "p1_bridge_helped": p[1],
                "p2_bridge_hurt": p[2],
                "argmax": int(max(range(3), key=lambda i: p[i])),
            }
        )
    save_jsonl(out, ERRORS_DIR / "3way_classifier_predictions.jsonl")

    n = len(out)
    n_true1 = sum(1 for o in out if o["true_label"] == 1)
    n_true2 = sum(1 for o in out if o["true_label"] == 2)

    # argmax confusion matrix
    confusion = [[0, 0, 0] for _ in range(3)]
    for o in out:
        confusion[o["true_label"]][o["argmax"]] += 1
    argmax_acc = sum(confusion[i][i] for i in range(3)) / n

    pr_rows = []
    for t in THRESHOLDS:
        overridden = [o for o in out if o["p1_bridge_helped"] >= t]
        n_over = len(overridden)
        tp = sum(1 for o in overridden if o["true_label"] == 1)
        harm = sum(1 for o in overridden if o["true_label"] == 2)
        precision = tp / n_over if n_over else 0.0
        recall = tp / n_true1 if n_true1 else 0.0
        harm_rate = harm / n_over if n_over else 0.0
        pr_rows.append(
            {
                "threshold": t,
                "n_override": n_over,
                "precision": precision,
                "recall": recall,
                "harm_rate": harm_rate,
            }
        )

    lines = [
        "# B-v4 (3-way 라벨 + 예측/confidence 입력 추가) — 평가 결과\n",
        f"n = {n} (test set) · true label=1(bridge_helped) {n_true1}건 · label=2(bridge_hurt) {n_true2}건\n",
        "## Argmax 기준 3-way confusion matrix (행=정답, 열=예측)",
        "| | pred 0 (변화없음) | pred 1 (helped) | pred 2 (hurt) |",
        "|---|---|---|---|",
        f"| true 0 | {confusion[0][0]} | {confusion[0][1]} | {confusion[0][2]} |",
        f"| true 1 (helped) | {confusion[1][0]} | {confusion[1][1]} | {confusion[1][2]} |",
        f"| true 2 (hurt) | {confusion[2][0]} | {confusion[2][1]} | {confusion[2][2]} |",
        "",
        f"argmax accuracy: {argmax_acc:.4f}",
        "",
        "## PR curve — threshold별 override(=bridge_helped로 판단해 답을 바꿈) 성능",
        "| threshold | n_override | precision | recall | harm_rate |",
        "|---|---|---|---|---|",
    ]
    for row in pr_rows:
        lines.append(
            f"| {row['threshold']:.1f} | {row['n_override']} | {row['precision']:.4f} "
            f"| {row['recall']:.4f} | {row['harm_rate']:.4f} |"
        )
    lines += [
        "",
        (
            "precision = override한 것 중 실제로 도움 된 비율, recall = 전체 "
            "bridge_helped 케이스 중 잡아낸 비율, harm_rate = override한 것 중 "
            "실제로는 해로웠던(bridge_hurt) 비율. B-v3(이진, threshold 0.5 고정)는 "
            "precision 0.7049 / recall 0.1620이었다 — 이 표에서 recall을 그만큼 "
            "확보하려면 threshold를 어디까지 낮춰야 하고, 그때 precision·harm_rate가 "
            "어떻게 무너지는지 확인."
        ),
    ]

    summary = {
        "n": n,
        "n_true_bridge_helped": n_true1,
        "n_true_bridge_hurt": n_true2,
        "confusion_matrix": confusion,
        "argmax_accuracy": argmax_acc,
        "pr_curve": pr_rows,
    }
    with open(ERRORS_DIR / "3way_classifier_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ERRORS_DIR / "3way_classifier_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
