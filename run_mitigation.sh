#!/usr/bin/env bash
# Runs the mitigation experiments (docs/mitigation-experiment.md) in order.
# Requires run_pipeline.sh to have completed first (needs its trained model
# and test_predictions.jsonl). adversarial_training and
# bridge_relatedness_classifier don't depend on each other and were
# originally run in parallel on two GPUs; counterfactual_classifier reads
# the original model, so it doesn't need adversarial_training either - the
# sequential order below is just for a single-GPU run.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
M=scripts/mitigation

# A - adversarial training (targets H3, the accepted fix)
$PY $M/adversarial_training/build_data.py
$PY $M/adversarial_training/train.py
$PY $M/adversarial_training/evaluate.py

# B-v1 - support classifier, real-vs-random bridge pairing (failed - see docs)
$PY $M/bridge_relatedness_classifier/build_data.py
$PY $M/bridge_relatedness_classifier/train.py
$PY $M/bridge_relatedness_classifier/evaluate.py

# B-v3 - support classifier, counterfactual P1-vs-P2 labeling (partial improvement)
$PY $M/counterfactual_classifier/build_data.py
$PY $M/counterfactual_classifier/train.py
$PY $M/counterfactual_classifier/evaluate.py

# B-v4 (eval/ - exploratory follow-up on B-v3, not part of the accepted pipeline)
$PY eval/add_p1_confidence.py
$PY eval/train_3way_classifier.py
$PY eval/evaluate_3way_classifier.py
$PY eval/evaluate_veto_pipeline.py  # closes the loop: does the veto actually move final EM/F1?
