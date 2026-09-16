#!/usr/bin/env bash
# Runs the diagnostic pipeline in pipeline/ in order. File names are purely
# descriptive (no numeric prefixes) - this script is the single source of
# truth for execution order; see README.md "파이프라인" for what each step
# does and why the order matters.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python

# 데이터 준비
$PY pipeline/load_hotpotqa.py
$PY pipeline/build_eval_conditions.py
$PY pipeline/split_dataset.py
$PY pipeline/analyze_lengths.py

# 학습 (Full 조건만)
$PY pipeline/train_bert.py

# 가설 1·2 평가 (핵심 결과)
$PY pipeline/evaluate_conditions.py
$PY pipeline/confidence_bias_analysis.py
$PY pipeline/error_taxonomy.py

# 가설 3·4·5 원인 진단
$PY pipeline/question_masking_probe.py
$PY pipeline/self_containment_split.py
$PY pipeline/fame_bias_analysis.py
