# v2 Two-stage 실험 (Stage 1 그대로 + Stage 2 support classifier)

n = 4734

## Ablation — 진짜 bridge vs 랜덤 bridge
| | support score (mean) |
|---|---|
| 진짜 bridge_hop | 0.9958 |
| 랜덤 bridge_hop (다른 질문에서) | 0.0028 |
| 격차 | +0.9930 |

격차가 크면(양수, 유의미) → Stage 2가 실제로 bridge 내용을 이용해 지지 여부를 구분한다는 뜻. 격차가 작으면 → Stage 2도 표면적 패턴만 학습했을 뿐, bridge 정보를 실제로 검증하지 못한다는 뜻.

## Veto 가능성 — Stage 1이 맞았을 때 vs 틀렸을 때, 진짜 bridge에 대한 support score
| | n | support score (mean) |
|---|---|---|
| Stage 1 정답(EM=1) | 2915 | 0.9960 |
| Stage 1 오답(EM=0) | 1819 | 0.9954 |
| 격차 | | +0.0006 |

오답 쪽 support score가 눈에 띄게 낮으면 → Stage 2가 Stage 1의 shortcut 오답을 사후에 flag/veto할 수 있는 신호를 제공한다는 뜻 (top-k 재순위화까지는 이번 실험 범위 밖 - 시간 제약).