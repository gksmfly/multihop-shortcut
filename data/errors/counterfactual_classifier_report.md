# B-v3 Counterfactual 라벨링 — 평가 결과

n = 4734 (test set, 원본 모델 Stage 1 그대로)

## 1. 분류기 성능 (ground-truth P1≠P2 라벨 대비)
| | 값 |
|---|---|
| accuracy | 0.7968 |
| precision (label=1) | 0.7049 |
| recall (label=1) | 0.1620 |
| 다수결 baseline(항상 다수 클래스 예측 시 accuracy) | 0.7757 |

accuracy가 다수결 baseline과 비슷하면 → 분류기가 사실상 '항상 no_change로 찍는' 수준이라는 뜻(라벨 불균형 90.8:9.2 때문에 흔한 함정). recall이 낮으면 실제 '영향 있음' 케이스를 거의 못 잡아낸다는 뜻.

## 2. Veto gap (B-v1과 동일 지표)
| | n | pred_score (mean) |
|---|---|---|
| Stage 1 정답 | 2915 | 0.0677 |
| Stage 1 오답 | 1819 | 0.0981 |
| 격차(오답 − 정답) | | +0.0304 |

## 3. 한계 2 실증 — 분류기가 '영향 있음(1)'으로 flag한 케이스의 실제 구성
| bucket | n | flag된 것 중 비율 |
|---|---|---|
| bridge_helped | 61 | 25.0% |
| bridge_hurt | 72 | 29.5% |
| changed_still_wrong | 39 | 16.0% |
| no_change | 72 | 29.5% |

이 표가 veto 시스템의 실제 가치를 보여준다 — flag된 케이스 중 bridge_helped 비율이 낮고 bridge_hurt/changed_still_wrong 비율이 높으면, 이 flag를 신뢰해서 답을 바꾸는 건 득보다 실이 크다는 뜻.