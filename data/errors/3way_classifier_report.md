# B-v4 (3-way 라벨 + 예측/confidence 입력 추가) — 평가 결과

n = 4734 (test set) · true label=1(bridge_helped) 260건 · label=2(bridge_hurt) 400건

## Argmax 기준 3-way confusion matrix (행=정답, 열=예측)
| | pred 0 (변화없음) | pred 1 (helped) | pred 2 (hurt) |
|---|---|---|---|
| true 0 | 4017 | 53 | 4 |
| true 1 (helped) | 169 | 90 | 1 |
| true 2 (hurt) | 339 | 1 | 60 |

argmax accuracy: 0.8802

## PR curve — threshold별 override(=bridge_helped로 판단해 답을 바꿈) 성능
| threshold | n_override | precision | recall | harm_rate |
|---|---|---|---|---|
| 0.1 | 1077 | 0.1959 | 0.8115 | 0.0975 |
| 0.2 | 455 | 0.3231 | 0.5654 | 0.0747 |
| 0.3 | 238 | 0.4706 | 0.4308 | 0.0462 |
| 0.4 | 159 | 0.5786 | 0.3538 | 0.0126 |
| 0.5 | 140 | 0.6286 | 0.3385 | 0.0000 |
| 0.6 | 129 | 0.6589 | 0.3269 | 0.0000 |
| 0.7 | 123 | 0.6748 | 0.3192 | 0.0000 |
| 0.8 | 117 | 0.7094 | 0.3192 | 0.0000 |
| 0.9 | 110 | 0.7364 | 0.3115 | 0.0000 |

precision = override한 것 중 실제로 도움 된 비율, recall = 전체 bridge_helped 케이스 중 잡아낸 비율, harm_rate = override한 것 중 실제로는 해로웠던(bridge_hurt) 비율. B-v3(이진, threshold 0.5 고정)는 precision 0.7049 / recall 0.1620이었다 — 이 표에서 recall을 그만큼 확보하려면 threshold를 어디까지 낮춰야 하고, 그때 precision·harm_rate가 어떻게 무너지는지 확인.