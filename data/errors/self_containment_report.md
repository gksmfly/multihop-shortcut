# 가설 4 — Answer-hop 문단 자기완결성 분석

Full 정답 2775건 기준

| 그룹 | n | Answer-hop only EM |
|---|---|---|
| bridge 제목이 answer_hop에 언급됨 | 664 | 0.7801 |
| bridge 제목이 answer_hop에 언급 안 됨 | 2111 | 0.9460 |

## 가설 3(질문)과 2x2 교차

| 질문에 bridge 언급 | answer_hop에 bridge 언급 | n | Answer-hop only EM |
|---|---|---|---|
| True | True | 155 | 0.9226 |
| True | False | 1188 | 0.9520 |
| False | True | 509 | 0.7367 |
| False | False | 923 | 0.9382 |

두 마진 비율 차이가 크면 그쪽(질문 vs 문단)이 shortcut의 더 강한 예측 인자 - scripts/09의 마스킹 결과와 함께 해석할 것.