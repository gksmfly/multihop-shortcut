# 가설 2 — Confidence / Bias 분석

n = 4734

## 조건별 평균 confidence / cls_prob

| 조건 | 평균 confidence | 평균 cls_prob |
|---|---|---|
| full | 0.6905 | 0.0001 |
| answer_only | 0.7256 | 0.0001 |
| bridge_only | 0.4541 | 0.0013 |

Full 대비 Bridge-hop only confidence 평균 하락폭: 0.2364
(하락폭이 0.1 이하인 샘플 비율: 42.73% — 이 비율이 높으면 '증거가 없어도 확신도가 안 떨어진다'는 뜻)

Bridge-hop only 오답 중 엔티티 타입이 정답과 일치하는 비율: 69.48% (4712건 중) — 높으면 무작위 추측이 아니라 '그럴듯한' 오답을 고르는 편향이 있다는 뜻