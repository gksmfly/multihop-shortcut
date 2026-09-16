# 완화(mitigation) 실험 — Before / After

H3에서 확인된 원인(질문의 타입 제약 누출)을 겨냥한 adversarial training (Answer-hop only를 학습 시 unanswerable로 라벨링)이 shortcut 의존도를 실제로 줄이는지 확인.

| 조건 | EM (원본) | EM (완화 후) | F1 (원본) | F1 (완화 후) |
|---|---|---|---|---|
| full | 0.5862 | 0.5809 | 0.7597 | 0.7495 |
| answer_only | 0.6158 | 0.5756 | 0.7670 | 0.7370 |
| bridge_only | 0.0046 | 0.0042 | 0.1091 | 0.1058 |

**Answer-hop only − Full (EM 격차)**: 원본 +0.0296 → 완화 후 -0.0053

격차가 양수(+)면 여전히 shortcut이 우세(Answer-hop only가 Full보다 높음), 음수(−)로 뒤집히거나 0에 가까워지면 완화가 통했다는 뜻.