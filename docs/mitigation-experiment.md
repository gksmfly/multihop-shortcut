# Shortcut 완화(Mitigation) 실험 기록

이 문서는 [README.md](../README.md)의 진단 연구(가설 1~5, scripts 01~11) 이후,
"진단에서 그치지 말고 실제로 해결하라"는 과제 요건에 따라 추가로 진행한
완화(mitigation) 실험의 설계·실행·결과를 기록한다. 원본 진단 연구의 데이터/모델/
가설 검증 결과는 README.md가 최종 소스이며, 이 문서는 그 이후의 작업만 다룬다.

## 배경 — 왜 완화까지 필요했는가

과제 요건("해결하고자 하는 문제를 자유롭게 정의하고, 이를 해결하기 위한 방법을
설계")을 재검토한 결과, "shortcut이 존재하는지 진단한다"만으로는 부족하고 실제
해결 방법을 설계·실행해서 결과를 보여줘야 한다고 판단했다. 문제 정의는 다음과
같이 재구성된다:

> BERT는 Multi-hop QA를 풀 때 두 문단을 실제로 종합하는가, 아니면 질문의 타입
> 제약 단서만으로 shortcut을 타는가 — 이를 진단하고, 확인된 원인을 완화한다.

## 원인 선정 — 왜 H3만 타겟했는가

원본 연구의 가설 3~5(원인 진단) 중 실제로 **재현 가능한 인과관계로 지지된 것은
H3뿐**이다.

- **H3(질문의 정보 누출)**: bridge 엔티티 이름을 질문에서 마스킹해도 Answer-hop
  only EM이 0.646 → 0.641로 거의 불변 → 원인으로 강하게 지지됨
- **H4(문단 자기완결성)**: 예상과 반대 방향으로 나와 반증됨
- **H5(사전학습 지식/유명도 편향)**: n=22로 표본 부족, 신호도 약하고 방향도 반대라 미확정

원인이 불확실한 H4·H5를 겨냥해 해결책을 설계하면 무엇을 고치는지 근거가 없다.
그래서 유일하게 확실한 인과관계(질문 마스킹 → 성능 불변)를 갖는 **H3만 타겟**으로
확정했다.

## 기각된 설계 — Two-stage v1

최초 초안은 다음과 같은 2단계 파이프라인이었다.

```
Stage 1: Question + Bridge hop → BERT → Bridge Entity 추출
Stage 2: Question + Bridge Entity + Answer hop → BERT → Final Answer
```

검토 결과 두 가지 구멍이 발견되어 기각했다.

1. **검증 안 됨**: Stage 2가 bridge entity를 입력으로 "받기만" 할 뿐, 실제로
   그 값에 의존해서 답하는지 확인할 방법이 설계에 없었다(엔티티를 무시하고
   기존 타입 제약 shortcut을 그대로 써도 구조상 걸러지지 않음).
2. **학습 라벨 없음**: Stage 1의 지도학습 라벨("bridge 문단에서 정확히 어느
   span이 bridge entity인가")이 데이터에 없다. 대리 라벨로
   `answer_hop_title`이 `bridge_hop_text`에 문자 그대로 등장하는지를 확인한
   결과 **54.7%(29,022/53,089)** 샘플에서만 성립해, 나머지 45.3%는 학습도
   평가도 불가능했다.

## 실행한 실험

두 가지를 병렬로 설계·실행했다: 학습 단계 개입(A)과 추론 단계 개입(B).

### 실험 A — Adversarial Training (학습 단계 개입)

**아이디어**: H3의 원인(질문의 타입 제약만으로 answer_hop에서 답을 찾을 수
있음)을 학습 데이터 자체에서 차단한다. Jiang & Bansal(2019)류의 adversarial
augmentation 기법 — Answer-hop only로 주어진 경우를 "답 없음(unanswerable,
CLS 라벨)"으로 학습시켜, 모델이 bridge 없이 답하는 행동 자체를 억제한다.

**구현**:
- [`scripts/mitigation/adversarial_training/build_data.py`](../scripts/mitigation/adversarial_training/build_data.py) —
  원본 Full 조건 학습 데이터(47,780/5,309건)에, 동일 qid의 `answer_hop_text`만을
  context로 주고 `answer=""`, `start/end position=0(CLS)`으로 라벨링한
  adversarial 행을 1:1로 추가(→ train 95,560 / val 10,618건).
- [`scripts/mitigation/adversarial_training/train.py`](../scripts/mitigation/adversarial_training/train.py) —
  동일 하이퍼파라미터(3 epoch, lr 3e-5, batch 16)로 재학습.
  `is_unanswerable=True`인 행은 offset 매칭 없이 강제로 CLS(0,0) 라벨을 부여.
  모델은 `models/multihop_shortcut_qa_mitigated/best`에 저장(원본 모델은
  보존, 덮어쓰지 않음).
- [`scripts/mitigation/adversarial_training/evaluate.py`](../scripts/mitigation/adversarial_training/evaluate.py) —
  원본과 동일한 test_conditions.jsonl 3조건에 대해 재평가, 원본
  `condition_comparison.json`과 비교.

**결과** (`data/errors/mitigated_condition_comparison.json`,
`data/errors/mitigation_report.md`):

| 조건 | EM (원본) | EM (완화 후) | 변화 | mean_cls_prob (완화 후) |
|---|---|---|---|---|
| Full | 0.5862 | 0.5809 | −0.0053 | 0.0070 |
| Answer-hop only | 0.6158 | 0.5756 | **−0.0402** | **0.9960** |
| Bridge-hop only | 0.0046 | 0.0042 | −0.0004 | 0.9877 |

**Answer-hop only − Full (EM 격차)**: 원본 **+0.0296**(shortcut 우세) →
완화 후 **−0.0053**(사실상 역전/소멸).

**해석**: Full 조건 성능은 거의 그대로 유지하면서(진짜 추론 능력은 손상 안 됨),
Answer-hop only에서만 선택적으로 −0.0402의 큰 하락이 발생했다 — 정확히 의도한
타겟팅. 추가로 answer_only의 `mean_cls_prob`이 0.0001(원본) → 0.9960(완화
후)으로 폭증한 것은 H2(증거 없어도 확신도가 안 떨어지는 문제)까지 부수적으로
개선됐다는 증거다. 단, 평가 파이프라인(`run_qa_inference`)이 CLS를 답으로
선택하지 못하고 항상 context 내 span을 강제로 고르는 구조라, "모른다"는 인지는
cls_prob 상승으로만 드러나고 EM/F1이 0으로 수렴하지는 않는다는 점은
감안해야 한다.

**→ 실험 A는 성공.** shortcut 신호(Answer-hop only ≥ Full)가 해소됐다.

### 실험 B-v1 — Two-stage Support Classifier (추론 단계 개입)

**아이디어**: Stage 1(QA 모델)은 재학습하지 않고 원본(shortcut이 있는) 모델을
그대로 사용하되, Stage 2로 "이 bridge_hop이 이 질문에 실제로 연결되는가"를
판별하는 이진 분류기를 얹어 shortcut 오답을 사후에 탐지/veto할 수 있는지
검증한다. v1의 두 구멍을 다음과 같이 고쳤다:

- **구멍 1(검증 안 됨) 해결**: "거치기"가 아니라 "지지 점수(support score)"로
  설계하고, 랜덤 bridge_hop을 주입하는 ablation을 처음부터 포함시켰다.
- **구멍 2(라벨 없음) 해결**: `answer_hop_title` 문자열 매칭 대신, **진짜
  (question, bridge_hop) 쌍 = positive(1)**, **다른 질문의 bridge_hop을
  무작위로 붙인 쌍 = negative(0)**으로 라벨을 구성해 100% 커버리지를 확보했다.

**구현**:
- [`scripts/mitigation/bridge_relatedness_classifier/build_data.py`](../scripts/mitigation/bridge_relatedness_classifier/build_data.py) —
  질문마다 진짜 bridge_hop(positive) + 무작위 다른 행의 bridge_hop(negative)
  쌍으로 데이터 구성(train 95,560 / val 10,618건).
- [`scripts/mitigation/bridge_relatedness_classifier/train.py`](../scripts/mitigation/bridge_relatedness_classifier/train.py) —
  `BertForSequenceClassification`(이진 분류)을 동일 하이퍼파라미터로 학습,
  `models/support_classifier/best`에 저장. train_loss 0.0144까지 수렴.
- [`scripts/mitigation/bridge_relatedness_classifier/evaluate.py`](../scripts/mitigation/bridge_relatedness_classifier/evaluate.py) —
  test set(4,734건)에 대해 (1) 진짜 bridge vs 랜덤 bridge ablation,
  (2) Stage 1(원본 모델)의 정답/오답 여부에 따른 support score 격차(veto
  가능성)를 각각 측정.

**결과** (`data/errors/two_stage_report.json`,
`data/errors/two_stage_report.md`):

| Ablation | support score |
|---|---|
| 진짜 bridge_hop | 0.9958 |
| 랜덤 bridge_hop | 0.0028 |
| 격차 | **+0.9930** |

| Veto 가능성 | n | support score |
|---|---|---|
| Stage 1 정답(EM=1) | 2,915 | 0.9960 |
| Stage 1 오답(EM=0) | 1,819 | 0.9954 |
| 격차 | | **+0.0006** |

**→ 실험 B-v1은 실패.** Ablation은 거의 완벽하게 통과했지만(진짜 vs 랜덤 격차
0.9930), Stage 1의 정답/오답에 따른 support score 격차는 사실상 0(+0.0006)이다.

#### 왜 실패했는가 — 문제 정의 자체의 불일치

B가 **풀려던 문제**는 "Stage 1이 bridge를 실제로 활용해서 답했는가?"였다.
하지만 B가 **실제로 푼 문제**는 "이 bridge_hop이 이 질문과 주제적으로
관련 있는가?"였다.

원인은 학습 데이터 구성 자체에 있다:

- positive = 진짜 bridge_hop (질문과 주제 일치)
- negative = 무작위 bridge_hop (질문과 주제 불일치)

이 라벨링은 **토픽 매칭**을 구분하도록 분류기를 학습시킨 것이지, **Stage 1의
추론 경로**(bridge를 실제로 썼는지 vs 타입 제약 shortcut을 탔는지)를 구분하도록
만든 게 아니다. 진짜 bridge_hop은 Stage 1이 shortcut을 타든 실제로 두 홉을
종합해서 추론하든 상관없이 **항상** 질문과 주제적으로 관련 있다 — 그래서
support score가 Stage 1의 정답 여부(0.9960 vs 0.9954)와 무관하게 똑같이
높게 나온 것이다. train_loss가 0.0144까지 떨어진 것도, 이 분류기가
(어려운 "실제 의존성 판별"이 아니라) 상대적으로 쉬운 "주제 일치 판별"을
학습했다는 정황과 일치한다.

**한 줄 요약**: "bridge가 진짜냐"와 "bridge를 실제로 썼냐"는 서로 다른
신호인데, 학습 데이터가 전자만 구분하도록 설계됐다.

### 실험 B-v3 — Counterfactual 라벨링 (실제 실행)

이론적 검토(위 "향후 과제"의 두 한계)만으로 끝내지 않고 실제로 구현·실행했다.

**구현**:
- [`scripts/mitigation/counterfactual_classifier/build_data.py`](../scripts/mitigation/counterfactual_classifier/build_data.py) —
  원본(비완화) 모델을 Stage 1로 두 번 실행(Answer-hop only → P1, Full → P2),
  `P1 != P2`(정규화 비교)이면 label=1. 정답 여부까지 교차해
  `bridge_helped`/`bridge_hurt`/`changed_still_wrong`/`no_change` 4버킷으로 분류.
- [`scripts/mitigation/counterfactual_classifier/train.py`](../scripts/mitigation/counterfactual_classifier/train.py) —
  이 라벨로 Stage 2를 재학습(`models/support_classifier_counterfactual/best`).
- [`scripts/mitigation/counterfactual_classifier/evaluate.py`](../scripts/mitigation/counterfactual_classifier/evaluate.py) —
  test set에서 분류기 성능 + veto gap + flag된 케이스의 실제 구성을 평가.
  test set의 ground-truth 라벨은 pipeline/evaluate_conditions.py가 이미 저장해 둔
  `full_pred`/`answer_only_pred`로 재추론 없이 바로 계산.

**한계 2가 라벨 분포에서부터 실증됨** (`data/errors/train_counterfactual.log` 통계,
`scripts/mitigation/counterfactual_classifier/build_data.py` 출력):

| train (n=47,780) | 비율 |
|---|---|
| no_change (shortcut) | 90.8% |
| bridge_helped | 5.4% |
| bridge_hurt | 2.1% |
| changed_still_wrong | 1.6% |

`P1≠P2`(9.2%) 중 실제로 정답 쪽으로 바뀐 건 5.4%p뿐이고, 나머지 3.7%p는 오히려
틀리게 만들거나(2.1%) 여전히 틀림(1.6%). val set에서는 `bridge_hurt`(5.6%)가
`bridge_helped`(5.2%)보다 **높아**, bridge가 예측을 바꿀 때 도움이 되는 만큼
해가 될 수 있다는 걸 실제 데이터로 보여준다.

**평가 결과** (`data/errors/counterfactual_classifier_report.json`,
`.md`, test set n=4,734):

| 지표 | 값 |
|---|---|
| accuracy | 0.7968 (다수결 baseline 0.7757) |
| precision(label=1) | 0.7049 |
| recall(label=1) | **0.1620** |
| **veto gap**(오답 − 정답 평균 score) | **+0.0304** |

B-v1의 veto gap(+0.0006, 사실상 0)과 비교하면 **방향이 맞고 30배 커졌다** —
counterfactual 라벨이 실제로 "Stage 1이 틀릴 만한 케이스"에 더 높은 점수를
주는 신호를 학습했다는 뜻. 다만 recall 0.162로 진짜 "영향 있음" 케이스의
84%를 놓친다(라벨 불균형 90.8:9.2의 대가).

flag(예측=1)한 244건의 실제 구성:

| bucket | 비율 |
|---|---|
| bridge_helped | 25.0% |
| bridge_hurt | 29.5% |
| changed_still_wrong | 16.0% |
| no_change (오탐) | 29.5% |

**해석**: 한계 1·2가 정확히 수치로 재현됐다. veto 신호는 실재하지만
(1) 대부분의 "영향 있음" 케이스를 놓치고(recall 0.162), (2) flag된 것 중에서도
실제로 도움 되는 비율(25.0%)이 해가 되는 비율(29.5%)보다 낮다 — naive하게
"flag되면 Full 조건 답으로 바꾼다"는 전략을 쓰면 득보다 실이 클 수 있다.
**B-v1보다는 개선됐지만, 독립적인 해결책으로 쓰기엔 아직 약하다.**

### 실험 B-v4 — 3-way 라벨 + 입력 확장 (eval/ 폴더, 정식 파이프라인 밖)

B-v3의 두 한계(정답과 무관하게 label=1로 뭉뚱그림, Stage 2가 (질문,
bridge_hop)만 보고 Stage 1의 실제 예측/확신도를 모름)를 직접 겨냥한 개선을
실제로 구현·실행했다(`eval/add_p1_confidence.py` →
`eval/train_3way_classifier.py` → `eval/evaluate_3way_classifier.py`,
파이프라인 재현용 `scripts/`가 아니라 탐색적 실험이라 `eval/`에 분리).

**변경점**:
1. **3-way 라벨**: `{0: no_change/changed_still_wrong(전환 이득 없음),
   1: bridge_helped(틀림→맞음, 잡아야 할 케이스), 2: bridge_hurt(맞음→틀림,
   override하면 오히려 해로운 케이스)}`로 세분화(`metrics.BUCKET_TO_3WAY`).
2. **입력 확장**: Stage 2 입력을 `(질문, bridge_hop_text)`에서
   `(질문 + Stage 1의 answer_hop-only 예측 텍스트 + confidence, bridge_hop_text)`로
   확장(`classifier_training.build_augmented_question`).

**결과** (`data/errors/3way_classifier_report.json`, test n=4,734,
true label=1 260건 / label=2 400건):

| | B-v3(이진, threshold 0.5) | B-v4(3-way, threshold 0.5) |
|---|---|---|
| precision | 0.7049 | 0.6286 |
| recall | 0.1620 | **0.3385**(2배 이상) |
| harm_rate | 미측정 | **0.0000** |

confusion matrix상 true=bridge_hurt(400건) 중 단 1건만 "helped"로 오분류됨 —
**3-way 분리가 "도움 되는 변화"와 "해로운 변화"를 실제로 다른 패턴으로
학습**하게 만들었다(이진 라벨에서는 이 둘이 label=1로 뭉쳐서 신호가
흐려졌었다).

**PR curve** (threshold 0.1~0.9, `data/errors/3way_classifier_report.md`):
threshold를 낮출수록 recall은 0.81까지 오르지만 harm_rate도 9.75%까지
같이 오른다. threshold 0.4~0.5 구간이 "harm_rate ≈ 0을 유지하면서 recall을
어느 정도(33~35%) 확보하는" 실용적 지점으로 보인다.

**해석**: B-v4는 여전히 recall이 완벽하지 않지만(대부분의 threshold에서
50% 이상의 bridge_helped 케이스를 놓침), **B-v1(쓸모없음)·B-v3(약한 신호,
harm 위험 있음)보다 실질적으로 안전하고 개선된 veto 신호**를 만들었다.
다만 A(학습 단계 개입)의 완결성·확실성에는 여전히 못 미친다.

#### B-v4를 실제 파이프라인에 적용하면 최종 EM/F1이 오르는가

지금까지는 분류기 자체의 precision/recall/harm_rate만 봤을 뿐, "flag된
답을 실제로 어떻게 처리하는가"(다운스트림 로직)가 없었다 — 분류기 성능이
좋아도 최종 태스크 지표가 안 오르면 실전에서는 의미가 없다는 지적을 받아
직접 확인했다. 결정 규칙: `p(bridge_helped) >= threshold`면 Full 조건 예측으로
교체, 아니면 Answer-hop only 예측(원본 shortcut-prone 기본값)을 유지.
재학습 없이 이미 저장된 예측 파일만 결합
([`eval/evaluate_veto_pipeline.py`](../eval/evaluate_veto_pipeline.py)).

| | EM | F1 |
|---|---|---|
| baseline: Answer-hop only 전부 유지 | 0.6158 | 0.7670 |
| baseline: Full 전부로 교체 | 0.5862 | 0.7597 |
| **veto 적용(threshold 0.2, 최적)** | **0.6396** | **0.7880** |
| veto 적용(threshold 0.5) | 0.6343 | 0.7837 |

**모든 threshold(0.1~0.9)에서 일관되게 EM/F1이 상승**했다(threshold 0.2에서
EM +0.0239로 최대). 특히 threshold 0.2의 0.6396은 Answer-hop only 단독
(0.6158)과 Full 단독(0.5862) **둘 다보다 높다** — veto가 "언제 bridge를
믿어야 하는가"를 선택적으로 잘 골라내고 있다는 뜻이다. recall 34%라는
한계에도 불구하고, **B-v4는 실제 파이프라인에 적용했을 때 측정 가능한
개선을 만든다** — B-v1·B-v3와 달리 정직한 실패로 끝나지 않은 유일한 B
버전이다.

**왜 두 단독 조건보다 높은가**: 이 결과는 단순 평균 개선이 아니라
**선택적 개입이 두 조건의 단점을 상호 보완**하기 때문이다.
- Answer-hop only 단독(0.6158): bridge를 아예 안 씀 → bridge가 진짜
  필요한 9.4%(`bridge_needed`, 핵심 결과 절 참고) 케이스를 놓친다.
- Full 단독(0.5862): bridge를 항상 씀 → shortcut만으로 이미 충분한
  케이스에서 오히려 더 긴 context에 방해받는다(`bridge_hurt` 2.1%).
- veto: "이 케이스는 bridge를 믿어도 될 것 같다"는 신호가 있을 때만
  선택적으로 Full로 전환 → 두 조건의 장점만 취한다.

즉 B-v4는 단순한 성능 개선이 아니라 **모델이 스스로 언제 shortcut을
벗어나야 하는지 판단하게 만든 것**이다. 이 지점에서 이 프로젝트 전체의
서사 — H1·H2(shortcut 존재 확인) → H3~H5(원인 규명) → 완화(A 성공, B는
v1·v3 실패를 거쳐 v4에서 성공) — 가 완결된다.

## 종합 결론

| 실험 | 개입 시점 | 결과 |
|---|---|---|
| A — Adversarial training | 학습 단계 | ✅ 성공 — shortcut 격차(+0.0296 → −0.0053) 해소, Full 성능 유지, H2도 부수적 개선 |
| B-v1 — Support classifier(관련성 라벨) | 추론 단계 | ❌ 실패 — veto gap 사실상 0(+0.0006), 관련성 판별과 의존성 판별은 다른 과제임을 확인 |
| B-v3 — Support classifier(counterfactual 이진 라벨) | 추론 단계 | ▲ 부분 개선 — veto gap +0.0304, 방향은 맞으나 recall 0.162로 약함 |
| B-v4 — Support classifier(3-way 라벨 + 입력 확장) | 추론 단계 | ✅ **성공** — precision 0.629/recall 0.339/harm_rate 0.000, 실제 파이프라인 적용 시 EM 0.6158→0.6396(threshold 0.2), 모든 threshold에서 일관되게 상승 |

**A(adversarial training)가 가장 직접적이고 확실한 해결책이지만, B도 v4에서
실제로 성공했다.** B는 세 버전(v1→v3→v4)을 거치며 "관련성 판별"에서 "실제
의존성 판별"로, 다시 "안전한 override 판별"로 계속 개선됐고, v4는 단순
분류기 지표뿐 아니라 **실제 다운스트림 EM/F1 개선까지 확인**됐다(+0.0239,
threshold 0.2). 다만 A처럼 문제를 근본(학습 단계)에서 없애는 게 아니라
원본 Stage 1의 shortcut을 그대로 둔 채 사후에 일부만(recall 34%) 건져내는
방식이라 개선 폭이 A보다 작고 상한이 있다. v1→v3의 실패, v4의 성공이라는
대비 자체가 "post-hoc 검증을 제대로 설계하려면 무엇이 필요한지, 그리고
분류기 지표만으로는 부족하고 다운스트림까지 닫아서 확인해야 한다"는
교훈을 정직하게 보여주는 게 B 실험 전체의 기여다.

## 향후 과제

- **B-v3 한계를 실제로 다시 검증**: 이론 검토에서 예측했던 두 한계가
  실측으로 확인됐다.
  1. **Stage 1 종속성**: B-v3의 라벨은 실험 A 적용 전 원본 모델 기준이다.
     A를 최종 배포 모델로 쓴다면 counterfactual 라벨도 A 모델 기준으로
     다시 만들어야 한다(스코프 밖 — A 자체가 이미 shortcut을 줄였으므로
     우선순위 낮음).
  2. **정답과 무관**: flag된 244건 중 bridge_helped 25.0% vs
     bridge_hurt+changed_still_wrong 45.5% — naive하게 "flag되면 답을
     바꾼다"는 전략은 채택하지 않는다. flag를 "확신도를 낮춘다/사람에게
     넘긴다" 정도의 약한 신호로만 쓰는 게 현재 근거 수준에 맞다.
  3. recall(0.162)을 올리려면 라벨 불균형(90.8:9.2)에 class weighting이나
     oversampling을 적용해볼 수 있으나, 위 2번 한계 때문에 recall을 올려도
     precision/flag 구성 문제가 남아 우선순위를 A 쪽에 뒀다.
- A의 성공을 top-k 재순위화나 실제 서비스 파이프라인에 통합하는 것은 스코프
  밖으로 남겨둔다(README "스코프" 절과 동일한 원칙).
