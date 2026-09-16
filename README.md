# Multihop-Shortcut

BERT 기반 multi-hop QA 모델은 실제로 두 홉의 정보를 모두 종합해서 답하는가,
아니면 정답이 있는 한쪽 홉만 보고도 맞히는 shortcut을 타는가?

## 연구 질문

Multi-hop QA(예: "이순신을 그린 영화의 감독은 어느 나라 사람인가?" → 영화
찾기(1홉) → 감독 국적 찾기(2홉))는 두 단계 추론을 요구하도록 설계된
태스크다. 하지만 모델이 실제로는 정답이 직접 담긴 문단(answer hop)만 찾아
읽고, 연결 역할을 하는 문단(bridge hop)은 활용하지 않는 shortcut learning
현상이 선행 연구(Min et al. 2019, Chen & Durrett 2019, Jiang & Bansal
2019)에서 지적된 바 있다. 이 프로젝트는 이 현상이 실제로 재현되는지
`bert-base-cased` 기반 extractive QA 모델로 직접 검증한다 — 완전히 새로운
발견을 주장하는 게 아니라 재현(replication) 연구이며, 핵심 기여는 8단계
에러 분석과 9~11단계 원인 진단에서 "언제, 왜" shortcut이 통하는지 유형화하는 데 있다.

**가설 1:** 두 홉 문단을 모두 준 조건(Full)과, 정답이 있는 홉만 준 조건
(Answer-hop only)의 정확도(EM/F1) 차이가 작다면 → 모델이 bridge 홉을
실제로 활용하지 않는다는 뜻.

**가설 2:** 정답이 없는 bridge 홉만 준 조건(Bridge-hop only)에서는 정답이
아예 context에 없으므로 EM/F1은 설계상 거의 0이 나올 수밖에 없다 — 이건
가설 검증에 쓸 수 없다. 대신 모델이 이 조건에서도 여전히 **높은 확신도로
그럴듯한(엔티티 타입이 맞는) 오답**을 뱉는지를 본다. 이게 유의미하게
나타난다면, "bridge 정보 부재를 인지하지 못하는" 별도의 구조적 편향으로
해석한다.

가설 1·2는 "shortcut이 존재하는가"를 확인한다. 아래 가설 3~5는 그 결과가
나온 **이후** "왜 그런가"를 규명하는 원인 진단이다.

**가설 3(원인 규명 — 질문 자체의 정보 누출):** Answer-hop only 성능이 유지되는
이유가 bridge 문단이 아니라 **질문 문장 자체**에 이미 답을 좁힐 단서가 있기
때문일 수 있다(예: "~를 그린 영화의 감독은 어느 나라 사람인가?"에서
"감독"·"어느 나라"만으로도 답 후보 타입이 국적으로 좁혀진다). 검증:
질문에서 bridge 엔티티 언급(대개 `bridge_hop_title`이 질문에 그대로
등장한다)을 플레이스홀더로 마스킹한 뒤에도 Answer-hop only 성능이 유지되면
→ 질문의 나머지 부분만으로 이미 답이 좁혀진다는 뜻으로, 가설 3을 지지한다.

**가설 4(원인 규명 — 문단 자체의 자기완결성):** 반대로, 성능 유지의 원인이
질문이 아니라 **answer_hop 문단 자체**가 이미 자기완결적이기 때문일 수
있다(문단 안에 bridge 엔티티가 다시 언급되면서 문맥이 완성되는 경우).
검증: `answer_hop_text` 안에 `bridge_hop_title`이 언급되는 샘플과 안 되는
샘플로 나눠 Answer-hop only 성능을 비교 — 언급되는 그룹에서만 성능이
유지된다면 가설 4를 지지한다. 가설 3·4는 상호 배타적이지 않다(둘 다 부분적
원인일 수 있다) — 두 그룹을 교차해서(질문에 언급/문단에 언급의 2x2) 어느
쪽이 더 강한 예측 인자인지 본다.

**가설 5(원인 규명 — 사전학습 지식과의 상호작용):** Bridge-hop only에서
나오는 "확신도 높은 그럴듯한 오답"(가설 2)이 무작위가 아니라, BERT가
사전학습 때 암기한 사실 지식을 끌어다 쓰는 것일 수 있다(예: bridge 문단에
이름만 나와도 사전학습 지식으로 국적을 맞히는 경우). 검증: Bridge-hop only에서
정답(또는 정답과 같은 개체)을 맞춘 샘플들이, 코퍼스 전체에서 자주 언급되는
"유명" 엔티티에 편중되는지 빈도 분석으로 확인한다.

## Oracle 설정 (스코프 결정)

HotpotQA distractor 설정은 질문당 문단 10개(정답 관련 2개 + 방해 문단 8개)를
준다. 전부 이어 붙이면 평균 1,000~1,500 토큰으로 `bert-base`의 512 토큰
한계를 넘겨 truncation이 발생한다. 별도의 문단 랭커(reranker)를 두지 않는
한, truncation 자체가 "bridge 정보 활용 여부"와 뒤섞이는 교란 요인이 된다.

그래서 이 프로젝트는 **처음부터 끝까지 방해 문단을 전혀 쓰지 않는
oracle 설정**으로 범위를 한정한다 — 세 조건 모두 golden 문단만으로
구성한다(Full = 정답 없는 문단 + 정답 있는 문단, 단일 조건들은 그중 하나만).
이렇게 하면 세 조건 간 유일한 차이는 "bridge 홉이 있는가"뿐이고,
truncation·문단 개수·문서 검색(retrieval) 성능 같은 다른 변수는 실험에
아예 등장하지 않는다. (문단 검색을 포함한 end-to-end 시스템 평가는
스코프 밖 — 아래 "스코프" 참고.)

## 데이터

**HotpotQA** (distractor 설정, HuggingFace `hotpotqa/hotpot_qa`) — 원래
`hotpot_qa`(네임스페이스 없는 레포)는 `datasets>=4`의 스크립트 기반 로딩
폐지로 실패하므로, 네임스페이스가 있는 미러 `hotpotqa/hotpot_qa`를 쓴다
([`pipeline/load_hotpotqa.py`](pipeline/load_hotpotqa.py)).

**필터링** (`type == "bridge"`인 샘플만; comparison형은 홉 구조가 다르고
답이 보통 yes/no라 스코프 밖):

1. `supporting_facts`가 정확히 서로 다른 2개 문단 제목에서만 나오는 샘플만
   유지(순수 2-hop).
2. 정답 문자열이 두 golden 문단 중 **정확히 하나에만** (대소문자 구분,
   substring) 등장하는 샘플만 유지 — 정답이 담긴 쪽을 `answer_hop`, 나머지를
   `bridge_hop`으로 태깅한다.
   - 정답이 **어느 쪽에도** 없으면 제외한다(답이 두 문단의 정보를 합성해야만
     나오는 경우 — extractive span 모델로는 애초에 학습이 불가능해서 스코프
     밖이다).
   - 정답이 **양쪽 다**에 있으면 제외한다(대명사·흔한 숫자·일반 명사의
     우연한 일치 — "bridge_hop에는 정답이 없다"는 가설 2의 전제 자체가
     깨지므로 반드시 걸러야 한다).

HF `train` split → `data/processed/train_pool.jsonl`(추후 우리가 직접
train/val로 재분할). HF `validation` split(공식 dev set, 정답 라벨 있음) →
`data/processed/test.jsonl`(3조건 평가에 쓸 최종 held-out set, 건드리지
않고 그대로 둔다).

**필터링 결과** (`data/processed/load_filter_stats.json`):

| split | 원본 | bridge 아님 | 순수 2-hop 아님 | 답 0곳 | 답 양쪽 | 최종 |
|---|---|---|---|---|---|---|
| train_pool | 90,447 | 17,456 | 0 | 0 | 19,902 | **53,089** |
| test | 7,405 | 1,487 | 0 | 0 | 1,184 | **4,734** |

이 데이터셋의 bridge형 샘플은 이미 전부 순수 2-hop이었다(3개 이상 문단에
걸친 경우 없음). "답이 양쪽에 있어서 제외"된 비율이 전체 bridge 샘플의
약 27%로 상당히 크다 — 흔한 단어·연도 등의 우연한 일치가 의외로 잦다는
뜻이며, 이 필터를 빼면 가설 2 검증이 처음부터 오염됐을 것이다.

## 평가 조건

같은 질문에 대해 context만 다르게 구성(전부 oracle — 방해 문단 없음):

| 조건 | Context 구성 | 정답이 context에 있는가 |
|---|---|---|
| Full | answer_hop + bridge_hop | 있음 |
| Answer-hop only | answer_hop만 | 있음 |
| Bridge-hop only | bridge_hop만 | **없음(설계상)** |

## 파이프라인

번호가 아니라 `pipeline/` 안에서 이 표의 순서대로 실행한다(실행 순서는
아래 "사용법"의 `run_pipeline.sh` 참고).

| 순서 | 스크립트 | 내용 | 상태 |
|---|---|---|---|
| 1 | [`pipeline/load_hotpotqa.py`](pipeline/load_hotpotqa.py) | HotpotQA 로드, bridge형·순수 2-hop 필터링, answer_hop/bridge_hop 태깅. | 완료 |
| 2 | [`pipeline/build_eval_conditions.py`](pipeline/build_eval_conditions.py) | test set의 각 질문마다 Full/Answer-hop only/Bridge-hop only 3조건 context를 생성. | 완료 |
| 3 | [`pipeline/split_dataset.py`](pipeline/split_dataset.py) | train_pool을 train/val로 분할(qid 단위). | 완료 |
| 4 | [`pipeline/analyze_lengths.py`](pipeline/analyze_lengths.py) | 토큰화된 (question, context) 길이 분포 확인, max_length 결정. | 완료 |
| 5 | [`pipeline/train_bert.py`](pipeline/train_bert.py) | `bert-base-cased`를 SQuAD 스타일 extractive QA로 파인튜닝. **Full 조건 train 데이터로만 학습** — shortcut 여부 조작은 테스트 단계에서만. | 완료 |
| 6 | [`pipeline/evaluate_conditions.py`](pipeline/evaluate_conditions.py) | 학습된 모델을 test set의 3조건 각각에 대해 추론, EM/F1 + 예측 span의 confidence(softmax) 기록. | 완료 |
| 7 | [`pipeline/confidence_bias_analysis.py`](pipeline/confidence_bias_analysis.py) | 가설 2 전용: Full/Answer-hop-only 대비 Bridge-hop-only의 confidence 하락폭 분포, 예측 span의 엔티티 타입이 기대 답 타입과 맞는 비율. | 완료 |
| 8 | [`pipeline/error_taxonomy.py`](pipeline/error_taxonomy.py) | **핵심 결과.** Answer-hop-only가 Full과 동일하게 맞춘 샘플(질문의 타입 제약, n-gram 단서)과 Bridge-hop-only 오답(타입 일치 여부)을 유형화 + 케이스 스터디. | 완료 |
| 9 | [`pipeline/question_masking_probe.py`](pipeline/question_masking_probe.py) | 가설 3 전용: 질문에서 bridge 엔티티 언급을 마스킹한 뒤 Answer-hop only를 재평가, 마스킹 전후 EM/F1 비교. | 완료 |
| 10 | [`pipeline/self_containment_split.py`](pipeline/self_containment_split.py) | 가설 4 전용: `answer_hop_text`에 `bridge_hop_title` 언급 여부로 그룹을 나눠 Answer-hop only 성능 비교(가설 3과 2x2 교차). | 완료 |
| 11 | [`pipeline/fame_bias_analysis.py`](pipeline/fame_bias_analysis.py) | 가설 5 전용: Bridge-hop only에서 정답을 맞힌 샘플의 정답 엔티티가 코퍼스 전체에서 얼마나 자주 언급되는지(유명도 프록시) 빈도 분석. | 완료 |

진단 이후의 완화(mitigation) 실험(shortcut을 실제로 줄이려는 시도, 성공/실패
포함)은 [`docs/mitigation-experiment.md`](docs/mitigation-experiment.md) 참고.

## 실험 설계 원칙

- **Oracle 설정으로 변수를 하나로 좁힌다.** 방해 문단을 아예 안 써서
  truncation·검색 성능이라는 다른 변수가 섞이지 않게 한다(위 "Oracle 설정"
  참고).
- **Bridge-hop only는 EM/F1으로 채점하지 않는다.** 정답이 context에 없으므로
  EM/F1은 정의상 0에 수렴하도록 설계돼 있다 — 이 조건에서 의미 있는 지표는
  confidence와 예측 span의 편향(엔티티 타입 일치 여부)이다(가설 2, 7단계).
- **양쪽 문단에 정답이 우연히 겹치는 샘플은 제외한다.** bridge_hop에 정답
  문자열이 우연히 들어 있으면 "bridge에는 정답이 없다"는 전제 자체가
  깨지므로, 1단계 필터링에서 반드시 걸러낸다.
- **학습은 Full 조건으로만, shortcut 조작은 테스트에서만.** 표준적인
  학습 절차를 그대로 쓰고, "어느 홉을 보여주는가"는 오직 평가 단계의
  변수로만 조작한다(5~6단계).
- **재현 연구임을 숨기지 않는다.** 이 현상 자체는 선행 연구에 이미
  보고돼 있다. 이 프로젝트의 기여는 "존재를 증명"하는 데 있지 않고, 8단계
  에러 분석과 9~11단계 원인 진단에서 shortcut이 통하는/안 통하는 조건과
  그 이유를 구체적으로 유형화하는 데 있다.
- **질문 마스킹은 삭제가 아니라 플레이스홀더 치환.** 가설 3 검증(9단계)에서
  bridge 엔티티 언급을 질문에서 그냥 지우면 질문 길이·구문 구조가 깨져서
  "정보가 없어져서"가 아니라 "문장이 어색해져서" 성능이 떨어질 수 있다.
  `[ENTITY]` 같은 플레이스홀더로 치환해 길이와 구문은 보존하고 엔티티
  정보만 제거한다.

## 환경 설정

이 프로젝트는 이 기기의 다른 프로젝트와 분리된 자체 venv를 쓴다. 로컬
드라이버에 맞는 CUDA 빌드의 torch가 필요하기 때문이다:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install transformers datasets scikit-learn numpy accelerate
.venv/bin/python -m pip install -e . --no-deps
```

(`pip install torch`만 실행하면 최신 CUDA 빌드를 받아온다 — 작성 시점 기준
CUDA 13 — 그런데 이게 오래된 드라이버에서는 `torch.cuda.is_available()`이
조용히 실패한다. 드라이버가 실제로 지원하는 CUDA 빌드로 고정할 것;
`nvidia-smi`로 드라이버가 지원하는 최대 CUDA 버전을 확인할 수 있다.)

마지막 줄은 이 저장소 자체의 [`src/multihop_shortcut/`](src/multihop_shortcut)
패키지를 editable 모드로 설치한다 — 이 덕분에 `pipeline/`·`scripts/mitigation/`·
`eval/`의 모든 진입점 파일이 동일한 경로/IO 헬퍼를 중복 정의하는 대신
`from multihop_shortcut import ...`로 가져다 쓸 수 있다.

## 디렉터리 구조

```
.
├── pyproject.toml          # editable 패키지 설치 설정 (src 레이아웃)
├── README.md
├── .gitignore
├── run_pipeline.sh          # pipeline/ 실행 순서(아래 표와 동일)
├── run_mitigation.sh        # scripts/mitigation/, eval/ 실행 순서
├── src/
│   └── multihop_shortcut/  # 여러 진입점이 공유하는 라이브러리 코드
│       ├── __init__.py
│       ├── paths.py              # ROOT 및 data/models 하위 경로 상수
│       ├── io_utils.py           # JSONL 읽기/쓰기 (load_jsonl, save_jsonl)
│       ├── constants.py          # 베이스 모델명, hop 라벨 상수
│       ├── metrics.py            # SQuAD 스타일 EM/F1 (normalize_answer 포함)
│       ├── qa_training.py        # 추출형 QA 학습 공용 (Dataset, feature 변환)
│       ├── classifier_training.py# 이진/다중 분류기 학습 공용
│       ├── inference.py          # 배치 추론 (QA·분류기 공용)
│       └── typing_heuristics.py  # 규칙 기반 answer 타입 분류(가설 2/5용)
├── pipeline/                # 진단 연구 진입점 — run_pipeline.sh 순서대로 실행
│   ├── load_hotpotqa.py              ┐
│   ├── build_eval_conditions.py      │
│   ├── split_dataset.py              │ 데이터 준비
│   ├── analyze_lengths.py            │
│   ├── train_bert.py                 ┘ 학습(Full 조건만)
│   ├── evaluate_conditions.py        ┐
│   ├── confidence_bias_analysis.py   │ 가설 1·2 평가
│   ├── error_taxonomy.py             ┘ (핵심 결과)
│   ├── question_masking_probe.py     ┐
│   ├── self_containment_split.py     │ 가설 3·4·5 원인 진단
│   └── fame_bias_analysis.py         ┘
├── scripts/mitigation/      # 완화 실험 — 기법별 하위 폴더, 각각 build_data→train→evaluate
│   ├── adversarial_training/            (A, 채택된 해결책)
│   ├── bridge_relatedness_classifier/   (B-v1)
│   └── counterfactual_classifier/       (B-v3)
├── eval/                    # 완화 실험의 탐색적 후속 실험(정식 파이프라인 아님, B-v4)
├── docs/
│   └── mitigation-experiment.md  # 완화 실험 설계·결과 전체 기록
├── data/                    # 파이프라인 입출력 (raw/processed/splits/errors)
└── models/                  # 파인튜닝 체크포인트 (.gitignore로 추적 제외)
```

각 진입점 파일(`pipeline/*.py`, `scripts/mitigation/*/*.py`, `eval/*.py`)은
재사용 로직을 담는 곳이 아니라 `src/multihop_shortcut/`를 불러와 실행만
하는 얇은 진입점이다. 여러 곳에서 같은 코드가 필요해지면 그 로직은
`src/multihop_shortcut/`에 추가한다. 파일명에 실행 순서를 숫자로 박아넣지
않는다 — 순서는 `run_pipeline.sh`/`run_mitigation.sh`와 파일명 자체의
동사(`build_data`→`train`→`evaluate`)가 말해준다.

## 사용법

```bash
./run_pipeline.sh
```

개별 실행:

```bash
.venv/bin/python pipeline/load_hotpotqa.py
.venv/bin/python pipeline/build_eval_conditions.py
.venv/bin/python pipeline/split_dataset.py
.venv/bin/python pipeline/analyze_lengths.py
.venv/bin/python pipeline/train_bert.py
.venv/bin/python pipeline/evaluate_conditions.py
.venv/bin/python pipeline/confidence_bias_analysis.py
.venv/bin/python pipeline/error_taxonomy.py
.venv/bin/python pipeline/question_masking_probe.py
.venv/bin/python pipeline/self_containment_split.py
.venv/bin/python pipeline/fame_bias_analysis.py
```

완화 실험은 `./run_mitigation.sh` — 자세한 설계·결과는
[`docs/mitigation-experiment.md`](docs/mitigation-experiment.md) 참고.

`train_bert.py`·`evaluate_conditions.py`·`question_masking_probe.py`는 GPU가
필요하고 `CUDA_VISIBLE_DEVICES=1`을 코드 안에서 고정한다(이 기기에 GPU가
2개 있고, 1번을 쓰기로 했다 — 필요하면 스크립트 상단의
`os.environ.setdefault(...)` 줄을 바꾼다).

## 결과

`bert-base-cased`를 Full 조건 train 47,780건으로 3 epoch 파인튜닝했다
(train loss 0.641 → eval loss 0.725, val loss 기준 best checkpoint 선택).
Test set 4,734건에 3조건 평가를 돌린 결과는 다음과 같다.

### 조건별 비교 (`data/errors/condition_comparison.json`)

| 조건 | EM | F1 | 평균 confidence | 평균 cls_prob |
|---|---|---|---|---|
| Full | 0.5862 | 0.7597 | 0.6905 | 0.0001 |
| Answer-hop only | **0.6158** | **0.7670** | 0.7256 | 0.0001 |
| Bridge-hop only | 0.0046 | 0.1091 | 0.4541 | 0.0013 |

**가설 1은 강하게 지지된다.** Answer-hop only가 Full보다 오히려 EM +3.0pt,
F1 +0.7pt 더 높다 — bridge 문단을 더 준다고 도움이 되기는커녕, 정답 문단을
더 긴 context 속에서 찾아야 하는 부담만 늘어 성능이 살짝 낮아진다. 모델이
두 홉을 종합하는 게 아니라 사실상 답이 있는 쪽 문단만 활용한다는 뜻이다.

### 가설 2 — Confidence/Bias (`data/errors/confidence_bias_report.md`)

Bridge-hop only는 정답이 context에 없는데도 평균 confidence가 0.454에
그친다(Full 대비 하락폭 0.236에 불과). 샘플의 **42.7%는 confidence
하락폭이 0.1 이하**다. cls_prob(모델이 "답 없음"에 해당하는 [CLS] 위치에
준 확률)는 세 조건 모두 사실상 0(0.0001~0.0013)이라, 증거가 없어도 모델은
"모르겠다"는 신호를 거의 내지 않는다. Bridge-hop only 오답 중
**69.5%는 정답과 같은 coarse 타입**(고유명사/숫자/날짜)을 골랐다 — 무작위
추측이 아니라 그럴듯한 오답을 고르는 구조적 편향으로 해석된다.

### 핵심 결과 — 에러 taxonomy (`data/errors/error_taxonomy_report.md`)

Full을 맞춘 2,775건 중 **90.6%(2,515건)는 Answer-hop only로도 그대로
맞혔다**(`shortcut_success`). 나머지 9.4%(260건)만 bridge 문단이 실제로
필요했다(`bridge_needed`).

| 그룹 | n | 질문 타입-제약 단서 비율 | 질문-문단 단어 중첩(평균) | bridge 제목이 질문에 | bridge 제목이 answer_hop에 |
|---|---|---|---|---|---|
| shortcut_success | 2,515 | 25.1% | 0.433 | 50.7% | 20.6% |
| bridge_needed | 260 | 11.9% | 0.424 | 26.5% | 56.2% |

### 가설 3 — 질문 마스킹 (`data/errors/question_masking_report.md`)

질문에 bridge 엔티티 이름이 문자 그대로 등장하는 서브셋(2,223건, 47.0%)에서
그 이름을 `[ENTITY]`로 마스킹해도 Answer-hop only의 EM은 0.646 → 0.641
(하락폭 0.005), F1은 0.804 → 0.800(하락폭 0.003)으로 거의 변화가 없다.
**가설 3을 강하게 지지한다** — 모델은 질문 속 구체적인 bridge 엔티티 이름을
몰라도, 질문의 나머지 어휘(타입 제약)만으로 answer_hop 문단에서 답을
찾아낸다.

### 가설 4 — 문단 자기완결성 (`data/errors/self_containment_report.md`)

예상과 반대로 나왔다. bridge 제목이 answer_hop 문단 안에 재언급되는
그룹의 Answer-hop only EM(0.780)이, 언급되지 않는 그룹(0.946)보다 오히려
**낮았다**. 즉 "문단이 bridge 엔티티를 다시 언급해서 자기완결적이 된다"는
가설 4의 메커니즘은 이 데이터에서 지지되지 않는다 — 오히려 bridge 제목이
answer_hop에 재언급되는 경우는(위 taxonomy에서 `bridge_needed` 그룹의
56.2%를 차지했듯) 정말로 두 홉을 종합해야 하는 어려운 문제와 상관관계가
있어 보인다. 가설 3(질문 마스킹)의 신호가 훨씬 강했다는 점에서, shortcut의
주된 원인은 문단의 자기완결성보다는 **질문 자체의 타입 제약**으로 잠정
결론짓는다.

### 가설 5 — 유명도 편향 (`data/errors/fame_bias_report.md`)

Bridge-hop only인데도 정답을 정확히 맞힌 사례가 22/4,734건(0.46%) 있었다
— bridge_hop_text에는 정답 문자열이 없도록 이미 필터링했으므로, 이 22건은
문맥 복사로는 설명되지 않는다. 다만 "코퍼스 전체에서 bridge 엔티티가 다른
질문의 context로 얼마나 자주 등장하는가"를 유명도 프록시로 썼을 때, 이
22건의 평균 유명도(2.27)는 나머지(3.24)보다 오히려 **낮았다** — 이
프록시로는 가설 5가 지지되지 않는다(n=22로 표본도 작다). 케이스 스터디를
직접 보면 여러 건은 순수 암기라기보다 bridge_hop_text 안의 표현을 다른
말로 바꿔 쓴 것에 가깝다(예: 질문 자체에 답의 일부 단어가 이미 들어있는
경우) — 유명도 프록시 자체의 한계일 수 있다. 실제 "사전학습 지식 사용
여부"를 가리려면 별도 방법(예: context 자체를 아예 안 주고 질문만으로
zero-shot 추론)이 필요하다(아래 "향후 과제").

## 스코프

의도적으로 제외했고, 향후 과제로 남겨둔 것들:
- **문단 검색(retrieval)/랭킹.** distractor 문단을 포함한 end-to-end
  시스템 평가는 하지 않는다 — 이 프로젝트는 "모델이 golden 문단이 주어졌을
  때 실제로 두 홉을 종합하는가"만 본다(위 "Oracle 설정" 참고).
- comparison형 질문(다른 홉 구조, yes/no 답).
- 3개 이상 문단에 걸친 supporting_facts 샘플(순수 2-hop만 다룬다).

## 향후 과제

- **가설 5 재검증.** 코퍼스 등장 빈도 기반 fame 프록시는 신호가 약했다(오히려
  방향이 반대). context를 아예 주지 않고 질문만으로 zero-shot 추론시켜
  "모델이 애초에 이 사실을 아는가"를 직접 물어보는 방식이 더 직접적인
  검증이 될 것이다.
- **가설 4의 반직관적 결과를 더 파보기.** bridge 제목이 answer_hop에
  재언급되는 그룹이 왜 오히려 더 어려운 그룹과 겹치는지(질문 유형·난이도
  `level`과의 상관관계 등) 추가 분석이 필요하다.
- entity-type 판별에 규칙 기반 휴리스틱(`typing_heuristics.py`) 대신 별도
  NER 모델이 필요한지 재검토.
- Oracle 설정을 벗어나 실제 문단 검색(retrieval)을 포함한 end-to-end
  평가(스코프 밖으로 남겨둔 부분).
