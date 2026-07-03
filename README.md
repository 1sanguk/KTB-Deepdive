# KTB DeepDive

카카오테크부트캠프 딥다이브 스터디 저장소입니다.  
매 주차별로 주제를 선정하고, 해당 주제를 깊이 있게 탐구합니다.

---

## 목차

| 주차 | 주제 | 언어 | 링크 |
|:----:|------|:----:|:----:|
| Project | 한국어 Mini-GPT 챗봇 (자체 GPT + RAG + LangChain + LangGraph) | LLM | [바로가기](LLM_Project/chatbot/README.md) |
| 08 | MCP Context Isolation (보안·권한·데이터 오염 문제 분석) | LLM | [바로가기](08/README.md) |
| 06 | Hybrid Search (Sparse + Dense Vector 검색) | Python | [바로가기](06/README.md) |
| 05 | 트랜스포머의 위치 인코딩 3가지 비교 | Python | [바로가기](05/README.md) |
| 04 | 데이터 전처리 방식에 따른 머신러닝 모델 성능 변화 | Python | [바로가기](04/README.md) |
| 03 | NumPy 배열의 생성과 연산 (브로드캐스팅 포함) | Python | [바로가기](03/README.md) |
| 02 | 이터레이터와 제너레이터의 메모리 관리 효율성 | Python | [바로가기](02/README.md) |

---

## 주차별 요약

### LLM_Project — 한국어 Mini-GPT 챗봇

> 자모(NFD) 단위 BPE 토크나이저부터 GPT 아키텍처, 학습, RAG, LangChain 파이프라인, LangGraph, FastAPI 서빙, LangSmith 트레이싱까지 전부 직접 구현한 한국어 챗봇 프로젝트.

**핵심 결론**

- 직접 구현한 GPT 디코더(~97M params, 12층)를 단계별로 학습: **Stage 1** 이어쓰기 → **Stage 2** Q&A 파인튜닝 → **Stage 4** 추출형 QA(정답 스팬 위치 분류) → **Stage 5** DPO 선호 학습(진행 중)
- 검색 방식에 따라 **Basic / RAG(TF-IDF) / LangChain(BM25+FAISS 하이브리드) / LangGraph** 4가지 모드를 제공하고, 검색 점수가 임계값 미만이면 Stage 2(잡담형) 모델로, 이상이면 Stage 4(추출형) 모델로 라우팅
- 하이브리드 검색(BM25+FAISS) 도입으로 라우팅 정확도가 TF-IDF 단독 73.3% → **82.7%**로 향상 (held-out 검증 기준)
- **LangGraph**로 별도의 상태 기반(StateGraph) 파이프라인도 구현 — 검색 실패 시 단순 폴백이 아니라 **노드 재시도(retry) 루프**를 거치며, `build_graph`(SOP_GPT) / `build_claude_graph` / `build_claude_agent_graph`(Claude Agent 도구 호출) 3가지 그래프를 노드 팩토리 8개(SOP_GPT 4 + Claude 4)로 조립
- LangChain LCEL로 체인을 표준화하고 FastAPI + SSE 스트리밍으로 서빙, 동일 검색기를 재사용해 SOP_GPT와 Claude API 응답을 분할화면으로 실시간 비교
- LangSmith로 체인 실행 트레이싱 자동 수집

| 모드 | 검색기 | 라우팅 임계값 | 미달 시 폴백 | 비고 |
|---|---|:--:|---|---|
| Basic | 없음 | - | Stage 2 직접 생성 | |
| RAG | TF-IDF + 코사인 유사도 | ≥ 0.25 | Stage 2 직접 생성 | |
| LangChain | BM25 + FAISS 하이브리드 | ≥ 0.515 | Stage 2 직접 생성 | LCEL 체인 |
| LangGraph | BM25 + FAISS 하이브리드 | ≥ 0.515 | Stage 2 직접 생성 | 노드 재시도 루프 + Claude Agent 도구 호출 지원 |

자세한 내용 → [LLM_Project/chatbot/README.md](LLM_Project/chatbot/README.md)

---

### Week 08 — MCP Context Isolation

> MCP의 Context Isolation이 AI Agent와 외부 도구·데이터 소스를 연결할 때 필요한 이유를 설명하고, 컨텍스트가 분리되지 않았을 때 발생할 수 있는 보안·권한·데이터 오염 문제를 구체적으로 분석하시오.

**핵심 결론**

- **MCP(Model Context Protocol)**는 LLM이 외부 도구·데이터와 통신하기 위해 Anthropic이 만든 JSON-RPC 기반 표준 규약. 도구마다 제각각인 연동 방식을 하나의 계층으로 통일한다.
- LLM은 **최신성·환각성·실행성**이라는 근본적 한계가 있어 외부 도구 연결이 필수인데, 이 연결 지점을 **Context Isolation**(데이터 분리, 권한 분리, 도구 분리, 실행 환경 분리)으로 걸러내지 않으면 문제가 생긴다.
- Context Isolation이 없을 때 발생하는 문제는 **보안·권한·데이터 오염** 세 갈래로 번지며, 셋 다 "무엇을 보여줄지"를 미리 걸러내지 못한 동일한 원인에서 파생된다.

| 구분 | 핵심 원인 | 대표 시나리오 |
|---|---|---|
| 보안 | 도구/데이터를 통해 들어온 텍스트를 지시로 오인 | 프롬프트 인젝션, Tool Poisoning, 데이터 유출 |
| 권한 | Agent의 권한과 사용자의 권한이 분리되지 않음 | 과도한 권한 부여, Confused Deputy, 세션 간 권한 누수 |
| 데이터 오염 | 컨텍스트·메모리가 출처·세션별로 분리되지 않음 | 컨텍스트 희석, 크로스 유저 오염, 캐시 포이즈닝 |

자세한 내용 → [08/README.md](08/README.md)

---

### Week 06 — Hybrid Search

> Hybrid Search가 Sparse Vector 검색과 Dense Vector 검색을 결합해 검색 품질을 높이는 원리를 설명하고, 키워드/의미 기반 검색 각각의 실패를 Hybrid Search가 어떻게 보완하는지 분석하시오.

**핵심 결론**

- **Sparse(BM25)**는 키워드 일치만 보기 때문에 동의어/의역에는 점수 0 (검색 결과에서 완전히 누락), 동음이의어에는 거짓 양성(false positive)을 낸다.
- **Dense(임베딩)**는 의미 유사도를 보기 때문에 동의어는 잡아내지만, 학습 데이터에 없는 고유 코드/식별자는 의미를 담지 못해 다른 문서에 밀린다.
- 두 점수는 스케일이 달라서(BM25는 상한 없음, 코사인 유사도는 0~1) 그냥 더하면 안 되고, **정규화(min-max) 또는 RRF(순위 기반 결합)** 를 거쳐야 한다.
- 단, 한쪽 점수의 격차가 극단적으로 크면 정규화 후 가중합도 틀릴 수 있다 — 이런 경우 점수 크기 대신 순위만 보는 **RRF**가 더 안정적이다.

| | Sparse (키워드) | Dense (의미) | Hybrid |
|---|---|---|---|
| 잘하는 것 | 고유명사, 코드, 정확한 키워드 | 동의어, 의역, 문맥적 의미 | 둘 다 |
| 실패하는 경우 | 동의어 매칭 실패(점수 0), 동음이의어 거짓 양성 | 고유 코드/식별자 의미 손실 | 한쪽 점수가 극단적으로 튀면 가중합도 실패 가능 → RRF 필요 |

자세한 내용 → [06/README.md](06/README.md)

---

### Week 05 — 트랜스포머의 위치 인코딩

> 트랜스포머의 위치 인코딩 방식 3가지(기본 Sinusoidal, BERT의 학습형 절대 위치, RoPE 상대 위치)를 설명하고 비교하시오.

**핵심 결론**

- 트랜스포머는 문장을 병렬로 처리해서 "몇 번째 단어인지" 정보가 입력 임베딩에 없다 → Positional Encoding 필요
- **Sinusoidal**: sin/cos로 직접 계산하는 고정 함수. 학습 파라미터 없음, 더 긴 문장에도 그대로 계산 가능
- **BERT 학습형**: 위치별 벡터를 테이블에서 조회(lookup). 성능은 좋지만 테이블 크기(`max_position_embeddings`)를 넘는 위치는 처리 불가
- **RoPE**: 임베딩에 더하지 않고 Query/Key 벡터를 위치만큼 회전. 내적 결과가 절대 위치가 아니라 "위치 차이(상대 위치)"에만 의존 → 긴 문맥 일반화에 유리

| | Sinusoidal (기본) | BERT (학습형 절대 위치) | RoPE (상대 위치) |
|---|---|---|---|
| 학습 파라미터 | 없음 | 있음 (테이블) | 없음 |
| 더 긴 문장 처리 | 가능 | 불가능 (테이블 크기 고정) | 가능 |
| 표현하는 위치 정보 | 절대 위치 | 절대 위치 | 상대 위치 (명시적) |

자세한 내용 → [05/README.md](05/README.md)

---

### Week 04 — 데이터 전처리 방식에 따른 성능

> 머신러닝 모델이 동일한 데이터셋에서 전처리 방식(정규화, 표준화, 결측치 처리)에 따라 성능이 어떻게 달라질 수 있는지 사례를 들어 설명하시오.

**핵심 결론**

- PUBG 데이터(SVM, 클래스 불균형 포함)로 직접 실험: 결측치 제거만 했을 때 정확도 0.8667
- **정규화(MinMaxScaler)는 이상치에 민감**해서 오히려 결측치 제거보다 낮은 0.8000이 나옴 — 상위 플레이어(이상치)가 나머지 데이터를 좁은 범위로 몰아넣기 때문
- **표준화(StandardScaler)는 이상치에 상대적으로 강건** → 0.8889
- 전체 기준 IQR을 적용하면 소수 클래스가 전부 이상치로 삭제될 수 있어 **클래스별 IQR**이 필요 → 이상치 제거 + 표준화로 정확도 1.0000까지 향상
- SMOTE로 데이터를 증강해도 accuracy는 오히려 낮아질 수 있음 — accuracy 자체가 불균형 데이터에 적합하지 않은 지표이기 때문 (recall/F1-score로 봐야 함)

| 전처리 방식 | 정확도 |
|---|:---:|
| 결측치 제거 | 0.8667 |
| 결측치 제거 + 정규화 | 0.8000 |
| 결측치 제거 + 표준화 | 0.8889 |
| 이상치 제거(클래스별) + 표준화 | **1.0000** |

자세한 내용 → [04/README.md](04/README.md)

---

### Week 03 — NumPy 배열의 생성과 연산

> NumPy 배열의 생성과 연산이 데이터를 어떻게 처리하는지 설명하시오 (브로드캐스팅 포함)

**핵심 결론**

- Python List는 **객체 주소값 배열**, NumPy는 **연속 원시 데이터 버퍼** — 메모리 구조가 근본적으로 다름
- **벡터화(Vectorization)**: Python 루프를 건너뛰고 CPU SIMD 명령어로 다수 원소를 한 번에 처리
- **슬라이싱**: NumPy는 실제 복사 없이 `strides` 메타데이터만 변경 → 메모리 절약
- **브로드캐스팅**: shape이 다른 배열도 메모리 복사 없이 확장 연산 가능 (규칙: 오른쪽 정렬 후 크기가 같거나 1이어야 함)

| 항목 | Python List | NumPy ndarray |
|------|:-----------:|:-------------:|
| 메모리 구조 | 객체 주소값 배열 | 연속 원시 데이터 버퍼 |
| 연산 방식 | 인터프리터 루프 | SIMD 벡터화 (C레벨) |
| 슬라이싱 | 실제 데이터 복사 | strides 메타데이터만 변경 |
| 브로드캐스팅 | 미지원 | 지원 (메모리 복사 없음) |

자세한 내용 → [03/README.md](03/README.md)

---

### Week 02 — 이터레이터와 제너레이터

> 이터레이터와 제네레이터가 메모리 공간 효율성을 개선하는 방식을 설명하고,  
> 대규모 데이터 처리(예: 로그 파일 분석)에서 이를 활용하는 구체적인 시나리오를 제시하시오.

**핵심 결론**

- 리스트 대신 제너레이터를 사용하면 메모리를 **약 96% 절약**
- 처리 속도도 **약 2배 향상** (대규모 CSV 벤치마크 기준)
- 즉시 평가(Eager) vs **지연 평가(Lazy Evaluation)** 의 차이가 핵심

| 방식 | 메모리 사용량 | 처리 시간 |
|------|:------------:|:--------:|
| 리스트 기반 | ~595 MB | 10.0 sec |
| 제너레이터 (iter 래핑) | ~24 MB | 5.3 sec |
| 제너레이터 (순수) | ~23 MB | **4.7 sec** |

자세한 내용 → [02/README.md](02/README.md)

---

## 저장소 구조

```
KTB-DeepDive/
├── LLM_Project/       # 한국어 Mini-GPT 챗봇 (GPT 아키텍처 + RAG + LangChain, 별도 프로젝트)
│   └── chatbot/
│       ├── README.md
│       ├── version.md
│       ├── images/           # 아키텍처 다이어그램
│       ├── ragdata/          # RAG 검색용 커스텀 문서
│       └── source/
│           ├── app/          # FastAPI 서빙 (chat.py, stream.py, ui.py)
│           ├── lc/           # LangChain 통합 레이어 (retriever, llm, chain)
│           ├── lg/           # LangGraph 파이프라인 레이어
│           ├── rag/          # TF-IDF 검색기
│           └── model/        # BPE 토크나이저, GPT 모델, 학습 루프, 체크포인트
├── 02/               # Week 02 — 이터레이터 & 제너레이터
│   ├── README.md
│   ├── example1.py   # 기본 크기 비교
│   ├── example2.py   # 참조 객체 포함 크기 비교
│   ├── example3.py   # 제너레이터 표현식
│   ├── example4.py   # 스트리밍 파이프라인
│   └── example5.py   # 대규모 CSV 처리 벤치마크
├── 03/               # Week 03 — NumPy 배열 생성과 연산
│   └── README.md
├── 04/               # Week 04 — 데이터 전처리 방식에 따른 성능
│   ├── README.md
│   └── example.py
├── 05/               # Week 05 — 트랜스포머의 위치 인코딩
│   ├── README.md
│   └── example.py
├── 06/               # Week 06 — Hybrid Search
│   ├── README.md
│   └── example.py
└── 08/               # Week 08 — MCP Context Isolation
    └── README.md
```
