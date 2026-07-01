# 버전 기록

날짜별로 무엇이 바뀌었는지 정리한 문서입니다.

- [2026-07-01 — Claude API 연동 + 스트리밍 + 분할화면 UI + app.py 모듈화 + RPG 문서](#2026-07-01--claude-api-연동--스트리밍--분할화면-ui--apppy-모듈화--rpg-문서)
- [2026-06-30 — AI Hub 대화 데이터 추가 + RAG 확장](#2026-06-30--ai-hub-대화-데이터-추가--rag-확장)
- [2026-06-28 — 모델 확장 재학습 + LangSmith 트레이싱 (3번 과제)](#2026-06-28--모델-확장-재학습--langsmith-트레이싱-3번-과제)
- [2026-06-27 — LangChain 전체 파이프라인 마이그레이션 (1번·2번 과제 완료)](#2026-06-27--langchain-전체-파이프라인-마이그레이션-1번2번-과제-완료)
- [2026-06-25 — RAG 검색기 LangChain 마이그레이션](#2026-06-25--rag-검색기-langchain-마이그레이션)
- [2026-06-20 — RAG 아키텍처 적용 (6주차 위클리 챌린지)](#2026-06-20--rag-아키텍처-적용-6주차-위클리-챌린지)
- [2026-06-14 — 최초 구현 (5주차 위클리 챌린지)](#2026-06-14--최초-구현-5주차-위클리-챌린지)
- [회고](#회고)

---

## 2026-07-01 — Claude API 연동 + 스트리밍 + 분할화면 UI + app.py 모듈화 + RPG 문서

### `/chat/langchain` segfault 수정

- `app.py` 최상단에 `os.environ["TOKENIZERS_PARALLELISM"] = "false"` + `OMP_NUM_THREADS = "1"` 추가
- 원인: uvicorn 요청 처리 중 FAISS가 쿼리 임베딩 시 HuggingFace 토크나이저가 멀티프로세싱 세마포어를 잘못 처리 → segfault
- 환경변수로 병렬 토크나이저를 비활성화해 해결. `/chat/basic`, `/chat/rag`는 정상이었으나 `/chat/langchain`만 임베딩 추론을 실시간으로 호출하는 구조였기 때문

### QA LLM 생성 파라미터 조정

| 파라미터 | 변경 전 | 변경 후 | 이유 |
|---|---|---|---|
| `stop_on` | `"line"` | `"sentence"` | `\n` 대신 `.?!`에서 멈춤 — 학습 데이터 포맷의 `질문:` 반복을 방지 |
| `temperature` | 0.8 | 0.7 | 더 일관성 있는 출력 |
| `top_p` | 없음 | 0.9 | nucleus sampling 추가 |
| `max_new_tokens` | 60 | 250 | 더 긴 답변 허용 |

`stop_on="none"`으로 설정하면 모델이 학습 데이터 포맷(`"질문: ...\n답변: ..."`)을 무한 반복 생성하는 현상을 확인. `"sentence"`가 실용적 최적값.

### Claude API 연동 (`source/lc/claude_llm.py` 신설)

- `claude-haiku-4-5-20251001` 모델 사용 (가장 저렴하고 빠른 티어, $0.80/1M input)
- `ask_claude(question)`: RAG 없이 Claude가 직접 답변
- `ask_claude_with_context(question, context)`: 검색된 문서를 참고해 Claude가 답변
- `stream_claude(question, context="")`: 비동기 스트리밍 제너레이터 (매 yield마다 누적 텍스트)
- `build_claude_rag_chain(retriever, threshold)`: 기존 retriever 재사용, 점수에 따라 컨텍스트 제공 여부 결정
- Claude 엔드포인트 3개 추가: `POST /chat/claude/basic`, `/chat/claude/rag`, `/chat/claude/langchain`

### 분할화면 UI 개편

- 기존: 홈 화면에 SOP 카드 3개 + Claude 카드 3개 총 6개
- 변경: 홈 화면에 카드 3개만 (모드 선택), 입장 후 좌/우 분할화면
  - 왼쪽: SOP_GPT 응답
  - 오른쪽: Claude (Haiku) 응답
- `.app` 최대 너비 760px → 1100px로 확장
- 메시지 전송 시 두 패널이 독립적으로 각자 요청을 시작, 먼저 끝난 쪽이 먼저 표시됨

### SSE 스트리밍 구현

**`source/model/model.py`** — `SOP_GPT.generate_stream()` 추가:
- 기존 `generate()`와 동일한 로직이지만 매 토큰 생성마다 `yield next_id.item()`

**`source/lc/llm.py`** — `SOP_GPT_LLM.stream_tokens()` 추가:
- 동기 제너레이터. 토큰을 생성할 때마다 `decode(accumulated_ids, itos).strip()`으로 현재까지 디코딩된 전체 텍스트를 yield
- BPE NFD 특성상 토큰 하나씩 delta를 보내면 부분 한글이 깨질 수 있어, 전체 누적 텍스트를 yield하는 방식 선택

**`source/app/app.py`** — SOP 스트리밍 3개 + Claude 스트리밍 3개 엔드포인트 추가:
- `POST /chat/{basic|rag|langchain}/stream`
- `POST /chat/claude/{basic|rag|langchain}/stream`
- SSE 포맷: `data: {"type": "text"|"rag_context"|"done", "text": "..."}` 형식
- `_sop_stream()`: 동기 제너레이터를 `threading.Thread` + `queue.Queue` + `asyncio.run_in_executor()`로 감싸 비동기 FastAPI에서 논블로킹으로 실행
- RAG 모드는 검색 결과를 먼저 `rag_context` 이벤트로 보낸 뒤 답변 스트리밍

**프론트엔드 JS**:
- `Promise.allSettled()` 방식 → 각 패널별 독립 `fetch` + `ReadableStream` 파서로 교체
- `streamPanelWith()`: SSE 청크를 버퍼링해서 `\n\n` 기준으로 파싱, `evt.type`에 따라 텍스트 업데이트 또는 RAG 라벨 추가
- `pendingCount` 카운터로 두 패널 모두 완료 시 전송 버튼 재활성화

### `source/app/app.py` 622줄 → 모듈 분리

단일 파일이었던 `app.py`를 역할별로 6개 파일로 분리. 기능은 동일하게 유지.

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI 선언 + 라우터 등록 (30줄로 축소) |
| `state.py` | 모델·LLM·체인·검색기 초기화 (import 시점 1회 실행) |
| `models.py` | Pydantic 스키마 4개 |
| `streaming.py` | SSE 헬퍼 (`_sse`, `sop_stream`, `sop_rag_stream`, `claude_rag_stream`) |
| `ui.py` | `GET /` 엔드포인트 + 전체 HTML 문자열 |
| `routers/chat.py` | non-streaming 엔드포인트 7개 |
| `routers/stream.py` | SSE 스트리밍 엔드포인트 6개 |

**sys.path 순서 의존성**: `app.py`가 먼저 `sys.path.insert()`를 실행하고 나서 `import state`를 해야 한다. `routers/*.py`도 `state`를 import하지만 이 시점에는 이미 path가 등록되어 있다.

### `ragdata/rpg.md` 추가

- RPG(롤플레잉 게임) 한국어 설명 문서 신설 (자체 작성, GitHub 공개 가능)
- 포함 섹션: RPG 정의·특징·종류(JRPG/ARPG/MMORPG/TRPG/로그라이크/오픈월드)·역사·주요 용어·한국 RPG 문화
- `ragdata/.gitkeep` 삭제, 서버 재기동 시 `load_ragdata_passages()`가 자동으로 TF-IDF + LangChain 인덱스에 포함
- `.cache/` 전체 삭제 → 다음 기동 시 새 문서 포함해 재빌드

### README.md · basicdata/info.md 이미지 삽입

| 이미지 | 삽입 위치 |
|---|---|
| `images/basic_gpt.png` | README `## 개요` 아래 / info.md `source/app/` 섹션 시작부 |
| `images/rag_gpt.png` | README `## RAG 아키텍처` 아래 / info.md `TfidfRetriever` 설명 앞 |
| `images/langchain_gpt.png` | README `## LangChain 파이프라인 구조` 아래 / info.md `retriever.py` 섹션 시작부 |

---

## 2026-06-30 — AI Hub 대화 데이터 추가 + RAG 확장

### 학습 데이터 확장 (Stage 1)

- **AI Hub 한국어 대화 데이터 4종 추가**: 011(일상대화 멀티세션) / 020(주제별 텍스트 일상 대화) / 141(멀티세션 대화) / 297(SNS 데이터 고도화)
  - 기존 챗봇 데이터(songys/Chatbot_data, 12K쌍 감성 대화 위주)의 한계를 보완하기 위해 일상 대화 중심의 대규모 데이터 추가
  - `tokenizer.py`에 `load_aihub_conversation_data()` 추가: 압축 해제된 txt/json 파일을 직접 읽어 처리
  - `main.py` Stage 1 학습 데이터에 AI Hub 대화 텍스트 병합
  - 출처: 한국지능정보사회진흥원(NIA) AI Hub — 인공지능 학습 목적으로만 사용

### RAG 검색 범위 확장

- **KorQuAD train set 포함**: `build_tfidf_retriever` / `build_hybrid_retriever` 모두 `include_train=True`로 변경 — dev만(5,774쌍) → train+dev(~61K쌍)로 검색 커버리지 확대
- **`ragdata/` 폴더 신설**: 앱 시작 시 해당 폴더의 `.txt`/`.md` 파일을 자동으로 읽어 RAG 검색 인덱스에 포함. 도메인 특화 문서를 파일만 넣으면 재학습 없이 검색 범위 확장 가능

---

## 2026-06-28 — 모델 확장 재학습 + LangSmith 트레이싱 (3번 과제)

### 모델 아키텍처 확장 및 재학습

- **모델 하이퍼파라미터 확장**: `n_embd 256→512`, `n_head 4→8`, `n_layer 6→12`, `block_size 128→256` — 파라미터 수 약 600만 → 약 5천만으로 증가
- **RAM 절감 학습 최적화**: `batch_size 64→16` + `gradient accumulation(accum_steps=4)` — 유효 배치 크기(64) 유지하면서 활성화 메모리 4배 절감
- **데이터 파이프라인 메모리 최적화**: `chatbot_text * 15` 문자열 업샘플링 → 텐서 인코딩 후 `tensor.repeat(15)` 로 변경 — Python 리스트(토큰당 ~28B) 중간 복사본 제거로 피크 RAM 대폭 감소, 인코딩 후 즉시 `del` + `gc.collect()` 적용
- **순차 학습 스크립트 `train_all.sh`**: Stage 1 → Stage 2 → Stage 4를 하나의 스크립트로 순차 실행, 타임스탬프 포함 로그 기록. `caffeinate -w PID`로 맥북 뚜껑 닫은 상태에서도 학습 유지
- **재학습 결과**:
  - Stage 1 (이어쓰기): val loss **3.355** (step 3860 early stop)
  - Stage 2 (Q&A): val loss **2.657** (step 1280 early stop)
  - Stage 4 (추출형 QA): val loss **3.092** (step 1660 early stop)

### LangSmith 트레이싱 연동

- **환경 설정 분리**: `.env`(비밀 아닌 설정: tracing on/off, APAC 엔드포인트, 프로젝트명) + `api_keys`(실제 키 값) 두 파일로 분리 보관, 둘 다 gitignore 처리
- **`app.py` 최상단 `load_dotenv`**: 모든 langchain import보다 먼저 실행해야 트레이싱 env var가 제때 인식됨 — `LANGSMITH_TRACING` + `LANGCHAIN_TRACING_V2` 두 변수 모두 설정(버전 호환)
- **APAC 엔드포인트 사용**: `https://apac.api.smith.langchain.com` — 기본 US 엔드포인트 대신 APAC으로 지정
- **트레이싱 확인**: `retrieve_and_answer`, `SOP_GPT_LLM` 체인 실행 기록이 LangSmith `adapterz-langchain-textbook` 프로젝트에 실시간 수집됨

### Dataset 기반 평가 (`source/evaluate.py`)

- **Dataset**: KorQuAD v1.0 dev set 30개 질문/정답을 LangSmith `sop-gpt-korquad`로 업로드
- **Evaluator**: `contains_match` — 정답 텍스트가 예측 답변에 포함되면 1점
- **결과**: 검색 없이는 KorQuAD 사실형 질문에 정답 포함률 0%, 검색을 붙이면 20%로 상승. 수치 및 해석은 [basicdata/eval.md](basicdata/eval.md) 참고

---

## 2026-06-27 — LangChain 전체 파이프라인 마이그레이션 (1번·2번 과제 완료)

> SOP_GPT 모델 자체를 LangChain `LLM`으로 래핑하고, LCEL(`|` 연산자)로 검색기와 LLM을 연결하는 체인을 조립했습니다. 서버를 3개 모드로 나누고, 파이프라인 전체를 스모크 테스트로 검증했습니다.

### LangChain LLM 래퍼 및 LCEL 체인 (`source/lc/`)

- **`source/lc/llm.py` 신설**: `SOP_GPT_LLM(LLM)` — PyTorch 모델을 LangChain `BaseLLM`으로 래핑. `stop_on="line"`(QA, `\n`에서 중단) / `stop_on="sentence"`(이어쓰기, `.?!`에서 중단) 두 가지 동작 모드. PyTorch Tensor를 Pydantic 필드로 두기 위해 `model_config = ConfigDict(arbitrary_types_allowed=True)` 적용
- **`source/lc/llm.py` — `make_span_extractor()`**: `SOP_GPT_Span`을 `RunnableLambda` 주입용 클로저로 래핑. `{"question": str, "context": str} → str` 인터페이스라 LangChain 표준 LLM(`str → str`)과 맞지 않아 별도 팩토리로 분리
- **`source/lc/chain.py` 신설**:
  - `build_basic_chain(llm)`: `{"question": RunnablePassthrough()} | QA_PROMPT | llm | StrOutputParser()` — LCEL `|` 연산자만으로 완성
  - `build_rag_chain(retriever, llm, span_fn, threshold)`: `RunnableLambda(retrieve_and_answer)` — 조건 분기(임계값 라우팅)가 있어 순수 `|` 체인으로 표현이 어렵고, `RunnableLambda`로 감싸되 반환값은 표준 Runnable로 유지
- **`source/lc/` 디렉터리명**: 원래 `source/langchain/`으로 지었다가 Python이 `source/`를 `sys.path`에 넣자 실제 `langchain` 패키지를 가려버리는 import 섀도잉 문제 발생 → `source/lc/`로 이름 변경

### TF-IDF 검색기 분리 (`source/rag/rag.py`)

- `source/rag/__init__.py`에 몰려있던 코드를 `source/rag/rag.py`로 이동 (`__init__.py`는 `from .rag import *`만 남김)
- **`TfidfRetriever` 클래스 추가**: `LangChain EnsembleRetriever`와 동일한 `retrieve(query)` / `best_match(query)` 인터페이스를 맞춰 `build_rag_chain()`에 그대로 꽂을 수 있도록 설계. 임계값 0.25(raw 코사인 유사도)
- `build_tfidf_retriever()` 팩토리 추가

### 3개 모드 FastAPI 서빙 (`source/app/app.py`)

- **엔드포인트 3개로 확장**: `/chat/basic`(검색 없음, QA LLM 직접) / `/chat/rag`(TF-IDF 체인) / `/chat/langchain`(LangChain 하이브리드 체인)
- **웹 UI**: 모드 선택 카드 화면 → 선택 후 채팅 화면으로 전환. 세 모드 공통 채팅 뷰를 재사용

### 스모크 테스트 (`source/test.py`)

- 7개 테스트: BPE 토크나이저 왕복, `SOP_GPT_LLM.invoke()`, `basic_chain`, TF-IDF 라우팅, `tfidf_rag_chain` (RAG/폴백 경로 각각), `HybridRetriever`, `lc_rag_chain`
- `--skip-hybrid` 플래그로 임베딩 모델 로드(1~2분)를 건너뛸 수 있음

### 의존성 확정

- `rank_bm25`: `langchain_community.BM25Retriever`가 내부적으로 요구하지만 `langchain-community` 설치 시 자동으로 딸려오지 않음 — 별도 설치 필요
- `faiss-cpu`: macOS에서는 `faiss-gpu` 미지원, `faiss-cpu`로 대체

---

## 2026-06-25 — RAG 검색기 LangChain 마이그레이션

> 손으로 짠 하이브리드 검색기(TF-IDF + ko-sroberta)를 LangChain 표준 컴포넌트로 교체하고, 서빙용 검색 코드를 모델/학습 코드와 디렉터리 단위로 분리했습니다.

- **새 디렉터리 `source/langchain_rag/` 신설**: `HybridRetriever` 클래스가 `BM25Retriever`(sparse) + `FAISS`(dense, ko-sroberta 임베딩) + `EnsembleRetriever`로 검색을 결합. `source/model/`(아키텍처/학습/체크포인트)은 건드리지 않고 서빙 시점 검색 로직만 이쪽으로 옮김
- **라우팅 임계값(0.515) 로직은 그대로 보존**: `EnsembleRetriever`는 RRF(rank fusion) 방식이라 질문 간 비교 가능한 절대 점수를 안 주기 때문에, `/chat`의 폴백 판단(`best_match`)은 BM25 원시 점수 + FAISS 코사인 유사도에 기존 `calibrate()` 고정-보정 정규화 산식을 그대로 포팅해서 유지
- **라이브러리 함정 발견**: 설치된 `langchain-community` 버전의 FAISS `distance_strategy=COSINE`이 `normalize_L2`를 무시하는 미구현 상태였음 → 정규화된 벡터 + 기본 EUCLIDEAN 전략(`_euclidean_relevance_score_fn`)으로 대체해 코사인 유사도와 단조 대응되는 점수를 얻음
- **`source/model/rag.py` 정리**: 검색 관련 함수(`build_index`/`calibrate`/`_hybrid_scores`/`retrieve`/`best_match`) 제거, 학습(`train_stage4`)이 쓰는 코퍼스 로딩 함수만 남김. `app.py`/`chat.py`/`main.py`의 5-tuple 호출부를 `HybridRetriever` 객체 하나로 단순화
- **검증**: 실제 KorQuAD 질문 8개 + 잡담 질문 3개로 라우팅 정확도 8/11 확인 — 기존 TF-IDF+임베딩 하이브리드의 held-out 정확도(82.7%)와 비슷한 수준
- 의존성 추가: `langchain`, `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `rank_bm25`

---

## 2026-06-20 — RAG 아키텍처 적용 (6주차 위클리 챌린지)

> 5주차 챗봇에 RAG(검색 증강 생성)를 적용한 개인 프로젝트입니다. 아래 세 단계를 순서대로 거쳤습니다.

### 1. RAG 도입: 검색 결과를 실제로 활용하는 Q&A로 발전

- **지식 베이스 추가**: KorQuAD v1.0 dev set(한국어 위키 기반 reading comprehension 데이터셋)을 새로 도입, `urllib`로 다운로드 후 ~90자 단위 청크로 분할
- **검색기 구현**: scikit-learn `TfidfVectorizer` + 코사인 유사도 기반 검색 모듈 `rag.py` 신규 작성 (추가 설치 없이 기존 환경에서 동작)
- **Stage 3 파인튜닝 추가**: `SOP_GPT_qa.pt`에서 이어서 KorQuAD (참고/질문/답변) 트리플로 한 번 더 파인튜닝한 `SOP_GPT_rag.pt` 도입
  - 단순히 검색 결과를 프롬프트에 끼워넣는 것이 아니라, 모델이 `"참고: ..."` 필드를 실제로 읽고 답변에 반영하도록 학습
  - 880 step에서 early stop, val loss 5.406 → 3.538
- **`main.py` / `chat.py` 확장**: `train_rag`, `chat_rag` 모드 추가 (REPL에서 검색된 참고 문서와 답변을 함께 출력)
- **`app.py` `/chat` 엔드포인트 교체**: 질문 → 검색 → `"참고: ...\n질문: ...\n답변: "` 프롬프트 조합 → RAG 모델로 생성하는 파이프라인으로 변경, 응답에 `retrieved_context` 필드 추가, 웹 UI에 검색된 참고 문서 표시
- **검증 결과**: 검색(retrieval)은 질문과 관련된 KorQuAD 문단을 정확히 찾아냄. 다만 답변 생성 정확도는 모델 규모(256dim/6layer/8000vocab) 한계로 사실 추출형 질문에서는 부정확한 경우가 있었고, 잡담형 질문에서는 관련 없는 문서가 검색되어 오히려 Stage 2보다 부자연스러운 답변이 나오는 경우도 확인 → 유사도 임계값 기반 폴백 필요성을 다음 단계 과제로 남김

### 2. 생성형 RAG의 한계를 해결하기 위한 구조 변경

- **디코딩 파라미터 튜닝**: `model.py`의 `generate()`에 top-p(nucleus) 샘플링 추가, Stage 1 이어쓰기 기본값을 `temperature=0.8, top_k=40` → `temperature=0.7, top_p=0.9`로 변경 — 문장이 짧아지고 깨진 인용부호 등 잡음은 줄었지만 의미적 개선은 제한적
- **Stage 1 코퍼스 비중 재조정**: kowikitext(위키체)가 챗봇 데이터(구어체)보다 250배 많아 이어쓰기에 위키 말투가 섞이던 문제 → kowiki 비중을 80M→15M자로 줄이고 챗봇 데이터를 15배 업샘플링(train/val 분리 후 업샘플링해 데이터 누수 방지), val loss 3.514→3.458로 개선되고 위키 마크업 잔존은 일부 남음
- **Stage 3(생성형 RAG) 폐기, Stage 4(추출형 RAG QA) 도입**: GPT가 답을 토큰 단위로 새로 "생성"하는 대신, `SOP_GPT_Span`(같은 transformer 본체 + 정답 시작/끝 위치 분류 head)으로 참고 문단 안에서 정답 구간을 직접 찾아 그대로 잘라 쓰도록 변경
  - `bpe.py`에 `tokenize_with_offsets()` 추가해 토큰 ↔ 원문 문자 위치를 매핑
  - `rag.py`에 `answer_window`/`build_span_examples`/`best_match`(유사도 점수 포함 검색) 추가
  - `SOP_GPT_qa.pt`의 본체 가중치로 초기화 후 qa_head만 새로 학습, 520 step에서 early stop, val loss 3.175
  - 검증셋(578개) 기준 **정확히 일치 31.5%, 정답 포함/겹침 72.8%** — 이전 생성형 방식(정답을 통째로 틀리던 수준)보다 명확히 개선
- **유사도 임계값 기반 폴백 라우팅**: `/chat`에서 검색 유사도가 0.25 이상이면 Stage 4(추출형) 사용, 그보다 낮으면 Stage 2(잡담형) 모델로 자동 전환 — "오늘 기분은 어때?", "사랑해" 같은 질문이 더는 무관한 위키 문서를 끼워 넣지 않고 자연스럽게 답하도록 개선, 응답에 `used_rag` 필드로 어느 경로를 탔는지 노출
- 안 쓰게 된 `SOP_GPT_rag.pt`(Stage 3 생성형 체크포인트) 삭제

### 3. 임계값 라우팅 오분류 개선: 검색기 하이브리드화

> 잡담 질문이 RAG로 새는 오분류를 줄이기 위해 검색기를 개선했습니다.

- **임베딩 검색 시도**: `transformers`의 `klue/bert-base`로 1차 시도 — 문장 유사도용으로 학습된 모델이 아니라 모든 문장이 비슷한 점수로 뭉치는 anisotropy 문제 확인, 문장 유사도(STS)로 파인튜닝된 `jhgan/ko-sroberta-multitask`로 교체 (SKT KoBERT는 별도 sentencepiece 토크나이저 패키지가 필요해 AutoTokenizer로 바로 쓸 수 있는 동급 모델로 대체)
- **순수 임베딩도 단독으로는 부족함을 확인**: 격식체(위키)와 구어체(잡담) 사이의 문체 유사도가 주제 유사도보다 점수에 더 큰 영향을 줘서, "사랑해"가 진짜 사실형 질문보다 유사도가 높게 나오는 역전이 발생
- **하이브리드 점수 도입 + 정규화 버그 수정**: `hybrid_score = α·normalize(sparse) + (1-α)·normalize(dense)` (α=0.5) 자체는 맞는 접근이었으나, 처음엔 질문마다 그 질문 내부에서 min-max 정규화를 해서 "1등 청크"가 관련 여부와 무관하게 항상 1.0 근처로 나오는 버그가 있었음
  - `rag.calibrate()` 추가: KorQuAD 질문(relevant) vs 챗봇 데이터 질문(irrelevant) 80개씩 샘플로 sparse/dense 점수의 정상 범위를 **한 번만 고정** 측정 → 이 고정값으로 정규화해야 질문들 사이에 비교 가능한 절대 점수가 됨
- **held-out 검증으로 효과 확인**: 보정용 샘플과 평가용 샘플을 분리해서 측정 — TF-IDF 단독 분류 정확도 73.3%(임계값 0.26) → 하이브리드+고정보정 82.7%(임계값 0.515)
- `app.py`/`main.py`/`chat.py`: `rag.build_index()`가 `(tfidf_vectorizer, tfidf_matrix, embed_matrix, passages, norm_bounds)` 5종을 반환하도록 변경, `RAG_SIM_THRESHOLD`를 0.25 → 0.515로 갱신

---

## 2026-06-14 — 최초 구현 (5주차 위클리 챌린지)

**한국어 Mini-GPT 챗봇 최초 구현**

- 자모(NFD) 단위 BPE 토크나이저 직접 구현 (`bpe.py`, `tokenizer.py`)
- `mini_gpt.py` 아키텍처를 이식한 GPT 디코더 `SOP_GPT` 구현 (`model.py`)
- **Stage 1 (이어쓰기)**: kowikitext + 챗봇 데이터(88.58M자)로 다음 토큰 예측 학습 → val loss 3.514
- **Stage 2 (Q&A 응답)**: songys/Chatbot_data로 `"질문: ...\n답변: ..."` 포맷 파인튜닝 → val loss 2.185
- FastAPI로 `/generate`, `/chat` 엔드포인트 서빙 + 웹 채팅 UI 구현

---

## 회고

<details>
<summary><b>2026-06-28 — 모델을 키운다고 항상 빠르게 좋아지는 건 아니다</b></summary>

- **메모리 제약이 오히려 코드를 더 잘 이해하게 만들었다.** RAM 부족으로 학습이 안 된다는 문제를 마주쳤을 때, 처음엔 단순히 숫자를 줄이려 했다. 그런데 제대로 해결하려면 "왜 메모리가 부족한가"를 추적해야 했다 — `batch_size`가 활성화 메모리에 선형 비례한다는 것, `chatbot_text * 15`처럼 문자열을 Python에서 통째로 복사하면 Python int 객체(~28B/개)가 토큰 수만큼 쌓인다는 것. gradient accumulation이 왜 "유효 배치는 그대로, RAM 피크는 줄이는" 방법인지도 직접 연산해보고 나서야 설득이 됐다.
- **순차 실행 vs 병렬 실행 선택이 "어느 게 빠른가"가 아니라 "RAM이 얼마나 있는가"로 결정됐다.** 세 학습을 동시에 돌리면 각자 모델·데이터를 따로 올리니 3배 RAM이 필요하다. 병렬이 무조건 좋은 게 아니라, 자원 제약 안에서 병렬화 가능한 수준을 따져야 한다는 걸 실제로 부딪히고 배웠다.
- **LangSmith 트레이싱은 "설정만 하면 끝"이라는 인상과 달리 순서에 민감했다.** `load_dotenv`를 langchain import보다 나중에 두면 env var가 인식 안 된다. import 순서가 왜 중요한지, 모듈이 언제 env var를 읽는지를 트레이싱이 안 뜨는 상황에서 역추적하며 이해했다. 환경 설정도 코드처럼 "언제 실행되는가"가 중요하다.
- **API 키를 코드와 분리하는 건 습관의 문제다.** 처음에 키를 `.env`에 바로 넣었는데, 이후 `api_keys`로 따로 분리했다. 어느 파일이 git에 올라가는지, 어느 파일에 시크릿을 두어야 하는지를 매번 의식적으로 결정하는 연습이 됐다. 나중에 팀 프로젝트에서 실수하지 않으려면 지금 이 습관을 들여놓는 게 맞다.
- **val loss 숫자가 줄었다고 체감 품질이 같은 비율로 좋아지지 않는다.** 모델을 8배 키웠지만 답변의 시원찮은 느낌은 크게 달라지지 않았다. 데이터 규모와 모델 규모 사이에 균형이 있고, 지금 학습 데이터(챗봇 1만 쌍 + KorQuAD)는 5천만 파라미터 모델을 충분히 학습시키기엔 부족하다. 모델 크기와 데이터 크기는 함께 키워야 한다는 scaling law를 숫자로 체감했다.

</details>

<details>
<summary><b>2026-06-20 — 검색기 하이브리드화: "정규화"라는 단어 하나로도 결과가 완전히 달라진다</b></summary>

- **검색은 잘 됐지만, "이해"는 조금 덜 됬다.** TF-IDF 기반 검색기는 질문과 관련된 KorQuAD 문단을 거의 정확하게 찾아냈다. 문제는 그 다음이었다 — 찾아낸 문단에서 정답을 뽑아내는 건 순수 생성 모델인 SOP_GPT(256dim/6layer)에게는 버거운 과제였다. "검색을 잘하는 것"과 "검색한 걸로 잘 답하는 것"은 RAG 안에서 완전히 분리된 두 개의 능력이라는 걸 직접 확인한 셈.
- **포맷만 같다고 모델이 새 필드를 이해하는 건 아니다.** 처음에 검색된 문단을 그냥 프롬프트에 끼워넣을까 생각했지만, `SOP_GPT_qa.pt`는 `"질문: ...\n답변: ..."` 포맷만 학습해서 `"참고: ..."`라는 새 필드를 본 적이 없다는 점을 미리 짚고 Stage 3 파인튜닝을 추가한 게 옳은 선택이었다. 실제로 학습 후에는 모델이 검색된 문단의 단어를 답변에 가져다 쓰는 모습이 보였다 (정답 자체는 틀려도, 문맥과 무관한 말을 하지는 않음).
- **지식 베이스의 성격이 챗봇의 성격과 안 맞으면 RAG가 오히려 독이 된다.** KorQuAD는 위키 기반 사실형 QA 데이터셋이라, 본래 챗봇이 잘하던 위로/잡담형 질문에는 항상 관련 없는 문서가 검색되어 끼워 넣어졌고, 결과적으로 Stage 2보다 답변이 부자연스러워지는 역효과가 났다. RAG를 "무조건 켜놓는 것"이 아니라 유사도 임계값으로 "검색이 도움될 때만 쓰는" 제어가 필요하다는 걸 배웠다.
- **작은 모델로 RAG를 직접 구현해본 의미.** sentence-transformers나 FAISS 같은 기성 라이브러리 없이 TF-IDF+코사인 유사도로도 검색-증강-생성 파이프라인 전체를 끝까지 만들어볼 수 있었다. 답변 품질은 모델 규모의 한계가 명확했지만, RAG라는 아키텍처가 "검색→증강→생성"의 3단 구조이고 각 단계에서 무엇이 망가질 수 있는지를 체감하는 데는 충분했다.
- **"정확한 답"이 필요하면 생성이 아니라 분류로 풀어야 한다.** GPT가 정답을 한 토큰씩 새로 만들어내는 방식은, 자모 단위 BPE 특성상 한 글자만 틀려도 그 뒤가 전부 무너지는 구조였다. KorQuAD처럼 정답이 항상 주어진 문단 안에 그대로 들어있는 추출형 QA는 "정답의 시작/끝 위치를 분류"하는 문제로 바꾸면 훨씬 안정적이라는 걸 직접 비교해보고 확인했다(검증셋 정확히 일치 31.5%, 겹침 72.8% — 생성형 때는 정답을 통째로 틀리던 수준에서 개선).
- **causal(디코더 전용) 구조에서도 추출형 QA가 가능했다.** BERT 같은 양방향 인코더가 표준이지만, 프롬프트 순서를 "질문 먼저, 참고 문단 나중"으로 두면 causal 모델도 문단의 각 위치에서 질문 전체를 이미 보고 있는 상태가 되어 위치 분류가 잘 작동했다. 굳이 아키텍처를 통째로 새로 만들지 않고 기존 SOP_GPT의 transformer 본체(tok_emb/pos_emb/blocks/ln_f)를 그대로 재사용하고 head만 교체한 게 효율적이었다.
- **유사도 임계값은 정밀한 분류기가 아니라 거친 안전장치다.** 실제 점수를 찍어보면 사실형 질문과 잡담형 질문의 TF-IDF 유사도가 깔끔하게 갈리지 않았다(0.37 vs 0.24처럼 겹치는 구간이 있음). 임계값 하나로 완벽히 분리되길 기대하면 안 되고, "최악의 케이스(전혀 무관한 질문)만 걸러내는 보험"으로 이해하고 값을 실험적으로 조정하는 게 맞는 접근이었다(0.2 → 0.25로 조정해서 "오늘 기분은 어때?" 같은 경계 사례를 잡아냄).
- **디코딩 파라미터나 코퍼스 비중 조정 같은 "저비용 손잡이"는 한계가 명확하다.** temperature/top-p 조정, kowiki:챗봇 비중 재조정 모두 의미 있는 개선을 줬지만, 결국 "토큰을 하나씩 생성한다"는 근본 구조의 한계는 못 넘었다. 구조적 한계는 구조를 바꿔야 풀린다는 걸 단계적으로 체감했다.
- **임베딩으로 바꾼다고 다 좋아지는 건 아니다.** "TF-IDF는 단어만 본다"는 약점을 고치려고 사전학습 임베딩(klue/bert-base)을 붙였는데, 일반 MLM 모델은 문장 유사도용으로 학습된 게 아니라서 오히려 모든 점수가 비슷하게 뭉치는 anisotropy 문제가 생겼다. "사전학습 모델"과 "유사도 검색에 쓸 수 있는 임베딩 모델"은 다른 거였다 — STS로 파인튜닝된 모델(ko-sroberta)로 바꾸고서야 의미 있는 신호가 나왔다.
- **정규화를 "언제" 계산하느냐가 정규화 공식보다 더 중요했다.** `hybrid_score = α·normalize(sparse) + (1-α)·normalize(dense)` 공식 자체는 맞았지만, 그 정규화를 질문마다 새로 계산(그 질문 안에서의 min-max)하면 "1등 청크"는 관련이 있든 없든 항상 1.0 근처로 나와버린다 — 절대적인 관련도가 아니라 그 질문 내부의 상대 순위만 보여주는 셈이라 임계값으로 못 쓴다. relevant/irrelevant 질문 전체를 기준으로 정규화 범위를 한 번 고정(calibration)해야 질문들 사이에 비교 가능한 점수가 된다는 걸 직접 실험으로 확인했다.
- **느낌이 아니라 held-out 평가로 확인해야 한다.** "괜찮아진 것 같다"는 인상만으로 끝내지 않고, 보정용 샘플과 평가용 샘플을 분리해서 분류 정확도를 직접 측정했다(73.3% → 82.7%). 같은 데이터로 보정하고 평가하면 결과가 과장되기 쉬워서, 이 분리가 평가의 신뢰도를 좌우했다.

</details>
