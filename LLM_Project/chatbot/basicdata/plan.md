# 한국어 Mini-GPT 프로젝트 계획

## 목표
- 1차: `mini_gpt.py`처럼 글을 자연스럽게 "이어쓰는" 한국어 모델
- 2차: 입력(질문/프롬프트)에 대해 완전한 문장으로 "응답"하는 모델로 확장
- 토크나이저는 직접 구현하는 BPE/subword 사용
- 최종적으로 FastAPI로 서빙

## Phase 0 — 환경 & 프로젝트 구조 ✅
- 가상환경 세팅 후 PyTorch CUDA 버전 설치 (NVIDIA GPU + CUDA 13.1 확인됨)
- 모듈 분리: `tokenizer.py` / `bpe.py` / `model.py` / `train.py` / `chat.py` / `train_utils.py`
- 데이터 → 배치 → forward/backward → 체크포인트 저장 파이프라인 동작 확인

## Phase 1 — 코퍼스 준비 (Stage 1: 이어쓰기용) ✅
- kowikitext dev+test 전체 + train split 앞부분 80M자 + 한국어 챗봇 Q&A 데이터 → 총 88.58M자
- NFD로 분해 후 BPE 적용, NFC로 재조합해서 출력 (decompose/compose)

## Phase 2 — BPE 토크나이저 직접 구현 ✅
- 자모(NFD) 단위로 분해 후 BPE 적용
- **base alphabet은 빈도 상위 300자로 제한**하고 나머지 희귀 문자는 `UNK`(U+FFFD)로 치환 — 그래야 merge가 일어날 vocab 공간이 확보됨
  - (처음엔 코퍼스의 고유 문자 전체(9188개)를 base alphabet으로 써서 BPE_VOCAB_SIZE(8000)보다 커져 merge가 0번 일어나는 버그가 있었음 — 위 방식으로 수정)
- vocab_size=8000 (base 301 + merge 7699), 압축률(글자/토큰) 약 1.5x, encode→decode roundtrip 정상

## Phase 3 — mini_gpt.py 아키텍처 이식 ✅
- `CausalSelfAttention` / `Block` / `SOP_GPT` 구조 재사용, vocab_size=8000에 맞춰 임베딩/출력층 구성
- block_size=128, n_embd=256, n_head=4, n_layer=6, batch_size=64, lr=1e-3, dropout=0.2
- early stopping(patience=10, min_delta=1e-3, best checkpoint 저장) 도입

## Phase 4 — Stage 1 검증 (이어쓰기) ✅
- 88.58M자 → 58.9M 토큰으로 학습, val loss 3.514 (perplexity ~33.6)에서 early stop
- `chat()` REPL: 프롬프트 → 자모 분해 → BPE encode → 생성 → NFC 재조합 → 출력
- `.`/`?`/`!`로 끝나는 토큰이 나오면 멈추는 stop 조건 + temperature/top-k/repetition_penalty 샘플링 적용
- 예) "안녕하" → "안녕하세요!" 처럼 짧고 문법적으로 완결된 문장으로 종료됨을 확인

## Phase 5 — Stage 2: "완전한 문장 응답"으로 확장 ✅
- **데이터**: 한국어 챗봇 Q&A 데이터셋(songys/Chatbot_data, 11,823쌍) → `"질문: {q}\n답변: {a}\n\n"` 포맷으로 변환 (`tokenizer.load_chatbot_qa_data`)
  - Stage 1과 동일한 BPE vocab(8000) 재사용 — "질문"/"답변"이 이미 단일 토큰으로 인코딩됨, 압축률 1.36x
- **학습 전략**: B안 채택 — Stage 1 가중치(`SOP_GPT.pt`)에서 이어서 fine-tuning (`finetune_qa.py`)
  - lr을 1e-3 → 3e-4로 낮춰 급격한 망각 방지, early stopping(patience=10, min_delta=1e-3)으로 QA val loss 기준 best checkpoint를 `SOP_GPT_qa.pt`로 저장
  - 공통 학습 루프(get_batch/estimate_loss/train_loop)는 `train_utils.py`로 추출해 `train.py`/`finetune_qa.py`에서 공유
- **종료 조건**: "\n"으로 끝나는 토큰이 나오면 "답변: ..." 한 줄이 끝난 것으로 보고 멈춤 (`chat_qa`)
- **생성 로직**: `chat_qa(model, ...)` — `"질문: {input}\n답변: "`을 prompt로 넣고 답변 부분만 출력
- 기대치 관리: 이 규모(수백만 파라미터대)에서는 "똑똑한 답변"보다 "Q→A 턴 구조를 흉내내는" 수준이 현실적 목표
- **결과**: 1100 step에서 early stop, val loss 3.738 → 2.185 (perplexity ~8.9)
  - "취업 준비 잘하고 있어?" → "열심히 하세요.", "사랑해" → "사랑은 소유하는게 아니라죠." 등
  - 모든 응답이 짧고 문법적으로 완결되며, 학습 데이터의 위로/격려 톤을 잘 따라감

## Phase 6 — FastAPI 서빙 ✅
- `app.py`: 서버 시작 시 BPE vocab + `SOP_GPT.pt`(이어쓰기) / `SOP_GPT_qa.pt`(Q&A) 1회 로드
- `POST /generate`(이어쓰기, `.`/`?`/`!`에서 멈춤), `POST /chat`(Q&A, 줄바꿈에서 멈춤) — Pydantic 스키마
- 빈 입력 처리: `/generate`는 START_ID로 대체, `/chat`은 빈 질문도 템플릿 토큰으로 정상 인코딩됨
- `GET /` 테스트용 HTML 페이지(이어쓰기/Q&A 폼) + `/docs`(Swagger UI)
- 실행: `app/` 디렉터리에서 `uvicorn app:app --reload`
- 테스트 결과: "안녕하" → "세요!", "/chat 취업 준비 잘하고 있어?" → "계획적이고 필요한게 더 많겠죠." 등 정상 응답

## Phase 7 — RAG(검색 증강 생성) 도입 ✅
- **지식 베이스**: KorQuAD v1.0 dev set을 다운로드해 ~90자 단위 청크로 분할
- **검색기**: scikit-learn `TfidfVectorizer` + 코사인 유사도 기반 `rag.py` 신규 작성
- **Stage 3 파인튜닝**: `SOP_GPT_qa.pt`에서 이어서 KorQuAD (참고/질문/답변) 트리플로 학습한 `SOP_GPT_rag.pt` 도입 — 모델이 `"참고: ..."` 필드를 읽고 답변에 반영하도록 학습, val loss 5.406 → 3.538
- `main.py`/`chat.py`에 `train_rag`/`chat_rag` 모드 추가, `app.py` `/chat`을 검색→증강 프롬프트→생성 파이프라인으로 교체 (`retrieved_context` 필드 추가)
- 검증 결과: 검색은 정확했으나 답변 생성은 모델 규모 한계로 사실 추출형 질문에서 부정확, 잡담형 질문은 무관한 문서가 끼어들어 오히려 부자연스러워짐

## Phase 8 — 생성형 RAG 폐기 → 추출형 RAG(Stage 4)로 구조 변경 ✅
- 디코딩 파라미터 튜닝(top-p 추가), Stage 1 코퍼스 비중 재조정(kowiki 80M→15M자, 챗봇 데이터 15배 업샘플링) — 부분적 개선에 그침
- **Stage 3(생성형) 폐기, Stage 4(추출형 QA) 도입**: `SOP_GPT_Span`(transformer 본체 재사용 + 정답 시작/끝 위치 분류 head)으로 참고 문단에서 정답 구간을 직접 추출
  - `bpe.py`에 `tokenize_with_offsets()` 추가, `rag.py`에 `answer_window`/`build_span_examples`/`best_match` 추가
  - `SOP_GPT_qa.pt` 본체로 초기화 후 qa_head만 학습, val loss 3.175, 검증셋 정확히 일치 31.5% / 겹침 72.8%
- **유사도 임계값 폴백 라우팅**: `/chat`에서 검색 유사도 ≥0.25면 Stage4(추출형), 미만이면 Stage2(잡담형)로 자동 전환, 응답에 `used_rag` 필드 노출
- 안 쓰는 `SOP_GPT_rag.pt` 삭제

## Phase 9 — 검색기 하이브리드화 (오분류 개선) ✅
- 사전학습 임베딩(`klue/bert-base`)은 anisotropy 문제로 부적합 → STS 파인튜닝 모델 `jhgan/ko-sroberta-multitask`로 교체
- **하이브리드 점수**: `hybrid_score = α·normalize(sparse) + (1-α)·normalize(dense)` (α=0.5)
  - 질문별 그때그때 min-max 정규화하던 버그 수정 → `rag.calibrate()`로 relevant/irrelevant 샘플 기준 정규화 범위를 한 번만 고정
- held-out 검증: TF-IDF 단독 73.3%(임계값 0.26) → 하이브리드+고정보정 82.7%(임계값 0.515)
- `rag.build_index()`가 `(tfidf_vectorizer, tfidf_matrix, embed_matrix, passages, norm_bounds)` 반환, `RAG_SIM_THRESHOLD` 0.25 → 0.515로 갱신

## 디렉토리 구조
```
source/
├── app/
│   └── app.py        # FastAPI 서빙 (앱소스)
└── model/            # 토크나이저/아키텍처/학습/체크포인트 (모델소스)
    ├── bpe.py
    ├── tokenizer.py
    ├── model.py
    ├── rag.py            # 검색기 (TF-IDF + 임베딩 하이브리드, calibrate)
    ├── train_utils.py
    ├── main.py           # 진입점: train / chat / train_qa / chat_qa / train_rag / chat_rag
    ├── chat.py
    ├── bpe_vocab.json
    ├── SOP_GPT.pt
    ├── SOP_GPT_qa.pt
    └── SOP_GPT_Span.pt   # Stage4 추출형 QA head
```
- `model/main.py`는 그 디렉터리에서 실행 (상대경로로 `bpe_vocab.json`/체크포인트 참조)
  - `train`/`chat`은 Stage1, `train_qa`/`chat_qa`는 Stage2, `train_rag`/`chat_rag`는 RAG 흐름 — `chat`/`chat_qa`/`train_qa`는 88M자 Stage1 코퍼스를 불러오지 않아 빠르게 시작됨
- `app/app.py`는 `__file__` 기준으로 `../model`을 `sys.path`에 추가해 어디서 실행해도 동작
- `/chat`은 검색 유사도(hybrid_score)에 따라 Stage4(추출형, ≥0.515) ↔ Stage2(잡담형, <0.515)로 자동 라우팅

## Phase 10 — Stage 5: DPO (Direct Preference Optimization) ✅
- **목적**: Stage 2 SFT 모델이 "어떻게 대답할지"는 알지만 "좋은 답 vs 나쁜 답"을 구분하지 못하는 한계를 개선
- **구현**: `source/model/dpo.py` 신규 작성, `main.py`에 `train_dpo` 모드 추가
- **레퍼런스 모델**: `SOP_GPT_qa.pt`를 `deepcopy` 후 freeze → policy 모델이 SFT에서 너무 벗어나지 않도록 KL 페널티 역할
- **선호 쌍 생성 전략**: 별도 인간 레이블링 없이 배치 내 circular shift로 rejected 자동 생성 (chosen=정답 답변, rejected=다른 질문의 답변)
- **학습 데이터**: chatbot Q&A(11,823쌍) + KorQuAD 쌍 전체
- **손실 함수**: `-log sigmoid(β · ((log π_θ(chosen) - log π_θ(rejected)) - (log π_ref(chosen) - log π_ref(rejected))))`
- **하이퍼파라미터**: lr=1e-5, beta=0.1, steps=1000
- **결과물**: `SOP_GPT_dpo.pt`
- 실행: `python main.py train_dpo`

## 스트레치 골 (여유 있을 때)
- temperature / top-k / top-p 샘플링
- KV cache로 생성 속도 개선
- 파이프라인 검증 후 모델/데이터 규모 확장
- FastAPI `StreamingResponse`로 토큰 단위 스트리밍 출력
- sentence-transformers/FAISS 등 전용 라이브러리로 검색기 교체해 정확도/속도 비교
