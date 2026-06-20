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

## 디렉토리 구조
```
source/
├── app/
│   └── app.py        # FastAPI 서빙 (앱소스)
└── model/            # 토크나이저/아키텍처/학습/체크포인트 (모델소스)
    ├── bpe.py
    ├── tokenizer.py
    ├── model.py
    ├── train_utils.py
    ├── main.py           # 진입점: train / chat / train_qa / chat_qa
    ├── chat.py
    ├── bpe_vocab.json
    ├── SOP_GPT.pt
    └── SOP_GPT_qa.pt
```
- `model/main.py`는 그 디렉터리에서 실행 (상대경로로 `bpe_vocab.json`/체크포인트 참조)
  - `train`/`chat`은 Stage1, `train_qa`/`chat_qa`는 Stage2 — `chat`/`chat_qa`/`train_qa`는 88M자 Stage1 코퍼스를 불러오지 않아 빠르게 시작됨
- `app/app.py`는 `__file__` 기준으로 `../model`을 `sys.path`에 추가해 어디서 실행해도 동작

## 스트레치 골 (여유 있을 때)
- temperature / top-k / top-p 샘플링
- KV cache로 생성 속도 개선
- 파이프라인 검증 후 모델/데이터 규모 확장
- FastAPI `StreamingResponse`로 토큰 단위 스트리밍 출력
