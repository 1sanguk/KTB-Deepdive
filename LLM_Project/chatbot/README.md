# 한국어 Mini-GPT 챗봇 (RAG 적용)

자모(NFD) 단위 BPE 토크나이저부터 GPT 아키텍처, 학습, RAG, FastAPI 서빙까지 전부 직접 구현한 한국어 챗봇 프로젝트.

## 개요
> 하나로 합쳐진 "최종 모델"은 없다. 체크포인트 3개(`SOP_GPT.pt`/`SOP_GPT_qa.pt`/`SOP_GPT_span.pt`)가 역할을 나눠 쓰이는 구조 — `/generate`는 Stage1만, `/chat`은 검색 유사도로 Stage2/Stage4 중 하나를 골라 응답한다.

- 토크나이저: 한글을 NFD로 분해(초성/중성/종성 자모)한 뒤 BPE 적용, NFC로 재조합해 출력 (직접 구현)
- 모델: `CausalSelfAttention` / `Block` / `SOP_GPT`로 구성된 GPT 디코더 (block_size=128, n_embd=256, n_head=4, n_layer=6)
- 학습 단계
  - **Stage 1 (이어쓰기)**: kowikitext + 한국어 챗봇 Q&A 데이터로 다음 토큰 예측 학습 (kowiki:chatbot 비중을 재조정해 구어체 비중 강화)
  - **Stage 2 (Q&A 응답)**: songys/Chatbot_data(11,823쌍)로 `"질문: ...\n답변: ..."` 포맷 파인튜닝 — 잡담형 질문에 사용
  - **Stage 4 (추출형 RAG QA)**: KorQuAD v1.0으로 정답의 시작/끝 토큰 **위치를 분류**하는 `SOP_GPT_Span` 학습 — 사실형 질문에 사용
- 생성: temperature / top-k / top-p / repetition penalty 샘플링 + 종결 토큰(`.`/`?`/`!`/줄바꿈) 기준 stop 조건
- 서빙: FastAPI로 `/generate`(이어쓰기), `/chat`(RAG + 잡담 폴백 Q&A) 엔드포인트 제공 + 웹 채팅 UI

## RAG 아키텍처
1. **지식 베이스**: KorQuAD v1.0 dev set(한국어 위키 기반 reading comprehension 데이터셋)의 문단들을 ~90자 단위로 분할해 인덱싱
2. **검색(Retrieve)**: TF-IDF(`scikit-learn`, 단어 겹침)와 의미 임베딩(`jhgan/ko-sroberta-multitask`, 문장 유사도 파인튜닝 모델) 두 점수를 평균 낸 **하이브리드 검색**
   - TF-IDF 단독은 단어만 같으면 오검색(예: "게임 개발자" vs "디지털화폐 개발자"), 임베딩 단독은 격식체/구어체 같은 말투 유사도가 주제 유사도를 압도하는 약점이 있어 서로 보완
3. **라우팅(임계값 보정)**: KorQuAD 질문(관련 있음) vs 챗봇 데이터 질문(관련 없음) 샘플로 sparse/dense 점수의 "정상 범위"를 한 번 고정 측정(`rag.calibrate`)한 뒤, 그 기준으로 정규화한 하이브리드 점수가 임계값(0.515) 이상이면 RAG 경로, 아니면 Stage 2(잡담형) 모델로 폴백
   - 질문마다 점수를 다시 min-max 정규화하면 "1등 청크"가 관련 있든 없든 항상 1.0 근처로 나와버려 임계값으로 못 쓰는 문제가 있었음 — relevant/irrelevant 질문 전체를 기준으로 한 번 고정해야 질문들 사이에 비교 가능한 절대 점수가 됨
   - held-out 검증 기준 분류 정확도: TF-IDF 단독 73.3% → 하이브리드+고정보정 82.7%
4. **추출(Extract)**: 생성 대신 `SOP_GPT_Span`이 `"질문: {질문}\n참고: {청크}"` 안에서 정답의 시작/끝 토큰 위치를 직접 분류해 그 구간을 그대로 잘라 답으로 사용
   - 이전엔 GPT가 답을 한 토큰씩 새로 생성했는데(자기회귀), 토큰 하나만 틀려도 뒤가 다 망가지는 문제가 있었음 — 위치를 분류하는 방식으로 바꿔서 작은 모델로도 더 정확한 정답을 뽑음 (검증셋 기준 정확히 일치 31.5%, 정답 포함/겹침 72.8%)

## 디렉토리 구조
```
chatbot/
├── README.md
├── version.md
├── basicdata/            # 참고 자료, 작업 계획, 세션 기록
└── source/
    ├── app/
    │   └── app.py        # FastAPI 서빙
    └── model/
        ├── bpe.py         # BPE 토크나이저 직접 구현 (+ tokenize_with_offsets)
        ├── tokenizer.py   # 코퍼스 로딩 (kowikitext, 챗봇 Q&A)
        ├── rag.py         # KorQuAD 다운로드/청크/하이브리드 검색 인덱스+임계값 보정/Stage4 학습 데이터
        ├── model.py       # GPT 아키텍처(SOP_GPT) + 추출형 QA 아키텍처(SOP_GPT_Span)
        ├── train_utils.py # 학습 루프 (early stopping)
        ├── chat.py        # REPL (chat / chat_qa / chat_span)
        ├── main.py        # 진입점 (train / train_qa / train_span / chat / chat_qa / chat_span)
        ├── bpe_vocab.json
        ├── SOP_GPT.pt       # Stage 1 체크포인트
        ├── SOP_GPT_qa.pt    # Stage 2 체크포인트
        └── SOP_GPT_span.pt  # Stage 4(추출형 RAG QA) 체크포인트
```

## 실행
```bash
# 모델 학습/대화 (source/model 디렉터리에서 실행)
python main.py train       # Stage 1 학습
python main.py train_qa    # Stage 2 파인튜닝
python main.py train_span  # Stage 4 추출형 RAG QA 학습
python main.py chat        # Stage 1 이어쓰기 REPL
python main.py chat_qa     # Stage 2 Q&A REPL
python main.py chat_span   # Stage 4 RAG QA REPL (검색된 문서 + 추출된 답 함께 출력)

# 웹 서버 (source/app 디렉터리에서 실행)
uvicorn app:app --reload
```

## API
- `POST /generate` — `{"prompt": str, "max_new_tokens": int}` → 이어쓰기 결과
- `POST /chat` — `{"question": str}` → `{"answer": str, "retrieved_context": str, "used_rag": bool}`
  - 하이브리드(TF-IDF+임베딩) 유사도가 임계값(0.515) 이상이면 `used_rag=true`로 RAG(추출형 QA) 응답, 아니면 `used_rag=false`로 Stage2 잡담형 응답
- `GET /` — 테스트용 웹 채팅 UI
- `GET /docs` — Swagger UI

자세한 개발 과정은 [basicdata/plan.md](basicdata/plan.md), 단계별 변경 이력은 [version.md](version.md) 참고.
