# 평가 결과 기록서

- [Qwen3-1.7B BF16 vs Q4\_K\_M 벤치마크](#qwen3-17b-bf16-vs-q4_k_m-벤치마크)
- [2026-07-09](#2026-07-09)
- [2026-07-03](#2026-07-03)
- [2026-06-28](#2026-06-28)

---

## Qwen3-1.7B BF16 vs Q4\_K\_M 벤치마크

**측정 환경**: Apple M-series (MPS/Metal 통합 메모리), KorQuAD dev 100문항 random.seed(2026)

### 속도 · 메모리 (3문항 직접 측정)

| 항목 | BF16 (Transformers+MPS) | Q4\_K\_M (GGUF+Metal) |
|---|---|---|
| 로딩 시간 | **7.4s** | 11.8s |
| MPS 점유 | 3,282 MB | 0 MB (\*) |
| CPU RSS 증가 | ~0 MB | 1,351 MB |
| 단어 생성 속도 | 4.6 단어/s | **15.3 단어/s** |

(\*) llama-cpp는 자체 Metal 메모리를 관리해 `torch.mps`로 측정 불가. 실제론 Metal을 사용함.

### 품질 · 응답 시간 (KorQuAD 100문항)

#### Q4\_K\_M 설정 변경 전후 비교

| 항목 | Q4 구버전 | Q4 신버전 | BF16 |
|---|---|---|---|
| 정확도 (contains\_match) | 77.0% | **80.0%** | **82.0%** |
| 평균 응답 시간 | 3.62s | **0.81s** | 2.68s |
| 총 소요 | 361.5s | **80.9s** | 267.9s |
| 평균 응답 길이 | 55.3단어 | **12.0단어** | 12.3단어 |

**구버전 문제**: temperature 미설정, thinking 제어 없음 → 길고 느린 답변, `<think>` 태그 누출 가능성

**신버전 변경 사항**:
- `ask_with_context` → 항상 `/no_think` (context 추출은 reasoning 불필요)
- `ask` → 30자 이상 + `?` 포함 시 `/think` 자동 활성화, 나머지는 `/no_think`
- thinking 후 빈 답변 → `/no_think` 폴백
- temperature 0.7 / top_p 0.9 명시
- `MAX_TOKENS_THINK = 1024` (thinking 토큰 공간 확보)
- `_strip_think` 에서 `</think>` 없는 잘린 블록도 제거

### 일치 분석 (BF16 vs Q4 구버전)

| 케이스 | 개수 |
|---|---|
| 둘 다 정답 | 71 |
| BF16만 정답 | 11 |
| Q4만 정답 | 6 |
| 둘 다 오답 | 12 |

### 결론

- **속도**: Q4\_K\_M 신버전이 BF16보다 3배 빠름 (0.81s vs 2.68s) — `ask_with_context` NO\_THINK 덕분
- **정확도**: BF16 82% > Q4 신버전 80% — 2%p 차이로 거의 동등
- **메모리**: Q4가 MPS 3.3GB 대신 RSS 1.4GB 사용 — 다른 모델(SOP\_GPT)과 메모리 경합 감소
- **용도 권장**: 속도·메모리 우선이면 Q4\_K\_M, 정확도 최우선이면 BF16

---

## 2026-07-09

### 평가 설정

| 항목 | 값 |
|------|----|
| Dataset | `sop-gpt-korquad` (LangSmith) |
| 예제 수 | 30개 (KorQuAD v1.0 dev set) |
| Evaluator | `contains_match` — 정답 텍스트가 예측 답변에 포함되면 1점 |
| 모델 | n_embd=768, n_head=12, n_layer=12, block_size=256, vocab=8,003 (~91.4M) |
| 학습 데이터 | kowikitext(15M자) + chatbot(×15) + KorQuAD train+dev |
| 변경사항 | 모델 파일 `sop_model/` 하위 디렉토리로 이동 후 첫 평가 |

### 결과

| 체인 | contains_match | 정답 포함 수 | 전회(07-03) 대비 |
|------|---------------|-------------|-----------------|
| `basic` (검색 없음, QA LLM 직접) | **0.0%** | 0 / 30 | — |
| `tfidf_rag` (TF-IDF 검색 + Span 추출) | **40.0%** | 12 / 30 | ↑ +3.3%p |
| `lc_rag` (BM25+FAISS 하이브리드 + Span 추출) | **56.7%** | 17 / 30 | ↑ +10.0%p |
| `lg_graph` (LangGraph + BM25+FAISS + retry) | **70.0%** | 21 / 30 | ↑ +13.3%p |

### 해석

- **basic 0%**: 이전과 동일. 검색 없이 QA 모델만으로는 KorQuAD 사실형 질문에 정답 텍스트가 포함된 답을 생성할 수 없음.
- **전 체인 일괄 개선**: 코드 변경 없이 수치가 올랐다. 모델 파라미터가 512→768 dim으로 늘어난 것(이전 평가는 512-dim 체크포인트) 외에는 동일한 조건이므로, n_embd 확장이 Span 추출 품질에 긍정적으로 작용한 것으로 보임.
- **lc_rag 56.7% (↑ +10%p)**: BM25+FAISS 검색 품질 자체는 동일하지만, 더 큰 임베딩 차원의 SOP_GPT_Span이 정답 위치를 더 정확히 분류하는 효과.
- **lg_graph 70.0% (↑ +13.3%p, 신규 최고)**: retry 루프가 단순 lc_rag 대비 13.3%p 추가. 1회 검색으로 임계값 미달 시 임계값을 낮춰 재시도하는 구조가 규모가 커진 모델과 시너지를 낸 것으로 보임.

### LangSmith 실험 링크

- `sop-gpt-basic-fb11c323`
- `sop-gpt-tfidf_rag-bba25536`
- `sop-gpt-lc_rag-1803d314`
- `sop-gpt-lg_graph-fc3eeed2`

---

### DPO 실험 (Stage 5)

eval 수치는 DPO 전후 동일 (0% / 40% / 56.7% / 70%). DPO 모델(`SOP_GPT_dpo.pt`)이 실제 추론 경로에 들어가지 않아서다. `tfidf_rag` / `lc_rag` / `lg_graph` 세 체인은 Span 모델(`SOP_GPT_span.pt`)로 정답 위치를 추출하므로, DPO로 파인튜닝한 QA 모델은 파이프라인에 영향을 주지 않는다.

**실험 1 — KorQuAD 기반 (실패)**

| 항목 | 값 |
|---|---|
| 데이터 | KorQuAD dev 1,498쌍 (오답률 99.9%) |
| chosen | KorQuAD 정답 (짧은 명사구, 예: "1989년 2월 15일") |
| rejected | 모델이 RAG 없이 생성한 엉터리 단어 (예: "총장") |
| best val loss | 0.2735 (step 600) |
| 결과 | DPO 후 생성 품질 오히려 저하. 영어 혼입·반복 패턴·질문 이어쓰기 등 이상 출력 증가 |
| 원인 | chosen(명사구)과 rejected(단어)의 스타일 자체가 달라 DPO가 "어떤 패턴을 억제해야 하는지" 혼란. 도메인 불일치가 기존 생성 능력까지 훼손 |

**실험 2 — 챗봇 데이터 기반 (부분 개선)**

| 항목 | 값 |
|---|---|
| 데이터 | 챗봇 Q&A 1,937쌍 (불일치율 96.9%) |
| chosen | 원래 챗봇 정답 (대화체 완전 문장, 예: "위로해 드립니다.") |
| rejected | 모델이 같은 질문에 생성한 다른 답변 (같은 대화체) |
| best val loss | **0.0250 (step 450)**, early stopping step 700 |
| 결과 | 챗봇 답변이 더 길어지는 경향. 그러나 문장 후반부 품질 불안정 ("살았나봐요." 등 이상한 접속) |
| 원인 | 스타일 일치로 DPO 학습 자체는 성공했으나, RAG 없이 사실형 질문을 풀 능력 자체가 없는 91M 모델에서 생성 선호도 정렬의 효과는 제한적 |

**결론**: DPO는 이미 어느 정도 올바른 답변을 생성할 수 있는 모델을 전제로 한다. 현재 SOP_GPT는 RAG 없이 사실형 QA를 아예 못 풀어 DPO가 정렬할 "좋은 출력 후보"가 없다. 파이프라인 평가 수치 개선은 Span 모델 품질·검색 정확도·retry 구조에 달려 있으므로, DPO는 현 단계에서 효과 없음으로 결론. `SOP_GPT_qa.pt`를 계속 사용한다.

---

## 2026-07-03

### 평가 설정

| 항목 | 값 |
|------|----|
| Dataset | `sop-gpt-korquad` (LangSmith) |
| 예제 수 | 30개 (KorQuAD v1.0 dev set) |
| Evaluator | `contains_match` — 정답 텍스트가 예측 답변에 포함되면 1점 |
| 모델 | n_embd=512, n_head=8, n_layer=12, block_size=256 |
| 학습 데이터 | kowikitext(15M자) + chatbot(×15) + KorQuAD train+dev |

### 결과

| 체인 | contains_match | 정답 포함 수 | 전회 대비 |
|------|---------------|-------------|----------|
| `basic` (검색 없음, QA LLM 직접) | **0.0%** | 0 / 30 | — |
| `tfidf_rag` (TF-IDF 검색 + Span 추출) | **36.7%** | 11 / 30 | ↑ +16.7%p |
| `lc_rag` (BM25+FAISS 하이브리드 + Span 추출) | **46.7%** | 14 / 30 | ↑ +26.7%p |
| `lg_graph` (LangGraph + BM25+FAISS + retry) | **56.7%** | 17 / 30 | 신규 |

### 해석

- **basic 0%**: 이전과 동일. 검색 없이 QA 모델만으로는 KorQuAD 사실형 질문에 정답 텍스트가 그대로 나올 가능성 없음.
- **tfidf_rag 36.7% (↑ 16.7%p)**: 6월 28일 20.0% → 36.7%로 개선. `_expand_span()` 도입(span 30자 미만 시 context 전체 반환, 이상 시 주변 문장 확장) 효과로 보임.
- **lc_rag 46.7% (↑ 26.7%p)**: 6월 28일 20.0% → 46.7%로 개선. `_expand_span()` 외에 FAISS `normalize_embeddings=True` 수정(dense 점수 정상화), calibration 도메인 50:50 균형 샘플링, BM25 소문자 정규화, Cross-Encoder 제거 등이 복합적으로 기여.
- **lg_graph 56.7% (신규 최고)**: lc_rag 대비 10%p 추가 개선. 동일한 BM25+FAISS 검색기를 쓰지만 임계값 단계적 완화 retry 루프(0.25 → 0.20 → 0.15)가 1회 검색으로 점수 미달인 경우를 재시도로 구제하는 효과로 보임. StateGraph 구조로 검색 실패를 명시적으로 처리하는 것이 단순 체인보다 유리함을 확인.

### LangSmith 실험 링크

- `sop-gpt-basic-10bf95f6`
- `sop-gpt-tfidf_rag-0f06ef8c`
- `sop-gpt-lc_rag-fb2f36c4`
- `sop-gpt-lg_graph-51863cfa`

---

## 2026-06-28

### 평가 설정

| 항목 | 값 |
|------|----|
| Dataset | `sop-gpt-korquad` (LangSmith) |
| 예제 수 | 30개 (KorQuAD v1.0 dev set) |
| Evaluator | `contains_match` — 정답 텍스트가 예측 답변에 포함되면 1점 |
| 모델 | n_embd=512, n_head=8, n_layer=12, block_size=256 |
| 학습 데이터 | kowikitext(15M자) + chatbot(×15) + KorQuAD train+dev |

### 결과

| 체인 | contains_match | 정답 포함 수 |
|------|---------------|-------------|
| `basic` (검색 없음, QA LLM 직접) | **0.0%** | 0 / 30 |
| `tfidf_rag` (TF-IDF 검색 + Span 추출) | **20.0%** | 6 / 30 |
| `lc_rag` (BM25+FAISS 하이브리드 + Span 추출) | **20.0%** | 6 / 30 |

### 해석

- **basic 0%**: 검색 없이 QA 모델만으로는 KorQuAD 사실형 질문에 정확한 답을 생성할 수 없다. 모델이 학습한 잡담 Q&A 패턴(챗봇 데이터)과 사실 추출 질문의 성격이 완전히 달라서 정답 텍스트가 그대로 나올 가능성이 없다.
- **tfidf_rag = lc_rag = 20%**: 검색을 붙이면 정답 포함률이 0% → 20%로 오른다. TF-IDF와 BM25+FAISS의 점수가 같은 이유는 30개 질문 모두 RAG 라우팅 임계값을 넘겨 Span 추출 경로로 들어갔고, 검색-추출 성능이 두 검색기 간에 큰 차이가 없었기 때문으로 보인다.
- **20%의 한계**: 검증셋(578개) 기준으로는 정확 일치 31.5%, 포함/겹침 72.8%였는데 여기서 20%가 나온 건 두 가지 차이 때문이다. 첫째, 샘플 크기가 다르다(578개 vs 30개). 둘째, 평가 방식이 다르다 — `contains_match`는 정답이 답변에 **완전히 포함**되어야 1점이라 모델이 부분적으로 맞게 추출해도 0점으로 처리된다. 두 수치는 직접 비교할 수 없다.

### LangSmith 실험 링크

- `sop-gpt-basic-185cfe05`
- `sop-gpt-tfidf_rag-86f64b4f`
- `sop-gpt-lc_rae-b6ece450`
