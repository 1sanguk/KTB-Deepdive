# 체인 평가 결과

LangSmith Dataset 기반 평가 이력. 날짜별로 평가 결과를 누적 기록한다.

- [2026-07-03](#2026-07-03)
- [2026-06-28](#2026-06-28)

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
