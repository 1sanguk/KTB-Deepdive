# 체인 평가 결과

LangSmith Dataset 기반 평가. KorQuAD v1.0 dev set 앞 30개 질문으로 세 체인을 비교했다.

## 평가 설정

| 항목 | 값 |
|------|----|
| Dataset | `sop-gpt-korquad` (LangSmith) |
| 예제 수 | 30개 (KorQuAD v1.0 dev set) |
| Evaluator | `contains_match` — 정답 텍스트가 예측 답변에 포함되면 1점 |
| 실행일 | 2026-06-28 |

## 결과

| 체인 | contains_match | 정답 포함 수 |
|------|---------------|-------------|
| `basic` (검색 없음, QA LLM 직접) | **0.0%** | 0 / 30 |
| `tfidf_rag` (TF-IDF 검색 + Span 추출) | **20.0%** | 6 / 30 |
| `lc_rag` (BM25+FAISS 하이브리드 + Span 추출) | **20.0%** | 6 / 30 |

## 해석

- **basic 0%**: 검색 없이 QA 모델만으로는 KorQuAD 사실형 질문에 정확한 답을 생성할 수 없다. 모델이 학습한 잡담 Q&A 패턴(챗봇 데이터)과 사실 추출 질문의 성격이 완전히 달라서 정답 텍스트가 그대로 나올 가능성이 없다.
- **tfidf_rag = lc_rag = 20%**: 검색을 붙이면 정답 포함률이 0% → 20%로 오른다. TF-IDF와 BM25+FAISS의 점수가 같은 이유는 30개 질문 모두 RAG 라우팅 임계값을 넘겨 Span 추출 경로로 들어갔고, 검색-추출 성능이 두 검색기 간에 큰 차이가 없었기 때문으로 보인다.
- **20%의 한계**: 검증셋(578개) 기준으로는 정확 일치 31.5%, 포함/겹침 72.8%였는데 여기서 20%가 나온 건 평가 방식 차이 때문이다. `contains_match`는 정답이 답변에 **완전히 포함**되어야 1점이라 모델이 부분적으로 맞게 추출해도 0점으로 처리된다.

## LangSmith 실험 링크

- `sop-gpt-basic-185cfe05`
- `sop-gpt-tfidf_rag-86f64b4f`
- `sop-gpt-lc_rag-b6ece450`
