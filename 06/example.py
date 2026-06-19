import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. 예시 문서 / 쿼리 데이터
# ============================================================
# D1, D2 : "그래픽카드"와 "GPU"처럼 같은 의미를 다른 단어로 표현한 동의어 케이스
# D3, D4 : "A1234" 같은 고유 코드(식별자)가 들어간 케이스
# D5, D6 : "파이썬"이라는 단어가 겹치지만 뜻이 다른(동음이의어/다의어) 케이스
docs = {
    "D1": "RTX 4090 그래픽카드 가격 정리",
    "D2": "고성능 GPU 모델 리스트",
    "D3": "주문번호 A1234 환불 절차 안내",
    "D4": "환불 신청 방법 처리 기간",
    "D5": "파이썬 사육사 자격증 안내",
    "D6": "코딩 프로그래밍 자격증 시험 안내",
}

queries = {
    "Q1 (그래픽카드 추천)": "그래픽카드 추천",
    "Q2 (A1234 주문 환불)": "A1234 주문 환불",
    "Q3 (파이썬 자격증)": "파이썬 자격증",
}

doc_tokens = {name: text.split() for name, text in docs.items()}


# ============================================================
# 1. Sparse Vector 검색 (BM25)
# ============================================================
# 실제 어휘 사전 크기에 비해 쿼리/문서에 등장하는 단어는 극히 일부뿐이라
# "희소(sparse)"하다고 부른다. 여기서는 그 중 BM25를 직접 구현한다.
def bm25_scores(query, doc_tokens, k1=1.5, b=0.75):
    tokens = list(doc_tokens.values())
    N = len(tokens)
    avgdl = sum(len(t) for t in tokens) / N

    def df(term):
        return sum(1 for t in tokens if term in t)

    def idf(term):
        n = df(term)
        return np.log(1 + (N - n + 0.5) / (n + 0.5))

    scores = {}
    for name, t in doc_tokens.items():
        dl = len(t)
        score = 0.0
        for term in query.split():
            f = t.count(term)
            if f == 0:
                continue
            score += idf(term) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[name] = score
    return scores


# ============================================================
# 2. Dense Vector 검색 (의미 기반)
# ============================================================
# 실제로는 BERT 계열 임베딩 모델이 문장을 통째로 의미 벡터로 변환하지만,
# 여기서는 핵심 아이디어("의미가 비슷하면 벡터도 가깝다")만 보기 위해
# 직접 손으로 짠 4차원 토이 단어 벡터를 사용한다.
# - 그래픽카드 / GPU 처럼 동의어는 벡터를 가깝게 배치했다.
# - A1234 같은 고유 코드는 의미가 없는 임의의 식별자라서 0 벡터(=정보 없음)로 둔다.
#   -> 실제 임베딩 모델도 학습 데이터에 없던 코드/ID는 이렇게 의미를 거의 담지 못한다.
# - "파이썬"처럼 뜻이 여러 개인 단어는, 단어 하나만으로는 어느 뜻인지 알 수 없으므로
#   두 의미(프로그래밍 vs 동물) 중간값(애매한 벡터)으로 두고, 문장에 같이 쓰인 다른
#   단어들(사육사/자격증/코딩 등)이 평균을 어느 쪽으로 끌어당기는지로 의미를 가른다.
#   (실제 BERT 같은 문맥 임베딩은 이걸 훨씬 정교하게 하지만, 핵심 아이디어는 같다.)
word_vectors = {
    "그래픽카드": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "GPU": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    "RTX": [0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
    "4090": [0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
    "가격": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "정리": [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
    "고성능": [0.6, 0.2, 0.0, 0.0, 0.0, 0.0],
    "모델": [0.5, 0.3, 0.0, 0.0, 0.0, 0.0],
    "리스트": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "추천": [0.4, 0.4, 0.0, 0.2, 0.0, 0.0],
    "주문번호": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "환불": [0.0, 1.0, 0.1, 0.0, 0.0, 0.0],
    "절차": [0.0, 0.8, 0.0, 0.2, 0.0, 0.0],
    "안내": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    "주문": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "신청": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "방법": [0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
    "처리": [0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
    "기간": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
    "A1234": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 고유 코드 -> 의미 정보 없음
    "파이썬": [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],  # 다의어 -> 어느 뜻인지 모르니 애매한 중간값
    "자격증": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 교육/시험 도메인
    "시험": [0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
    "코딩": [0.0, 0.0, 0.0, 0.0, 0.95, 0.05],
    "프로그래밍": [0.0, 0.0, 0.0, 0.0, 0.95, 0.05],
    "사육사": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 동물 도메인
}
ZERO_VEC = [0.0] * 6


def sentence_vector(text):
    vecs = [word_vectors.get(tok, ZERO_VEC) for tok in text.split()]
    return np.mean(vecs, axis=0)


def cosine_sim(a, b):
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dense_scores(query, doc_tokens):
    q_vec = sentence_vector(query)
    return {name: cosine_sim(q_vec, sentence_vector(" ".join(t))) for name, t in doc_tokens.items()}


# ============================================================
# 3. Hybrid Search: 두 점수를 정규화해서 합치기
# ============================================================
def min_max_normalize(score_dict):
    values = np.array(list(score_dict.values()))
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return {k: 0.0 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def hybrid_scores(sparse, dense, alpha=0.5):
    sparse_n = min_max_normalize(sparse)
    dense_n = min_max_normalize(dense)
    hybrid = {k: alpha * sparse_n[k] + (1 - alpha) * dense_n[k] for k in sparse}
    return sparse_n, dense_n, hybrid


def rrf_scores(sparse, dense, k=60):
    # 점수 대신 "몇 위였는지(rank)"만 사용 -> 한쪽 점수가 극단적으로 튀어도 영향이 작다.
    def ranks(score_dict):
        ordered = sorted(score_dict.keys(), key=lambda name: score_dict[name], reverse=True)
        return {name: i + 1 for i, name in enumerate(ordered)}

    sparse_rank = ranks(sparse)
    dense_rank = ranks(dense)
    return {
        name: 1 / (k + sparse_rank[name]) + 1 / (k + dense_rank[name])
        for name in sparse
    }


# ============================================================
# 4. 쿼리별 실행 + 비교
# ============================================================
def print_ranking(title, score_dict):
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"  [{title}]")
    for name, score in ranked:
        print(f"    {name} ({docs[name]}): {score:.4f}")


for q_idx, (q_label, q_text) in enumerate(queries.items(), start=1):
    print(f"=== {q_label} ===")

    sparse = bm25_scores(q_text, doc_tokens)
    dense = dense_scores(q_text, doc_tokens)
    sparse_n, dense_n, hybrid = hybrid_scores(sparse, dense)

    print_ranking("1. Sparse (BM25) - 키워드 일치만 봄", sparse)
    print_ranking("2. Dense (의미 임베딩) - 의미 유사도만 봄", dense)
    print_ranking("3. Hybrid (정규화 후 0.5:0.5 결합)", hybrid)
    if q_idx == 3:
        rrf = rrf_scores(sparse, dense)
        print_ranking("4. Hybrid - RRF (순위 기반 결합)", rrf)
    print()

    # 시각화: 문서별 정규화 점수 비교 (Sparse vs Dense vs Hybrid)
    names = list(docs.keys())
    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, [sparse_n[n] for n in names], width=width, label="Sparse (BM25)")
    plt.bar(x, [dense_n[n] for n in names], width=width, label="Dense")
    plt.bar(x + width, [hybrid[n] for n in names], width=width, label="Hybrid")
    plt.xticks(x, names)
    plt.ylabel("Normalized score")
    plt.title(f"Query {q_idx}: Sparse vs Dense vs Hybrid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"06/output_{q_idx}_query{q_idx}.png")
    plt.show()

"""
관찰:
- Q1: D2(GPU 문서)는 쿼리와 겹치는 키워드가 하나도 없어 Sparse 점수가 0이다.
      Dense는 '그래픽카드'~'GPU'의 의미 유사도를 잡아내 D2를 가장 높게 평가한다.
      -> 키워드 검색만 썼을 때의 실패(동의어/의역 검색 실패)를 Dense가 보완.

- Q2: D3에는 정확한 주문번호 'A1234'가 있지만, 토이 임베딩은 그 코드에 의미를
      담지 못해(0벡터) Dense가 오히려 일반적인 환불 문서인 D4를 더 높게 평가한다.
      Sparse(BM25)는 'A1234'라는 희귀하고 고유한 토큰을 정확히 매칭해 D3을 1위로 둔다.
      -> 의미 기반 검색만 썼을 때의 실패(고유명사/코드/오탈자 검색 실패)를 Sparse가 보완.

- Q3: D5('파이썬 사육사 자격증 안내')는 쿼리 단어 '파이썬', '자격증'을 둘 다 그대로 담고 있어
      Sparse(BM25)에서 가장 높은 점수를 받는다. 하지만 실제로는 동물(파충류) 사육사 자격증으로,
      쿼리의 의도(프로그래밍 자격증)와는 무관한 거짓 양성(false positive)이다.
      D6('코딩 프로그래밍 자격증 시험 안내')는 '파이썬'이라는 단어를 전혀 안 썼지만 의미상 정답이다.
      Dense는 '파이썬'이라는 단어 하나만으로는 의미를 확정하지 않고, 함께 쓰인 다른 단어들
      ('사육사' vs '코딩/프로그래밍')을 같이 봐서 D6을 더 높게 평가한다(D6 0.9405 > D5 0.8485).
      그런데 0.5:0.5 가중합 Hybrid는 여전히 D5를 1위로 둔다(0.9511 > 0.6816).
      Sparse 쪽 점수 격차(D5가 D6의 약 2.75배)가 Dense 쪽 격차(약 1.1배)보다 훨씬 커서,
      정규화 후에도 Sparse의 극단적인 점수가 결합 결과를 끌고 가버리기 때문이다.
      순위만 보는 RRF로 바꾸면 D5와 D6가 정확히 동점(0.0325)이 된다 -> 점수 크기 차이에
      휘둘리지 않고, 두 문서가 똑같이 유력한 후보라는 더 합리적인 결론을 낸다.
      -> 가중합 방식의 한계(한쪽 점수가 극단적으로 튀면 결합 결과가 쏠림)를 RRF가 보완.

- Hybrid는 두 점수를 합쳐서 Q1, Q2에서는 실제로 의도된 문서(Q1->D2/D1, Q2->D3)를
  한쪽 방식만 썼을 때보다 더 안정적으로 상위로 끌어올린다.
  다만 Q3처럼 한쪽 점수의 격차가 극단적으로 클 때는 0.5:0.5 가중합도 틀릴 수 있고,
  이런 경우엔 RRF(순위 기반 결합)가 더 안정적인 결과를 준다.
"""
