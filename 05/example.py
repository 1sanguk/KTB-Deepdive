import numpy as np
import matplotlib.pyplot as plt

# 실습용으로 작은 값을 사용 (실제 모델은 d_model=512~12288, seq_len도 훨씬 큼)
d_model = 16   # 임베딩 벡터의 차원
seq_len = 50   # 문장 길이 (토큰 개수)

# 사용 예시에서 공통으로 쓸 3개 토큰짜리 문장의 단어 임베딩 (예: "나는", "학교에", "간다")
np.random.seed(1)
word_embeddings = np.random.randn(3, d_model) * 0.1


# ============================================================
# 1. 트랜스포머의 기본 위치 인코딩 (Sinusoidal Positional Encoding)
# ============================================================
# pos: 토큰 위치, i: 임베딩 차원 인덱스
# 짝수 차원은 sin, 홀수 차원은 cos을 사용하고, 차원마다 주기(파장)가 다르다.
def sinusoidal_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]                     # (seq_len, 1)
    div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)   # (d_model/2,)

    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)
    return pe

pe_sinusoidal = sinusoidal_positional_encoding(seq_len, d_model)

print("=== 1. 기본 위치 인코딩 (Sinusoidal) ===")
print(f"shape: {pe_sinusoidal.shape}  (seq_len, d_model)")
print(f"0번째 위치 벡터: {np.round(pe_sinusoidal[0], 3)}")
print(f"1번째 위치 벡터: {np.round(pe_sinusoidal[1], 3)}")
print("-> 위치가 1만 달라져도 벡터 전체가 달라진다 (위치마다 고유한 패턴)")
print()

# --- 사용 예시: 단어 임베딩 + Sinusoidal PE = 최종 입력 벡터 (더하기) ---
final_input_sinusoidal = word_embeddings + pe_sinusoidal[:3]
print("--- 사용 예시: 단어 임베딩 + Sinusoidal PE ---")
for pos in range(3):
    print(f"위치 {pos}")
    print(f"  단어 임베딩    : {np.round(word_embeddings[pos], 3)}")
    print(f"  + PE(pos={pos})  : {np.round(pe_sinusoidal[pos], 3)}")
    print(f"  = 최종 입력    : {np.round(final_input_sinusoidal[pos], 3)}")
print()

plt.figure(figsize=(10, 4))
plt.imshow(pe_sinusoidal.T, cmap="RdBu", aspect="auto")
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("1. Sinusoidal Positional Encoding")
plt.colorbar()
plt.tight_layout()
plt.savefig("05/output_1_sinusoidal.png")
plt.show()

# 1-1. 차원별 sin 곡선 (차원마다 진동 속도(주기)가 다르다 = 시계의 시/분/초침)
plt.figure(figsize=(10, 5))
dims_to_show = [0, 2, 6, 10, 14]  # 짝수(=sin) 차원 중 일부만 비교
for dim in dims_to_show:
    plt.plot(pe_sinusoidal[:, dim], label=f"dim {dim}", marker="o", markersize=3)
plt.xlabel("Position")
plt.ylabel("sin value")
plt.title("1-1. sin")
plt.legend(title="Imbedding Dimension")
plt.tight_layout()
plt.savefig("05/output_1_1_waves.png")
plt.show()

# 1-2. 위치별 인코딩 벡터 비교 (위치마다 고유한 패턴(지문)을 가진다)
plt.figure(figsize=(10, 4))
positions_to_show = [0, 1, 10, 30]
x = np.arange(d_model)
width = 0.2
for idx, pos in enumerate(positions_to_show):
    plt.bar(x + idx * width, pe_sinusoidal[pos], width=width, label=f"position {pos}")
plt.xlabel("Imbedding Dimension")
plt.ylabel("Encoding Value")
plt.title("1-2. Positional Encoding Vectors Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("05/output_1_2_position_vectors.png")
plt.show()


# ============================================================
# 2. BERT의 학습 가능한 절대 위치 인코딩 (Learned Absolute PE)
# ============================================================
# sin/cos 같은 고정 함수 대신, 위치(0, 1, 2, ...)마다 학습 가능한 벡터를
# "테이블(임베딩 매트릭스)"에 담아두고 학습 과정에서 값을 업데이트한다.
# 구조는 단어 임베딩 테이블과 완전히 동일하고, input이 단어 ID 대신 위치 인덱스일 뿐이다.
max_position = 50  # BERT는 보통 512. 여기서는 seq_len과 동일하게 설정

np.random.seed(42)
learned_pe_table = np.random.randn(max_position, d_model) * 0.02  # 학습 전 랜덤 초기화

print("=== 2. BERT 학습 가능한 절대 위치 인코딩 ===")
print(f"position embedding table shape: {learned_pe_table.shape}  (max_position, d_model)")
print(f"0번째 위치 벡터(학습 전 랜덤값): {np.round(learned_pe_table[0], 3)}")
print()

# max_position을 넘어가는 위치를 조회하면 테이블에 행 자체가 없다.
over_position = max_position  # 0-index 기준으로 테이블 범위를 벗어난 위치
try:
    _ = learned_pe_table[over_position]
except IndexError as e:
    print(f"위치 {over_position} 조회 시 에러 발생: {e}")
    print("-> 학습 때 본 적 없는 길이의 문장은 처리할 수 없다 (sinusoidal은 가능했음)")
print()

# --- 사용 예시: 단어 임베딩 + BERT 학습형 PE = 최종 입력 벡터 (더하기) ---
final_input_bert = word_embeddings + learned_pe_table[:3]
print("--- 사용 예시: 단어 임베딩 + BERT 학습형 PE (테이블 조회) ---")
for pos in range(3):
    print(f"위치 {pos}")
    print(f"  단어 임베딩       : {np.round(word_embeddings[pos], 3)}")
    print(f"  + 위치 임베딩[{pos}]  : {np.round(learned_pe_table[pos], 3)}")
    print(f"  = 최종 입력       : {np.round(final_input_bert[pos], 3)}")
print()


# ============================================================
# 3. RoPE (Rotary Positional Embedding) - 상대 위치 인코딩
# ============================================================
# 위치 정보를 벡터에 "더하지" 않고, Query/Key 벡터를 위치에 따라 회전시킨다.
# 임베딩 벡터를 2개씩 짝지어 2차원 좌표로 보고, 각 좌표쌍을 (위치 * 회전각도)만큼 회전한다.
# 짝마다 회전 속도(각도)가 다른데, 이 속도는 sinusoidal과 동일한 주파수를 사용한다.

def rotate_half(x):
    # (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...) : 짝마다 90도 방향의 짝을 만든다
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return np.stack([-x2, x1], axis=-1).reshape(x.shape)

def apply_rope(x, position, d_model):
    theta = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))  # 짝(pair)마다 다른 회전 속도
    angles = position * theta                                       # 위치 * 속도 = 회전 각도

    cos = np.repeat(np.cos(angles), 2)
    sin = np.repeat(np.sin(angles), 2)

    return x * cos + rotate_half(x) * sin

# 임의의 query, key 벡터 (실제로는 학습된 attention 벡터라고 생각하면 된다)
np.random.seed(0)
q = np.random.randn(d_model)
k = np.random.randn(d_model)

print("=== 3. RoPE (회전 위치 인코딩) ===")

# --- 사용 예시: RoPE는 "더하기"가 아니라 Q/K 벡터 자체를 회전시켜서 사용 ---
print("--- 사용 예시: RoPE는 임베딩에 더하지 않고, Q/K 벡터를 위치에 따라 회전시켜 사용 ---")
print("(1, 2번처럼 '최종 입력 벡터'를 따로 만들지 않는다. 입력 임베딩은 그대로 두고,")
print(" attention 계산 시 Query 벡터에만 위치별 회전을 적용한다고 생각하면 된다.)")
for pos in range(3):
    q_rot = apply_rope(q, pos, d_model)
    print(f"위치 {pos}")
    print(f"  Query 원본      : {np.round(q, 3)}")
    print(f"  회전 후 q_rot   : {np.round(q_rot, 3)}")
print()

print("query/key를 각자의 위치로 회전시킨 뒤 내적(attention score)을 구하면,")
print("결과는 두 위치의 '차이(상대 위치)'에만 의존한다.")
print()

pairs = [(0, 0), (5, 5), (0, 5), (10, 15), (100, 105)]
for pos_q, pos_k in pairs:
    q_rot = apply_rope(q, pos_q, d_model)
    k_rot = apply_rope(k, pos_k, d_model)
    score = np.dot(q_rot, k_rot)
    print(f"query 위치={pos_q:>4}, key 위치={pos_k:>4}  (상대거리={pos_k - pos_q:>3})  ->  내적 점수: {score:.4f}")

print()
print("-> 상대거리가 0인 (0,0)과 (5,5)의 점수가 같고,")
print("   상대거리가 5인 (0,5), (10,15), (100,105)의 점수도 모두 같다.")
print("   즉 절대 위치가 아니라 '위치 차이'만 attention 점수에 반영된다.")
