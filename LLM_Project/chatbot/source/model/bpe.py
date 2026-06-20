import json
import re
import unicodedata
from collections import Counter, defaultdict

# 텍스트를 "공백이 아닌 연속 구간" 또는 "공백 연속 구간"으로 분리한다.
# 이렇게 쪼개두면 나중에 그냥 이어붙이기만 해도 원본 텍스트가 그대로 복원된다.
WORD_RE = re.compile(r"\S+|\s+")

# base alphabet으로 쓸 문자 수. 코퍼스에서 가장 자주 등장하는 BASE_ALPHABET_SIZE개 문자만
# base alphabet에 포함하고, 나머지 희귀 문자(한자/이모지 등)는 모두 UNK 하나로 합친다.
# 이렇게 해야 base alphabet 크기가 vocab_size보다 커져서 merge가 한 번도 일어나지 못하는
# 상황(= 그냥 자모 단위 토크나이저가 되어버리는 상황)을 피할 수 있다.
BASE_ALPHABET_SIZE = 300
UNK = "�"


def decompose(text):
    """NFD 정규화: 한글 음절을 초성/중성/종성 자모로 분해한다 (base alphabet이 약 70~80개로 줄어듦)."""
    return unicodedata.normalize("NFD", text)


def compose(text):
    """NFC 정규화: 분해된 자모를 다시 한글 음절로 합친다. decompose()의 역변환."""
    return unicodedata.normalize("NFC", text)


def _split_words(text):
    return WORD_RE.findall(text)


def _get_pairs(tokens):
    """토큰 리스트에서 인접한 (토큰, 다음 토큰) 쌍을 모두 뽑는다."""
    return list(zip(tokens, tokens[1:]))


def train_bpe(text, vocab_size):
    """`text`로부터 BPE merge 규칙을 학습한다.

    반환값 (vocab, merges):
      - vocab: list[str]. 기본 자모/문자들 뒤에 학습 순서대로 merge된 토큰이 이어붙은 리스트
      - merges: dict[(str, str) -> int]. (쌍) -> 학습 순서(rank, 0이 가장 먼저 학습됨)
    """
    decomposed = decompose(text)
    words = _split_words(decomposed)
    word_freq = Counter(words)  # 동일한 단어가 코퍼스에 몇 번 등장하는지

    # 코퍼스에서 가장 자주 등장하는 문자들만 base alphabet으로 쓰고, 나머지는 UNK로 치환
    char_freq = Counter(decomposed)
    base_chars = sorted(c for c, _ in char_freq.most_common(BASE_ALPHABET_SIZE))
    base_set = set(base_chars)
    word_tokens = {w: [c if c in base_set else UNK for c in w] for w in word_freq}

    vocab = base_chars + [UNK]  # base alphabet (merge 0개 상태)
    merges = {}

    # pair_counts: (토큰a, 토큰b) 쌍이 코퍼스 전체에서 등장한 총 빈도
    # pair_to_words: 그 쌍을 포함하고 있는 "단어 문자열" 집합 (다음 merge 때 어떤 단어를 건드려야 하는지 추적)
    # 매 merge마다 전체 코퍼스를 다시 스캔하면 너무 느리므로, 이 두 자료구조를 증분(incremental)으로 갱신한다.
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for w, tokens in word_tokens.items():
        freq = word_freq[w]
        for pair in _get_pairs(tokens):
            pair_counts[pair] += freq
            pair_to_words[pair].add(w)

    while len(vocab) < vocab_size and pair_counts:
        # 가장 자주 등장하는 쌍을 하나 골라 새 토큰으로 합친다.
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:  # 더 합쳐봐야 의미 없는 수준이면 조기 종료
            break
        new_token = best_pair[0] + best_pair[1]
        merges[best_pair] = len(merges)  # 학습 순서 = rank
        vocab.append(new_token)

        # best_pair를 포함하고 있던 단어들만 골라서 갱신 (전체 스캔 X)
        for w in pair_to_words.pop(best_pair):
            tokens = word_tokens[w]
            freq = word_freq[w]

            # 1) 이 단어가 가지고 있던 기존 인접쌍들의 카운트를 먼저 제거
            for pair in _get_pairs(tokens):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    pair_to_words.pop(pair, None)

            # 2) best_pair가 나오는 부분을 new_token으로 합친 새 토큰 리스트 생성
            merged = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            word_tokens[w] = merged

            # 3) merge 후 새로 생긴 인접쌍들의 카운트를 추가
            for pair in _get_pairs(merged):
                pair_counts[pair] += freq
                pair_to_words[pair].add(w)

    return vocab, merges


def base_alphabet(vocab):
    """vocab에서 길이 1인 토큰들 = 학습 시 선택된 base alphabet(+UNK)."""
    return {t for t in vocab if len(t) == 1}


def _word_to_tokens(word, merges, base_set):
    """한 단어를 merge 규칙에 따라 BPE 토큰들로 쪼갠다.

    먼저 base alphabet에 없는 문자는 모두 UNK로 치환한 뒤, 현재 토큰들 중 merges에
    등록된 쌍을 찾아 rank(학습 순서)가 가장 낮은(= 가장 먼저 학습된, 우선순위가 높은)
    쌍부터 합친다. 더 합칠 쌍이 없으면 종료.
    """
    tokens = [c if c in base_set else UNK for c in word]
    while True:
        ranked = [(merges[pair], i) for i, pair in enumerate(_get_pairs(tokens)) if pair in merges]
        if not ranked:
            return tokens
        _, i = min(ranked)
        tokens = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2:]


def tokenize(text, merges, base_set):
    """`text`를 자모 단위로 분해한 뒤 BPE 토큰(list[str])으로 변환한다."""
    decomposed = decompose(text)
    tokens = []
    for word in _split_words(decomposed):
        tokens.extend(_word_to_tokens(word, merges, base_set))
    return tokens


def tokenize_with_offsets(text, merges, base_set):
    """tokenize()와 동일하지만, 각 토큰이 decompose(text) 안에서 차지하는 [start, end) 구간도 같이 반환한다.
    (추출형 QA에서 정답 문자 위치 -> 토큰 인덱스로 변환할 때 사용)"""
    decomposed = decompose(text)
    tokens, offsets = [], []
    pos = 0
    for word in _split_words(decomposed):
        for t in _word_to_tokens(word, merges, base_set):
            tokens.append(t)
            offsets.append((pos, pos + len(t)))
            pos += len(t)
    return tokens, offsets, decomposed


def build_vocab(vocab):
    """vocab 리스트로부터 토큰<->id 매핑(stoi/itos)을 만든다."""
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def encode(text, merges, stoi, base_set):
    """텍스트 -> BPE 토큰 -> id 리스트."""
    return [stoi[t] for t in tokenize(text, merges, base_set)]


def decode(ids, itos):
    """id 리스트 -> BPE 토큰 문자열 결합 -> NFC로 재조합한 텍스트."""
    return compose("".join(itos[i] for i in ids))


def save_bpe(path, vocab, merges):
    """학습된 vocab/merges를 JSON으로 저장 (다음 실행 시 재학습 없이 로드)."""
    ordered_pairs = [list(pair) for pair, _rank in sorted(merges.items(), key=lambda kv: kv[1])]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": ordered_pairs}, f, ensure_ascii=False)


def load_bpe(path):
    """save_bpe로 저장한 JSON을 읽어 (vocab, merges)로 복원."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    merges = {(a, b): rank for rank, (a, b) in enumerate(data["merges"])}
    return data["vocab"], merges
