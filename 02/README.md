# 02 — 이터레이터와 제너레이터

---

## 주제

> 이터레이터와 제네레이터가 메모리 공간 효율성을 개선하는 방식을 설명하고,  
> 대규모 데이터 처리(예: 로그 파일 분석)에서 이를 활용하는 구체적인 시나리오를 제시하시오.

기본 언어: **Python**

---

## 이터레이터 (Iterator)

| 구분 | 설명 |
|------|------|
| 일반명사 | 반복하는 하나 (One Which Iterates) |
| 고유명사 | 순서대로 다음 값을 가져올 수 있는 객체 |
| 사용 이유 | 데이터 구조와 별개로, 데이터를 어떻게 하나씩 꺼내서 쓸지 판단하기 위함 |
| 사용 방법 | 반복 가능한 객체(리스트, 큐, 딕셔너리 등)에서 순서대로 값을 꺼내어 사용 |

이터레이터 객체는 기본적으로 두 가지 메서드를 가진다.

- `__iter__()` — 자기 자신을 이터레이터로 반환
- `__next__()` — 다음 값을 반환하며, 다음 값이 없을 경우 `StopIteration` 에러 발생

```python
lst = [1, 2, 3, 4, 5]
li = iter(lst)

print(next(li))  # 1
print(next(li))  # 2
print(next(li))  # 3
print(next(li))  # 4
print(next(li))  # 5
print(next(li))  # StopIteration Error
```

> **StopIteration Error**: 반환할 값이 더 이상 없을 때 발생하는 에러

---

## 제너레이터 (Generator)

| 구분 | 설명 |
|------|------|
| 일반명사 | 발전기 |
| 고유명사 | 함수의 실행 상태를 보존해 호출 시점마다 값을 생성하는 객체 |
| 사용 이유 | 이터레이터보다 구현이 짧고 간결하며 메모리 절약에 적합 |
| 사용 방법 | `yield`를 사용하여 값을 반환 |

> 제너레이터는 이터레이터의 한 종류이다.  
> 이터레이터는 `__iter__()`와 `__next__()`를 가진 인터페이스(포괄 개념)이고,  
> 제너레이터는 이를 `yield`로 구현한 특수한 이터레이터이다.

---

## 메모리 효율성 비교

```python
nums_list      = [x for x in range(1_000_000)]   # 즉시 평가 — 100만 개 전부 메모리에 저장
nums_iter      = iter(nums_list)                 # 리스트-이터레이터 객체
nums_generator = (x for x in range(1_000_000))   # 지연 평가 — 필요할 때만 생성
```

### 즉시 평가 vs 지연 평가

|  | 즉시 평가 (Eager) | 지연 평가 (Lazy) |
|--|:-----------------:|:----------------:|
| **메모리** | 데이터 크기에 비례하여 증가 | 일정 수준 유지 |
| **응답 속도** | 전체 계산 후 반환 | 첫 값 즉시 반환 |
| **데이터 한계** | RAM 용량에 종속 | 무한 데이터 스트림도 처리 가능 |
| **접근 방식** | 인덱스로 어디든 접근 가능 | 순차적 접근만 가능 |
| **재사용성** | 여러 번 반복 가능 | 한 번 소비 후 소진 |

**비유**
- `nums_list` — 책 100만 장을 한꺼번에 펼쳐놓은 것
- `nums_iter` / `nums_generator` — 필요할 때마다 책 페이지 한 장을 펼치는 것

---

## Example 1 — 기본 크기 비교

[example1.py](example1.py) 참고

```
nums_list      = 8,448,728 bytes  ≈ 8 MB
nums_iter      =        48 bytes
nums_generator =       200 bytes
```

---

## Example 2 — 참조 객체 포함 크기 비교

[example2.py](example2.py) 참고

```python
nums_iter = iter([x for x in range(10_000_000)])   # 이터레이터
nums_gene = (x for x in range(10_000_000))         # 제너레이터
```

| 측정 방식 | `lst` | `nums_iter` | `nums_gene` |
|-----------|------:|------------:|------------:|
| `sys.getsizeof` | 85,176 B | 48 B | 200 B |
| `asizeof` (참조 포함) | 405,176 B | 405,224 B | **240 B** |

- `nums_iter`는 내부적으로 원본 리스트를 참조하므로 참조 포함 크기가 리스트와 거의 동일하다.
- `nums_gene`는 생성 규칙만 저장하므로 참조 포함 크기도 매우 작다.
- **결론**: 메모리 절약 효과는 `generator > iterator`

---

## Example 3 — 제너레이터 표현식

[example3.py](example3.py) 참고

---

## Example 4 — 제너레이터 함수를 통한 스트리밍 파이프라인

[example4.py](example4.py) 참고

---

## Example 5 — 대규모 CSV 처리 벤치마크

[example5.py](example5.py) 참고

```
=== LIST BASED ===
before:              20.875 MB
after list create:  595.043 MB
asizeof(rows):    591,650,176 B
matched:               15,899
time:               10.006 sec
after del:           23.578 MB

=== GENERATOR BASED ===
before:              23.578 MB
asizeof(reader):         832 B
matched:               15,899
time:                5.267 sec
after processing:    23.746 MB

=== GENERATOR BASED 2 (REAL GENERATOR) ===
before:              22.781 MB
asizeof(reader):         312 B
matched:               15,899
time:                4.680 sec
after processing:    22.785 MB
```

| 방식 | 메모리 사용량 | 처리 시간 |
|------|:------------:|:--------:|
| 리스트 기반 | ~595 MB | 10.0 sec |
| 제너레이터 (iter 래핑) | ~24 MB | 5.3 sec |
| 제너레이터 (순수) | ~23 MB | **4.7 sec** |

---

## 요약

이터레이터와 제너레이터는 값을 한꺼번에 만들지 않고 **필요한 시점에 하나씩 생성**하는 방식으로 메모리 사용을 줄인다.  
이 원리는 **지연 평가(Lazy Evaluation)** 와 스트리밍 처리에 기반하며,  
구현 방식(제너레이터 표현식, `itertools`, 사용자 정의 이터레이터)에 따라 실제 절약 효과가 달라진다.

> 대규모 데이터 처리에서 리스트 대신 제너레이터를 사용하면  
> 메모리를 **약 96% 절약**하고 속도도 **약 2배 향상**된다. (example5 기준)
