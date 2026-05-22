from pympler import asizeof
import sys

N = 10000  # 실제 실행 시 메모리 주의.

# 데이터/이터레이터/제너레이터 생성
lst = [x for x in range(N)]
numsiter = iter(lst)
numsgene = (x for x in range(N))

# 타입 확인
print(type(numsiter))    # <class 'list_iterator'>
print(type(numsgene))    # <class 'generator'>

# sys.getsizeof: 객체 자체(얕은 크기)
print("sys.getsizeof(lst):", sys.getsizeof(lst))
print("sys.getsizeof(numsiter):", sys.getsizeof(numsiter))
print("sys.getsizeof(numsgene):", sys.getsizeof(numsgene))

# pympler.asizeof.asizeof: 객체 그래프 전체(깊이 있는 크기)
print("asizeof.asizeof(lst):", asizeof.asizeof(lst))
print("asizeof.asizeof(numsiter):", asizeof.asizeof(numsiter))
print("asizeof.asizeof(numsgene):", asizeof.asizeof(numsgene))

# 제너레이터를 한 번 소비해보고(주의: 큰 N에서는 메모리 폭주)
# 아래는 예시로 작은 N에서만 실행 권장
# lst_from_gen = list(numsgene)
# print("asizeof.asizeof(lst_from_gen):", asizeof.asizeof(lst_from_gen))