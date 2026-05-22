# 제너레이터 표현식
# 메모리 효율적: 값은 필요할 때만 생성
gen = (x*x for x in range(10**7))
for i, v in enumerate(gen):
    if i >= 5:
        break
    print(v)