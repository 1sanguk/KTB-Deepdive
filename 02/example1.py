import sys

nums_list = [x for x in range(1000000)]
nums_iter = iter(nums_list)
nums_generator = (x for x in range(1000000))

print(nums_list)
print(nums_iter)
print(nums_generator)

print(sys.getsizeof(nums_list))  # 리스트 크기 (MB)
print(sys.getsizeof(nums_iter))  # 이터레이터 크기 (바이트)
print(sys.getsizeof(nums_generator))  # 제너레이터 크기 (바이트)