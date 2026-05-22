# pip install psutil pympler
import os, time, csv, gc
import psutil
from pympler import asizeof
import tracemalloc

proc = psutil.Process(os.getpid())

def mem_mb():
    return proc.memory_info().rss / 1024**2

def sample_file(path, n=10):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            print(f"LINE {i}:", repr(line.rstrip('\n')))
            if i >= n-1:
                break

def list_based(path):
    print("=== LIST BASED START ===")
    print("before:", mem_mb(), "MB")
    t0 = time.time()
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print("after list create:", mem_mb(), "MB")
    print("asizeof(rows):", asizeof.asizeof(rows))
    matched = 0
    for r in rows:
        try:
            if int(r.get('age','')) > 79:
                matched += 1
        except ValueError:
            continue
    t1 = time.time()
    print("matched:", matched)
    print("time:", t1 - t0, "sec")
    del rows
    gc.collect()
    print("after del:", mem_mb(), "MB")
    print("=== LIST BASED END ===\n")

def generator_based(path):
    print("=== GENERATOR BASED START ===")
    print("before:", mem_mb(), "MB")
    t0 = time.time()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    matched = 0
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        print("asizeof(reader):", asizeof.asizeof(reader))
        for r in reader:
            try:
                if int(r.get('age','')) > 79:
                    matched += 1
            except ValueError:
                continue

    t1 = time.time()
    snap2 = tracemalloc.take_snapshot()
    top_stats = snap2.compare_to(snap1, 'lineno')[:10]
    print("matched:", matched)
    print("time:", t1 - t0, "sec")
    print("after processing:", mem_mb(), "MB")
    print("tracemalloc top 10:")
    for stat in top_stats:
        print(stat)
    tracemalloc.stop()
    gc.collect()
    print("=== GENERATOR BASED END ===\n")

# 새로 추가된 진짜 제너레이터 함수
def iter_matching_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                if int(r.get('age','')) > 79:
                    yield r
            except ValueError:
                continue

def generator_based2(path):
    print("=== GENERATOR BASED 2 START (REAL GENERATOR) ===")
    print("before:", mem_mb(), "MB")
    t0 = time.time()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    matched = 0
    # iter_matching_rows는 yield를 사용하는 제너레이터
    print("asizeof(reader):", asizeof.asizeof(iter_matching_rows(path)))
    for r in iter_matching_rows(path):
        # 여기서 필요한 최소한의 처리만 수행
        matched += 1

    t1 = time.time()
    snap2 = tracemalloc.take_snapshot()
    top_stats = snap2.compare_to(snap1, 'lineno')[:10]
    print("matched:", matched)
    print("time:", t1 - t0, "sec")
    print("after processing:", mem_mb(), "MB")
    print("tracemalloc top 10:")
    for stat in top_stats:
        print(stat)
    tracemalloc.stop()
    gc.collect()
    print("=== GENERATOR BASED 2 END ===\n")

if __name__ == "__main__":
    path = "big.csv"
    print("File sample:")
    sample_file(path, n=10)
    print()
    list_based(path)
    generator_based(path)
    generator_based2(path)