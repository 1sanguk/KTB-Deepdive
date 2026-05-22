# 실행 전 주의
# - 이 스크립트는 큰 파일을 생성합니다. 기본값 N = 1_000_000(백만 행)으로 설정되어 있습니다.
# - 먼저 작은 N(예: 10000)으로 테스트하세요.
# - 실행 환경의 디스크 여유 공간과 메모리를 확인하세요.
# - Python 3.7+ 권장.

import csv
import random
from datetime import datetime, timedelta

def generate_big_csv(
    path="big.csv",
    n_rows=10_000_000,
    chunk_size=100_000,
    seed=42
):
    """
    스트리밍 방식으로 큰 CSV 파일을 생성합니다.
    - path: 출력 파일명
    - n_rows: 생성할 총 행 수
    - chunk_size: 한 번에 생성해서 쓰는 행 수 (메모리 제어용)
    - seed: 재현 가능한 난수 시드
    컬럼: id, name, age, country, score, created_at
    """
    random.seed(seed)

    countries = [
        "South Korea", "United States", "China", "Japan", "Germany",
        "France", "United Kingdom", "India", "Brazil", "Canada"
    ]

    # 간단한 이름 생성기 (의존성 없음)
    first_names = ["Min", "Ji", "Seo", "Hyun", "Jin", "Soo", "Young", "Hye", "Dong", "Eun"]
    last_names = ["Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Lim", "Han"]

    start_time = datetime(2020, 1, 1)

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 헤더
        writer.writerow(["id", "name", "age", "country", "score", "created_at"])

        written = 0
        while written < n_rows:
            this_chunk = min(chunk_size, n_rows - written)
            rows = []
            base_id = written + 1
            for i in range(this_chunk):
                idx = base_id + i
                name = f"{random.choice(last_names)} {random.choice(first_names)}"
                age = random.randint(18, 80)
                country = random.choice(countries)
                score = round(random.random() * 100, 2)
                # created_at: start_time + random offset up to 2000 days, plus random seconds
                offset_days = random.randint(0, 2000)
                offset_seconds = random.randint(0, 86400)
                created_at = (start_time + timedelta(days=offset_days, seconds=offset_seconds)).isoformat(sep=" ")
                rows.append([idx, name, age, country, score, created_at])

            writer.writerows(rows)
            written += this_chunk
            # 진행 출력 (콘솔에서 확인)
            print(f"Written {written}/{n_rows} rows")

    print(f"Finished writing {n_rows} rows to {path}")

if __name__ == "__main__":
    # 안전을 위해 기본 N은 10000으로 설정되어 있습니다.
    # 실제로 큰 파일을 원하면 아래 n_rows 값을 변경하세요.
    generate_big_csv(path="big.csv", n_rows=1000000, chunk_size=100000, seed=42)