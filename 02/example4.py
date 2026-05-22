# make_big_csv.py를 통해 우선 파일을 만들어 두자.
import csv

def read_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def filter_rows(rows):
    for r in rows:
        age_str = r.get('age', '')
        try:
            if int(age_str) > 79:
                yield r
        except ValueError:
            # 잘못된 값은 건너뜀
            continue

def process(row):
    # 예: 출력하거나 DB에 저장
    print(row)

if __name__ == "__main__":
    rows = read_rows('big.csv')
    filtered = filter_rows(rows)
    for row in filtered:
        process(row)