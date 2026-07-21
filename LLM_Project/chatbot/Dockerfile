FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 (llama-cpp-python 빌드에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (코드 변경 시 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY source/ ./source/
COPY ragdata/ ./ragdata/

# 데이터 디렉토리 생성 (볼륨 마운트 전 기본 구조)
RUN mkdir -p source/data/history source/.cache

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# app.py가 sys.path를 source/ 기준으로 구성하므로 source/ 에서 실행
WORKDIR /app/source/app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
