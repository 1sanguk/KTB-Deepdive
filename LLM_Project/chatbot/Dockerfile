FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 (llama-cpp-python 빌드에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (코드 변경 시 캐시 활용)
COPY requirements.txt .
# llama-cpp-python은 prebuilt wheel이 AVX512VNNI를 요구할 수 있으므로 requirements에서 제외하고
# VNNI 없는 CPU(EC2 Xeon 8259CL 등)를 위해 소스 빌드로 설치
RUN grep -v "llama-cpp-python" requirements.txt > /tmp/requirements_no_llama.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_llama.txt
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=OFF" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python==0.3.33 --no-binary llama-cpp-python

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
