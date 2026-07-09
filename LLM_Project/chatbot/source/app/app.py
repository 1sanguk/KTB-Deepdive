import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# HuggingFace 토크나이저 멀티프로세싱 비활성화 — 요청 처리 중 segfault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "api_keys")

# 프로젝트 내 패키지 경로 등록 (state, routers 등의 import보다 먼저)
_SOURCE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SOURCE / "model" / "sop_model"))
sys.path.insert(0, str(_SOURCE))

from fastapi import FastAPI

import state  # 모델·체인·검색기 초기화 (import 시점에 실행)
from routers import chat, stream
from ui import router as ui_router

app = FastAPI(title="SOP_GPT 한국어 챗봇")

app.include_router(chat.router)
app.include_router(stream.router)
app.include_router(ui_router)
