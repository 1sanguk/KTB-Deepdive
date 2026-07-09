"""히스토리 영속화 유틸리티 — JSON 파일 기반 대화 이력 관리."""

import json
from pathlib import Path

_history_dir = Path(__file__).resolve().parent.parent / "data" / "history"
_history_dir.mkdir(parents=True, exist_ok=True)


def load_history(thread_id: str) -> list:
    """저장된 JSON 히스토리 반환. 없으면 빈 리스트."""
    path = _history_dir / f"{thread_id}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(thread_id: str, messages: list) -> None:
    """LangGraph용: MemorySaver 전체 메시지 리스트를 저장. 세션 내 누적이 더 길면 덮어쓰기."""
    _history_dir.mkdir(parents=True, exist_ok=True)
    path = _history_dir / f"{thread_id}.json"
    existing = load_history(thread_id)
    to_save = messages if len(messages) >= len(existing) else existing + messages
    path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(thread_id: str, new_messages: list) -> None:
    """Basic/RAG/LangChain용: 기존 히스토리에 새 메시지 이어붙이기."""
    _history_dir.mkdir(parents=True, exist_ok=True)
    existing = load_history(thread_id)
    path = _history_dir / f"{thread_id}.json"
    path.write_text(json.dumps(existing + new_messages, ensure_ascii=False, indent=2), encoding="utf-8")
