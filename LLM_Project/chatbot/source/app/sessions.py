"""사용자별 세션 목록 관리 — JSON 파일 기반."""

import json
import uuid
from datetime import datetime
from pathlib import Path

_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _path(user_id: str) -> Path:
    return _SESSIONS_DIR / f"{user_id}.json"


def list_sessions(user_id: str) -> list:
    p = _path(user_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(user_id: str, sessions: list) -> None:
    _path(user_id).write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")


def create_session(user_id: str, name: str = "새 채팅") -> dict:
    session = {
        "id": str(uuid.uuid4()),
        "name": name[:40],
        "created_at": datetime.now().isoformat(),
    }
    sessions = list_sessions(user_id)
    sessions.insert(0, session)
    _save(user_id, sessions)
    return session


def rename_session(user_id: str, session_id: str, name: str) -> bool:
    sessions = list_sessions(user_id)
    for s in sessions:
        if s["id"] == session_id:
            s["name"] = name[:40]
            _save(user_id, sessions)
            return True
    return False
