"""사용자 ID + 비밀번호 기반 인증. 동일 ID에는 하나의 비밀번호만 허용."""

import json
from pathlib import Path

_DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
_USERS_FILE = _DATA_DIR / "users.json"


def _load() -> dict:
    if not _USERS_FILE.exists():
        return {}
    try:
        return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(users: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def login_or_register(user_id: str, password: str) -> tuple[bool, str]:
    """
    Returns (ok, msg).
      New ID           → register  → (True,  "registered")
      Existing + 정확한 PWD → login    → (True,  "logged_in")
      Existing + 틀린 PWD  → reject   → (False, "wrong_password")
    """
    users = _load()
    if user_id not in users:
        users[user_id] = password
        _save(users)
        return True, "registered"
    if users[user_id] == password:
        return True, "logged_in"
    return False, "wrong_password"
