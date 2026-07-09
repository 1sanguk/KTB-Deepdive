"""사용자 ID + 비밀번호 기반 인증. 동일 ID에는 하나의 비밀번호만 허용.

비밀번호는 PBKDF2-HMAC-SHA256 (salt 16 B, 반복 600,000회)으로 단방향 해시 후 저장.
저장 형식: {"salt": "<hex>", "hash": "<hex>"}
"""

import hashlib
import json
import os
from pathlib import Path

_DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
_USERS_FILE = _DATA_DIR / "users.json"
_ITERATIONS = 600_000


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


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, _ITERATIONS
    ).hex()


def login_or_register(user_id: str, password: str) -> tuple[bool, str]:
    """
    Returns (ok, msg).
      New ID           → register  → (True,  "registered")
      Existing + 정확한 PWD → login    → (True,  "logged_in")
      Existing + 틀린 PWD  → reject   → (False, "wrong_password")
    """
    users = _load()

    if user_id not in users:
        salt = os.urandom(16)
        users[user_id] = {"salt": salt.hex(), "hash": _hash(password, salt)}
        _save(users)
        return True, "registered"

    entry = users[user_id]
    # 기존 평문 레코드 자동 마이그레이션
    if isinstance(entry, str):
        if entry != password:
            return False, "wrong_password"
        salt = os.urandom(16)
        users[user_id] = {"salt": salt.hex(), "hash": _hash(password, salt)}
        _save(users)
        return True, "logged_in"

    salt = bytes.fromhex(entry["salt"])
    if _hash(password, salt) == entry["hash"]:
        return True, "logged_in"
    return False, "wrong_password"
