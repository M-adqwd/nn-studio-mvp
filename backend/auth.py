"""认证：用户表 + PBKDF2 密码哈希 + HMAC 签名 Bearer token（仅标准库）。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid

from fastapi import APIRouter, Body, Depends, Header, HTTPException

import db

TOKEN_TTL = 12 * 3600  # 12 小时

SECRET_PATH = db.BASE_DIR / ".secret"


def _load_secret() -> bytes:
    """签名密钥持久化到文件，避免服务重启后全员掉线。"""
    if SECRET_PATH.exists():
        return SECRET_PATH.read_bytes().strip()
    secret = secrets.token_hex(32).encode()
    SECRET_PATH.write_bytes(secret)
    return secret


SECRET = _load_secret()


# ---------- 密码 ----------

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 60_000)
    return dk.hex(), salt


def verify_password(password: str, expected_hash: str, salt: str) -> bool:
    actual, _ = hash_password(password, salt)
    return hmac.compare_digest(actual, expected_hash)


# ---------- Token ----------

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(user_id: str) -> str:
    payload = _b64e(json.dumps(
        {"uid": user_id, "exp": int(time.time()) + TOKEN_TTL},
        separators=(",", ":"),
    ).encode())
    sig = hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def parse_token(token: str) -> str | None:
    try:
        payload, sig = token.split(".", 1)
    except ValueError:
        return None
    expected = hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        data = json.loads(_b64d(payload))
    except Exception:  # noqa: BLE001
        return None
    if int(data.get("exp", 0)) < time.time():
        return None
    return data.get("uid")


# ---------- 用户 ----------

def public_user(u: dict) -> dict:
    return {
        "id": u["id"],
        "account": u["account"],
        "name": u["name"],
        "role": u["role"],
        "org": u.get("org"),
    }


def create_user(account: str, password: str, name: str, role: str = "成员", org: str = "") -> dict:
    pw_hash, salt = hash_password(password)
    user = {
        "id": f"u_{uuid.uuid4().hex[:10]}",
        "account": account,
        "name": name or account,
        "role": role,
        "org": org,
        "password_hash": pw_hash,
        "salt": salt,
    }
    db.insert_user(user)
    return user


def ensure_admin() -> None:
    """首次启动写入管理员账号。密码取环境变量，不写死。"""
    if not db.get_user_by_account("admin"):
        import os
        pwd = os.environ.get("NN_ADMIN_PASSWORD") or "Admin@123"
        create_user("admin", pwd, name="管理员", role="平台管理员", org="")


# ---------- FastAPI 依赖 ----------

def current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "未登录或登录已过期")
    uid = parse_token(authorization.split(None, 1)[1].strip())
    user = db.get_user(uid) if uid else None
    if not user:
        raise HTTPException(401, "未登录或登录已过期")
    return user


# ---------- 路由 ----------

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login")
def login(body: dict = Body(...)) -> dict:
    account = (body.get("account") or "").strip()
    password = body.get("password") or ""
    if not account or not password:
        raise HTTPException(400, "请输入账号与密码")
    user = db.get_user_by_account(account)
    if not user or not verify_password(password, user["password_hash"], user["salt"]):
        raise HTTPException(401, "账号或密码不正确")
    return {"token": issue_token(user["id"]), "user": public_user(user)}


@router.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    return public_user(user)


@router.post("/register")
def register(body: dict = Body(...), user: dict = Depends(current_user)) -> dict:
    """仅管理员可创建新账号。"""
    if user.get("role") != "平台管理员":
        raise HTTPException(403, "仅平台管理员可创建账号")
    account = (body.get("account") or "").strip()
    password = body.get("password") or ""
    name = (body.get("name") or "").strip()
    if not account or len(account) < 3:
        raise HTTPException(400, "账号至少 3 个字符")
    if len(password) < 8:
        raise HTTPException(400, "密码至少 8 位")
    if db.get_user_by_account(account):
        raise HTTPException(409, "账号已存在")
    new_user = create_user(account, password, name=name or account,
                             role=body.get("role") or "成员", org=body.get("org") or "")
    return public_user(new_user)
