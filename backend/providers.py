"""大模型 Provider 配置：OpenAI 兼容接口的 CRUD 与真实连通性测试。"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException

import auth
import db

router = APIRouter()


def _masked(p: dict) -> dict:
    """列表/详情返回时遮蔽 api_key，仅保留后 4 位。"""
    p = dict(p)
    key = p.get("api_key") or ""
    p["api_key"] = ("****" + key[-4:]) if key else ""
    p["has_key"] = bool(key)
    p["is_default"] = bool(p.get("is_default"))
    return p


def _test_connection(base_url: str, api_key: str, model: str) -> dict:
    """调用 {base_url}/chat/completions 做真实连通性测试（短超时）。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            latency = (time.time() - t0) * 1000
            if resp.status == 200:
                return {"ok": True, "status": "active", "latency_ms": round(latency, 1), "detail": "连接成功，模型响应正常"}
            return {"ok": False, "status": "unreachable", "latency_ms": round(latency, 1),
                    "detail": f"HTTP {resp.status}"}
    except urllib.error.HTTPError as e:
        latency = (time.time() - t0) * 1000
        if e.code in (401, 403):
            return {"ok": False, "status": "auth_failed", "latency_ms": round(latency, 1),
                    "detail": f"鉴权失败（HTTP {e.code}），请检查 API Key"}
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "status": "unreachable", "latency_ms": round(latency, 1),
                "detail": f"HTTP {e.code}：{body or e.reason}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": "unreachable", "latency_ms": None,
                "detail": f"无法连接：{e}"}


# ---------- CRUD ----------

@router.get("/api/providers")
def list_providers(user: dict = Depends(auth.current_user)) -> list[dict]:
    return [_masked(p) for p in db.list_providers()]


@router.post("/api/providers")
def create_provider(body: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    name = (body.get("name") or "").strip()
    base_url = (body.get("base_url") or "").strip()
    model = (body.get("model") or "").strip()
    if not name or not base_url or not model:
        raise HTTPException(400, "名称、Base URL 与模型名为必填项")
    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(400, "Base URL 需以 http(s):// 开头（OpenAI 兼容接口，如 https://api.openai.com/v1）")
    if body.get("is_default"):
        db.clear_default_provider()
    rec = {
        "id": f"prv_{uuid.uuid4().hex[:10]}",
        "name": name,
        "provider_type": "openai_compat",
        "base_url": base_url,
        "api_key": body.get("api_key") or "",
        "model": model,
        "is_default": bool(body.get("is_default")),
    }
    db.insert_provider(rec)
    return _masked(db.get_provider(rec["id"]))


@router.put("/api/providers/{pid}")
def update_provider(pid: str, body: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    p = db.get_provider(pid)
    if not p:
        raise HTTPException(404, "Provider 不存在")
    fields = {}
    for k in ("name", "base_url", "model"):
        if body.get(k) is not None and str(body[k]).strip():
            fields[k] = str(body[k]).strip()
    if body.get("api_key"):  # 空字符串不覆盖已有 key
        fields["api_key"] = body["api_key"]
        fields["status"] = "unknown"  # key 变更后需重新测试
    if body.get("is_default"):
        db.clear_default_provider()
        fields["is_default"] = True
    db.update_provider(pid, **fields)
    return _masked(db.get_provider(pid))


@router.delete("/api/providers/{pid}")
def delete_provider(pid: str, user: dict = Depends(auth.current_user)) -> dict:
    if not db.get_provider(pid):
        raise HTTPException(404, "Provider 不存在")
    db.delete_provider(pid)
    return {"ok": True}


@router.post("/api/providers/{pid}/test")
def test_provider(pid: str, user: dict = Depends(auth.current_user)) -> dict:
    p = db.get_provider(pid)
    if not p:
        raise HTTPException(404, "Provider 不存在")
    if not p.get("api_key"):
        raise HTTPException(400, "未配置 API Key，无法测试连接；请先编辑并填入有效 Key")
    result = _test_connection(p["base_url"], p["api_key"], p["model"])
    db.update_provider(pid, status=result["status"], latency_ms=result.get("latency_ms"))
    return {**result, "provider": _masked(db.get_provider(pid))}
