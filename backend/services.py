"""服务发布：把模型以固定路由 /api/svc/{name}/predict 对外提供，支持启停与调用计数。"""
from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException, Request

import auth
import db
import models as M

router = APIRouter()

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")


def _with_model(s: dict) -> dict:
    m = db.get_model(s["model_id"])
    s["model_name"] = m["name"] if m else None
    s["model_version"] = m["version"] if m else None
    s["task_hint"] = m.get("task_hint") if m else None
    s["endpoint"] = f"/api/svc/{s['name']}/predict"
    return s


@router.post("/api/services")
def create_service(body: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    name = (body.get("name") or "").strip().lower()
    if not NAME_RE.match(name):
        raise HTTPException(400, "服务名需为小写字母开头的 slug（a-z0-9-，2~41 字符）")
    if db.get_service_by_name(name):
        raise HTTPException(409, f"服务名「{name}」已被占用")
    m = db.get_model(body.get("model_id") or "")
    if not m:
        raise HTTPException(404, "模型不存在")
    rec = {
        "id": f"svc_{uuid.uuid4().hex[:10]}",
        "name": name,
        "model_id": m["id"],
        "status": "running",
        "created_by": user.get("account"),
    }
    db.insert_service(rec)
    return _with_model(db.get_service(rec["id"]))


@router.get("/api/services")
def list_services(user: dict = Depends(auth.current_user)) -> list[dict]:
    return [_with_model(s) for s in db.list_services()]


@router.post("/api/services/{sid}/start")
def start_service(sid: str, user: dict = Depends(auth.current_user)) -> dict:
    s = db.get_service(sid)
    if not s:
        raise HTTPException(404, "服务不存在")
    db.update_service(sid, status="running")
    return _with_model(db.get_service(sid))


@router.post("/api/services/{sid}/stop")
def stop_service(sid: str, user: dict = Depends(auth.current_user)) -> dict:
    s = db.get_service(sid)
    if not s:
        raise HTTPException(404, "服务不存在")
    db.update_service(sid, status="stopped")
    return _with_model(db.get_service(sid))


@router.delete("/api/services/{sid}")
def delete_service(sid: str, user: dict = Depends(auth.current_user)) -> dict:
    if not db.get_service(sid):
        raise HTTPException(404, "服务不存在")
    db.delete_service(sid)
    return {"ok": True}


# ---------- 对外推理路由（服务消费者使用） ----------

@router.post("/api/svc/{name}/predict")
async def svc_predict(name: str, request: Request, user: dict = Depends(auth.current_user)) -> dict:
    s = db.get_service_by_name(name)
    if not s:
        raise HTTPException(404, f"服务「{name}」不存在")
    if s["status"] != "running":
        raise HTTPException(409, f"服务「{name}」已停止，请先启动")
    m = db.get_model(s["model_id"])
    if not m:
        raise HTTPException(500, "服务关联的模型已被删除")
    out = await M.run_inference(m, request)
    db.update_service(s["id"], requests_total=int(s.get("requests_total") or 0) + 1)
    out["service"] = {"name": s["name"], "model": m["name"], "version": m["version"]}
    return out
