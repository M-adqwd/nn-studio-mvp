"""通用 AI 网关：把训练平台（ModelForge）的模型直接暴露给阅片平台（MIIS）。

## 为什么需要这一层

现状是两套系统各说各话：
- ModelForge 的推理接口是 `/api/models/{mid}/predict`，返回 `label + confidence`，
  面向「有模型 id 的人」；
- MIIS 的 AI 适配器 `InferRequest` 只认 `study_instance_uid + targets`，
  是 DICOM 序列分割的形态（MONAI Deploy 那套）。

于是「ModelForge 训出来的模型」和「MIIS 病例工作台」之间没有任何一条路：
模型训得再好，医生在阅片台上也看不到它的结论。

本模块补的就是这一环，且只补最小必要面：
1. `GET  /api/gateway/catalog`  —— 模型目录，带「模态 + 任务 + 类别」，供阅片端选模型；
2. `POST /api/gateway/infer`    —— 统一推理入口，输入可以是图像 / 波形 / 表格行，
   输出统一信封（含结论文本），调用方不需要知道背后是 CNN 还是 GBDT；
3. `POST /api/gateway/diagnose` —— 面向临床：给一段上下文（检查号/患者/部位），
   直接产出可直接写进报告 `findings` / `impression` 的文本，并附免责声明。

## 边界（必须写清楚，否则会被当成「AI 出报告」）

网关只产出 **建议性结论**（suggestion）。报告落库、状态流转、医师确认
全部由 MIIS 负责，网关不写任何临床库。任何对外口径都应表述为
「辅助提示，需医师确认」。
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

import auth
import db
from models import run_inference

router = APIRouter(tags=["gateway"])

# 任务类型 → 输入模态（阅片端据此决定从哪里取数据）
MODALITY = {
    "image_classification": "image",
    "object_detection": "image",
    "signal_classification": "signal",
    "tabular_classification": "table",
    "tabular_regression": "table",
    "text_classification": "text",
}

# 分级任务的中文语义：给报告文本用，缺省时退回 label 原文
# 心搏类别（MIT-BIH AAMI 五分类）的中文语义
BEAT_TEXT = {
    "N": "正常心搏",
    "S": "室上性早搏",
    "V": "室性早搏",
    "F": "融合波",
    "Q": "不可分类",
}

GRADE_TEXT = {
    "0": "无可见病变",
    "1": "轻度",
    "2": "中度",
    "3": "重度",
    "4": "增殖期",
}

DISCLAIMER = "本结论由 AI 模型生成，仅供辅助参考，须经具备资质的医师复核后方可用于临床。"


_META_CACHE: dict[str, dict] = {}


def _checkpoint_meta(path: str | None) -> dict:
    """从 .pt 里读出 prep（类别、图像尺寸、导联列），带缓存。

    模型表里只有 metrics，类别与输入规格存在 checkpoint 的 meta 里。
    网关必须知道这些：阅片端要显示「这个模型判哪几类」，报告要把 label 翻译成语义。
    """
    if not path:
        return {}
    key = str(path)
    if key in _META_CACHE:
        return _META_CACHE[key]
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta") or {}
    except Exception:  # noqa: BLE001
        meta = {}
    _META_CACHE[key] = meta
    return meta


def _model_brief(m: dict) -> dict:
    meta = m.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:  # noqa: BLE001
            meta = {}
    if not meta:
        meta = _checkpoint_meta(m.get("model_path"))
    prep = (meta.get("prep") or {}) if isinstance(meta, dict) else {}
    metrics = m.get("metrics") or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:  # noqa: BLE001
            metrics = {}
    test = (metrics.get("test_metrics") or {}) if isinstance(metrics, dict) else {}
    return {
        "id": m["id"],
        "name": m["name"],
        "version": m.get("version") or 1,
        "task_hint": m.get("task_hint"),
        "modality": MODALITY.get(m.get("task_hint") or "", "unknown"),
        "architecture": m.get("architecture"),
        "classes": prep.get("classes") or [],
        "input": {
            "image_size": prep.get("img_size"),
            "leads": len(prep.get("columns") or []) if prep.get("columns") else None,
            "length": prep.get("length"),
            "multilabel": bool(prep.get("multilabel")),
        },
        "quality": {
            "test_acc": test.get("test_acc"),
            "macro_f1": test.get("macro_f1") or test.get("test_f1"),
            "macro_auc": test.get("macro_auc"),
            "qwk": test.get("qwk"),
            "test_samples": test.get("test_samples"),
        },
    }


def _pick_model(ref: str) -> dict:
    """按 id 或「名称[#版本]」找模型；名称不唯一时取最高版本。

    已归档模型一律拒绝：留着记录是为了追溯，不是为了让旧的错模型继续出结果。
    """
    m = db.get_model(ref)
    if m:
        stage = (m.get("stage") or "")
        if stage == "archived":
            raise HTTPException(
                409, f"模型已归档，不再提供推理：{m.get('name')} v{m.get('version')}"
                     "（请改用同任务的在役模型）")
        if stage == "rejected":
            raise HTTPException(
                409, f"模型未通过发布门（rejected），不提供推理：{m.get('name')} v{m.get('version')}"
                     "（请改用通过评估的模型）")
        return m
    name, ver = ref, None
    if "#" in ref:
        name, v = ref.rsplit("#", 1)
        ver = int(v) if v.isdigit() else None
    cands = [x for x in db.list_models() if x["name"] == name]
    if ver is not None:
        cands = [x for x in cands if int(x.get("version") or 1) == ver]
    if not cands:
        raise HTTPException(404, f"模型不存在：{ref}")
    return sorted(cands, key=lambda x: int(x.get("version") or 1))[-1]


def _top(preds: list[dict]) -> dict:
    return preds[0] if preds else {}


def _grade_of(label: str, classes: list[str]) -> str | None:
    """从类别名里取分级号。

    实际数据集的类别名是 `0_healthy` / `2_moderate` 这种「序号_语义」形式，
    也有纯数字的。两种都要认，否则报告里会直接冒出 `2_moderate` 这种
    给医生看很别扭的字符串。
    """
    if len(classes) != 5:
        return None
    if label.isdigit():
        return label
    head = label.split("_")[0].strip()
    return head if head.isdigit() else None


def _findings_text(brief: dict, top: dict, preds: list[dict]) -> str:
    """把「标签 + 概率」翻译成一句医生能读的结论。"""
    label = str(top.get("label") or "")
    conf = float(top.get("confidence") or 0.0)
    cls = brief.get("classes") or []
    grade = _grade_of(label, cls)
    if brief.get("task_hint") == "image_classification" and grade is not None:
        g = GRADE_TEXT.get(grade, "")
        later = "，具体分期请医师结合临床判断" if grade == "4" else ""
        return (f"眼底图像自动分析：糖尿病视网膜病变 {grade} 级"
                f"（{g}），模型置信度 {conf:.2%}{later}。")
    if brief.get("task_hint") == "signal_classification":
        if "|" in label:  # 多标签：PTB-XL 五大超类
            sup = {"NORM": "正常", "MI": "心肌梗死", "STTC": "ST-T 改变",
                   "CD": "传导阻滞", "HYP": "肥厚"}
            names = "、".join(f"{sup.get(x, x)}（{x}）" for x in label.split("|"))
            return f"心电图自动判读：检出 {names}，按类阈值输出。"
        beat = BEAT_TEXT.get(label)
        if beat and len(brief.get("classes") or []) <= 6:
            return f"心电信号自动分类：{beat}（{label}），置信度 {conf:.2%}。"
        return f"心电信号自动分类：{label}，置信度 {conf:.2%}。"
    return f"模型判定为「{label}」，置信度 {conf:.2%}。" if conf is not None else f"模型判定为「{label}」。"


def _impression_text(brief: dict, top: dict) -> str:
    label = str(top.get("label") or "")
    grade = _grade_of(label, brief.get("classes") or [])
    if brief.get("task_hint") == "signal_classification" and BEAT_TEXT.get(label):
        if label == "N":
            return "该心搏形态正常，未见异位搏动。"
        return f"检出{BEAT_TEXT[label]}（{label}），建议结合节律与临床判断。"
    if brief.get("task_hint") == "image_classification" and grade is not None:
        g = int(grade)
        if g == 0:
            return "未见糖尿病视网膜病变征象。"
        if g <= 2:
            return f"糖尿病视网膜病变 {g} 级，建议眼科随访。"
        return f"糖尿病视网膜病变 {g} 级（重度及以上），建议尽快眼科专科评估。"
    return f"AI 提示：{label}，请结合临床与影像复核。"


@router.get("/api/gateway/catalog")
def catalog(user: dict = Depends(auth.current_user)) -> dict:
    """模型目录：阅片端据此知道有哪些模型可用、吃什么输入、效果如何。"""
    items = [_model_brief(m) for m in db.list_models()]
    return {"models": items, "count": len(items)}


@router.post("/api/gateway/infer")
async def infer(model: str, request: Request,
                user: dict = Depends(auth.current_user)) -> dict:
    """统一推理入口。

    入参完全沿用 `/api/models/{id}/predict`：图像走 multipart（字段 file），
    信号/表格走 JSON（rows / samples）。这样既不用改编排层，也不破坏既有客户端。
    """
    m = _pick_model(model)
    brief = _model_brief(m)
    raw = await run_inference(m, request)
    preds = raw.get("predictions") or []
    if not preds:
        raise HTTPException(500, "模型未返回预测结果")
    top = _top(preds)
    return {
        "model": brief,
        "predictions": preds,
        "top": top,
        "report": {
            "findings": _findings_text(brief, top, preds),
            "impression": _impression_text(brief, top),
            "refs": [{"kind": "ai_job", "uid": f"model:{brief['id']}",
                      "version": brief["version"]}],
        },
        "disclaimer": DISCLAIMER,
    }


@router.post("/api/gateway/diagnose")
async def diagnose(payload: dict = Body(...),
                   user: dict = Depends(auth.current_user)) -> dict:
    """带上下文的诊断建议。

    `context` 里的字段会写进结论文本（检查号 / 患者标识 / 检查部位），
    但**不参与推理**——推理只看信号或图像本身。这样既方便报告追溯，
    也避免把患者信息喂进模型造成偏倚。
    """
    model_ref = payload.get("model")
    if not model_ref:
        raise HTTPException(400, "缺少 model")
    ctx = payload.get("context") or {}
    m = _pick_model(model_ref)
    brief = _model_brief(m)

    # 这里不重复实现推理：直接复用 /api/gateway/infer 的产物格式，
    # 由调用方先调 infer 再把 top 结果带进来；若只给了 samples 则就地推理。
    samples = payload.get("samples")
    rows = payload.get("rows")
    if samples is None and rows is None and not payload.get("top"):
        raise HTTPException(400, "需要 samples / rows，或直接传已算好的 top")

    if payload.get("top"):
        top = payload["top"]
        preds = payload.get("predictions") or [top]
    else:
        # 构造一个最小 JSON 请求对象，复用同一套推理实现
        class _Shim:
            headers = {"content-type": "application/json"}

            async def json(self_inner):  # noqa: N805
                return {"samples": samples, "rows": rows}

        raw = await run_inference(m, _Shim())  # type: ignore[arg-type]
        preds = raw.get("predictions") or []
        top = _top(preds)

    findings = _findings_text(brief, top, preds)
    if ctx.get("study_uid"):
        findings += f"（检查号 {ctx['study_uid']}）"
    return {
        "model": brief,
        "context": ctx,
        "top": top,
        "report": {
            "findings": findings,
            "impression": _impression_text(brief, top),
            "refs": [{"kind": "ai_job", "uid": f"model:{brief['id']}",
                      "version": brief["version"]}],
        },
        "disclaimer": DISCLAIMER,
    }
