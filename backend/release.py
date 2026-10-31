"""G5 发布门 / 模型包 / G7 退役门。

解决审计里最后一段链路的问题：模型训练完就是一个裸 `.pt`，
没有预处理参数、没有标签顺序、没有输入 schema、没有溯源，
部署端只能靠「约定」去猜。猜错的后果是把 2 级读成 3 级。

模型包（bundle）把这些约定变成文件，缺一个就发不出去。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import db
import spec
from spec import BLOCK, WARN, GateReport, Issue

BUNDLE_ROOT = db.STORAGE_DIR / "bundles"
BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

REQUIRED = ("manifest.json", "weights.pt", "preprocess.json", "labels.json",
            "input_schema.json", "metrics.json", "provenance.json", "card.md")

_MODALITY_INPUT = {
    "image": {"type": "image", "field": "file", "formats": ["png", "jpg", "jpeg"],
              "note": "单张图像；尺寸由 preprocess.img_size 决定，服务端负责缩放"},
    "tabular": {"type": "json_rows", "field": "rows",
                "note": "行数组，字段名与 features 一致；数值列做标准化，类别列做 one-hot"},
    "text": {"type": "json_rows", "field": "rows", "text_field": "text",
             "note": "每条样本一个 text 字段"},
    "signal": {"type": "json_rows", "field": "rows",
               "note": "每条样本若干个采样点列，列序必须与 preprocess.signal_cols 一致"},
    "detection": {"type": "image", "field": "file", "note": "单张图像，返回框列表"},
}


def _ckpt_meta(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return ck.get("meta") or {}
    except Exception:  # noqa: BLE001
        return {}


def _load(mid: str) -> dict:
    m = db.get_model(mid)
    if not m:
        raise FileNotFoundError(f"模型不存在：{mid}")
    return m


def _metrics(m: dict) -> dict:
    v = m.get("metrics") or {}
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            v = {}
    return v if isinstance(v, dict) else {}


def bundle_name(m: dict) -> str:
    return f"{m['id']}_v{m.get('version') or 1}"


def bundle_zip(mid: str) -> Path:
    m = _load(mid)
    return BUNDLE_ROOT / f"{bundle_name(m)}.zip"


def _write_json(p: Path, obj: object) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_bundle(mid: str, force: bool = False) -> dict:
    """生成模型包目录与 zip。返回 {dir, zip, files, sha256}。"""
    m = _load(mid)
    src = Path(m.get("model_path") or "")
    if not src.exists():
        raise FileNotFoundError(f"模型权重不存在：{src}")

    meta = _ckpt_meta(str(src))
    prep = meta.get("prep") or {}
    classes = prep.get("classes") or (meta.get("classes") or [])
    task_hint = meta.get("task_hint") or m.get("task_hint") or ""
    sp = spec.spec_of(task_hint)
    modality = sp.modality if sp else "tabular"

    out = BUNDLE_ROOT / bundle_name(m)
    if out.exists() and force:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, out / "weights.pt")
    _write_json(out / "preprocess.json", {
        "prep": prep,
        "image_size": prep.get("img_size"),
        "normalize": prep.get("normalize") or "imagenet",
        "signal_cols": prep.get("signal_cols"),
        "sampling_rate": prep.get("fs"),
        "note": "推理端必须按本文件做与训练一致的预处理，否则结果不可比",
    })
    _write_json(out / "labels.json", {
        "classes": classes,
        "order": "sorted",
        "task_hint": task_hint,
        "ordinal": spec.is_ordinal(classes),
        "thresholds": (meta.get("test_metrics") or {}).get("thresholds"),
    })
    _write_json(out / "input_schema.json", _MODALITY_INPUT.get(modality, {"type": "unknown"}))

    metrics = meta.get("test_metrics") or _metrics(m).get("test_metrics") or {}
    _write_json(out / "metrics.json", {"test_metrics": metrics,
                                       "history_tail": (_metrics(m).get("history") or [])[-3:]})

    ds = db.get_dataset(m["dataset_id"]) if m.get("dataset_id") else None
    split_info = meta.get("split_info") or ((ds or {}).get("meta") or {}).get("split") or {}
    _write_json(out / "provenance.json", {
        "model_id": m["id"], "model_name": m["name"], "version": m.get("version") or 1,
        "dataset_id": m.get("dataset_id"), "dataset_name": (ds or {}).get("name"),
        "dataset_version": (ds or {}).get("version"),
        "train_job_id": m.get("train_job_id"),
        "split_info": split_info,
        "config": meta.get("config"),
        "rounds": meta.get("rounds"),
        "platform": "ModelForge",
        "sop_version": "2.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    })
    test = metrics or {}
    card = [
        f"# {m['name']} v{m.get('version') or 1}", "",
        f"- 任务：{task_hint or '未知'}（{modality}）",
        f"- 类别：{', '.join(map(str, classes)) if classes else '未记录'}",
        f"- 训练数据：{(ds or {}).get('name') or m.get('dataset_id')}",
        f"- 测试集样本：{test.get('test_samples') or test.get('samples') or '未记录'}",
        f"- 主指标：{spec.spec_of(task_hint).primary_metric if spec.spec_of(task_hint) else '未知'}"
        f" = {_fmt(test)}", "",
        "## 用途", "科研与辅助判读，不构成诊断结论。", "",
        "## 限制",
        "- 单中心数据训练，跨机构性能未验证。",
        "- 训练数据未覆盖的人群/设备上表现可能显著下降。",
        "- 输出必须由具备资质的医师复核后使用。",
    ]
    (out / "card.md").write_text("\n".join(card) + "\n", encoding="utf-8")

    _write_json(out / "manifest.json", {
        "bundle_version": "2.0",
        "model_id": m["id"], "name": m["name"], "version": m.get("version") or 1,
        "task_hint": task_hint, "modality": modality,
        "architecture": m.get("architecture"),
        "files": list(REQUIRED),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    })

    zp = BUNDLE_ROOT / f"{bundle_name(m)}.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in REQUIRED:
            z.write(out / f, arcname=f)

    h = hashlib.sha256(zp.read_bytes()).hexdigest()
    return {"dir": str(out), "zip": str(zp), "files": list(REQUIRED), "sha256": h}


def verify_bundle(mid: str) -> GateReport:
    """G5：包是否完整、可信、可追溯。"""
    rep = GateReport("bundle")
    try:
        zp = bundle_zip(mid)
    except FileNotFoundError as e:
        rep.issues.append(Issue("E6001", str(e), "先训练出模型再打包"))
        return rep
    if not zp.exists():
        rep.issues.append(Issue("E6001", "模型包尚未生成", "调用 build_bundle 生成"))
        return rep
    try:
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
    except zipfile.BadZipFile:
        rep.issues.append(Issue("E6001", "模型包损坏", "重新生成"))
        return rep

    missing = [f for f in REQUIRED if f not in names]
    if missing:
        rep.issues.append(Issue("E6001", f"模型包缺少文件：{', '.join(missing)}",
                                "重新打包；bundle 必须自包含"))
        return rep

    with zipfile.ZipFile(zp) as z:
        man = json.loads(z.read("manifest.json"))
        lab = json.loads(z.read("labels.json"))
        prov = json.loads(z.read("provenance.json"))
        met = json.loads(z.read("metrics.json"))

    if not lab.get("classes"):
        rep.issues.append(Issue("E6001", "labels.json 没有类别表", "推理端无法把输出映射成语义"))
    if not prov.get("dataset_id"):
        rep.issues.append(Issue("E6001", "缺少训练数据溯源（dataset_id）",
                                "无法回答「这个模型是用什么数据训的」"))
    if not prov.get("train_job_id"):
        rep.issues.append(Issue("W6002", "缺少训练任务溯源（job_id）", "建议补齐", WARN))
    if not (met.get("test_metrics") or {}):
        rep.issues.append(Issue("E6003", "模型包内没有测试集指标", "必须带独立测试集评估"))
    return rep


def _fmt(test: dict) -> str:
    for k in ("macro_f1", "macro_auc", "qwk", "acc", "test_acc", "r2"):
        if test.get(k) is not None:
            try:
                return f"{k}={float(test[k]):.4f}"
            except (TypeError, ValueError):
                return f"{k}={test[k]}"
    return "未记录"


def _coverage(test: dict, n_classes: int) -> float | None:
    per = test.get("per_class") or []
    if not per or not n_classes:
        return None
    covered = sum(1 for c in per if (c.get("support") or 0) > 0)
    return covered / n_classes


def release(mid: str, force: bool = False) -> GateReport:
    """G4+G5：评估门槛 + 包完整性，双通过才允许 stage=production。"""
    m = _load(mid)
    task = m.get("task") or ""
    # models.task 是 classification/regression，真正的任务类型在 meta 里
    meta = _ckpt_meta(m.get("model_path"))
    task_hint = meta.get("task_hint") or m.get("task_hint") or ""
    metrics = _metrics(m)
    test = metrics.get("test_metrics") or meta.get("test_metrics") or {}
    classes = (meta.get("prep") or {}).get("classes") or []
    ordinal = spec.is_ordinal(classes)

    rep = spec.gate_release(task_hint, {"test_metrics": test}, has_test=bool(test),
                            coverage=_coverage(test, len(classes)), ordinal=ordinal)
    rep.gate = "release"

    # 先打包再校验：包是发布门的证据，不能拿「还没打包」当拒绝理由。
    try:
        build_bundle(mid, force=True)
    except FileNotFoundError as e:
        rep.issues.append(Issue("E6001", f"模型权重缺失，无法打包：{e}",
                                "重新导出权重后再发布"))
        return rep
    except Exception as e:  # noqa: BLE001
        rep.issues.append(Issue("E6001", f"模型包生成失败：{e}", "检查权重文件是否可读"))
        return rep

    rep.issues.extend(verify_bundle(mid).issues)

    if rep.ok:
        db.update_model(mid, stage="production")
    elif not force and m.get("stage") == "candidate":
        # 有服务在用的模型不能被自动降级：那会让线上服务指向一个"被拒绝"的模型，
        # 状态自相矛盾。先让它保持 candidate，由人决定是补评估还是下线服务。
        in_use = any(s.get("model_id") == mid for s in db.list_services())
        if not in_use:
            db.update_model(mid, stage="rejected")
        else:
            rep.issues.append(Issue(
                "W8001", "该模型仍被服务引用，未自动置为 rejected",
                "先停止引用它的服务，再重新评估或退役", WARN))
    return rep


def retire(mid: str, replacement: str | None = None) -> GateReport:
    """G7：退役前检查引用。被服务占用的模型不能直接下线。"""
    rep = GateReport("retire")
    m = db.get_model(mid)
    if not m:
        rep.issues.append(Issue("E8001", f"模型不存在：{mid}"))
        return rep

    users = [s for s in db.list_services() if s.get("model_id") == mid]
    if users and not replacement:
        names = ", ".join(s["name"] for s in users[:3])
        rep.issues.append(Issue(
            "E8001", f"仍有服务在引用该模型：{names}",
            "先停止这些服务，或指定替代模型（replacement）后再退役"))
        return rep

    if users and replacement:
        if not db.get_model(replacement):
            rep.issues.append(Issue("E8001", f"替代模型不存在：{replacement}"))
            return rep
        for s in users:
            db.update_service(s["id"], model_id=replacement)

    db.update_model(mid, stage="retired")
    return rep
