"""模型仓库：从训练任务登记模型、版本管理、在线推理。"""
from __future__ import annotations

import io
import uuid

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from PIL import Image

import auth
import db
import release
import spec
import train as T

router = APIRouter()

STAGES = {"candidate", "staging", "production", "archived"}


# ---------- 推理工具 ----------

def _load(model: dict):
    """加载 state_dict 并重建网络结构。返回 (nn.Module, meta)。

    结构参数必须取自训练时保存的 arch_params —— 若在此处硬编码，
    用户改了隐藏层宽度等超参后，推理端会按不同结构重建并导致
    load_state_dict 形状不匹配。
    """
    if not model.get("model_path"):
        raise HTTPException(400, "模型文件缺失")

    # 表格梯度提升树是 pickle 产物，不是 torch state_dict
    if model["model_path"].endswith(".pkl"):
        import pickle
        with open(model["model_path"], "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], {"arch": bundle.get("plan", {}).get("model", "hgb"),
                                 "task": bundle.get("task", "classification"),
                                 "prep": bundle.get("prep", {}),
                                 "pickle": True}

    try:
        bundle = torch.load(model["model_path"], map_location="cpu", weights_only=False)
    except FileNotFoundError:
        raise HTTPException(500, "模型文件已被移除") from None
    meta = bundle["meta"]
    arch, task = meta["arch"], meta["task"]
    ap = meta.get("arch_params") or {}
    nclass = len(meta["prep"].get("classes") or [0])
    if arch == "transfer_head":
        # 迁移学习模型：backbone 权重按名取自 torchvision，只加载训练好的分类头
        import automl
        net = automl.TransferNet(str(meta.get("backbone") or "mobilenet_v3_small"),
                                 nclass, int(ap.get("hidden", 128)),
                                 float(ap.get("dropout", 0.2)),
                                 int(ap.get("img_size") or 0) or None)
        # 微调过的模型：backbone 权重也被更新了，必须一并恢复，
        # 否则推理会用回预训练权重，准确率凭空掉一大截。
        bs = bundle.get("backbone_state")
        if bs:
            try:
                net.backbone.load_state_dict(bs)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    500, f"模型主干权重加载失败：{e}") from e
        net.head.load_state_dict(bundle["state_dict"])
        net.eval()
        return net, meta
    if arch == "text_transfer_head":
        # 文本迁移学习：只存分类头，encoder 权重由 transformers 缓存提供
        import automl
        net = automl.Head(int(ap.get("feat_dim") or 768),
                          int(ap.get("hidden", 128)), nclass,
                          float(ap.get("dropout", 0.2)))
        net.load_state_dict(bundle["state_dict"])
        net.eval()
        return net, meta
    if arch == "cnn":
        net = T.SmallCNN(nclass, int(ap.get("img_size", 64)), float(ap.get("dropout", 0.0)))
    elif arch == "cnn1d":
        # 1D 信号模型（心电等）：结构参数必须从 ckpt 还原，
        # 导联数、宽度、输出维度任一不符都会导致 load_state_dict 形状失配。
        import automl_signal as asig
        net = asig.build_net(meta)
    elif arch == "emb_mlp":
        net = T.TextClf(T.TEXT_VOCAB, int(ap.get("text_dim", T.TEXT_DIM)), nclass, float(ap.get("dropout", 0.0)))
    else:
        dout = nclass if task == "classification" else 1
        net = T.MLP(int(meta["prep"]["input_dim"]), int(ap.get("hidden", 64)), dout, float(ap.get("dropout", 0.0)))
    net.load_state_dict(bundle["state_dict"])
    net.eval()
    return net, meta


def _tabular_matrix(rows: list[dict], prep: dict) -> np.ndarray:
    """按训练期保存的统计量把任意 dict 行转换为特征矩阵。"""
    parts = []
    for col in prep["num_cols"]:
        st = prep["stats"][col]
        vals = pd.to_numeric(pd.Series([r.get(col) for r in rows]), errors="coerce")
        parts.append(((vals.fillna(st["mean"]) - st["mean"]) / st["std"]).values.reshape(-1, 1))
    for col in prep["cat_cols"]:
        cats = prep["stats"][col]["cats"]
        idx = {v: i for i, v in enumerate(cats)}
        onehot = np.zeros((len(rows), len(cats)), dtype=np.float32)
        for r_i, row in enumerate(rows):
            j = idx.get(str(row.get(col)))
            if j is not None:
                onehot[r_i, j] = 1.0
        parts.append(onehot)
    return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((len(rows), 0), np.float32)


def _classify_out(net, logits: torch.Tensor, classes: list[str]) -> list[dict]:
    proba = torch.softmax(logits, dim=1).numpy()
    out = []
    for p in proba:
        top = int(p.argmax())
        out.append({
            "label": classes[top],
            "confidence": round(float(p[top]), 4),
            "probabilities": {classes[i]: round(float(p[i]), 4) for i in range(len(classes))},
        })
    return out


# ---------- API ----------

@router.post("/api/models")
def register_model(body: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    """从已完成的训练任务登记模型，同名自动递增版本号。"""
    job_id = body.get("train_job_id")
    job = db.get_job(job_id or "")
    if not job:
        raise HTTPException(404, "训练任务不存在")
    if job["status"] != "done":
        raise HTTPException(400, f"训练任务尚未完成（当前状态 {job['status']}）")
    if not job.get("model_path"):
        raise HTTPException(400, "训练任务未产出模型文件")

    d = db.get_dataset(job["dataset_id"])
    metrics = job.get("metrics") or {}
    base_name = (body.get("name") or "").strip()
    if not base_name:
        arch = (metrics.get("arch") or "mlp").upper()
        base_name = f"{(d or {}).get('name', '模型')}-{arch}"
    siblings = [m for m in db.list_models() if m["name"] == base_name]
    version = max([m["version"] for m in siblings], default=0) + 1

    rec = {
        "id": f"mdl_{uuid.uuid4().hex[:10]}",
        "name": base_name,
        "version": version,
        "task": metrics.get("task"),
        "task_hint": (d or {}).get("task_hint"),
        "architecture": metrics.get("arch"),
        "dataset_id": job["dataset_id"],
        "train_job_id": job_id,
        "metrics": metrics,
        "model_path": job["model_path"],
    }
    db.insert_model(rec)
    return db.get_model(rec["id"])


@router.get("/api/models")
def list_models(include_archived: bool = False,
                user: dict = Depends(auth.current_user)) -> list[dict]:
    """模型列表。默认隐藏已归档模型，管理视角可带 include_archived=true 查看。"""
    return db.list_models(include_archived=include_archived)


@router.get("/api/models/{mid}")
def get_model(mid: str, user: dict = Depends(auth.current_user)) -> dict:
    m = db.get_model(mid)
    if not m:
        raise HTTPException(404, "模型不存在")
    d = db.get_dataset(m["dataset_id"]) if m.get("dataset_id") else None
    m["dataset_name"] = d["name"] if d else None
    return m


@router.put("/api/models/{mid}")
def update_model(mid: str, body: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    m = db.get_model(mid)
    if not m:
        raise HTTPException(404, "模型不存在")
    stage = body.get("stage")
    if stage and stage not in STAGES:
        raise HTTPException(400, f"无效阶段：{stage}（可选 {sorted(STAGES)}）")
    fields = {}
    if stage:
        fields["stage"] = stage
    db.update_model(mid, **fields)
    return db.get_model(mid)


@router.delete("/api/models/{mid}")
def remove_model(mid: str, user: dict = Depends(auth.current_user)) -> dict:
    m = db.get_model(mid)
    if not m:
        raise HTTPException(404, "模型不存在")
    db.delete_model(mid)
    return {"ok": True}


async def run_inference(m: dict, request: Request) -> dict:
    """推理核心：图像模型走 multipart（字段名 file），其余走 JSON。供模型/服务路由共用。"""
    net, meta = _load(m)
    prep = meta["prep"]
    task, arch = meta["task"], meta["arch"]
    ctype = request.headers.get("content-type", "")

    # 图像模型：multipart 文件上传（支持多张，字段名 file）
    if arch in ("cnn", "transfer_head"):
        if "multipart/form-data" not in ctype:
            raise HTTPException(400, "图像模型需以 multipart 上传图像文件（字段名 file，可重复以批量提交）")
        form = await request.form()
        uploads = [v for v in form.getlist("file") if hasattr(v, "read")]
        if not uploads:
            raise HTTPException(400, "缺少 file 字段")
        size = int(prep.get("img_size", 64))
        # 迁移学习模型必须使用与预训练一致的标准化，
        # 只除以 255 会让输入分布偏离，迁移收益大打折扣。
        tf = net.infer_transform() if arch == "transfer_head" else None
        batch: list[torch.Tensor] = []
        for up in uploads:
            raw = await up.read()
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    if tf is not None:
                        batch.append(tf(im))
                    else:
                        arr = np.asarray(im.resize((size, size)), dtype=np.float32) / 255.0
                        batch.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"图像解析失败: {e}") from e
        x = torch.stack(batch)
        with torch.no_grad():
            logits = net(x)
        return {"task": task, "arch": arch, "predictions": _classify_out(net, logits, prep["classes"])}

    # JSON 请求
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体需为 JSON") from None

    if arch == "cnn1d":
        # 1D 信号：rows 里每个 dict 是一条记录（12 导联 × 1000 点 = 12000 个键），
        # 也可以直接传 samples: [[...]] 按保存时的列顺序平铺。
        rows = body.get("rows")
        flat = body.get("samples")
        prep = meta["prep"]
        leads = prep.get("columns") or []
        if rows:
            arr = np.stack([
                np.stack([[float(r.get(c) or 0.0) for c in cols] for cols in leads], axis=0)
                for r in rows[:256]
            ], axis=0).astype(np.float32)
        elif flat:
            width = sum(len(c) for c in leads)
            arr = np.array(flat[:256], dtype=np.float32).reshape(
                -1, len(leads), width // len(leads))
        else:
            raise HTTPException(
                400, "请提供 rows（每行一条记录的字段字典）或 samples（按导联平铺的数组）")
        mu = arr.mean(axis=2, keepdims=True)
        sd = arr.std(axis=2, keepdims=True)
        sd[sd < 1e-6] = 1.0
        arr = (arr - mu) / sd
        with torch.no_grad():
            logits = net(torch.from_numpy(arr)).numpy()
        classes = prep.get("classes") or []
        if prep.get("multilabel"):
            # 多标签：按类给概率与阈值判定，不套 softmax
            thr = np.array(prep.get("thresholds") or [0.5] * len(classes))
            prob = 1.0 / (1.0 + np.exp(-logits))
            preds = []
            for row in prob:
                # 按概率降序排列：否则 labels 是按类别顺序拼的，
                # 而 confidence 取的是全类最大概率，会出现
                # 「报 MI，但最高概率其实是 STTC」这种自相矛盾的结论（联调实测）。
                order = np.argsort(-row)
                hit_idx = [int(i) for i in order if row[int(i)] >= thr[int(i)]]
                hit = [classes[i] for i in hit_idx]
                preds.append({
                    "label": "|".join(hit) if hit else "无阳性发现",
                    "labels": hit,
                    # confidence 必须对应 label 里第一个标签的概率。
                    # 取全类最大概率会出现「label=MI，confidence 却是 STTC 的 0.62」
                    # ——数值和标签对不上，医生一眼就会怀疑系统坏了（联调实测）。
                    "confidence": round(float(row[hit_idx[0]]) if hit_idx
                                        else float(row[int(order[0])]), 4),
                    "top_label": classes[int(order[0])],
                    "probabilities": {classes[i]: round(float(row[i]), 4)
                                      for i in range(len(classes))},
                })
            return {"task": task, "arch": arch, "predictions": preds}
        prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        preds = []
        for row in prob:
            top = int(row.argmax())
            preds.append({"label": classes[top] if top < len(classes) else str(top),
                          "confidence": round(float(row[top]), 4),
                          "probabilities": {classes[i]: round(float(row[i]), 4)
                                            for i in range(len(classes))}})
        return {"task": task, "arch": arch, "predictions": preds}

    if arch == "text_transfer_head":
        # 文本迁移学习：先用预训练 encoder 编码，再过分类头
        texts = body.get("texts") or ([body["text"]] if body.get("text") else None)
        if not texts:
            raise HTTPException(400, "请提供 text 或 texts 字段")
        import automl_text as at
        enc = str(meta.get("encoder") or at.DEFAULT_ENCODER)
        feats = at.encode_texts([str(t) for t in texts[:64]], enc)
        with torch.no_grad():
            logits = net(torch.from_numpy(feats))
        return {"task": task, "arch": arch,
                "predictions": _classify_out(net, logits, prep["classes"])}

    if arch in ("hgb", "gbdt"):
        # 表格梯度提升树：直接用 sklearn 的 predict
        rows = body.get("rows") or ([body["features"]] if body.get("features") else None)
        if not rows:
            raise HTTPException(400, "请提供 rows（dict 数组）或 features（单条 dict）")
        X = _tabular_matrix(rows[:256], prep)
        pred = net.predict(X)
        if task == "classification":
            classes = prep.get("classes") or []
            out = []
            for p in pred:
                i = int(p)
                out.append({"label": classes[i] if i < len(classes) else str(i),
                            "confidence": 1.0})
            return {"task": task, "arch": arch, "predictions": out}
        return {"task": task, "arch": arch,
                "predictions": [{"value": round(float(v), 5)} for v in pred]}

    if arch == "emb_mlp":
        texts = body.get("texts") or ([body["text"]] if body.get("text") else None)
        if not texts:
            raise HTTPException(400, "请提供 text 或 texts 字段")
        preds = []
        for t in texts[:64]:
            ids = torch.tensor(T.hash_tokens(str(t)), dtype=torch.long)
            offs = torch.tensor([0], dtype=torch.long)
            with torch.no_grad():
                logits = net(ids, offs)
            preds.extend(_classify_out(net, logits, prep["classes"]))
        return {"task": task, "arch": arch, "predictions": preds}

    # 表格 MLP
    rows = body.get("rows") or ([body["features"]] if body.get("features") else None)
    if not rows:
        raise HTTPException(400, "请提供 rows（dict 数组）或 features（单条 dict）")
    X = torch.from_numpy(_tabular_matrix(rows[:256], prep))
    with torch.no_grad():
        logits = net(X)
    if task == "classification":
        preds = _classify_out(net, logits, prep["classes"])
    else:
        ys = prep.get("y_stats") or {"mean": 0.0, "std": 1.0}
        preds = [{"value": round(float(v) * ys["std"] + ys["mean"], 4)} for v in logits.squeeze(-1).numpy()]
    return {"task": task, "arch": arch, "predictions": preds}


@router.post("/api/models/{mid}/predict")
async def predict(mid: str, request: Request, user: dict = Depends(auth.current_user)) -> dict:
    m = db.get_model(mid)
    if not m:
        raise HTTPException(404, "模型不存在")
    return await run_inference(m, request)


# ---------------- SOP v2：模型包 / 发布 / 退役 ----------------

@router.get("/api/models/{mid}/bundle")
def download_bundle(mid: str, user: dict = Depends(auth.current_user)):
    """导出模型包（zip）：权重 + 预处理 + 标签表 + 输入 schema + 指标 + 溯源 + 模型卡。

    部署端只需要这个包，不再靠"约定"猜类别顺序和预处理参数。
    """
    from fastapi.responses import FileResponse
    if not db.get_model(mid):
        raise HTTPException(404, "模型不存在")
    try:
        info = release.build_bundle(mid, force=True)
    except FileNotFoundError as e:
        raise HTTPException(404, f"模型权重缺失，无法打包：{e}") from e
    m = db.get_model(mid) or {}
    fname = f"{m.get('name') or mid}_v{m.get('version') or 1}.zip"
    return FileResponse(info["zip"], filename=fname, media_type="application/zip")


@router.get("/api/models/{mid}/bundle/manifest")
def bundle_manifest(mid: str, user: dict = Depends(auth.current_user)) -> dict:
    """查看模型包内容与完整性校验结果（不下载）。"""
    if not db.get_model(mid):
        raise HTTPException(404, "模型不存在")
    try:
        info = release.build_bundle(mid, force=False)
    except FileNotFoundError as e:
        raise HTTPException(404, f"模型权重缺失，无法打包：{e}") from e
    rep = release.verify_bundle(mid)
    return {"model_id": mid, "bundle": info, "verify": rep.as_dict()}


@router.post("/api/models/{mid}:release")
def release_model(mid: str, body: dict = Body(default={}),
                  user: dict = Depends(auth.current_user)) -> dict:
    """发布门：测试集指标达标 + 模型包完整，才允许进入 production。"""
    if not db.get_model(mid):
        raise HTTPException(404, "模型不存在")
    rep = release.release(mid, force=bool(body.get("force")))
    if not rep.ok:
        raise spec.GateBlocked(rep, http_status=422)
    m = db.get_model(mid) or {}
    return {"model_id": mid, "stage": m.get("stage"), "gate": rep.as_dict()}


@router.post("/api/models/{mid}:retire")
def retire_model(mid: str, body: dict = Body(default={}),
                 user: dict = Depends(auth.current_user)) -> dict:
    """退役门：被服务引用的模型必须先摘流量或指定替代模型。"""
    rep = release.retire(mid, replacement=body.get("replacement"))
    if not rep.ok:
        raise spec.GateBlocked(rep, http_status=409)
    return {"model_id": mid, "stage": "retired", "gate": rep.as_dict()}
