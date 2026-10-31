"""FastAPI 入口：数据集 / 清洗 / EDA 真实后端。

启动：cd backend && uvicorn main:app --reload --port 8000

数据解析完全交给 AI（见 ai_pipeline.py）：平台不预设任何格式规则，
由 AI 判断结构、识别 train/val/test 划分并生成数据准备脚本。
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

import ai_pipeline
import analysis
import auth
import db
import llm
import gateway
import models
import preflight
import providers
import recall
import release
import services
import spec
import split
import train

app = FastAPI(title="ModelForge · Data Service", version="0.4.0")


# ---------- 统一错误信封 ----------
# SOP v2 要求：每个拒绝都带稳定错误码 + 人话 + 可执行建议。
# 前端不再解析「解析异常：…」这类自由文本。
@app.exception_handler(spec.GateBlocked)
async def _gate_blocked_handler(request, exc: spec.GateBlocked):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=exc.http_status, content=exc.envelope())


# 上传限制：单文件 3GB
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024
TABULAR_EXTS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl",
                ".parquet", ".pq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(train.router)
app.include_router(models.router)
app.include_router(services.router)
app.include_router(providers.router)
# 通用 AI 网关：把训练平台的模型以统一信封暴露给阅片平台（MIIS）
app.include_router(gateway.router)

db.init_db()
auth.ensure_admin()

# 清理上次异常退出残留的临时上传分片（被占用时跳过，下次再清）
for _tmp in db.STORAGE_DIR.glob("_upload_*"):
    try:
        _tmp.unlink(missing_ok=True)
    except OSError:
        pass


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _safe_unlink(path: Path) -> None:
    """删除临时文件，任何失败都忽略。

    不直接用 Path.unlink：部分运行环境会对删除动作加保护并抛 SystemExit，
    那会让异常处理路径本身再次失败，把真实错误掩盖掉。
    """
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except BaseException:  # noqa: BLE001  含 SystemExit
        pass


def load_df(dataset: dict) -> pd.DataFrame:
    path = Path(dataset["file_path"])
    if not path.exists():
        raise HTTPException(500, f"数据文件丢失: {path}")
    return pd.read_csv(path)


def save_version_csv(df: pd.DataFrame, did: str, version: int) -> Path:
    path = db.STORAGE_DIR / f"{did}_v{version}.csv"
    df.to_csv(path, index=False)
    return path


# ---------- 健康检查 ----------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "data", "version": "0.1.0"}


# ---------- 数据集 CRUD ----------

@app.get("/api/datasets")
def list_datasets(user: dict = Depends(auth.current_user)) -> list[dict]:
    return db.list_datasets()


@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    target_col: str = Form(None),
    task_hint: str = Form(None),
    ai_analyze: str = Form(None),
    user: dict = Depends(auth.current_user),
) -> dict:
    """数据集上传入口。

    **完全由 AI 解析**：不再有任何写死的格式规则。
    平台把文件交给 AI，由 AI 判断这是什么结构、哪些是训练/验证/测试集、
    该用什么列当目标，然后生成并执行数据准备脚本。
    同结构的数据集会命中记忆库，零 AI 调用直接复用已验证脚本。

    解析在后台线程执行，接口立即返回任务 ID —— 大数据集的解析可能长达几十分钟，
    同步执行会把整个后端占死（连 /api/health 都会无响应）。
    """
    fname = (file.filename or "").lower()
    did = uid("ds")
    ds_name = name or Path(file.filename or "dataset").stem
    # 保留参数位（前端旧版会传），AI 全自动已是默认行为
    _ = str(ai_analyze or "").lower()

    # 归档类（zip / tar.gz / tar）走流式落盘：大文件不能整个读进内存
    ARCHIVE_EXTS = (".zip", ".tar.gz", ".tgz", ".tar")
    is_archive = fname.endswith(ARCHIVE_EXTS)
    suffix = (".tar.gz" if fname.endswith((".tar.gz", ".tgz"))
              else ".tar" if fname.endswith(".tar") else ".zip") if is_archive else Path(fname).suffix

    tmp = db.STORAGE_DIR / f"_upload_{did}{suffix}"
    total = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        400, f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024) // 1024}GB 上限")
                out.write(chunk)
    except HTTPException:
        _safe_unlink(tmp)
        raise
    except OSError as e:
        _safe_unlink(tmp)
        raise HTTPException(400, f"文件接收失败，请重试：{e}") from e

    # ---- G0 预检：确定性错误在这里就拒掉，不再拖到异步解析 ----
    sha = preflight.sha256_of(tmp)
    dup = db.find_upload_by_hash(sha)
    if dup and db.get_dataset(dup["dataset_id"]):
        same = db.get_dataset(dup["dataset_id"]) or {}
        _safe_unlink(tmp)
        raise spec.GateBlocked(spec.GateReport("preflight", [spec.Issue(
            "E1009",
            f"内容与已有数据集「{same.get('name') or dup['dataset_id']}」完全相同",
            f"直接复用 {dup['dataset_id']} 即可，无需重复上传（重复上传会重复占用存储）")]),
            http_status=409)

    report = preflight.inspect_upload(tmp, file.filename or "dataset")
    if not report.ok:
        _safe_unlink(tmp)
        raise spec.GateBlocked(report, http_status=400)

    db.record_upload_hash(sha, did, file.filename or "dataset", total)

    task_id = ai_pipeline.start_parse_task(
        tmp, did, ds_name, file.filename or "dataset", total,
        user_target=target_col or None, user_task=task_hint or None,
    )
    return {"parsing": True, "task_id": task_id, "dataset_id": did,
            "name": ds_name, "sha256": sha,
            "preflight": report.as_dict(),
            "message": "文件已接收，AI 正在解析…"}


@app.get("/api/datasets/parse-status/{task_id}")
def parse_status(task_id: str, user: dict = Depends(auth.current_user)) -> dict:
    """查询后台解析任务的进度与结果。"""
    task = ai_pipeline.get_task(task_id)
    if not task:
        raise HTTPException(404, "解析任务不存在或已过期")
    return {k: v for k, v in task.items() if k != "dataset"}


# ---------------- SOP v2：数据集画像与门禁 ----------------

@app.get("/api/datasets/{did}/profile")
def dataset_profile(did: str, refresh: bool = False,
                    user: dict = Depends(auth.current_user)) -> dict:
    """数据集契约：画像 + 门禁结论。老数据集首次访问时补算。

    `refresh=true` 强制重算（改过数据文件后用）。
    """
    ds = db.get_dataset(did)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    contract = preflight.ensure_contract(ds, force=refresh)
    return {"dataset_id": did, "name": ds.get("name"), "status": ds.get("status"),
            "contract": contract}


@app.post("/api/datasets/{did}/recheck")
def dataset_recheck(did: str, body: dict = Body(...),
                    user: dict = Depends(auth.current_user)) -> dict:
    """修正目标列 / 任务类型 / 剔除泄漏列后重算画像门。

    这是 `needs_fix` 状态唯一的出口：把「系统判不了」交回给人，
    人做完决定后系统重新判定，而不是拿最后一列硬猜。
    """
    ds = db.get_dataset(did)
    if not ds:
        raise HTTPException(404, "数据集不存在")

    fields: dict = {}
    target = (body.get("target_col") or "").strip()
    task = (body.get("task_hint") or "").strip()
    drops = [str(c) for c in (body.get("drop_columns") or [])]

    if task:
        if not spec.spec_of(task):
            raise HTTPException(400, f"不支持的任务类型：{task}")
        fields["task_hint"] = task
    if target:
        fields["target_col"] = target

    if drops:
        df = preflight._read_csv_any(Path(ds["file_path"]))
        keep = [c for c in df.columns if c not in drops]
        new_ver = int(ds.get("version") or 1) + 1
        new_path = db.STORAGE_DIR / f"{did}_v{new_ver}.csv"
        df[keep].to_csv(new_path, index=False, encoding="utf-8-sig")
        fields.update(file_path=str(new_path), version=new_ver,
                      cols=len(keep), rows=int(len(df)))
        if target and target not in keep:
            raise HTTPException(400, f"目标列 {target} 在剔除列表里")

    if fields:
        meta = dict(ds.get("meta") or {})
        meta.pop("contract", None)   # 契约随数据/声明变化一起失效
        fields["meta"] = meta
        db.update_dataset(did, **fields)
        ds = db.get_dataset(did)

    contract = preflight.ensure_contract(ds, force=True)
    return {"dataset_id": did, "status": ds.get("status"),
            "contract": contract,
            "gate": contract.get("gate")}


# ---------------------------------------------------------------- G2 切分门

def _split_frame(ds: dict) -> "pd.DataFrame":
    """只读出切分需要的列（标签 + 患者分组），避免把 12000 列的大表整个读进内存。"""
    import pandas as pd
    from pathlib import Path as _P

    meta = ds.get("meta") or {}
    target = ds.get("target_col") or meta.get("target_col") or "label"
    path = _P(ds["file_path"])
    head = pd.read_csv(path, nrows=0)
    cols = [str(c) for c in head.columns]

    want: set[str] = set()
    if target in cols:
        want.add(target)
    if "label" in cols:
        want.add("label")
    if "split" in cols:
        want.add("split")
    for c in cols:
        if str(c).strip().lower() in split.GROUP_MARKER:
            want.add(c)
    if not want:
        want = set(cols[:3])
    return pd.read_csv(path, usecols=lambda c: c in want)


@app.post("/api/datasets/{did}/split-plan")
def dataset_split_plan(did: str, body: dict = Body(default={}),
                       user: dict = Depends(auth.current_user)) -> dict:
    """G2 切分预览：只计算不落库。

    人先看清楚「每折多少条、每个类在每折有多少、有没有患者跨折」，
    再决定要不要落成正式方案。这一步不做，切分就是黑箱。
    """
    ds = db.get_dataset(did)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    meta = ds.get("meta") or {}
    classes = [str(c) for c in (meta.get("classes") or [])] or None

    try:
        df = _split_frame(ds)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"读取数据集失败，无法计算切分：{e}") from e

    try:
        plan = split.plan_splits(
            df,
            label_col=ds.get("target_col") or meta.get("target_col"),
            classes=classes,
            multilabel=bool(meta.get("multilabel")),
            val_frac=float(body.get("val_frac", 0.2)),
            test_frac=float(body.get("test_frac", 0.15)),
            group_col=(body.get("group_col") or None),
            seed=body.get("seed"),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    out = plan.as_dict()
    out["dataset_id"] = did
    out["name"] = ds.get("name")
    return out


@app.post("/api/datasets/{did}/split-plan:apply")
def dataset_split_apply(did: str, body: dict = Body(default={}),
                        user: dict = Depends(auth.current_user)) -> dict:
    """确认切分方案并存进数据集契约，后续训练直接复用它。

    只存索引与指纹，不重写数据文件——12000 列的表重写一次要几分钟，
    而切分本质就是「哪些行去哪一折」，存索引足够。
    """
    ds = db.get_dataset(did)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    meta = ds.get("meta") or {}
    classes = [str(c) for c in (meta.get("classes") or [])] or None

    df = _split_frame(ds)
    plan = split.plan_splits(
        df,
        label_col=ds.get("target_col") or meta.get("target_col"),
        classes=classes,
        multilabel=bool(meta.get("multilabel")),
        val_frac=float(body.get("val_frac", 0.2)),
        test_frac=float(body.get("test_frac", 0.15)),
        group_col=(body.get("group_col") or None),
        seed=body.get("seed"),
    )
    if not plan.ok and not body.get("force"):
        raise spec.GateBlocked(plan.report, http_status=422)

    stored = plan.as_dict(with_index=True)
    stored["rows"] = int(len(df))
    new_meta = dict(meta)
    new_meta["split_plan"] = stored
    db.update_dataset(did, meta=new_meta)

    return {"dataset_id": did, "applied": True, "gate": plan.report.as_dict(),
            "index_hash": plan.index_hash(), "counts": stored["counts"],
            "per_class": stored["per_class"], "notes": plan.notes}


@app.get("/api/datasets/{did}/split-plan")
def dataset_split_current(did: str, user: dict = Depends(auth.current_user)) -> dict:
    """查看当前生效的切分方案（没有就是还没算过）。"""
    ds = db.get_dataset(did)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    plan = (ds.get("meta") or {}).get("split_plan")
    if not plan:
        raise HTTPException(404, "该数据集尚未确认切分方案，请先调用 split-plan 预览")
    return {"dataset_id": did, **{k: v for k, v in plan.items() if k != "splits"}}


@app.get("/api/datasets/parse-status/{task_id}/result")
def parse_result(task_id: str, user: dict = Depends(auth.current_user)) -> dict:
    """取解析完成后的数据集记录。"""
    task = ai_pipeline.get_task(task_id)
    if not task:
        raise HTTPException(404, "解析任务不存在或已过期")
    if task.get("status") != "done":
        raise HTTPException(409, "解析尚未完成")
    ds = dict(task.get("dataset") or {})
    ds["_parse_report"] = task.get("report") or {}
    return ds


def _save_tabular_dataset(df: "pd.DataFrame", did: str, ds_name: str, filename: str,
                          size_bytes: int, target_col: str | None, task_hint: str | None,
                          extra_meta: dict | None = None) -> dict:
    """表格数据集落库共用逻辑：目标列/任务推断、schema/stats、写 v1 CSV、入库。"""
    # 目标列若为空，取最后一个非 split 列（split 是划分标记，不是特征/目标）
    if not target_col or target_col not in df.columns:
        cols = [c for c in df.columns if c != "split"]
        target_col = str(cols[-1])

    task = analysis.detect_task(df, target_col)
    meta: dict = {"filename": filename}
    if extra_meta:
        meta.update(extra_meta)

    if task_hint == "text_classification":
        # 文本分类：正文列优先用 AI 判定的 text_col，其次自动推断
        text_col = None
        meta_text = (extra_meta or {}).get("text_col")
        if meta_text and meta_text in df.columns and meta_text != target_col:
            text_col = meta_text
        if not text_col:
            text_col = next(
                (c for c in df.columns if c != target_col and analysis.infer_col_type(df[c]) == "text"),
                None,
            )
        if not text_col:
            raise HTTPException(400, "未识别到文本列：文本分类数据集需包含一列长文本内容")
        task = "text_classification"
        meta["text_col"] = str(text_col)

    # 保存 v1：统一落为 CSV，保证下游预览/清洗/EDA/训练管线一致
    path = db.STORAGE_DIR / f"{did}_v1.csv"
    path.write_bytes(df.to_csv(index=False).encode("utf-8-sig"))

    meta.update({
        "schema": analysis.build_schema(df, target_col),
        "stats": analysis.build_stats(df),
    })
    # 分类任务记录类别列表与分布：EDA 洞察、向导类别展示、类别权重建议都依赖它
    # 注意：类别顺序必须与 train.prep_tabular 的 sorted() 完全一致（字典序），
    # 否则展示层顺序与模型输出层索引不对应，存在解读错类别的风险。
    if task in ("tabular_classification", "text_classification"):
        vc = df[target_col].astype(str).value_counts()
        if 1 < len(vc) <= 100:
            ordered = sorted(str(v) for v in vc.index.tolist())
            meta["classes"] = ordered
            meta["class_counts"] = {c: int(vc.get(c, 0)) for c in ordered}
        else:
            meta["classes"] = None
            meta["class_counts"] = None
    record = {
        "id": did,
        "name": ds_name,
        "source": "upload",
        "task_hint": task,
        "target_col": target_col,
        "file_path": str(path),
        "version": 1,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "size_bytes": size_bytes,
        "meta": meta,
    }
    db.insert_dataset(record)
    return db.get_dataset(did)


@app.get("/api/datasets/{did}")
def get_dataset(did: str, user: dict = Depends(auth.current_user)) -> dict:
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    return d


@app.get("/api/datasets/{did}/preview")
def preview(did: str, n: int = 20, user: dict = Depends(auth.current_user)) -> dict:
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    df = load_df(d)
    return analysis.preview_rows(df, n=n)


@app.get("/api/datasets/{did}/image")
def dataset_image(did: str, path: str, token: str | None = None):
    """按 manifest 相对路径返回图像文件。<img> 标签无法带 header，故用 query token 鉴权。"""
    uid = auth.parse_token(token or "")
    if not uid or not db.get_user(uid):
        raise HTTPException(401, "未授权")
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    root_str = (d.get("meta") or {}).get("image_root")
    if not root_str:
        raise HTTPException(400, "该数据集不是图像数据集")
    root = Path(root_str).resolve()
    fp = (root / path).resolve()
    if not str(fp).startswith(str(root)) or not fp.is_file():
        raise HTTPException(404, "图像不存在")
    return FileResponse(fp)


@app.delete("/api/datasets/{did}")
def delete_dataset(did: str, user: dict = Depends(auth.current_user)) -> dict:
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    # 删除物理文件
    for p in db.STORAGE_DIR.glob(f"{did}_v*.csv"):
        try:
            p.unlink()
        except OSError:
            pass
    # AI 解析产物的图像根目录（image_root 指向 _work_<did>）
    img_dir = db.STORAGE_DIR / f"{did}_images"
    if img_dir.exists():
        ai_pipeline._rm_tree(img_dir)
    work_dir = db.STORAGE_DIR / f"_work_{did}"
    if work_dir.exists():
        ai_pipeline._rm_tree(work_dir)
    db.delete_dataset(did)
    return {"ok": True}


# ---------- 解析记忆库 ----------

@app.get("/api/parse-memory")
def parse_memory(user: dict = Depends(auth.current_user)) -> dict:
    """返回解析记忆库概览：已记住的结构模式、复用次数、命中阈值。"""
    try:
        return recall.stats()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"读取记忆库失败：{e}") from e


@app.delete("/api/parse-memory/{mid}")
def delete_parse_memory(mid: str, user: dict = Depends(auth.current_user)) -> dict:
    """删除一条记忆（复用过失败时可人工清除，避免一直命中坏脚本）。"""
    ok = recall.forget(mid)
    if not ok:
        raise HTTPException(404, "该记忆不存在")
    return {"ok": True}


# ---------- EDA ----------

def _image_eda(d: dict, df: pd.DataFrame) -> dict:
    """图像数据集 EDA：样本网格 + 采样尺寸分布。"""
    root = Path(d["meta"]["image_root"])
    sample = df.sample(min(12, len(df)), random_state=42)
    samples = [{"path": str(r.filepath), "label": str(r.label)} for r in sample.itertuples()]
    widths: list[float] = []
    heights: list[float] = []
    probe = df.sample(min(200, len(df)), random_state=1)
    for r in probe.itertuples():
        try:
            with Image.open(root / str(r.filepath)) as im:
                widths.append(float(im.width))
                heights.append(float(im.height))
        except Exception:  # noqa: BLE001
            continue
    return {
        "samples": samples,
        "sizes": {
            "width": analysis.numeric_distribution(pd.Series(widths)),
            "height": analysis.numeric_distribution(pd.Series(heights)),
        },
    }


@app.patch("/api/datasets/{did}/target")
def update_target(did: str, body: dict = Body(...),
                  user: dict = Depends(auth.current_user)) -> dict:
    """修改数据集的「答案列」（目标列）。

    这是给不懂 AI 的用户留的纠错口子：AI 自动认错列时，
    用户应当能自己改，而不是只能重传一遍数据。
    """
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    target = str(body.get("target_col") or "").strip()
    if not target:
        raise HTTPException(400, "请提供目标列名")
    df = load_df(d)
    if target not in df.columns:
        raise HTTPException(400, f"数据里没有「{target}」这一列")
    if target == "split":
        raise HTTPException(400, "「split」是数据划分标记，不能作为答案列")

    meta = dict(d.get("meta") or {})
    old = d.get("target_col")
    # 改答案列 → 任务类型可能随之改变（分类 / 回归 / 文本分类）
    if d.get("task_hint") in ("tabular_classification", "tabular_regression"):
        try:
            import pandas as pd
            s = df[target]
            new_task = ("tabular_regression"
                        if pd.api.types.is_numeric_dtype(s)
                        and s.nunique() > max(20, len(df) * 0.5)
                        else "tabular_classification")
        except Exception:  # noqa: BLE001
            new_task = d.get("task_hint")
        meta["schema"] = analysis.build_schema(df, target)
        vc = df[target].astype(str).value_counts()
        if 1 < len(vc) <= 100:
            meta["classes"] = sorted(str(x) for x in vc.index.tolist())
            meta["class_counts"] = {c: int(vc.get(c, 0)) for c in meta["classes"]}
    else:
        new_task = d.get("task_hint")

    # 表格类数据里若存在明显的正文列，应判为文本分类 ——
    # 否则正文会被当成类别列做 one-hot，几十万行直接撑爆特征矩阵。
    # （历史数据可能是旧版解析器判错的，这里顺带纠正。）
    if new_task in ("tabular_classification", "tabular_regression"):
        try:
            import ai_pipeline as ap
            marker = set(ap._SPLIT_MARKER_COLS)
            text_col = ap._pick_text_column(df, target, marker)
            if text_col:
                new_task = "text_classification"
                meta["text_col"] = text_col
        except Exception:  # noqa: BLE001
            pass

    db.update_dataset(did, target_col=target, task_hint=new_task, meta=meta)
    return {"ok": True, "target_col": target, "task_hint": new_task,
            "old_target_col": old,
            "message": f"已把答案列改为「{target}」"}


@app.get("/api/datasets/{did}/eda")
def eda(did: str, user: dict = Depends(auth.current_user)) -> dict:
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    df = load_df(d)
    payload = analysis.eda_payload(df, d.get("target_col"))
    if d.get("task_hint") == "image_classification" and (d.get("meta") or {}).get("image_root"):
        payload["images"] = _image_eda(d, df)
    return payload


@app.get("/api/datasets/{did}/ai-inspect")
def ai_inspect(did: str, user: dict = Depends(auth.current_user)) -> dict:
    """调用默认大模型对数据集逐列做价值判断，返回保留/丢弃建议。

    仅支持表格与文本数据集（图像数据集不适用列级判断）。
    """
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    if d.get("task_hint") == "image_classification":
        raise HTTPException(400, "图像数据集不适用列级 AI 洞察（已按类别目录自动识别）")
    df = load_df(d)
    # 大列宽安全：只取前 5 行样本即可，避免 prompt 过长
    try:
        result = llm.inspect_dataset(df, d.get("target_col"), d.get("task_hint"))
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    return result


# ---------- 大模型对话 ----------

@app.post("/api/llm/chat")
def llm_chat(body: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    """通用对话接口：透传 messages 给默认大模型，返回文本回复。"""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages 不能为空")
    system = body.get("system")
    try:
        reply = llm.chat(messages, system=system)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    return {"reply": reply}


# ---------- 清洗 ----------

@app.post("/api/datasets/{did}/clean")
def clean(did: str, recipe: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    d = db.get_dataset(did)
    if not d:
        raise HTTPException(404, "数据集不存在")
    df = load_df(d)
    new_df, report = analysis.apply_recipe(df, recipe)
    if new_df.empty:
        raise HTTPException(400, "清洗后数据为空，请调整配方")

    new_version = int(d["version"]) + 1
    path = save_version_csv(new_df, did, new_version)
    db.update_dataset(
        did,
        version=new_version,
        file_path=str(path),
        rows=int(len(new_df)),
        cols=int(new_df.shape[1]),
        size_bytes=path.stat().st_size,
        meta={**d["meta"],
              "schema": analysis.build_schema(new_df, d.get("target_col")),
              "stats": analysis.build_stats(new_df)},
    )
    run = {
        "id": uid("cr"),
        "dataset_id": did,
        "version": new_version,
        "recipe": recipe,
        "report": report,
    }
    db.insert_cleaning_run(run)
    return {"dataset": db.get_dataset(did), "run": run}


@app.get("/api/datasets/{did}/cleaning-runs")
def cleaning_runs(did: str, user: dict = Depends(auth.current_user)) -> list[dict]:
    if not db.get_dataset(did):
        raise HTTPException(404, "数据集不存在")
    return db.list_cleaning_runs(did)
