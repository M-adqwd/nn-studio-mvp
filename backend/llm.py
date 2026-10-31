"""大模型调用封装：统一使用默认 Provider（OpenAI 兼容接口，含 MiniMax）。

提供能力：
1. chat(messages) —— 通用对话，供 AI 助手使用。
2. inspect_dataset(df, meta) —— 把数据集的字段结构、统计量与样本行发给 LLM，
   让模型逐列判断「有用 / 建议丢弃」并给出理由，返回结构化 JSON。
3. profile_archive(path) —— 本地结构画像（不调用 AI），供 AI 判断与指纹计算使用。
4. generate_parser_code / fix_parser_code —— 让 AI 写出数据准备脚本。
"""
from __future__ import annotations

import io
import json
import pickle
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

import db
import fingerprint as fp


def _call_chat(messages: list[dict], *, timeout: int = 60, prefer_fast: bool = False, **extra) -> dict:
    """调用默认 Provider 的 /chat/completions，返回完整响应 dict。

    prefer_fast=True 时走「快速模型优先」策略：
    依次尝试候选模型，第一个成功的即返回。

    关键点：候选列表里**优先放非强制 thinking 的模型**。
    MiniMax-M3 系列（含 M3.1-Flash）强制输出 reasoning，生成长代码时
    思考本身就可能超过 300 秒；M2.5 这类非 thinking 模型则能稳定在 1 分钟内出码。
    """
    p = db.get_default_provider()
    if not p:
        raise RuntimeError("尚未配置默认大模型服务，请先在「大模型配置」中新增并设为默认")
    url = p["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": p["model"],
        "messages": messages,
        "temperature": extra.get("temperature", 0.3),
    }
    # MiniMax M3.1-Flash 强制 thinking；这里不传 thinking/disabled，避免 400
    for k in ("max_tokens", "max_completion_tokens"):
        if k in extra and extra[k]:
            payload[k] = extra[k]

    candidates = _fast_candidates(p["model"]) if prefer_fast else [p["model"]]

    def _post(model: str) -> dict:
        payload["model"] = model
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {p['api_key']}"}
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    last_err: Exception | None = None
    for model in candidates:
        try:
            return _post(model)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            last_err = RuntimeError(f"大模型调用失败（HTTP {e.code}）：{detail or e.reason}")
            # 4xx 多为模型名不可用或参数不支持 → 试下一个候选
            continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            # 超时/网络抖动 → 试下一个候选（往往更快的模型能成）
            continue

    if last_err is not None:
        if isinstance(last_err, RuntimeError):
            raise last_err
        raise RuntimeError(f"无法连接大模型服务：{last_err}") from last_err
    raise RuntimeError("大模型调用失败：没有可用模型")


# 快速候选模型：按优先级排列，第一个成功即返回。
# 非 thinking 模型优先 —— 生成长代码时 thinking 开销极大且容易超时。
_FAST_CANDIDATES = [
    "MiniMax-M2.5",              # 非 thinking，实测生成 7K 字符代码约 80 秒
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2",                # 早期版本，部分账号可用
    "MiniMax-M3.1-Flash-Preview",  # 强制 thinking，兜底用
]


def _fast_candidates(default_model: str) -> list[str]:
    """构造快速模型候选列表（去重、保留默认模型兜底）。"""
    if "MiniMax" not in default_model:
        return [default_model]
    out: list[str] = []
    for m in _FAST_CANDIDATES:
        if m not in out:
            out.append(m)
    if default_model not in out:
        out.append(default_model)
    return out


def _fast_model_of(model: str) -> str | None:
    """由默认模型名推导同系列的快速模型；非 MiniMax 系列返回 None。"""
    if "MiniMax" not in model:
        return None
    return _FAST_CANDIDATES[0]


def chat(messages: list[dict], *, system: str | None = None, timeout: int = 60) -> str:
    """通用对话，返回模型文本内容（已剥离 thinking 推理块）。"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)
    resp = _call_chat(msgs, timeout=timeout)
    try:
        return _strip_reasoning(resp["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"大模型响应格式异常：{resp}") from e


# ---------- 数据集智能洞察 ----------

INSPECT_SYSTEM = (
    "你是一名资深数据科学家，擅长判断表格数据中每一列对建模的价值。"
    "请严格按用户给出的 JSON 格式返回，不要输出多余文字或 Markdown。"
)


_WIDE_LIMIT = 120        # 超过这个列数视为宽表（信号 / 传感器 / 特征工程产物）


def fold_wide_columns(cols: list[str]) -> list[dict]:
    """把宽表的列名折叠成「前缀 + 序号区间」。

    12000 列的 CSV（12 导联 × 1000 采样点）如果逐列塞进提示词，
    prompt 会到几 MB，不但超长而且没有信息量——对模型来说 I_0…I_999
    是同一条信息重复一千遍。折叠成 "I(0-999)×1000" 才是它需要的。
    """
    groups: dict[str, list[int]] = {}
    plain: list[str] = []
    for c in cols:
        m = re.match(r"^(.*?)[_\-]?(\d+)$", str(c))
        if m and m.group(1):
            groups.setdefault(m.group(1), []).append(int(m.group(2)))
        else:
            plain.append(str(c))
    out = []
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:80]:
        out.append({"prefix": k, "count": len(v), "range": [min(v), max(v)]})
    if plain:
        out.append({"prefix": "(无序号)", "count": len(plain), "names": plain[:20]})
    return out


def _build_inspect_user_prompt(df: pd.DataFrame, target_col: str | None, task_hint: str | None) -> str:
    """构造发给 LLM 的数据画像 prompt。

    宽表不逐列描述：先判定是不是「同一组列名 + 连续序号」的结构，
    是的话只给折叠后的分组摘要，外加少量真实列做样本，
    否则 12000 列的 JSON 会把请求撑到几 MB 且必然被截断。
    """
    all_cols = [str(c) for c in df.columns]
    wide = len(all_cols) > _WIDE_LIMIT
    # 宽表只逐列描述首尾各一小段，中间靠分组摘要概括
    detail_cols = all_cols if not wide else (all_cols[:6] + all_cols[-6:])

    schema = []
    for col in detail_cols:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        nulls = int(s.isna().sum())
        sample_vals = [
            None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
            for v in s.dropna().head(5).tolist()
        ]
        dtype = str(s.dtype)
        schema.append({
            "name": str(col),
            "dtype": dtype,
            "n_unique": nunique,
            "nulls": nulls,
            "sample_values": sample_vals,
        })

    stats = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "target_col": target_col,
        "task_hint": task_hint,
    }

    if wide:
        stats["wide_table"] = True
        stats["note"] = (
            f"该表共 {len(all_cols)} 列，属于宽表（典型形态：多导联生理信号、"
            "传感器多通道、或已展开的时序特征）。列名以下是分组摘要，"
            "每组内部是同一物理量按采样点序号展开，不需要逐列判断。"
        )
        cols_desc = (
            "列分组摘要（prefix=列名前缀，count=该组列数，range=序号区间）：\n"
            f"{json.dumps(fold_wide_columns(all_cols), ensure_ascii=False, indent=2)}\n\n"
            "首尾各 6 列的真实画像（用于确认数据类型与量纲）：\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            "对宽表的判断要求：整组信号列要么全保留要么全丢弃，"
            "不要只保留其中一部分采样点；`columns` 里请按「组」返回，"
            "name 写前缀（如 I、II、V1），不要展开成一千行。"
        )
    else:
        cols_desc = (
            "字段画像（每列含类型、唯一值数、缺失数、前5个非空样本值）：\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

    return (
        "以下是一个数据集的字段画像，请逐列判断它对建模是否有用，"
        "并给出建议。判定标准：\n"
        "- 主键 / ID / 流水号 / 全唯一无业务含义 → 建议丢弃\n"
        "- 缺失率极高（>80%）或单一值占比>95% → 建议丢弃\n"
        "- 与目标列重复或完全线性相关 → 建议丢弃\n"
        "- 含业务信息的类别/数值/文本列 → 保留\n"
        "- 时间列可保留（除非是纯序号）\n\n"
        f"数据集统计：{json.dumps(stats, ensure_ascii=False)}\n\n"
        f"{cols_desc}\n\n"
        "请返回如下 JSON（不要包含 Markdown 代码块标记）：\n"
        "{\n"
        '  "summary": "对该数据集的整体分析与建模建议（2-4句）",\n'
        '  "columns": [\n'
        '    {"name": "列名", "useful": true/false, "reason": "一句话理由", "suggestion": "保留/丢弃/作为目标/类型转换建议"}\n'
        "  ],\n"
        '  "recommended_drop": ["建议丢弃的列名列表"],\n'
        '  "target_suggestion": "若用户未指定目标列，建议哪一列作为目标（或说明无需调整）"\n'
        "}"
    )


def _strip_reasoning(text: str) -> str:
    """剥离 MiniMax-M3 等强制 thinking 模型输出的 <think>...</think> 推理块。

    未闭合的 <think> 也一并截断，避免推理内容混入最终结果。
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _parse_json_loose(text: str) -> dict:
    """容错解析 LLM 返回的 JSON：剥离 think 块与 ```json 包裹，截取 {..} 区间二次解析。"""
    cleaned = _strip_reasoning(text)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        lo, hi = cleaned.find("{"), cleaned.rfind("}")
        if lo >= 0 and hi > lo:
            try:
                return json.loads(cleaned[lo:hi + 1])
            except json.JSONDecodeError as e:
                raise RuntimeError(f"大模型返回内容无法解析为 JSON：{e}\n原始内容：{text[:300]}") from e
        raise RuntimeError(f"大模型未返回 JSON 结构，原始内容：{text[:300]}")


def inspect_dataset(df: pd.DataFrame, target_col: str | None, task_hint: str | None,
                    timeout: int = 120) -> dict:
    """调用 LLM 对数据集做逐列价值判断，返回结构化建议。"""
    prompt = _build_inspect_user_prompt(df, target_col, task_hint)
    messages = [
        {"role": "system", "content": INSPECT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = chat(messages, timeout=timeout)
    result = _parse_json_loose(text)

    # 标准化字段，保证前端可用
    cols = result.get("columns") or []
    norm_cols = []
    for c in cols:
        if not isinstance(c, dict):
            continue
        norm_cols.append({
            "name": str(c.get("name", "")),
            "useful": bool(c.get("useful", True)),
            "reason": str(c.get("reason", "")),
            "suggestion": str(c.get("suggestion", "")),
        })
    result["columns"] = norm_cols
    result.setdefault("recommended_drop", [c["name"] for c in norm_cols if not c["useful"]])
    result.setdefault("summary", "")
    result.setdefault("target_suggestion", "")
    return result


# ---------- 数据集结构画像（本地，不调用 AI） ----------

# 画像时最多采样的小文件数量，控制耗时与 prompt 体积
_MAX_TREE = 400
_MAX_SAMPLES = 3
_SAMPLE_TEXT_BYTES = 1500

# 划分相关目录名（用于在画像里高亮提示 AI，不作为解析规则）
_SPLIT_HINTS = ("train", "val", "valid", "validation", "test", "eval", "dev")


def _is_split_like(name: str) -> bool:
    """判断路径段是否像划分目录（train2017 / val_set / TEST 等）。"""
    s = str(name).strip().lower()
    s = re.sub(r"[_\-\.\s]+\d*$", "", s)
    return s in ("train", "training", "trainset", "val", "valid", "validation",
                 "validset", "test", "testing", "testset", "eval", "dev")


def _split_dirs_of(tree: list[str]) -> list[str]:
    """从目录树里找出所有像划分的目录段，供 AI 快速定位。"""
    seen: list[str] = []
    for p in tree:
        for seg in p.replace("\\", "/").split("/")[:-1]:
            if _is_split_like(seg) and seg not in seen:
                seen.append(seg)
    return seen[:20]


def _profile_zip(src: Path) -> dict:
    prof: dict = {"entries": 0, "ext_counts": {}, "tree": [], "samples": {},
                  "top_dirs": {}, "no_ext_probe": [], "read_errors": []}
    with zipfile.ZipFile(src) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        prof["entries"] = len(infos)
        prof["ext_counts"] = dict(Counter(
            Path(i.filename).suffix.lower() or "(无扩展名)" for i in infos))

        for i in infos:
            top = i.filename.replace("\\", "/").split("/")[0]
            prof["top_dirs"][top] = prof["top_dirs"].get(top, 0) + 1
        prof["tree"] = [i.filename for i in infos[:_MAX_TREE]]

        # ---- 无扩展名文件：探测是否为 pickle ----
        for i in infos:
            if Path(i.filename).suffix:
                continue
            if i.file_size > 8 * 1024 * 1024:
                continue
            try:
                head = zf.read(i)[:64]
            except Exception:  # noqa: BLE001
                continue
            kind = None
            if head[:2] == b"\x80\x02" or head[:1] in (b"(", b"}", b"]") or b"pickle" in head[:16]:
                kind = "pickle（Python pickle，可能为 CIFAR/MNIST 等批次格式）"
            elif head[:4] == b"PK\x03\x04":
                kind = "zip"
            elif head[:4] == b"\x89PNG":
                kind = "png"
            elif head[:2] == b"\xff\xd8":
                kind = "jpeg"
            if kind:
                prof["no_ext_probe"].append(f"{i.filename} → {kind}")
            if len(prof["no_ext_probe"]) >= 5:
                break

        # ---- 关键小文件优先采集 ----
        # 这一步决定 AI 能不能一次写对：类别表、划分清单、配置文件的「内容」
        # 比目录列表有用得多，必须优先喂给模型，不能只按遍历顺序取前几个。
        # 例：Darknet 的 classes.names / train.txt 说明了类别与划分，
        # YOLO 的 data.yaml 说明了数据组织 —— 漏掉它们 AI 只能瞎猜。
        priority = {"classes.names": 0, "data.yaml": 0, "data.yml": 0,
                    "labelnames.txt": 0, "labels.txt": 0}
        small_text = []
        for i in infos:
            if i.file_size > 256 * 1024:
                continue
            stem = Path(i.filename).name.lower()
            ext = Path(i.filename).suffix.lower()
            base = int(priority.get(stem, 1)) if ext == "" else (
                0 if stem in priority else
                (2 if stem.endswith((".txt", ".names", ".yaml", ".yml", ".cfg", ".json")) else 3))
            small_text.append((base, len(i.filename), i))
        small_text.sort(key=lambda t: (t[0], t[1]))

        picked = 0
        for _prio, _n, i in small_text:
            if picked >= _MAX_SAMPLES:
                break
            ext = Path(i.filename).suffix.lower()
            try:
                blob = zf.read(i)
            except Exception as e:  # noqa: BLE001
                prof["read_errors"].append(f"{i.filename}: {type(e).__name__}")
                continue
            # 表格类：解析出列名/类型/样本行
            if ext in (".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl"):
                try:
                    if ext in (".parquet", ".pq"):
                        tdf = pd.read_parquet(io.BytesIO(blob)).head(200)
                    elif ext == ".jsonl":
                        tdf = pd.read_json(io.BytesIO(blob), lines=True).head(200)
                    elif ext == ".json":
                        tdf = pd.read_json(io.BytesIO(blob)).head(200)
                    elif ext == ".tsv":
                        tdf = pd.read_csv(io.BytesIO(blob), sep="\t", nrows=200)
                    else:
                        tdf = pd.read_csv(io.BytesIO(blob), sep=None, engine="python", nrows=200)
                    if not tdf.empty:
                        cols = [str(c) for c in tdf.columns]
                        head = tdf.head(3).astype(str).to_dict("records")
                        dtypes = {str(c): str(t) for c, t in tdf.dtypes.items()}
                        prof["samples"][i.filename] = (
                            f"表格文件：{len(cols)} 列 / 读到 {len(tdf)} 行\n"
                            f"列名: {json.dumps(cols, ensure_ascii=False)}\n"
                            f"各列类型: {json.dumps(dtypes, ensure_ascii=False)}\n"
                            f"前 3 行: {json.dumps(head, ensure_ascii=False)[:900]}")
                        picked += 1
                        continue
                except Exception:  # noqa: BLE001
                    pass
            # 文本类：直接给内容
            prof["samples"][i.filename] = blob[:_SAMPLE_TEXT_BYTES * 2].decode("utf-8", "ignore")
            picked += 1

        # 其余小文件补位
        for i in infos:
            if len(prof["samples"]) >= _MAX_SAMPLES:
                break
            if i.file_size > 64 * 1024 or i.filename in prof["samples"]:
                continue
            if Path(i.filename).suffix.lower() not in (
                    ".yaml", ".yml", ".xml", ".txt", ".names", ".cfg", ".json", ".md"):
                continue
            try:
                prof["samples"][i.filename] = zf.read(i)[:_SAMPLE_TEXT_BYTES].decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                continue

        # 无扩展名 pickle 内容探测
        if not prof["samples"]:
            for i in infos:
                if Path(i.filename).suffix or i.file_size > 8 * 1024 * 1024:
                    continue
                try:
                    obj = pickle.loads(zf.read(i), encoding="latin1")
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(obj, dict):
                    prof["samples"][i.filename] = (
                        f"pickle 字典，键={list(obj.keys())!r}；"
                        f"各键类型={{{', '.join(f'{k!r}: {type(v).__name__}' for k, v in list(obj.items())[:6])}}}")
                    break
    return prof


def _profile_tar(src: Path) -> dict:
    import tarfile
    prof: dict = {"entries": 0, "ext_counts": {}, "tree": [], "samples": {},
                  "top_dirs": {}, "no_ext_probe": [], "read_errors": []}
    mode = "r:gz" if src.name.lower().endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(src, mode) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        prof["entries"] = len(members)
        prof["ext_counts"] = dict(Counter(
            Path(m.name).suffix.lower() or "(无扩展名)" for m in members))
        for m in members:
            top = m.name.replace("\\", "/").split("/")[0]
            prof["top_dirs"][top] = prof["top_dirs"].get(top, 0) + 1
        prof["tree"] = [m.name for m in members[:_MAX_TREE]]
        picked = 0
        for m in members:
            ext = Path(m.name).suffix.lower()
            if ext not in (".csv", ".tsv", ".yaml", ".yml", ".json", ".txt") or m.size > 4 * 1024 * 1024:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                blob = f.read(_SAMPLE_TEXT_BYTES)
            except Exception:  # noqa: BLE001
                continue
            prof["samples"][m.name] = blob.decode("utf-8", "ignore")
            picked += 1
            if picked >= _MAX_SAMPLES:
                break
    return prof


def profile_archive(src: Path) -> dict:
    """生成压缩包结构画像。**只读目录与小文件，不解压大文件**，GB 级包也是秒级。

    画像结果附带 structure_summary（人类可读的结构摘要）与 split_dirs（疑似划分目录），
    供 AI 判断使用。
    """
    src = Path(src)
    name = src.name.lower()
    if name.endswith(".zip"):
        prof = _profile_zip(src)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        prof = _profile_tar(src)
    else:
        raise ValueError("profile_archive 只支持 zip / tar / tar.gz")

    prof["split_dirs"] = _split_dirs_of(prof.get("tree") or [])
    prof["structure_summary"] = _structure_summary(prof)
    return prof


def profile_table_file(src: Path) -> dict:
    """单个表格文件（CSV / TSV / Excel / JSON / Parquet）的画像。

    与压缩包画像同构，但额外给出列结构 —— 列结构进入指纹，
    保证记忆库不会把列不同的表格误判为同一结构。
    """
    src = Path(src)
    ext = src.suffix.lower()
    try:
        if ext in (".parquet", ".pq"):
            df = pd.read_parquet(src).head(200)
        elif ext == ".jsonl":
            df = pd.read_json(src, lines=True).head(200)
        elif ext == ".json":
            df = pd.read_json(src).head(200)
        elif ext == ".tsv":
            df = pd.read_csv(src, sep="\t", nrows=200)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(src).head(200)
        else:
            # 编码必须先探测：国内业务表经常是 GBK/UTF-16（Excel 另存的「Unicode 文本」），
            # 直接用默认 utf-8 读会抛 "codec can't decode 0xff"，用户完全看不懂。
            last_err: Exception | None = None
            for enc in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
                try:
                    df = pd.read_csv(src, sep=None, engine="python", nrows=200,
                                     encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError) as e:
                    last_err = e
                    continue
            else:
                raise RuntimeError(f"表格编码无法识别（已尝试 UTF-8/UTF-16/GB18030）：{last_err}")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"表格文件画像失败：{e}") from e

    if df.empty:
        raise RuntimeError("表格文件内容为空")

    cols = [str(c) for c in df.columns]
    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
    head = df.head(3).astype(str).to_dict("records")
    # 宽表（信号/传感器）不能把上千个列名原样写进样本串：
    # 既占满提示词预算，又没有额外信息量，按分组折叠。
    if len(cols) > _WIDE_LIMIT:
        col_text = ("列分组摘要: "
                    + json.dumps(fold_wide_columns(cols), ensure_ascii=False)
                    + f"\n总列数: {len(cols)}")
    else:
        col_text = ("列名: " + json.dumps(cols, ensure_ascii=False)
                    + "\n各列类型: " + json.dumps(dtypes, ensure_ascii=False))

    prof: dict = {
        "entries": 1,
        "ext_counts": {ext or "(无扩展名)": 1},
        "tree": [src.name],
        "samples": {src.name: (
            f"表格文件：{len(cols)} 列 / 读到 {len(df)} 行\n"
            f"{col_text}\n"
            f"前 3 行: {json.dumps(head, ensure_ascii=False)[:1200]}")},
        "top_dirs": {src.name: 1},
        "no_ext_probe": [],
        "read_errors": [],
        "columns": [{"name": c, "dtype": dtypes.get(c, "")} for c in cols],
    }
    if len(cols) > _WIDE_LIMIT:
        prof["wide"] = True
        prof["column_groups"] = fold_wide_columns(cols)
    # 单表里也可能自带 split 列
    split_cols = [c for c in cols if str(c).strip().lower() in ("split", "subset", "fold", "set")]
    prof["split_dirs"] = split_cols
    prof["structure_summary"] = (
        f"单个表格文件 {src.name}：{len(cols)} 列（{', '.join(cols[:10])}）"
        + (f"；自带划分列 {split_cols[0]}" if split_cols else "")
    )
    return prof


def _structure_summary(prof: dict) -> str:
    """把画像压缩成一句人类可读的结构摘要。"""
    ext = prof.get("ext_counts") or {}
    top = sorted((prof.get("top_dirs") or {}).items(), key=lambda kv: -kv[1])[:6]
    bits = [f"共 {prof.get('entries', 0)} 个文件"]
    if top:
        bits.append("顶层目录：" + "、".join(f"{k}({v})" for k, v in top))
    if ext:
        bits.append("类型分布：" + "、".join(
            f"{k}×{v}" for k, v in sorted(ext.items(), key=lambda kv: -kv[1])[:6]))
    if prof.get("split_dirs"):
        bits.append("疑似划分目录：" + "、".join(prof["split_dirs"]))
    return "；".join(bits)


def fingerprint_of(profile: dict) -> dict:
    """由结构画像计算结构指纹（与具体文件名、样本数量无关）。"""
    return fp.dataset_fingerprint(profile)


def _profile_digest(profile: dict) -> str:
    """把结构画像渲染为紧凑文本，供 AI 判断格式与编写解析脚本。"""
    parts = [
        f"文件总数: {profile.get('entries')}",
        f"扩展名分布: {json.dumps(profile.get('ext_counts') or {}, ensure_ascii=False)}",
    ]
    td = profile.get("top_dirs")
    if td:
        top = dict(list(sorted(td.items(), key=lambda kv: -kv[1]))[:12])
        parts.append(f"顶层目录文件数: {json.dumps(top, ensure_ascii=False)}")
    if profile.get("split_dirs"):
        parts.append(f"疑似划分目录: {', '.join(profile['split_dirs'])}")
    probe = profile.get("no_ext_probe")
    if probe:
        parts.append("无扩展名文件类型探测:\n  " + "\n  ".join(probe))
    errs = profile.get("read_errors")
    if errs:
        parts.append("读取失败的文件:\n  " + "\n  ".join(errs))
    parts.append("目录结构（最多 400 条）:\n" + "\n".join((profile.get("tree") or [])[:400]))
    samples = profile.get("samples") or {}
    if samples:
        parts.append("抽样文件内容:\n" + "\n".join(
            f"--- {name} ---\n{head}" for name, head in list(samples.items())[:4]))
    return "\n".join(parts)


# ---------- AI 生成解析脚本（划分感知 + 通用可复用） ----------

GEN_PARSER_SYSTEM = (
    "你是一名资深机器学习数据工程师。你的任务是：看着用户上传的数据集真实结构与内容，"
    "写出一个完整可运行的数据准备脚本，把任意格式的数据集转换成平台统一约定的两份产物。"
    "脚本在受限子进程中执行（禁网、限时），只允许读取当前目录的数据并写出这两个文件。"
)

# 平台与 AI 的契约：AI 只需产出这几份文件，其余（清洗、训练）由平台完成
_OUTPUT_CONTRACT = (
    "【产物契约】你只需要产出两个文件，平台会接管后续的清洗与训练：\n"
    "1. manifest.csv —— 每个样本一行，必须包含且仅需这三列：\n"
    "     filepath : 样本文件相对于「数据集根目录」的相对路径，用正斜杠分隔\n"
    "     label    : 该样本的类别名（字符串）\n"
    "     split    : 该样本属于哪个划分，取值 train / val / test\n"
    "   —— 用于图像、音频等「文件型」数据集。\n"
    "2. dataset.csv —— 用于「表格型」数据集：把整理好的整张表写成 CSV（表头 + 数据行），\n"
    "   其中必须包含一列 split（取值 train / val / test）标记该行属于哪个划分。\n"
    "3. meta.json —— 一份 JSON，至少包含这些键：\n"
    "     kind        : image_classification / object_detection / text_classification / "
    "tabular_classification / tabular_regression\n"
    "     target_col  : 预测目标列名（图像任务固定为 label；表格任务填真实列名）\n"
    "     text_col    : 文本分类的正文列名；非文本任务填空字符串\n"
    "     classes     : 类别名列表（按字典序，分类任务必填）\n"
    "     summary     : 一句话说明你识别出的数据集是什么、怎么组织的、划分怎么来的\n"
    "4. **目标检测（YOLO 格式）专用**：除了上面三个字段，manifest.csv 还要多两列：\n"
    "     boxes    : 该图里所有目标的边界框，用分号分隔多个框，\n"
    "               每个框写成 '类别名,x_center,y_center,width,height'，\n"
    "               坐标是 0~1 的归一化值（YOLO 原生格式）。例如：\n"
    "               'cat,0.5,0.5,0.3,0.4;dog,0.2,0.3,0.1,0.2'\n"
    "     —— 一张图有多个目标时必须全部写出来，不能只取第一个，也不能只写类别丢弃框。\n"
    "   同时 meta.json 里加 'detection': true，并给出 'classes'（全部类别名）。\n"
)

# 划分识别的核心要求 —— 平台最关心的部分
_SPLIT_REQUIREMENT = (
    "【划分识别 · 最重要】用户的训练集 / 验证集 / 测试集通常已经划分好了，"
    "你的核心任务是**准确识别哪些数据属于哪个划分**，让平台直接拿这份划分去训练：\n"
    "1. 优先从目录名识别：train / training / val / valid / validation / test / eval / dev "
    "及其变体（如 train2017、val_set、TEST）。\n"
    "2. 其次从文件名识别：如 train.csv、val.csv、test.csv、train_data.parquet；"
    "也包括 train.txt / test.txt 这类「每行一个文件路径」的清单文件。\n"
    "3. 表格数据集也可能自带 split 列，或通过 train/test 两个文件区分 —— 都要识别出来。\n"
    "4. 目录结构形如 train/类别/图片、类别/train/图片、images/train/类别 —— 都要能处理。\n"
    "5. 如果存在「类别映射文件」（如 train.csv 含 文件名、类别 两列，或 classes.names），"
    "用它确定 label，同时该文件自身的划分名决定对应样本的 split。\n"
    "6. **没有划分信息是正常的**：如果数据里确实只有一类（比如全是一个 train 目录），"
    "就全部标成 train，**不要自己随机切分造出验证集/测试集** —— 平台会在训练时按比例切，"
    "你的任务是如实反映原始数据长什么样。\n"
    "7. 每行都写明 split，不要留空；如果数据集原本有 test，也照样写出来"
    "（平台只是不拿它训练，但会用于最终评估）。\n"
)

# 目标检测（YOLO 格式）的识别要求
_DETECTION_REQUIREMENT = (
    "【目标检测 · YOLO 格式识别】如果数据集带边界框标注（BoundingBox），"
    "**它就是目标检测任务，不是图像分类** —— 千万别当成分类做：\n"
    "1. 典型结构：images/train|val|test/xxx.jpg 与 labels/train|val|test/xxx.txt 一一对应，\n"
    "   或 images/train|classes/xxx.jpg 同级目录放 labels；也可能带 data.yaml 声明类别。\n"
    "2. 每个 .txt 标签文件每行一个目标：'class_id x_center y_center width height'，"
    "   坐标是 0~1 归一化值。class_id 从 0 开始，对应 data.yaml 的 names 列表顺序"
    "（或 classes.names 每行一个类别名）。\n"
    "3. **一张图可能有多个目标** —— 必须把每个目标都写进 manifest.csv 的 boxes 列，"
    "格式 '类别名,x,y,w,h;类别名,x,y,w,h'。丢掉多目标等于把检测降级成分类，"
    "边界框信息必须完整保留。\n"
    "4. meta.json 里 kind 填 'object_detection'，并加 'detection': true，"
    "   'classes' 填全部类别名（按 class_id 顺序，不是字典序）。\n"
    "5. COCO 格式（JSON 标注 + images/annotations）同样支持，"
    "   从 annotations 里取每个 image_id 的所有 bbox 与 category_name。\n"
    "6. 划分仍按目录名 / 清单文件识别，与其它任务一致。\n"
)

_GEN_PARSER_REQUIREMENT = (
    "【硬性要求】\n"
    "1. 当前工作目录就是解压后的数据集根目录，所有路径都相对于它，不要用绝对路径。\n"
    "2. 只读取数据、只写上面那几个产物文件。禁止联网、禁止执行外部命令、禁止删除文件、"
    "禁止写到当前目录之外。\n"
    "3. 允许使用：标准库、numpy、pandas、PIL、pyarrow。\n"
    "4. **绝对不要自己实现 pickle 安全白名单、不要用 RestrictedUnpickler 之类限制**"
    "（历史上就是这么把 numpy 数组误杀掉的）。"
    "读 pickle 直接用 pickle.load(f, encoding='latin1')，失败再试 encoding='bytes'。\n"
    "5. 每一个文件读取都要 try/except 容错跳过坏文件，脚本必须在任何情况下跑完并写出产物，"
    "不要因为个别坏文件整体崩溃。\n"
    "6. 图像数据集：如果原格式是二进制/向量/pickle 形态（如 CIFAR 的 3072 字节向量），"
    "你需要把它还原成真实图像文件保存到当前目录的 images_out/ 下，manifest 里的 filepath "
    "指向 images_out/ 里的文件。\n"
    "7. 数据量大时注意效率：批量用 numpy 处理，图像写盘可分批，避免一次性把几万张图读进内存。\n"
    "8. **写盘体积控制（非常重要）**：还原图像时一律用 JPEG（quality=90）而不是 PNG，"
    "并把图片缩放到 64×64 —— 训练侧本来就会缩放到 64×64，存原尺寸纯属浪费。"
    "PNG 无损会让 6 万张 32×32 图膨胀到 4GB 以上，JPEG 只需几百 MB。"
    "务必边读边写、分批处理，不要把整个批次攒在内存里再写。\n"
    "9. 只输出完整可运行的 Python 代码，不要任何解释文字、不要 Markdown 代码块标记。\n"
)

# 复用要求：脚本必须「自适应」，否则无法在同结构的新数据集上复用
_REUSABLE_REQUIREMENT = (
    "【可复用性 · 重要】这份脚本会被平台记住，同结构的新数据集到来时会被直接复用执行，"
    "因此脚本必须**自适应探测**而不是写死：\n"
    "1. 禁止把具体的数据集名、目录名、列名写死在代码里（例如不要写死 "
    "'cifar-10-batches-py'、'PetImages'、'flower_photos' 这类名字）。\n"
    "2. 列名、目录名都要从实际读到的内容里推断：先列出实际存在的目录名/列名，再据此判断。\n"
    "3. 划分目录名用**关键字匹配**（正则包含 train/val/test/valid/eval/dev），"
    "而不是精确等于某个名字，这样换一批目录名也能复用。\n"
    "4. 类别名、文件名同理：从目录内容读出来，不要预设。\n"
    "5. 路径一律用 os.walk / pathlib 相对遍历，不要假设固定层级。\n"
)

_FIX_REQUIREMENT = (
    "【修正要求】\n"
    "1. 只针对报错根因改动，保持原有正确逻辑不变。\n"
    "2. 如果报错来自你自己实现的安全限制（自定义 Unpickler、路径校验过严、编码判断过严），"
    "直接放宽该限制。\n"
    "3. 如果某个文件读不了，跳过它并继续，不要抛异常中断整个脚本。\n"
    "4. 无论发生什么，脚本结束前必须写出 manifest.csv / dataset.csv 与 meta.json，"
    "哪怕内容不完整也比直接崩溃好。\n"
)


def _clean_code(text: str) -> str:
    """从模型输出中提取纯代码：剥离 think 块、Markdown 包裹与前置散文。"""
    code = _strip_reasoning(text).strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    if not code.startswith(("import ", "from ", "def ", "class ", "#", '"""')):
        idx = min([i for i in (code.find("\nimport "), code.find("\ndef "), code.find("\nfrom "),
                               code.find("\nclass ")) if i > 0] or [-1])
        if idx > 0:
            code = code[idx:].lstrip("\n")
    return code


def generate_parser_code(profile: dict, timeout: int = 300) -> str:
    """让 LLM 依据结构画像生成完整的数据准备脚本（解析 + 还原 + 划分标注）。"""
    prompt = (
        "以下是用户上传的数据集压缩包的真实结构与内容样本：\n"
        f"{_profile_digest(profile)}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n"
        f"{_SPLIT_REQUIREMENT}\n\n"
        f"{_DETECTION_REQUIREMENT}\n\n"
        f"{_REUSABLE_REQUIREMENT}\n\n"
        f"{_GEN_PARSER_REQUIREMENT}\n\n"
        "请写出这个数据准备脚本。"
    )
    resp = _call_chat([{"role": "system", "content": GEN_PARSER_SYSTEM},
                       {"role": "user", "content": prompt}],
                      timeout=timeout, prefer_fast=True)
    try:
        text = _strip_reasoning(resp["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"大模型响应格式异常：{resp}") from e
    code = _clean_code(text)
    try:
        compile(code, "<ai_parser>", "exec")
    except SyntaxError as e:
        raise RuntimeError(f"AI 生成的脚本存在语法错误：第 {e.lineno} 行 {e.msg}")
    if "manifest.csv" not in code and "dataset.csv" not in code:
        raise RuntimeError("AI 生成的脚本未产出 manifest.csv 或 dataset.csv")
    return code


def improve_parser_code(profile: dict, code: str, feedback: str,
                        round_no: int = 1, timeout: int = 240) -> str:
    """把**真实的运行结果**反馈给 AI，让它像人一样看着报错改。

    这是「一次生成」与「迭代收敛」的分水岭：
    只给 AI 结构画像，它只能盲写盲跑；把脚本输出、产物体检、平台校验
    一并喂回去，它就能根据实际情况修正（就像在对话里处理数据集那样）。
    """
    prompt = (
        "你上一轮写的解析脚本没有产出合格结果。以下是它**实际运行**的情况：\n\n"
        f"{feedback}\n\n"
        f"【原脚本（第 {round_no} 轮）】\n{code}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n"
        f"{_SPLIT_REQUIREMENT}\n\n"
        "【修正原则】\n"
        "1. **先根据上面的真实输出判断问题出在哪**，针对性修改，不要推倒重写。\n"
        "2. 如果 train.txt 之类的清单里是别人机器上的绝对路径（如 /home/xxx/），"
        "只能提取文件名部分再去当前目录里匹配，不能直接用那个路径。\n"
        "3. 如果类别名来自 classes.names 之类的文件，就读它来映射，"
        "不要假设类别名等于目录名。\n"
        "4. 标注文件（与图片同名的 .txt）里第一列通常是类别索引，"
        "要用它结合类别表转成类别名。\n"
        "5. 一定要保证脚本能跑完并写出产物：每个文件读取都 try/except 跳过坏文件，"
        "不要因为个别文件报错就整体崩溃。\n"
        f"{_GEN_PARSER_REQUIREMENT}\n\n"
        "请输出修正后的完整脚本。"
    )
    resp = _call_chat([{"role": "system", "content": GEN_PARSER_SYSTEM},
                       {"role": "user", "content": prompt}],
                      timeout=timeout, prefer_fast=True)
    try:
        text = _strip_reasoning(resp["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"大模型响应格式异常：{resp}") from e
    fixed = _clean_code(text)
    try:
        compile(fixed, "<ai_parser>", "exec")
    except SyntaxError as e:
        # 语法坏了就退回原脚本，至少不比现在更差
        return code
    if "manifest.csv" not in fixed and "dataset.csv" not in fixed:
        return code
    return fixed


def fix_parser_code(profile: dict, code: str, error: str, timeout: int = 300) -> str:
    """把执行报错交回 LLM 修正脚本，返回修正后的完整代码。"""
    prompt = (
        "你之前为这个数据集写的数据准备脚本执行失败了，请修正。\n\n"
        f"数据集结构画像：\n{_profile_digest(profile)}\n\n"
        f"【原脚本】\n{code}\n\n"
        f"【执行报错】\n{error}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n"
        f"{_SPLIT_REQUIREMENT}\n\n"
        f"{_FIX_REQUIREMENT}\n"
        f"{_GEN_PARSER_REQUIREMENT}\n\n"
        "请输出修正后的完整脚本。"
    )
    resp = _call_chat([{"role": "system", "content": GEN_PARSER_SYSTEM},
                       {"role": "user", "content": prompt}],
                      timeout=timeout, prefer_fast=True)
    try:
        text = _strip_reasoning(resp["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"大模型响应格式异常：{resp}") from e
    fixed = _clean_code(text)
    try:
        compile(fixed, "<ai_parser>", "exec")
    except SyntaxError as e:
        raise RuntimeError(f"修正后的脚本仍有语法错误：第 {e.lineno} 行 {e.msg}")
    if "manifest.csv" not in fixed and "dataset.csv" not in fixed:
        raise RuntimeError("修正后的脚本未产出 manifest.csv 或 dataset.csv")
    return fixed
