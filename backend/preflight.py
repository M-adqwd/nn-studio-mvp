"""G0 接收预检 与 G1 画像门的具体实现。

这个模块解决审计里暴露的两个硬伤：

1. **所有内容级错误都拖到异步解析才暴露**。预检把「能确定性判定」的错误
   提前到同步阶段：空文件、假 zip、损坏压缩包、路径穿越、空包、嵌套包、
   编码错乱、后缀与内容不符。这样接口就不会再出现「200 + 后台失败」。
2. **数据集没有画像，什么都敢训**。`build_profile` 给出行数/类别/缺失/
   泄漏/可解码率，`spec.gate_dataset` 据此判定是否 ready。

注意：预检不做「是否适合训练」的价值判断，那是画像门的事。
预检只回答一个问题——**这份字节流能否被安全、完整地读进来**。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

import spec
from spec import GateReport, Issue, WARN, BLOCK

# 允许的扩展名（与 main.TABULAR_EXTS 对齐，另外加上压缩包）
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")
TABLE_EXTS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl",
              ".parquet", ".pq")
ALLOWED_EXTS = TABLE_EXTS + ARCHIVE_EXTS

# 解压后总体积上限（相对压缩包的放大倍数）——防 zip bomb
MAX_ZIP_RATIO = 60
MAX_ZIP_ENTRIES = 500_000


def sha256_of(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ext_of(filename: str) -> str:
    low = (filename or "").lower()
    for e in (".tar.gz", ".tgz"):
        if low.endswith(e):
            return e
    return Path(low).suffix


def _sniff(path: Path) -> str:
    """按魔数判断真实容器类型。返回 zip|gzip|parquet|xlsx|text|unknown。"""
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head[:2] == b"PK":
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:4] == b"PAR1":
        return "parquet"
    if head[:4] == b"Rar!":
        return "rar"
    if head[:5] == b"%PDF-":
        return "pdf"
    return "text"


# ---------------------------------------------------------------- G0 预检

def inspect_upload(path: Path, filename: str) -> GateReport:
    """同步预检。调用方在返回前必须保证 path 已完整落盘。"""
    rep = GateReport("preflight")
    ext = _ext_of(filename)
    size = path.stat().st_size if path.exists() else 0
    fname = filename or "dataset"

    if ext not in ALLOWED_EXTS:
        rep.issues.append(Issue(
            "E1001", f"不支持的文件类型：{ext or '(无扩展名)'}",
            "支持 CSV / TSV / TXT / XLSX / XLS / JSON / JSONL / Parquet，"
            "文件型数据集请打包为 ZIP / tar.gz"))
        return rep

    if size == 0:
        rep.issues.append(Issue("E1003", "文件为空（0 字节）", "请确认导出是否成功"))
        return rep

    kind = _sniff(path)

    # 后缀与内容一致性：只对强特征类型做检查，文本类放宽（csv/txt/json 都是文本）
    if ext == ".zip" and kind != "zip":
        rep.issues.append(Issue(
            "E1002", "文件后缀是 .zip，但内容不是 ZIP（可能被改过后缀或传输损坏）",
            "重新打包上传；若实际是其它格式，请改成正确后缀"))
        return rep
    if ext in (".xlsx",) and kind != "zip":
        rep.issues.append(Issue("E1002", ".xlsx 文件结构损坏（不是有效的 OOXML 包）",
                                "用 Excel 重新另存为 .xlsx"))
        return rep
    if ext == ".parquet" and kind != "parquet":
        rep.issues.append(Issue("E1002", ".parquet 文件魔数不符", "重新导出"))
        return rep
    if kind in ("rar", "pdf"):
        rep.issues.append(Issue(
            "E1002", f"文件实际是 {kind.upper()}，不在支持范围",
            "RAR 请先解压再以 ZIP 上传；PDF 请先转成表格或图像"))
        return rep

    if ext in (".zip",) or kind == "zip" and ext in (".zip",):
        _inspect_zip(path, rep)
    elif ext in (".tar", ".tar.gz", ".tgz") or kind == "gzip":
        _inspect_tar(path, rep)
    elif ext in (".csv", ".tsv", ".txt", ".json", ".jsonl"):
        _inspect_text(path, ext, rep)

    return rep


def _inspect_zip(path: Path, rep: GateReport) -> None:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        rep.issues.append(Issue("E1005", "压缩包已损坏，无法解压", "重新压缩后再上传"))
        return
    except OSError as e:
        rep.issues.append(Issue("E1005", f"压缩包无法读取：{e}", "检查文件是否被占用"))
        return

    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if not infos:
            rep.issues.append(Issue("E1007", "压缩包里没有任何文件",
                                    "确认打包时选中的是内容目录而不是空目录"))
            return
        if len(infos) > MAX_ZIP_ENTRIES:
            rep.issues.append(Issue(
                "E1004", f"压缩包条目过多：{len(infos)} > {MAX_ZIP_ENTRIES}",
                "请拆分上传"))
            return

        total = sum(i.file_size for i in infos)
        try:
            comp = max(path.stat().st_size, 1)
        except OSError:
            comp = 1
        if total / comp > MAX_ZIP_RATIO and total > 200 * 1024 * 1024:
            rep.issues.append(Issue(
                "E1004", f"解压后体积异常（{total / 1024 ** 3:.1f}GB，压缩比 {total / comp:.0f}:1）",
                "可能是压缩炸弹或大量重复文件，请检查后再上传"))
            return

        bad_paths: list[str] = []
        nested: list[str] = []
        real = 0
        for i in infos:
            name = i.filename
            posix = PurePosixPath(name.replace("\\", "/"))
            if posix.is_absolute() or ".." in posix.parts:
                bad_paths.append(name)
            low = name.lower()
            if low.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".rar", ".7z")):
                nested.append(name)
            if not low.startswith("__macosx/") and not low.endswith(".ds_store"):
                real += 1

        if bad_paths:
            rep.issues.append(Issue(
                "E1006", f"压缩包内含非法路径（目录穿越/绝对路径）：{bad_paths[0]}",
                "重新打包，确保条目是相对路径"))
        if real == 0:
            rep.issues.append(Issue("E1007", "压缩包里只有系统临时文件（__MACOSX/.DS_Store）",
                                    "重新打包，排除隐藏文件"))
        if nested:
            rep.issues.append(Issue(
                "E1008", f"压缩包内还有压缩包：{nested[0]}",
                "请先解开外层，把内容目录直接打包上传"))


def _inspect_tar(path: Path, rep: GateReport) -> None:
    import tarfile
    try:
        with tarfile.open(path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if not members:
                rep.issues.append(Issue("E1007", "压缩包里没有任何文件", "确认打包内容"))
                return
            bad = [m.name for m in members
                   if m.name.startswith("/") or ".." in PurePosixPath(m.name).parts]
            if bad:
                rep.issues.append(Issue(
                    "E1006", f"压缩包内含非法路径：{bad[0]}", "重新打包为相对路径"))
    except (tarfile.TarError, OSError) as e:
        rep.issues.append(Issue("E1005", f"压缩包无法读取：{e}", "重新压缩"))


def decode_text(path: Path, sample: int = 256 * 1024) -> tuple[str | None, str | None]:
    """嗅探文本编码。返回 (text, encoding)，失败返回 (None, None)。

    顺序很关键：先看 BOM（UTF-16 的 0xFF 0xFE 会被 utf-8 直接判死），
    再试 utf-8-sig / gb18030。gb18030 放最后是因为它几乎能解码任何字节，
    放前面会把 UTF-8 文件误判。
    """
    raw = path.open("rb").read(sample)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, None


def _inspect_text(path: Path, ext: str, rep: GateReport) -> None:
    text, enc = decode_text(path)
    if text is None:
        rep.issues.append(Issue(
            "E1010", "文件编码无法识别（不是 UTF-8 / UTF-16 / GB18030）",
            "用 Excel 或编辑器另存为 UTF-8 CSV 后重传"))
        return
    stripped = text.strip()
    if not stripped:
        rep.issues.append(Issue("E1003", "文件只有空白字符，没有内容", "确认导出是否成功"))
        return

    if ext in (".json", ".jsonl"):
        return

    # CSV/TSV：至少要有表头 + 1 行数据
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) < 2:
        rep.issues.append(Issue(
            "E1003", "表格只有表头没有数据行", "确认导出时是否包含了数据"))
        return

    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]),
                                      delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        # 单列文本也可能合法（例如只有一列文本语料），给警告而不是拒绝
        rep.issues.append(Issue(
            "W1001", "未能识别分隔符（可能是单列表格或非标准分隔）",
            "若确认是单列数据可忽略；否则请导出为标准 CSV", WARN))
        return
    header_cols = len(lines[0].split(delim))
    if header_cols < 2:
        rep.issues.append(Issue(
            "W1002", "表格只有 1 列，无法构成特征+标签", "检查导出设置", WARN))


# ---------------------------------------------------------------- G1 画像

def _read_csv_any(path: Path) -> pd.DataFrame:
    """按 BOM → utf-8-sig → utf-8 → gb18030 的顺序读表。"""
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 兜底：分隔符嗅探失败时按 python 引擎猜
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8",
                       encoding_errors="replace")


def build_profile(ds: dict, task: str | None = None, target: str | None = None) -> dict:
    """从数据集产物算出画像。失败时抛异常，由调用方决定是否阻断。"""
    meta = ds.get("meta") or {}
    task = task or ds.get("task_hint") or ""
    target = target or ds.get("target_col")
    prof: dict = {"rows": int(ds.get("rows") or 0), "cols": int(ds.get("cols") or 0)}
    # 目标列来源决定它是「事实」还是「猜测」：兜底猜出来的必须人工确认
    prof["target_source"] = meta.get("target_source") or "unknown"

    fp = Path(ds.get("file_path") or "")
    if not fp.exists():
        raise FileNotFoundError(f"数据集文件不存在：{fp}")

    df = _read_csv_any(fp)
    prof["rows"], prof["cols"] = int(df.shape[0]), int(df.shape[1])
    prof["duplicate_rows"] = int(df.duplicated().sum())
    prof["missing_pct"] = float(df.isna().sum().sum() / max(df.size, 1))

    # 图像数据集：标签在 label 列，像素在磁盘，抽检可解码率
    if "filepath" in df.columns and "label" in df.columns:
        root = Path(meta.get("image_root") or "")
        sample = df.sample(min(24, len(df)), random_state=42) if len(df) else df
        bad = 0
        checked = 0
        try:
            from PIL import Image
            for _, r in sample.iterrows():
                p = Path(str(r["filepath"]))
                if not p.is_absolute():
                    p = root / p
                checked += 1
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception:  # noqa: BLE001
                    bad += 1
        except Exception:  # noqa: BLE001
            checked = 0
        prof["decode_fail_pct"] = (bad / checked) if checked else 0.0
        prof["decode_checked"] = checked

    # 类别分布
    if target and target in df.columns:
        s = df[target].dropna().astype(str)
        prof["target"] = target
        vc = s.value_counts()
        prof["n_classes"] = int(len(vc))
        if 1 < len(vc) <= 200:
            prof["classes"] = sorted(str(v) for v in vc.index)
            prof["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        if task == "tabular_regression":
            try:
                prof["target_var"] = float(pd.to_numeric(df[target], errors="coerce").var())
            except Exception:  # noqa: BLE001
                prof["target_var"] = None

        # 标签泄漏：任一特征列与目标列逐行完全同值
        leaks: list[str] = []
        tvals = df[target].astype(str).values
        for c in df.columns:
            if c == target or c == "split":
                continue
            try:
                if (df[c].astype(str).values == tvals).all():
                    leaks.append(str(c))
            except Exception:  # noqa: BLE001
                continue
        prof["leak_suspects"] = leaks

    # 常量列（只对中小表做，宽表跳过）
    if df.shape[1] <= 500:
        const = [str(c) for c in df.columns
                 if c not in (target, "split") and df[c].nunique(dropna=False) <= 1]
        prof["constant_cols"] = const[:50]
    else:
        prof["constant_cols"] = []

    # 文本长度
    tcol = meta.get("text_col")
    if task == "text_classification" and tcol and tcol in df.columns:
        prof["text_len_mean"] = float(df[tcol].astype(str).str.len().mean())

    if not prof.get("classes") and meta.get("classes"):
        prof["classes"] = list(meta["classes"])
        prof["class_counts"] = meta.get("class_counts") or {}
        prof["n_classes"] = len(prof["classes"])

    return prof


def dataset_contract(ds: dict, profile: dict, report: GateReport) -> dict:
    """把画像 + 门禁结论固化成契约，写回 dataset.meta。"""
    task = ds.get("task_hint") or ""
    sp = spec.spec_of(task)
    return {
        "contract_version": "2.0",
        "kind": sp.modality if sp else None,
        "task": task,
        "target": ds.get("target_col"),
        "classes": profile.get("classes") or (ds.get("meta") or {}).get("classes"),
        "class_counts": profile.get("class_counts") or (ds.get("meta") or {}).get("class_counts"),
        "profile": {k: v for k, v in profile.items() if k != "classes"},
        "gate": report.as_dict(),
        "checked_at": report.checked_at,
    }


def ensure_contract(ds: dict, force: bool = False, persist: bool = True) -> dict:
    """保证数据集有可用的契约与门禁结论；返回 contract。

    老数据集（SOP v1 上传的）没有 contract，第一次用到时补算。
    补算失败不阻断主流程，但会以 W9001 标注——不能因为门禁自己出错
    就让整个平台停摆。
    """
    import db  # 延迟导入，避免循环依赖

    meta = dict(ds.get("meta") or {})
    contract = meta.get("contract")
    if contract and not force:
        return contract

    try:
        profile = build_profile(ds)
        report = spec.gate_dataset(ds.get("task_hint") or "", profile,
                                   target=ds.get("target_col"))
        contract = dataset_contract(ds, profile, report)
    except Exception as e:  # noqa: BLE001
        contract = {
            "contract_version": "2.0",
            "task": ds.get("task_hint"),
            "gate": {"gate": "dataset", "status": "warn",
                     "codes": ["W9001"],
                     "warnings": [{"code": "W9001", "level": "warn",
                                   "message": f"画像计算失败，按放行处理：{type(e).__name__}: {e}",
                                   "hint": "请检查数据集文件是否完整"}],
                     "blocking": []},
            "checked_at": time_str(),
        }
        report = None

    if persist:
        meta["contract"] = contract
        status = "ready"
        if report is not None and not report.ok:
            status = spec.NEEDS_FIX
        try:
            db.update_dataset(ds["id"], meta=meta, status=status)
            ds["status"] = status
            ds["meta"] = meta
        except Exception:  # noqa: BLE001
            pass
    return contract


def time_str() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def gate_of(ds: dict) -> GateReport:
    """取数据集的门禁报告（不重算）。没有 contract 时返回空通过报告。"""
    c = ((ds.get("meta") or {}).get("contract") or {})
    g = c.get("gate") or {}
    rep = GateReport("dataset", checked_at=g.get("checked_at") or "")
    for item in g.get("blocking") or []:
        rep.issues.append(Issue(item["code"], item["message"], item.get("hint") or "",
                                BLOCK, item.get("field")))
    for item in g.get("warnings") or []:
        rep.issues.append(Issue(item["code"], item["message"], item.get("hint") or "",
                                WARN, item.get("field")))
    return rep
