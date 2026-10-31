"""AI 解析统一管线：上传 → 画像 → 指纹 → 记忆命中 → AI 生成 → 受限执行 → 落库。

设计目标（对应产品诉求）：
1. **完全由 AI 决定怎么解析**：不再有写死的格式配方。平台只定义「产物契约」，
   AI 负责看懂任意目录结构 / 表格 / 二进制批次，并写出数据准备脚本。
2. **划分是解析结果的一部分**：AI 必须识别出哪些是训练集 / 验证集 / 测试集，
   并写进产物的 split 列。训练阶段直接消费该列，无需再次切分。
3. **结构记忆**：同一结构的第二个数据集不再调用 AI —— 复用已验证脚本，
   只重跑一次受限执行，秒级完成。
4. **快 + 大**：画像阶段只读目录与小样本（不解压），GB 级包也是秒级；
   执行阶段流式解压、限时强杀，避免拖垮服务。
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

import pandas as pd

import analysis
import db
import llm
import recall


def _rm_tree(path: Path) -> None:
    """递归删除目录。

    不用 shutil.rmtree：部分运行环境会对它做批量删除保护而直接抛 SystemExit，
    导致异常处理本身再次失败。这里逐层删除，任何失败都忽略。
    """
    path = Path(path)
    if not path.exists():
        return
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    (Path(root) / name).unlink()
                except BaseException:  # noqa: BLE001  含 SystemExit（删除保护）
                    pass
            for name in dirs:
                try:
                    (Path(root) / name).rmdir()
                except BaseException:  # noqa: BLE001
                    pass
        path.rmdir()
    except BaseException:  # noqa: BLE001
        pass

# ---------- 产物契约 ----------
# AI 生成的脚本必须产出下面两种之一，平台据此落库：
#   manifest.csv —— 文件型数据集（图像等）：filepath / label / split
#   dataset.csv  —— 表格型数据集：原始列 + split
MANIFEST_NAME = "manifest.csv"
DATASET_NAME = "dataset.csv"
META_NAME = "meta.json"

# 合法划分名。AI 允许输出别的别名，落库前统一归一到这三个。
CANON_SPLITS = ("train", "val", "test")

# 平台内置的极简别名归一表（只做词形归一，不涉及任何「格式配方」）
_SPLIT_ALIASES = {
    "train": "train", "training": "train", "trainset": "train", "train_set": "train",
    "tr": "train", "fit": "train",
    "val": "val", "valid": "val", "validation": "val", "val_set": "val",
    "dev": "val", "eval": "val", "evaluation": "val",
    "test": "test", "testing": "test", "testset": "test", "test_set": "test", "te": "test",
}

# 划分标记列名：这些列是「划分说明」，不是特征也不是目标。
# 刻意不含过于宽泛的 "set"/"group" —— 真实数据里确实存在叫 set/group 的业务列，
# 误排除会白白丢一个有效特征。
_SPLIT_MARKER_COLS = {"split", "subset", "partition", "fold", "dataset_split",
                      "data_split", "split_type", "set_split"}


class ParseError(Exception):
    """解析失败（对外转成 HTTP 400）。"""


# ---------- 划分归一 ----------

def canon_split(raw: object) -> str | None:
    """把任意划分写法归一为 train / val / test，无法识别返回 None。"""
    s = str(raw or "").strip().lower()
    if not s:
        return None
    if s in _SPLIT_ALIASES:
        return _SPLIT_ALIASES[s]
    # 形如 train2017 / val_set_2 / test-1：取前缀做匹配
    for key, val in _SPLIT_ALIASES.items():
        if s.startswith(key):
            return val
    return None


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名去重，并把空列名改成有意义的名字。

    AI 生成的 CSV 若出现重复列名，pandas 读回来后 df["split"] 会返回 DataFrame
    而非 Series，后续的 .value_counts().index[0] 就会拿到一个 Index 对象，
    引发难以定位的类型错误。这里在入口处一次性收拾干净。
    """
    df = df.copy()
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for i, raw in enumerate(df.columns):
        name = str(raw).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


def normalize_split_column(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把产物中的 split 列归一为 train/val/test，并统计各划分数量。

    返回 (新 DataFrame, 划分统计 dict)。
    """
    df = _dedupe_columns(df)

    if "split" not in df.columns:
        df["split"] = "train"
        return df, {"train": int(len(df)), "val": 0, "test": 0}

    raw = df["split"].astype(str).str.strip().str.lower()
    mapped = raw.map(canon_split)
    # 未识别的划分归入 train（宁可多训练，也不要静默丢样本）
    df["split"] = mapped.fillna("train")

    counts = {k: int((df["split"] == k).sum()) for k in CANON_SPLITS}
    if counts["train"] == 0 and len(df):
        # 全部被判成非训练集：取数量最多的那个当训练集，避免训练阶段空集
        vc = df["split"].value_counts()
        mode = str(vc.index[0])
        df.loc[df["split"] == mode, "split"] = "train"
        counts = {k: int((df["split"] == k).sum()) for k in CANON_SPLITS}
    return df, counts


# ---------- 受限执行器 ----------

# 沙箱守卫：注入到子进程最前面，阻断网络、子进程与越权写入。
_GUARD_SRC = r'''
import os, sys, builtins, socket

# --- 工作目录（一切读写的边界）---
_work = os.path.abspath(os.environ.get("MF_WORKDIR", "."))
os.chdir(_work)
_work_prefix = _work.rstrip("\\/") + os.sep

# --- 禁网：任何 socket 创建都直接失败 ---
class _NetworkBlocked(RuntimeError):
    pass

def _blocked(*a, **k):
    raise _NetworkBlocked("沙箱禁止网络访问")

# 注意：socket.socket 必须用「类」替换，不能用函数。
# ssl 模块里有 `class SSLSocket(socket)`，一旦 socket 变成函数，
# 任何间接 import ssl 的脚本（requests / urllib / torchvision 权重下载等）
# 都会在导入阶段抛出 TypeError: function() argument 'code' must be code。
# 这里继承真正的 socket.socket 再禁掉实例化，既保住了类语义（isinstance、
# 类属性照常），又让真正的建连动作失败。
class _BlockedSocket(socket.socket):
    def __init__(self, *a, **k):
        raise _NetworkBlocked("沙箱禁止网络访问")

socket.socket = _BlockedSocket
socket.SocketIO = _BlockedSocket
socket.create_connection = _blocked
socket.create_server = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked

# --- 禁子进程 / 禁动态代码执行 ---
def _no_popen(*a, **k):
    raise RuntimeError("沙箱禁止执行外部命令")
os.popen = _no_popen

# 同理，subprocess.Popen 也得用「类」替换：asyncio.windows_utils 里有
# `class Popen(subprocess.Popen)`，替换成函数会让 import asyncio 直接崩。
try:
    import subprocess as _sp

    class _BlockedPopen(_sp.Popen):
        def __init__(self, *a, **k):
            raise RuntimeError("沙箱禁止执行外部命令")

    _sp.Popen = _BlockedPopen
    _sp.run = _no_popen
    _sp.call = _no_popen
    _sp.check_output = _no_popen
except Exception:
    pass

# --- 越权写入拦截：只允许写工作目录内部 ---
_real_open = builtins.open
_real_os_open = os.open

def _in_work(path):
    try:
        ap = os.path.abspath(os.fspath(path))
    except Exception:
        return True   # 无法判定时放行读取类调用
    return ap.startswith(_work_prefix)

def _guarded_open(file, mode="r", *a, **k):
    if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
        if not _in_work(file):
            raise PermissionError("沙箱禁止写入工作目录之外的路径：%s" % (file,))
    return _real_open(file, mode, *a, **k)

def _guarded_os_open(path, flags, *a, **k):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND):
        if not _in_work(path):
            raise PermissionError("沙箱禁止写入工作目录之外的路径：%s" % (path,))
    return _real_os_open(path, flags, *a, **k)

builtins.open = _guarded_open
os.open = _guarded_os_open

# --- 环境变量白名单（保留 PATH/PYTHONIOENCODING 等必要项）---
_KEEP = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
         "PYTHONIOENCODING", "PYTHONHASHSEED", "HOME", "USERPROFILE",
         "LOCALAPPDATA", "APPDATA", "NUMBER_OF_PROCESSORS"}
for _k in list(os.environ.keys()):
    if _k not in _KEEP:
        os.environ.pop(_k, None)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# --- 资源限制（Unix 有效，Windows 退化）---
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 2048))
except Exception:
    pass

# --- 载入用户脚本 ---
_target = sys.argv[1]
sys.argv = [_target]
_g = {"__name__": "__main__", "__file__": _target}
with open(_target, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _target, "exec"), _g)
'''


def _inspect_products(work_dir: Path, exec_res: dict) -> str:
    """体检当前产物，生成给 AI 看的反馈。

    这是「让 AI 看到自己干了什么」的关键 —— 没有它，AI 只能盲写盲跑。
    反馈包含：脚本输出、报错、产物行数、split 分布、类别、路径是否存在。
    """
    lines: list[str] = []

    code_out = (exec_res.get("stdout") or "").strip()
    if code_out:
        lines.append(f"[脚本 stdout]\n{code_out[-1500:]}")
    err = (exec_res.get("stderr") or "").strip()
    if err:
        lines.append(f"[脚本 stderr / 报错]\n{err[-1500:]}")
    rc = exec_res.get("returncode")
    if rc not in (0, None):
        lines.append(f"[退出码] {rc}（非 0 表示脚本中途出错或抛异常）")

    mp = work_dir / MANIFEST_NAME
    dp = work_dir / DATASET_NAME
    for path, kind in ((mp, "manifest.csv"), (dp, "dataset.csv")):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            lines.append(f"[产物] {kind} 存在但无法解析：{e}")
            continue
        info = [f"[产物] {kind}: {len(df)} 行 × {len(df.columns)} 列",
                f"  列名: {list(df.columns)}"]
        if "label" in df.columns:
            vc = df["label"].astype(str).value_counts()
            info.append(f"  类别数: {len(vc)}；分布: {dict(list(vc.items())[:10])}")
        if "split" in df.columns:
            sc = df["split"].astype(str).value_counts()
            info.append(f"  split 分布: {dict(sc)}")
        if "filepath" in df.columns:
            root = work_dir.resolve()
            miss = [str(r) for r in df["filepath"].astype(str).head(50)
                    if not (root / str(r)).exists()]
            info.append(f"  filepath 指向不存在的文件: {len(miss)}"
                        + (f"，例: {miss[:3]}" if miss else "（全部有效）"))
        if "target_col" in df.columns or len(df.columns) > 3:
            if "filepath" not in df.columns:
                info.append(f"  前 3 行: {df.head(3).astype(str).to_dict('records')}")
        lines.append("\n".join(info))

    meta_p = work_dir / META_NAME
    if meta_p.exists():
        lines.append(f"[meta.json]\n{meta_p.read_text(encoding='utf-8')[:800]}")

    if not lines:
        lines.append("[产物检查] 工作目录里没有 manifest.csv / dataset.csv —— 脚本什么都没产出。")
    return "\n\n".join(lines)


def _run_in_sandbox(code: str, work_dir: Path, timeout: int,
                    heartbeat=None) -> dict:
    """在受限子进程中执行 AI 生成的解析脚本。

    heartbeat —— 长时间执行期间的进度回调（每 5 秒调一次），
    让前端能显示「脚本仍在运行 + 已用时/预计剩余」，而不是一片死寂。

    关键修复：使用 `-I`（隔离模式）但 **不加 `-S`**。
    `-S` 会禁用 site 模块，导致 pandas / numpy / PIL 全部无法 import，
    使 AI 生成的脚本 100% 失败。`-I` 只隔离环境变量与用户 site，保留标准 site-packages。

    工作目录即数据集根目录，脚本只在此目录内读写。
    """
    script = work_dir / "_ai_parser.py"
    script.write_text(code, encoding="utf-8")
    guard = work_dir / "_sandbox_guard.py"
    guard.write_text(_GUARD_SRC, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "TEMP": os.environ.get("TEMP", str(work_dir)),
        "TMP": os.environ.get("TMP", str(work_dir)),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MF_WORKDIR": str(work_dir),
    }

    # 边跑边监控写盘体积：AI 还原图像时可能写出远超预期的文件量
    # （如 6 万张 PNG = 4GB+），撑爆磁盘。超限立刻终止。
    proc = subprocess.Popen(
        [sys.executable, "-I", str(guard), str(script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(work_dir), env=env,
    )
    deadline = time.time() + timeout
    over_limit = False
    last_beat = time.time()
    while True:
        try:
            proc.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            now = time.time()
            if now > deadline:
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
                raise ParseError(
                    f"AI 解析脚本执行超过 {timeout} 秒已被终止（可能是死循环或数据量过大）")
            if not over_limit and _dir_size(work_dir) > MAX_PRODUCT_BYTES:
                over_limit = True
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
                raise ParseError(
                    "AI 解析脚本写出的数据超过 12GB 上限（可能用无损格式存了大量图像），"
                    "已终止以避免占满磁盘。请重试。")
            # 心跳：脚本可能跑好几分钟，期间必须持续回报，
            # 否则前端看起来像卡死，用户会以为失败
            if heartbeat and now - last_beat >= 5:
                last_beat = now
                try:
                    heartbeat()
                except Exception:  # noqa: BLE001
                    pass

    out, err = proc.communicate()
    if over_limit:
        raise ParseError("AI 解析脚本写盘超过上限，已终止。")

    return {
        "returncode": proc.returncode,
        "stdout": (out or b"").decode("utf-8", "ignore"),
        "stderr": (err or b"").decode("utf-8", "ignore"),
    }


def _dir_size(path: Path) -> int:
    """目录总字节数（用于写盘护栏）。

    目录可能有几十万文件，全量 stat 太慢；这里只统计顶层若干子项的体积，
    对「写了几个 GB 图像」这种场景足够可靠，且开销可控。
    """
    total = 0
    try:
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    # 只下探一层，避免在大目录上耗时过久
                    for sub in entry.iterdir():
                        try:
                            if sub.is_file():
                                total += sub.stat().st_size
                        except OSError:
                            continue
            except OSError:
                continue
    except OSError:
        pass
    return total


# 解析产物占用上限（GB）——超出说明 AI 写法有问题（如用 PNG 存大批量图像）
MAX_PRODUCT_BYTES = 12 * 1024 * 1024 * 1024


# ---------- 解压 ----------

def _safe_extract(archive: Path, dest: Path) -> None:
    """解压到 dest，拒绝路径穿越（Zip Slip / Tar Slip）。

    解压大包可能长达十几分钟，必须持续回报进度，
    否则用户只看到「卡住不动」，无法判断是在工作还是死了。
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    name = archive.name.lower()
    t0 = time.time()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            total = len(infos)
            # 先把路径安全检查做完（廉价），再真正解压
            for info in infos:
                target = (dest / info.filename).resolve()
                if not str(target).startswith(str(root)):
                    raise ParseError("压缩包内包含非法路径（目录穿越），已拒绝解压")
            last = time.time()
            for n, info in enumerate(infos, 1):
                zf.extract(info, dest)
                now = time.time()
                if now - last >= 1.0:
                    last = now
                    frac = n / max(total, 1)
                    eta = (now - t0) / max(n, 1) * (total - n)
                    _progress(
                        f"正在解压文件… {n}/{total}（{frac * 100:.0f}%）",
                        0.22 + frac * 0.33)
                    _extract_state(files_done=n, files_total=total,
                                   seconds=int(now - t0), eta=int(eta))
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        import tarfile
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive, mode) as tf:
            members = tf.getmembers()
            total = len(members)
            for member in members:
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(root)):
                    raise ParseError("压缩包内包含非法路径（目录穿越），已拒绝解压")
                if member.issym() or member.islnk():
                    raise ParseError("压缩包内包含链接条目，已拒绝解压")
            last = time.time()
            for n, member in enumerate(members, 1):
                tf.extract(member, dest)
                now = time.time()
                if now - last >= 1.0:
                    last = now
                    frac = n / max(total, 1)
                    eta = (now - t0) / max(n, 1) * (total - n)
                    _progress(
                        f"正在解压文件… {n}/{total}（{frac * 100:.0f}%）",
                        0.22 + frac * 0.33)
                    _extract_state(files_done=n, files_total=total,
                                   seconds=int(now - t0), eta=int(eta))
    else:
        raise ParseError("不支持的压缩格式")


# ---------- 产物读取与校验 ----------

def _clean_products(work_dir: Path) -> None:
    """执行前清空工作目录里的旧产物。

    关键：AI 脚本会扫描目录下的 csv 文件当作「标签映射文件」。
    若上一次的产物（manifest.csv / dataset.csv）残留在目录里，
    脚本会把它误当成输入数据，导致划分被错误重分配。
    """
    for name in (MANIFEST_NAME, DATASET_NAME, META_NAME):
        p = work_dir / name
        try:
            if p.exists():
                p.unlink()
        except BaseException:  # noqa: BLE001  含 SystemExit（删除保护）
            pass


def _split_consistency(df: pd.DataFrame, work_dir: Path) -> dict:
    """校验 AI 产出的划分与源目录结构是否一致。

    数据集自带 train/val/test 目录时，AI 应当原样保留划分。
    若产物把大量样本重新分配到与源目录不符的划分，说明 AI 误判了结构，
    这时必须拒绝落库 —— 用错误划分训练出来的模型没有意义。

    返回 {"checked": bool, "consistent": bool, "mismatched": int, "total": int}
    """
    if "filepath" not in df.columns or "split" not in df.columns:
        return {"checked": False, "consistent": True, "mismatched": 0, "total": 0}

    rows = df[["filepath", "split"]].astype(str).values.tolist()
    total = len(rows)
    if not total:
        return {"checked": False, "consistent": True, "mismatched": 0, "total": 0}

    mismatched = 0
    for rel, sp in rows:
        parts = Path(rel).parts[:-1]           # 去掉文件名，只看目录段
        src_sp = None
        for seg in parts:
            c = canon_split(seg)
            if c:
                src_sp = c
                break
        if src_sp and src_sp != sp:
            mismatched += 1

    # 只有当源目录确实带划分信息、且 AI 判定与源目录大范围冲突时才判为不一致
    consistent = mismatched == 0 or (mismatched / total) < 0.05
    return {"checked": True, "consistent": consistent,
            "mismatched": mismatched, "total": total}


def _split_from_source_names(work_dir: Path) -> dict[str, int]:
    """扫描工作目录里的源表格文件，按文件名推断各划分的数据量。

    用于 AI 未把划分写进产物时的兜底：
    用户的 train.csv / val.csv / test.csv 本来就是划分好的，
    平台必须把这层信息接住，不能让样本全落进训练集。
    """
    counts: dict[str, int] = {}
    for p in work_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in (MANIFEST_NAME, DATASET_NAME, META_NAME):
            continue
        if p.suffix.lower() not in (".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl"):
            continue
        stem = p.stem
        segs = re.split(r"[_\-\.\s]+", stem)
        kind = next((canon_split(s) for s in segs if canon_split(s)), None)
        if not kind:
            continue
        try:
            if p.suffix.lower() in (".parquet", ".pq"):
                n = int(len(pd.read_parquet(p)))
            elif p.suffix.lower() == ".jsonl":
                n = int(len(pd.read_json(p, lines=True)))
            elif p.suffix.lower() == ".json":
                n = int(len(pd.read_json(p)))
            else:
                n = int(len(pd.read_csv(p, sep=None, engine="python")))
        except Exception:  # noqa: BLE001
            continue
        counts[kind] = counts.get(kind, 0) + n
    return counts


def _recover_split_by_source(df: pd.DataFrame, work_dir: Path) -> tuple[pd.DataFrame, dict]:
    """AI 产物没有有效划分时，用源文件名重建划分。

    场景：AI 把 train/val/test 三个 csv 合并成一张表，但忘了写 split 列，
    或写成了无法识别的值。此时按源文件重新分配，保证划分不丢。
    """
    src_counts = _split_from_source_names(work_dir)
    if not src_counts or not (src_counts.get("train", 0) or src_counts.get("val", 0)):
        return df, {}

    # 源文件里 train/val/test 都有 → 按出现顺序把行依次分配
    order: list[str] = []
    for p in sorted(work_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name in (MANIFEST_NAME, DATASET_NAME, META_NAME):
            continue
        if p.suffix.lower() not in (".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl"):
            continue
        stem = p.stem
        segs = re.split(r"[_\-\.\s]+", stem)
        kind = next((canon_split(s) for s in segs if canon_split(s)), None)
        if kind:
            order.append(kind)
    if len(set(order)) < 2:
        return df, {}

    # 按源文件顺序重新读取并拼接，同时打上 split
    frames: list[pd.DataFrame] = []
    for p in sorted(work_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name in (MANIFEST_NAME, DATASET_NAME, META_NAME):
            continue
        if p.suffix.lower() not in (".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl"):
            continue
        stem = p.stem
        segs = re.split(r"[_\-\.\s]+", stem)
        kind = next((canon_split(s) for s in segs if canon_split(s)), None)
        if not kind:
            continue
        try:
            if p.suffix.lower() in (".parquet", ".pq"):
                sub = pd.read_parquet(p)
            elif p.suffix.lower() == ".jsonl":
                sub = pd.read_json(p, lines=True)
            elif p.suffix.lower() == ".json":
                sub = pd.read_json(p)
            else:
                sub = pd.read_csv(p, sep=None, engine="python")
        except Exception:  # noqa: BLE001
            continue
        sub = sub.copy()
        sub["split"] = kind
        frames.append(sub)

    if not frames:
        return df, {}
    out = pd.concat(frames, ignore_index=True)
    counts = {k: int((out["split"] == k).sum()) for k in CANON_SPLITS}
    return out, counts


def _derive_split_from_filepath(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """以源路径为准推导划分。

    文件型数据集里，样本所在的目录名（train/val/test）就是最权威的划分依据 ——
    这是用户在原始数据里就已经定好的，AI 的标注只是复述。
    因此平台自己从 filepath 的目录段推导划分，覆盖 AI 可能出错的标注。

    返回 (新 DataFrame, 划分统计 或 None 表示无法推导)。
    """
    if "filepath" not in df.columns:
        return df, None

    derived: list[str | None] = []
    for rel in df["filepath"].astype(str):
        parts = Path(rel).parts[:-1]
        hit = None
        for seg in parts:
            c = canon_split(seg)
            if c:
                hit = c
                break
        derived.append(hit)

    known = [d for d in derived if d]
    if not known:
        return df, None
    # 只有当相当比例的样本都能从路径推出划分时才采用
    if len(known) / max(len(derived), 1) < 0.5:
        return df, None

    out = df.copy()
    # 只覆盖能从路径推出的行，其余保留 AI 的标注
    cur = out["split"].astype(str).str.strip().str.lower().map(canon_split) \
        if "split" in out.columns else pd.Series([None] * len(out), index=out.index)
    new_sp = []
    for ai_v, path_v in zip(cur.tolist(), derived):
        new_sp.append(path_v if path_v else (ai_v if ai_v else "train"))
    out["split"] = new_sp
    counts = {k: int((out["split"] == k).sum()) for k in CANON_SPLITS}
    return out, counts


_TABLE_EXTS = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl", ".xlsx", ".xls"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def _detect_work_kind(work_dir: Path) -> str:
    """按工作目录里**实际存在的文件形态**判断任务类型。

    不依赖 AI 在 meta.json 里的声明 —— AI 可能把表格任务误标成图像分类，
    或漏写 kind。平台自己看一眼目录里是表还是图，才靠得住。
    返回 "tabular" / "file_based" / "mixed" / "unknown"
    """
    has_table = False
    has_image = False
    try:
        for p in work_dir.iterdir():
            if not p.is_file():
                continue
            if p.name in (MANIFEST_NAME, DATASET_NAME, META_NAME):
                continue
            ext = p.suffix.lower()
            if ext in _TABLE_EXTS:
                has_table = True
            elif ext in _IMAGE_EXTS:
                has_image = True
            if has_table and has_image:
                break
    except OSError:
        pass
    if has_table and has_image:
        return "mixed"
    if has_table:
        return "tabular"
    if has_image:
        return "file_based"
    return "unknown"


def _products_usable(df: pd.DataFrame | None) -> bool:
    """判断一份产物是否真的可用。"""
    if df is None or df.empty:
        return False
    if "filepath" in df.columns and "label" in df.columns:
        # 文件型：类别至少要 2 个，否则分类训练无从谈起
        return int(df["label"].astype(str).nunique()) >= 2
    return True


def _read_products(work_dir: Path) -> tuple[pd.DataFrame, dict]:
    """读取 AI 脚本产出的 manifest.csv / dataset.csv 与 meta.json。"""
    meta: dict = {}
    meta_path = work_dir / META_NAME
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                meta = {}
        except Exception:  # noqa: BLE001
            meta = {}

    manifest_path = work_dir / MANIFEST_NAME
    dataset_path = work_dir / DATASET_NAME

    # AI 可能同时产出两份（如表格任务也顺手写了 manifest），
    # 也可能把任务类型标错。选择依据以「工作目录实际形态」为主、AI 声明为辅，
    # 并且两份都要过一遍可用性检查，避免读到一份坏产物就整单失败。
    kind = str((meta or {}).get("kind") or "").strip()
    work_kind = _detect_work_kind(work_dir)

    def _read_manifest() -> pd.DataFrame | None:
        if not (manifest_path.exists() and manifest_path.stat().st_size > 0):
            return None
        try:
            mdf = pd.read_csv(manifest_path)
        except Exception:  # noqa: BLE001
            return None
        if "label" not in mdf.columns or "filepath" not in mdf.columns:
            return None
        return mdf

    def _read_dataset() -> pd.DataFrame | None:
        if not (dataset_path.exists() and dataset_path.stat().st_size > 0):
            return None
        try:
            ddf = pd.read_csv(dataset_path)
        except Exception:  # noqa: BLE001
            return None
        if ddf.empty:
            return None
        return ddf

    mdf = _read_manifest()
    ddf = _read_dataset()

    # 表格型优先读 dataset.csv，文件型优先读 manifest.csv
    prefer_dataset = (
        work_kind == "tabular"
        or kind in ("tabular_classification", "tabular_regression", "text_classification")
    )
    if work_kind == "file_based" and kind != "text_classification":
        prefer_dataset = False

    order = ([ddf, mdf] if prefer_dataset else [mdf, ddf])
    for cand in order:
        if _products_usable(cand):
            return cand, meta

    # 产物不是分类/表格，但工作目录里存在带边界框的标注 —— 说明这是检测任务。
    # 平台自己判断，不依赖 AI 声明（AI 常把 YOLO 数据误判成分类）。
    det = _try_detection_manifest(work_dir)
    if det is not None:
        return det, meta
    # 两份都不可用：给出更具体的提示
    if mdf is not None and "label" in mdf.columns:
        raise ParseError(
            f"AI 解析产物中类别只有 {int(mdf['label'].astype(str).nunique())} 个，"
            f"无法用于分类训练（至少需要 2 个类别）")
    if ddf is None and mdf is None:
        raise ParseError(
            "AI 解析脚本执行完毕但未产出 manifest.csv 或 dataset.csv，请重试或调整数据集结构")
    raise ParseError("AI 解析脚本未产出有效的数据，请重试或调整数据集结构")


def _try_detection_manifest(work_dir: Path, progress=None) -> pd.DataFrame | None:
    """平台独立识别目标检测数据集。

    YOLO 数据长得非常明确：图片旁有「每行 class_id x y w h」的 .txt 标签。
    如果不识别出来，AI 一旦把它当成分类处理，多目标会被压成单标签、
    **边界框信息全部丢失** —— 用户以为在训检测器，实际训的是分类器。

    返回带 boxes 列的 manifest，或 None。
    """
    try:
        import detection
    except Exception:  # noqa: BLE001
        return None
    try:
        df = detection.build_manifest(work_dir, progress=progress)
    except Exception:  # noqa: BLE001
        df = None
    if df is None or df.empty:
        return None
    if not detection.is_detection_manifest(df):
        return None
    return df


def _pick_text_column(df: pd.DataFrame, target: str | None, split_like: set[str]) -> str | None:
    """挑选正文列。

    判据（比 infer_col_type 更可靠）：正文字符串「长」且「多样」。
    - sample_id 这类 ID 列：每个值都唯一 → 唯一值率高但长度短，应排除
    - text / content / body 这类正文列：长度明显更长且重复度高
    """
    best: tuple[float, str] | None = None
    for c in df.columns:
        if c == target:
            continue
        low = str(c).strip().lower()
        if low in split_like or low == "filepath" or low == "label":
            continue
        s = df[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        vals = s.dropna().astype(str)
        if vals.empty:
            continue
        avg_len = float(vals.str.len().mean())
        if avg_len < 8:
            continue
        # 唯一值率接近 1 且长度短 → 典型 ID 列，跳过
        uniq_ratio = float(vals.nunique()) / max(len(vals), 1)
        if uniq_ratio > 0.95 and avg_len < 30:
            continue
        if best is None or avg_len > best[0]:
            best = (avg_len, str(c))
    if best:
        return best[1]

    # 兜底：没有一列达到「长度 ≥ 8」的正文标准，但存在明显的文本型列时，
    # 仍应判为文本分类 —— 短标题（如新闻标题）同样需要语义模型，
    # 漏判会让整列被当成类别特征做 one-hot，直接把训练打崩。
    return _pick_text_column_loose(df, target, split_like)


def _pick_text_column_loose(df: pd.DataFrame, target: str | None,
                            split_like: set[str]) -> str | None:
    """放宽标准的正文列识别：允许短文本，但排除 ID/编码类列。

    判据：字符串列 + 平均长度 ≥ 4 + 唯一值率不高（重复出现的文本）
    或 列名本身带正文语义（text/content/body/title/desc/message 等）。
    """
    name_hint = ("text", "content", "body", "title", "desc", "description",
                 "message", "review", "comment", "article", "news", "post",
                 "summary", "abstract", "sentence", "query", "question", "answer")
    best: tuple[float, str] | None = None
    for c in df.columns:
        if c == target:
            continue
        low = str(c).strip().lower()
        if low in split_like or low == "filepath" or low == "label":
            continue
        s = df[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        vals = s.dropna().astype(str)
        if vals.empty:
            continue
        avg_len = float(vals.str.len().mean())
        if avg_len < 4:
            continue
        uniq_ratio = float(vals.nunique()) / max(len(vals), 1)
        named = any(h in low for h in name_hint)
        # ID 列：几乎全唯一且没有正文语义
        if uniq_ratio > 0.98 and not named:
            continue
        # 数值型编码（category 1/2/3 这类）不该当正文
        if uniq_ratio < 0.05 and not named:
            continue
        score = avg_len + (100 if named else 0)
        if best is None or score > best[0]:
            best = (score, str(c))
    return best[1] if best else None


def _resolve_task(df: pd.DataFrame, meta: dict, user_target: str | None,
                  user_task: str | None) -> tuple[str, str | None, str | None]:
    """确定任务类型、目标列、正文列。

    优先级：用户显式指定 > 平台独立校验 > AI meta.json 声明。
    AI 的声明只作参考 —— 同一次解析 AI 可能给出不同的判断，
    平台必须自己能兜住，否则任务类型与划分会随 AI 心情漂移。
    """
    meta = meta or {}
    kind = str(meta.get("kind") or "").strip()
    target = str(meta.get("target_col") or "").strip() or None
    text_col = str(meta.get("text_col") or "").strip() or None

    # 文件型数据集：目标列固定为 label
    if "filepath" in df.columns and "label" in df.columns:
        # 带 boxes 列说明是目标检测（边界框）而不是分类
        if "boxes" in df.columns and df["boxes"].astype(str).str.strip().ne("").any():
            meta["target_source"] = "file"
            return "object_detection", "label", None
        meta["target_source"] = "file"
        return "image_classification", "label", None

    # 划分标记列名集合：绝不能被当成目标列或特征列
    split_like = _SPLIT_MARKER_COLS

    if user_target and user_target in df.columns:
        target = user_target
        meta["target_source"] = "user"
    # AI 把划分列当成目标列 —— 纠正：换用真正的目标列
    if target and str(target).strip().lower() in split_like:
        target = None
    if target and target in df.columns and not meta.get("target_source"):
        meta["target_source"] = "ai"
    if not target or target not in df.columns:
        # AI 没给或给错：取最后一列非 split 列
        candidates = [c for c in df.columns
                      if str(c).strip().lower() not in split_like
                      and str(c).strip().lower() != "filepath"]
        if not candidates:
            raise ParseError("AI 解析产物中未找到可用的目标列")
        target = str(candidates[-1])
        # 兜底推断必须留痕：宽表里最后一列通常是特征而不是标签。
        # 画像门据此要求人工确认，避免把一列信号当成标签训练。
        # 但如果列名本身就叫 label/target/标签，那属于「规则可判」，
        # 不必让用户为一件显而易见的事再点一次确认。
        _low = target.strip().lower()
        _HINTS = ("label", "target", "class", "category", "diagnosis", "outcome",
                  "标签", "类别", "分类", "诊断", "结果")
        meta["target_source"] = "rule" if any(h in _low for h in _HINTS) else "fallback"

    task = user_task or ""
    if not task:
        task = kind if kind in (
            "image_classification", "text_classification",
            "tabular_classification", "tabular_regression") else ""
    if not task:
        task = analysis.detect_task(df, target)

    # 平台独立校验：若表里存在明显的正文字段，应判为文本分类。
    # （AI 可能判成 tabular_classification，导致正文被当成类别特征）
    # 用户显式指定任务时不做覆盖 —— 否则手动指定的信号/表格任务会被抢走。
    if task in ("tabular_classification", "text_classification"):
        platform_text = _pick_text_column(df, target, split_like)
        if platform_text and task != "text_classification" and not user_task:
            # 正文列存在但被判成普通表格分类 —— 以平台判断为准
            if text_col != platform_text:
                task = "text_classification"
        if task == "text_classification":
            if (not text_col or text_col not in df.columns
                    or str(text_col).strip().lower() in split_like
                    or text_col == target):
                text_col = platform_text
            if not text_col:
                raise ParseError("判定为文本分类但未找到正文列（text_col）")
    return task, target, text_col


_IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm")
_ANNO_SUFFIX = (".csv", ".tsv", ".json", ".jsonl", ".txt", ".xlsx", ".xml")


def _diagnose_archive(work_dir: Path) -> str | None:
    """本地结构诊断：只回答「这个包有没有最小可用结构」。

    返回 None 表示结构没明显问题（继续交给 AI / 平台识别器）；
    返回字符串表示确定的不可解析原因，直接报给用户，不浪费一次 AI 调用。

    判据只有一条：**图像必须能找到标签来源**——要么有标注文件，
    要么按类别分了目录。这条规则简单，但正好覆盖审计里
    「images_no_labels.zip 被送去大模型、最后报 401」的场景。
    """
    imgs: list[Path] = []
    anno = 0
    scanned = 0
    for p in work_dir.rglob("*"):
        if not p.is_file():
            continue
        scanned += 1
        if scanned > 20000:      # 大包只抽样判断结构，不为了诊断扫全量
            break
        low = p.name.lower()
        if low.endswith(_IMG_SUFFIX):
            imgs.append(p)
        elif low.endswith(_ANNO_SUFFIX) and p.parent == work_dir:
            anno += 1

    if not imgs:
        return None              # 不是图像数据集，交给后续流程

    if anno:
        return None              # 有顶层标注文件

    parents = {p.parent for p in imgs}
    if len(parents) >= 2:
        return None              # 按类别分了目录（ImageFolder 结构）

    top_dirs = {p.relative_to(work_dir).parts[0] for p in imgs
                if len(p.relative_to(work_dir).parts) > 1}
    if len(top_dirs) >= 2:
        return None              # 多个顶层目录，可能是 train/val 划分

    return ("压缩包里只有图像，找不到标签来源：既没有标注文件（CSV/JSON/TXT），"
            "也没有按类别分目录。请按「类别名/图片」的结构重新打包，"
            "或在包内附一份至少包含「文件名,标签」两列的标注 CSV。")


def _read_table_direct(path: Path) -> pd.DataFrame:
    """直接读取单文件表格（csv/tsv/xlsx/parquet），不经过 AI。

    稳健性优先：
    - 编码依次尝试 utf-8 / utf-8-sig / gbk —— 国内业务表 GBK 很常见，
      直接用 utf-8 读会整表报错；
    - csv/tsv 用 sep=None 让 pandas 嗅探分隔符，兼容分号、制表符；
    - 自动丢掉 pandas 生成的 Unnamed 索引列，避免它被当成特征。
    """
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        last: Exception | None = None
        # 顺序有讲究：先看 BOM（UTF-16 的 0xFF 0xFE 用 utf-8 会直接判死），
        # 再 utf-8 / gbk。gb18030 放最后，因为它几乎能解码任何字节，放前面会误判。
        for enc in ("utf-8-sig", "utf-8", "utf-16", "gbk", "gb18030"):
            try:
                # 先只读表头嗅探分隔符，再用 C 引擎整表读取。
                # 直接用 sep=None + engine="python" 读 600MB 宽表要十几分钟，
                # C 引擎只要几十秒。
                head = path.open("r", encoding=enc, errors="strict").read(65536)
                sep = ","
                try:
                    sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
                except Exception:  # noqa: BLE001
                    if "\t" in head.split("\n")[0]:
                        sep = "\t"
                df = pd.read_csv(path, sep=sep, engine="c",
                                 encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last = e
                continue
            except pd.errors.EmptyDataError as e:
                raise ParseError("表格文件是空的（没有任何行），请确认导出是否成功") from e
            except pd.errors.ParserError as e:
                raise ParseError(f"表格结构无法解析：{e}") from e
        else:
            raise ParseError(f"表格编码无法识别（已尝试 UTF-8 / UTF-16 / GB18030）：{last}")
    drop = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if drop:
        df = df.drop(columns=drop)
    return df


def _materialize(df: pd.DataFrame, meta: dict, did: str, ds_name: str, filename: str,
                 size_bytes: int, work_dir: Path, task: str, target: str | None,
                 text_col: str | None) -> dict:
    """把解析产物落库为数据集记录。

    图像任务：产物中的 filepath 指向 work_dir 内的实际文件，
    因此 image_root 直接指向解压后的数据根目录。
    """
    is_file_based = "filepath" in df.columns and "label" in df.columns
    meta = dict(meta or {})
    meta["filename"] = filename
    meta["parsed_by"] = "ai"

    def _attach_plan() -> None:
        """生成并挂载智能体建模方案（数据卡 + 方案卡）。

        解析一结束就能定下来，前端第 5 步据此展示「智能体打算怎么训」，
        用户无需等训练开始才知道用了什么模型。规则兜底保证 AI 不可用时不阻塞。
        """
        # 这里刻意只用规则基线、不调 AI：解析阶段每多一次大模型调用，
        # 用户就要多等十几秒。精细调优留到训练阶段（automl.decide）再做。
        try:
            import automl as _am
            hint = "image_classification" if is_file_based else (task or "tabular")
            card = _am.inspect(df, hint, meta)
            meta["datacard"] = card
            if hint == "text_classification":
                import automl_text as _at
                # text_col 由外层 _resolve_task 判定后传入，比 meta 里的更可靠
                tc = text_col or meta.get("text_col")
                sample = (df[tc].astype(str).tolist()[:200]
                          if tc and tc in df.columns else [])
                meta["ml_plan"] = _at.text_baseline_plan(card, sample)
            elif hint in ("tabular_classification", "tabular_regression"):
                meta["ml_plan"] = _am._tabular_plan(card, 1)
            elif hint == "signal_classification":
                meta["ml_plan"] = {
                    "model": "cnn1d", "width": 48, "epochs": 24, "lr": 1e-3,
                    "normalize": "per_record",
                    "reason": (f"1D 卷积网络 · {card.get('rows')} 条记录 · "
                               "逐条 z-score 归一，按患者分组切分"),
                }
            else:
                meta["ml_plan"] = _am.baseline_plan(card)
        except Exception:  # noqa: BLE001
            pass

    if is_file_based:
        is_det = task == "object_detection"
        # filepath 可能是绝对路径（平台检测识别器给出），
        # 统一转成相对于 image_root 的相对路径，下游拼路径才不会出错。
        _root0 = work_dir.resolve()
        _rel = []
        for _p in df["filepath"].astype(str):
            _pp = Path(_p)
            if _pp.is_absolute():
                try:
                    _rel.append(str(_pp.relative_to(_root0)).replace("\\", "/"))
                except ValueError:
                    _rel.append(str(_pp).replace("\\", "/"))
            else:
                _rel.append(_p.replace("\\", "/"))
        df = df.copy()
        df["filepath"] = _rel
        labels = df["label"].astype(str)
        if labels.nunique() < 2 and not is_det:
            raise ParseError("AI 解析结果中类别少于 2 个，无法用于分类训练")
        # 检测任务的类别来自所有框，不只是每张图的主标签
        if is_det and "boxes" in df.columns:
            all_names: list[str] = []
            for b in df["boxes"].astype(str):
                for item in b.split(";"):
                    item = item.strip()
                    if not item:
                        continue
                    nm = item.split(",")[0]
                    if nm and nm not in all_names:
                        all_names.append(nm)
            classes = all_names or sorted(labels.unique().tolist())
        else:
            classes = sorted(labels.unique().tolist())
        class_counts = {c: int((labels == c).sum()) for c in classes}

        # 校验 filepath 真实存在（AI 可能写了不存在的路径）
        root = work_dir.resolve()
        missing = 0
        for rel in df["filepath"].astype(str).head(200):
            p = Path(rel)
            if not p.is_absolute() and not (root / rel).exists():
                missing += 1
            elif p.is_absolute() and not p.exists():
                missing += 1
        if missing > 0:
            raise ParseError(
                f"AI 解析产物中有 {missing} 个样本路径在数据目录中不存在，解析结果不可用")

        path = db.STORAGE_DIR / f"{did}_v1.csv"
        df.to_csv(path, index=False)
        meta.update({
            "kind": ("object_detection" if is_det
                     else (meta.get("kind") or "image_manifest")),
            "image_root": str(root),
            "classes": classes,
            "class_counts": class_counts,
            "schema": analysis.build_schema(df, "label"),
            "stats": analysis.build_stats(df),
        })
        if is_det:
            meta["detection"] = True
            meta.pop("summary", None)
            meta["summary"] = (f"识别为目标检测数据集：{len(df)} 张标注图、"
                               f"{len(classes)} 个类别、"
                               f"共 {int(sum(len([x for x in str(b).split(';') if x.strip()]) for b in df['boxes']))} "
                               f"个边界框。")
        _attach_plan()
        db.insert_dataset({
            "id": did, "name": ds_name, "source": "upload",
            "task_hint": "object_detection" if is_det else "image_classification",
            "target_col": "label",
            "file_path": str(path), "version": 1,
            "rows": int(len(df)), "cols": len(df.columns),
            "size_bytes": size_bytes, "meta": meta,
        })
        return db.get_dataset(did)

    # 表格型
    meta.update({
        "kind": meta.get("kind") or "tabular",
        "schema": analysis.build_schema(df, target),
        "stats": analysis.build_stats(df),
    })
    if text_col:
        meta["text_col"] = text_col
    if task in ("tabular_classification", "text_classification"):
        vc = df[target].astype(str).value_counts()
        if 1 < len(vc) <= 100:
            ordered = sorted(str(v) for v in vc.index.tolist())
            meta["classes"] = ordered
            meta["class_counts"] = {c: int(vc.get(c, 0)) for c in ordered}
        else:
            meta["classes"] = None
            meta["class_counts"] = None
    elif task == "signal_classification":
        # 信号任务的类别来自诊断码，不是整串标签：
        # 多标签时 `MI|STTC` 是一个组合，不是第 3 个类别，必须拆开统计。
        import automl_signal as _asig
        _y, _classes, _ml = _asig.parse_labels(df[target])
        meta["classes"] = _classes
        meta["multilabel"] = bool(_ml)
        if _ml:
            cnt = {c: int(_y[:, i].sum()) for i, c in enumerate(_classes)}
        else:
            vc = pd.Series(_y).value_counts().to_dict()
            cnt = {c: int(vc.get(i, 0)) for i, c in enumerate(_classes)}
        meta["class_counts"] = cnt
    path = db.STORAGE_DIR / f"{did}_v1.csv"
    df.to_csv(path, index=False)
    _attach_plan()
    db.insert_dataset({
        "id": did, "name": ds_name, "source": "upload",
        "task_hint": task, "target_col": target,
        "file_path": str(path), "version": 1,
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "size_bytes": size_bytes, "meta": meta,
    })
    return db.get_dataset(did)


# ---------- 异步解析任务 ----------

# 解析任务状态表：task_id -> 状态字典
# 大数据集的解析可能长达几十分钟（AI 生成脚本 + 批量还原图像），
# 若同步执行会把整个后端占死 —— 连 /api/health 都会无响应。
# 因此上传接口立即返回任务 ID，解析在后台线程跑，前端轮询进度。
_PARSE_TASKS: dict[str, dict] = {}
_TASK_LOCK = threading.Lock()
_TASK_MAX = 200  # 最多保留的历史任务数

# 当前解析线程的任务 ID（供 parse_upload 内部更新进度用）
_CTX = threading.local()


def _set_task(task_id: str, **fields) -> None:
    with _TASK_LOCK:
        task = _PARSE_TASKS.setdefault(task_id, {"id": task_id})
        task.update(fields)


def get_task(task_id: str) -> dict | None:
    with _TASK_LOCK:
        return dict(_PARSE_TASKS.get(task_id) or {}) or None


def _progress(message: str, progress: float) -> None:
    """更新当前解析任务的进度（仅在任务上下文中有效）。"""
    tid = getattr(_CTX, "task_id", None)
    if tid:
        _set_task(tid, message=message, progress=progress)


def _extract_state(files_done: int = 0, files_total: int = 0,
                   seconds: int = 0, eta: int = 0, **extra) -> None:
    """记录解压阶段的详细统计，供前端展示真实进度。"""
    tid = getattr(_CTX, "task_id", None)
    if not tid:
        return
    _set_task(tid, stage="extract",
              files_done=int(files_done), files_total=int(files_total),
              elapsed=int(seconds), eta_seconds=int(eta), **extra)


def _set_task_safe(**fields) -> None:
    """在任务上下文中写入附加字段（无上下文时静默跳过）。"""
    tid = getattr(_CTX, "task_id", None)
    if tid:
        _set_task(tid, **fields)


def _prune_tasks() -> None:
    with _TASK_LOCK:
        if len(_PARSE_TASKS) <= _TASK_MAX:
            return
        # 按创建时间淘汰最旧的
        ordered = sorted(_PARSE_TASKS.values(), key=lambda t: t.get("created_at", 0))
        for old in ordered[: len(_PARSE_TASKS) - _TASK_MAX]:
            _PARSE_TASKS.pop(old["id"], None)


def start_parse_task(archive: Path, did: str, ds_name: str, filename: str,
                     size_bytes: int, user_target: str | None = None,
                     user_task: str | None = None) -> str:
    """启动后台解析任务，立即返回 task_id。

    任务状态：running → done / failed
    完成后可通过 get_task() 取结果。
    """
    task_id = f"pt_{uuid.uuid4().hex[:12]}"
    _set_task(task_id, status="running", created_at=time.time(),
               dataset_id=did, name=ds_name, message="正在分析数据集结构…",
               progress=0.05)

    def _worker() -> None:
        _CTX.task_id = task_id
        try:
            _progress("AI 正在分析数据集结构…", 0.2)
            record = parse_upload(archive, did, ds_name, filename, size_bytes,
                                  user_target=user_target, user_task=user_task)
            _set_task(task_id, status="done", progress=1.0,
                      message="解析完成", dataset=record,
                      report=record.get("_parse_report") or {})
        except ParseError as e:
            _set_task(task_id, status="failed", message=str(e), progress=1.0)
        except Exception as e:  # noqa: BLE001
            _set_task(task_id, status="failed",
                      message=f"解析异常：{e}", progress=1.0)
        finally:
            # 上传的临时文件在解析结束后清理（工作目录由 parse_upload 自己管）
            try:
                if Path(archive).exists():
                    Path(archive).unlink()
            except BaseException:  # noqa: BLE001
                pass
            _prune_tasks()

    threading.Thread(target=_worker, daemon=True, name=f"parse_{task_id}").start()
    return task_id


# ---------- 主流程 ----------

def parse_upload(archive: Path, did: str, ds_name: str, filename: str,
                 size_bytes: int, user_target: str | None = None,
                 user_task: str | None = None) -> dict:
    """上传文件的统一解析入口。

    流程：画像 → 指纹 → 记忆命中则复用脚本（免 AI）→ 未命中则 AI 生成 → 受限执行 → 落库。

    返回落库后的数据集记录，附加 _parse_report 字段说明本次走了哪条路径。
    """
    # 注意：这里不再强制要求已配置大模型。
    # 纯表格文件平台能直读（见下方 3.6），图片包命中记忆库时也不需要 AI。
    # 真正需要 AI 写脚本时再检查，否则「key 失效」会把整条离线链路一起拖死。

    t0 = time.time()
    work_dir = db.STORAGE_DIR / f"_work_{did}"
    if work_dir.exists():
        _rm_tree(work_dir)

    is_archive = archive.name.lower().endswith((".zip", ".tar.gz", ".tgz", ".tar"))

    try:
        # ---- 1. 结构画像（只读，不解压，GB 级包也是秒级）----
        if is_archive:
            profile = llm.profile_archive(archive)
        else:
            profile = llm.profile_table_file(archive)
        fingerprint = llm.fingerprint_of(profile)

        # ---- 2. 记忆库命中检查：同结构直接复用脚本，跳过 AI 调用 ----
        hit = recall.recall(profile)
        report: dict = {
            "fingerprint": fingerprint.get("signature", ""),
            "memory_hit": bool(hit),
            "memory_score": hit.get("score") if hit else 0.0,
            "memory_source": hit.get("learned_from") if hit else "",
            # ai_called：是否为了生成主解析脚本调用了 AI
            "ai_called": False,
            # ai_fix_calls：因执行失败或划分冲突触发的 AI 修正次数
            "ai_fix_calls": 0,
            "ai_reason": "",
            "structure": profile.get("structure_summary", ""),
            "entries": profile.get("entries", 0),
        }

        # ---- 3. 解压到工作目录（单文件直接复制进去）----
        t_extract = time.time()
        if is_archive:
            _safe_extract(archive, work_dir)
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, work_dir / archive.name)
        report["extract_seconds"] = round(time.time() - t_extract, 2)

        code: str | None = None
        if hit:
            code = hit.get("code") or ""
            report["memory_id"] = hit.get("id")

        # ---- 3.5 目标检测数据：平台独立识别，绕过 AI 脚本 ----
        # YOLO 数据若交给 AI 脚本，多目标会被压成单标签、边界框全部丢失。
        # 平台自己能可靠地认出「图片旁有 class_id x y w h 的 .txt」，
        # 所以直接走平台识别器，不浪费一次 AI 调用，也不会被记忆库里的
        # 旧分类脚本带偏。
        _progress("正在识别数据集结构（扫描标注文件）…", 0.58)
        _set_task_safe(stage="scan")
        _scan_state = {"n": 0, "t0": time.time()}

        def _det_progress(done: int, total: int, label: str) -> None:
            _scan_state["n"] = done
            frac = (done / total) if total else 0.0
            el = time.time() - _scan_state["t0"]
            eta = int(el / max(done, 1) * max(total - done, 0))
            _progress(f"{label}… {done}/{total}（{frac * 100:.0f}%）",
                      0.58 + frac * 0.17)
            _extract_state(files_done=done, files_total=total,
                           seconds=int(el), eta=eta)

        det_df = _try_detection_manifest(work_dir, progress=_det_progress)
        if det_df is not None:
            report["detection_direct"] = True
            report["ai_called"] = False
            report["ai_reason"] = "平台直接识别出目标检测标注结构，跳过 AI 解析"
            det_meta = {"kind": "object_detection", "detection": True,
                        "target_col": "label", "text_col": "",
                        "classes": [], "summary": ""}
            det_df, det_counts = normalize_split_column(det_df)
            try:
                task, target, text_col = _resolve_task(det_df, det_meta, None, None)
            except ParseError:
                task, target, text_col = "object_detection", "label", None
            ds = _materialize(det_df, det_meta, did, ds_name,
                              archive.name if is_archive else archive.name,
                              size_bytes, work_dir, task, target, text_col)
            report["split_counts"] = det_counts
            report["split_source"] = "platform_detection"
            report["seconds"] = round(time.time() - t0, 2)
            report["attempts"] = 0
            report["task"] = task
            report["target_col"] = target
            ds["_parse_report"] = report
            return ds

        # ---- 3.6 纯表格单文件：平台直读，不消耗 AI ----
        # 一张 CSV 交给大模型写脚本去读，本身就是绕远路：
        # 慢（600MB 表要等 AI 往返）、贵、而且 key 一失效整条链路就断
        # （Day 4 实测：PTB-XL 宽表因 provider 返回 401 完全无法入库）。
        # 平台自己读表是确定性的，也顺带把「无大模型可用」的场景救回来。
        if (not is_archive) and Path(filename).suffix.lower() in (
                ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pq"):
            _progress("正在直接读取表格文件…", 0.6)
            _set_task_safe(stage="read_table")
            t_read = time.time()
            try:
                # 注意用 archive.name：上传的临时文件名是 _upload_ds_xxx.csv，
                # 第 3 步复制进工作目录时用的也是这个名字，
                # 用 filename（原始文件名）会找不到文件而静默落回 AI 路径。
                df_direct = _read_table_direct(work_dir / archive.name)
            except Exception as e:  # noqa: BLE001
                report["direct_read_error"] = str(e)[:300]
                df_direct = None
            if df_direct is not None and len(df_direct) and len(df_direct.columns) >= 2:
                meta_direct = {"kind": "tabular"}
                df_direct, dcounts = normalize_split_column(df_direct)
                d_task, d_target, d_text = _resolve_task(
                    df_direct, meta_direct, user_target, user_task)
                report["direct_table"] = True
                report["ai_called"] = False
                report["ai_reason"] = "平台直读表格文件，未调用 AI"
                report["split_counts"] = dcounts
                report["split_source"] = "platform_table"
                report["read_seconds"] = round(time.time() - t_read, 2)
                report["attempts"] = 0
                report["task"] = d_task
                report["target_col"] = d_target
                _progress("正在入库…", 0.9)
                ds = _materialize(df_direct, meta_direct, did, ds_name,
                                  Path(filename).name, size_bytes, work_dir,
                                  d_task, d_target, d_text)
                report["seconds"] = round(time.time() - t0, 2)
                ds["_parse_report"] = report
                return ds

        # ---- 3.7 本地结构诊断：先判断"有没有最小可用结构"，再决定要不要叫 AI ----
        # 审计发现：空包、只有图没有标签的包，也会被送去大模型，
        # 结果 provider 一 401，用户看到的是"大模型调用失败"，完全查不到病因。
        # 这类判定不需要任何语义理解，本地就能给结论。
        if is_archive:
            local_issue = _diagnose_archive(work_dir)
            if local_issue:
                raise ParseError(local_issue)

        # ---- 4. 执行：像人一样「跑一次、看结果、再改」，而不是盲写盲跑 ----
        # 每轮都把真实的运行输出与产物体检结果反馈给 AI，
        # 让它像在对话里那样根据实际结果迭代修正。
        attempts: list[dict] = []
        code = code or ""
        if not code:
            if not db.get_default_provider():
                raise ParseError("该数据集需要 AI 编写解析脚本，但尚未配置默认大模型服务，"
                                 "请先在「大模型配置」中新增并设为默认")
            report["ai_called"] = True
            _progress("AI 正在阅读数据结构并编写解析脚本…", 0.25)
            code = llm.generate_parser_code(profile)
            report["ai_reason"] = "结构指纹未命中记忆库，调用 AI 生成解析脚本"

        timeout = _timeout_for(profile, size_bytes)
        feedback = ""
        MAX_ROUNDS = 5

        for round_no in range(1, MAX_ROUNDS + 1):
            _clean_products(work_dir)
            # 先报「开始第 N 轮」，再执行。
            # 之前是执行完才更新进度，而沙箱执行可能长达数分钟，
            # 用户看到的就是「一直卡在上一轮」——以为死了。
            _progress(f"第 {round_no}/{MAX_ROUNDS} 轮：正在执行解析脚本…",
                      0.3 + 0.5 * (round_no - 1) / MAX_ROUNDS)
            _set_task_safe(round_no=round_no, max_rounds=MAX_ROUNDS,
                           stage="ai_exec", files_done=0, files_total=0)
            exec_res = _run_in_sandbox(code, work_dir, timeout,
                                       heartbeat=lambda: _progress(
                                           f"第 {round_no}/{MAX_ROUNDS} 轮："
                                           f"脚本仍在运行（最长可等 {timeout} 秒）…",
                                           0.3 + 0.5 * (round_no - 1) / MAX_ROUNDS))
            attempts.append({"round": round_no, "returncode": exec_res["returncode"]})
            _progress(f"第 {round_no}/{MAX_ROUNDS} 轮：正在检查解析结果…",
                      0.3 + 0.5 * round_no / MAX_ROUNDS)

            # 本轮体检：产物 + 划分一致性
            feedback = _inspect_products(work_dir, exec_res)
            ok = False
            try:
                cand, cand_meta = _read_products(work_dir)
                cand, cand_counts = normalize_split_column(cand)
                cand2, path_counts2 = _derive_split_from_filepath(cand)
                if path_counts2 is not None:
                    cand, cand_counts = normalize_split_column(cand2)
                else:
                    cons2 = _split_consistency(cand, work_dir)
                    if cons2["checked"] and not cons2["consistent"]:
                        feedback += (
                            f"\n\n[平台校验] 有 {cons2['mismatched']}/{cons2['total']} 个样本的 "
                            f"split 与它们所在的源目录不一致 —— 源目录里的 train/val/test "
                            f"就是标准划分，必须原样保留，不要自己随机重切。")
                    else:
                        # 没有可推导的划分时，尝试按源文件名兜底
                        if (cand_counts.get("val", 0) == 0 and cand_counts.get("test", 0) == 0
                                and "filepath" not in cand.columns):
                            rec, rc = _recover_split_by_source(cand, work_dir)
                            if rc.get("val", 0) or rc.get("test", 0):
                                cand, cand_counts = normalize_split_column(rec)
                if len(cand) == 0:
                    feedback += "\n\n[平台校验] 产物没有任何数据行。"
                elif "label" in cand.columns and int(cand["label"].astype(str).nunique()) < 2:
                    feedback += (
                        f"\n\n[平台校验] 类别只有 "
                        f"{int(cand['label'].astype(str).nunique())} 个，无法做分类训练。"
                        f"请确认标签是否读对了。")
                elif "filepath" in cand.columns:
                    root2 = work_dir.resolve()
                    miss2 = [str(r) for r in cand["filepath"].astype(str).head(50)
                             if not (root2 / str(r)).exists()]
                    if miss2:
                        feedback += (
                            f"\n\n[平台校验] 有 {len(miss2)}/50 个 filepath 在数据目录里不存在，"
                            f"例：{miss2[:3]}。路径必须相对于当前数据根目录。")
                    else:
                        ok = True
                else:
                    ok = True
            except ParseError as e:
                feedback += f"\n\n[平台校验] {e}"

            # 数据没有自带验证集是**正常情况** —— 训练时会按比例从训练集里切。
            # 不要求 AI 为了凑齐 train/val/test 而随机重切，那会破坏用户的原始划分。
            if ok and (cand_counts.get("val", 0) == 0 and cand_counts.get("test", 0) == 0):
                report["no_native_val_test"] = True

            if ok:
                break

            if round_no < MAX_ROUNDS:
                report["ai_fix_calls"] = report.get("ai_fix_calls", 0) + 1
                report["ai_reason"] = (report["ai_reason"] + "；" if report.get("ai_reason") else "") \
                    + f"第 {round_no} 轮产物未通过校验，AI 正在根据实际结果修正"
                code = llm.improve_parser_code(profile, code, feedback, round_no)

        else:
            raise ParseError(
                f"AI 解析经过 {MAX_ROUNDS} 轮迭代仍未产出合格结果。"
                f"最后一次的问题：{(feedback or '未知')[-500:]}")

        # ---- 5. 读取最终产物 + 锁定划分 ----
        # 划分以「源路径 / 源文件名」为准：这是用户在原始数据里就定好的，
        # AI 的标注只是复述。平台自己能定的部分绝不交给 AI 猜。
        df, meta = _read_products(work_dir)
        df, path_counts = _derive_split_from_filepath(df)
        if path_counts is not None:
            report["split_source"] = "source_path"
        else:
            report["split_source"] = "ai_product"
            cons = _split_consistency(df, work_dir)
            report["split_consistency"] = cons
            # AI 没写出有效划分时，用源文件名重建（train.csv / val.csv / test.csv）
            df2, cand_counts = normalize_split_column(df)
            if cand_counts.get("val", 0) == 0 and cand_counts.get("test", 0) == 0:
                rec_df, rec_counts = _recover_split_by_source(df, work_dir)
                if rec_counts.get("val", 0) or rec_counts.get("test", 0):
                    df = rec_df
                    report["split_recovered_by_source"] = True

        df, split_counts = normalize_split_column(df)
        meta["split_counts"] = split_counts
        meta.setdefault("split_counts", split_counts)
        meta["ai_notes"] = str(meta.get("summary") or "")[:400]

        task, target, text_col = _resolve_task(df, meta, user_target, user_task)
        meta["split_counts"] = split_counts


        # ---- 7. 落库 ----
        record = _materialize(df, meta, did, ds_name, filename, size_bytes,
                              work_dir, task, target, text_col)

        # ---- 8. 写入记忆库 ----
        try:
            recall.remember(
                profile, code,
                kind=task,
                summary=meta.get("ai_notes", "") or meta.get("summary", ""),
                dataset_name=ds_name,
                verdict=meta,
            )
        except Exception:  # noqa: BLE001  记忆失败不影响主流程
            pass

        report["seconds"] = round(time.time() - t0, 2)
        report["attempts"] = len(attempts)
        report["split_counts"] = split_counts
        report["task"] = task
        report["target_col"] = target
        record["_parse_report"] = report
        return record

    finally:
        # 工作目录被 image_root 引用时不能删；表格型已在 _materialize 落盘，可清理
        try:
            ds = db.get_dataset(did)
            keep = bool(ds and (ds.get("meta") or {}).get("image_root"))
            if not keep:
                _rm_tree(work_dir)
        except Exception:  # noqa: BLE001
            pass


def _timeout_for(profile: dict, size_bytes: int) -> int:
    """按数据体量自适应超时：小数据 120s，大包最多 30 分钟。"""
    entries = int(profile.get("entries") or 0)
    by_count = 120 + min(entries // 200, 1500)
    by_size = 120 + int(size_bytes / (8 * 1024 * 1024))
    return int(min(max(by_count, by_size), 1800))
