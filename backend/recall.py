"""解析记忆库：记住每个数据集结构对应的「已验证可用的 AI 解析脚本」。

流程：
1. 上传文件 → 计算结构指纹（fingerprint.py）
2. 在记忆库里找相似度最高的记录
3. 命中且相似度 ≥ 阈值 → 直接复用该脚本，跳过 AI 调用（秒级完成）
4. 未命中 → 调 AI 生成脚本 → 执行验证成功 → 写入记忆库

记忆库记录的是「结构 → 脚本」的映射，与具体数据集名无关，
因此同一个供应商/同一采集流程产出的第 N 个数据集可以零 AI 成本直接解析。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import fingerprint as fp

DB_PATH = Path(__file__).resolve().parent / "parse_memory.db"

# 相似度阈值：达到此值即认为结构相同，直接复用已验证脚本
SIMILARITY_THRESHOLD = 0.75

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parse_memory (
  id TEXT PRIMARY KEY,
  signature TEXT NOT NULL,
  shape TEXT NOT NULL,
  ext_mix TEXT NOT NULL,
  layout_sample TEXT NOT NULL,
  code TEXT NOT NULL,
  kind TEXT,
  summary TEXT,
  dataset_name TEXT,
  verdict_json TEXT NOT NULL DEFAULT '{}',
  use_count INTEGER NOT NULL DEFAULT 1,
  fail_count INTEGER NOT NULL DEFAULT 0,
  last_ok_at TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parse_memory_sig ON parse_memory(signature);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10.0)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=10000")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        # 兼容早期版本：补齐缺失列
        cols = {r["name"] for r in c.execute("PRAGMA table_info(parse_memory)")}
        if "verdict_json" not in cols:
            c.execute("ALTER TABLE parse_memory ADD COLUMN verdict_json TEXT NOT NULL DEFAULT '{}'")


def _row_to_fp(row: sqlite3.Row) -> dict:
    return {
        "signature": row["signature"],
        "shape": json.loads(row["shape"]),
        "ext_mix": json.loads(row["ext_mix"]),
        "layout_sample": json.loads(row["layout_sample"]),
    }


def remember(profile: dict, code: str, *, kind: str = "", summary: str = "",
             dataset_name: str = "", verdict: dict | None = None) -> str:
    """记住一份「结构 → 解析脚本」的映射。返回记录 id。"""
    init_db()
    f = fp.dataset_fingerprint(profile)
    mid = f"pm_{uuid.uuid4().hex[:12]}"
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO parse_memory"
            " (id, signature, shape, ext_mix, layout_sample, code, kind, summary,"
            "  dataset_name, verdict_json, use_count, fail_count, last_ok_at, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,1,0,?,?)",
            (mid, f["signature"], json.dumps(f["shape"]), json.dumps(f["ext_mix"]),
             json.dumps(f["layout_sample"]), code, kind, summary, dataset_name,
             json.dumps(verdict or {}, ensure_ascii=False),
             _now(), _now()),
        )
    return mid


def _column_part(sig: str) -> str:
    """从指纹签名里取出列结构段（以 C 开头，形如 Cid,age#12）。"""
    for part in (sig or "").split("|"):
        if part.startswith("C"):
            return part
    return ""


def _columns_compatible(a: dict, b: dict) -> bool:
    """表格类数据集的列结构必须一致才允许复用脚本。

    只看目录骨架和文件类型时，任意两张 CSV 的相似度都是 1.0
    （都是一层、都是 table），于是「孕妇健康数据集」的解析脚本会被套到
    「心电心搏数据集」上——task 和 target 都可能是错的。
    非表格数据（图片目录等）没有列结构，仍按形状匹配。
    """
    ca, cb = _column_part(a.get("signature", "")), _column_part(b.get("signature", ""))
    if not ca and not cb:
        return True
    return bool(ca) and ca == cb


def recall(profile: dict, threshold: float = SIMILARITY_THRESHOLD) -> dict | None:
    """在记忆库里查找结构相似的已验证脚本。

    优先精确命中 signature（完全同构），否则退化为相似度匹配。
    """
    init_db()
    f = fp.dataset_fingerprint(profile)
    with _conn() as c:
        rows = c.execute("SELECT * FROM parse_memory").fetchall()

    # 1) 精确命中
    for r in rows:
        if r["signature"] == f["signature"]:
            return _to_hit(r, 1.0)

    # 2) 相似度匹配（表格类必须先过列结构校验）
    best, best_score = None, 0.0
    for r in rows:
        row_fp = _row_to_fp(r)
        if not _columns_compatible(f, row_fp):
            continue
        score = fp.similarity(f, row_fp)
        if score > best_score:
            best, best_score = r, score
    if best is None or best_score < threshold:
        return None
    return _to_hit(best, round(best_score, 4))


def _to_hit(row: sqlite3.Row, score: float) -> dict:
    try:
        verdict = json.loads(row["verdict_json"] or "{}")
    except Exception:  # noqa: BLE001
        verdict = {}
    return {
        "id": row["id"],
        "score": score,
        "code": row["code"],
        "kind": row["kind"],
        "summary": row["summary"],
        "dataset_name": row["dataset_name"],
        "use_count": row["use_count"],
        "verdict": verdict,
        "learned_from": row["dataset_name"] or row["summary"] or "历史数据集",
    }


def mark_result(memory_id: str, ok: bool) -> None:
    """记录一次复用结果：成功累加使用次数，失败累加失败计数。"""
    init_db()
    with _conn() as c:
        if ok:
            c.execute("UPDATE parse_memory SET use_count=use_count+1, last_ok_at=? WHERE id=?",
                      (_now(), memory_id))
        else:
            c.execute("UPDATE parse_memory SET fail_count=fail_count+1 WHERE id=?", (memory_id,))


def stats() -> dict:
    """记忆库概览：模式数、复用次数、明细。"""
    init_db()
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM parse_memory").fetchone()[0]
        total_use = c.execute("SELECT COALESCE(SUM(use_count),0) FROM parse_memory").fetchone()[0]
        rows = c.execute(
            "SELECT id, dataset_name, kind, summary, signature, use_count, fail_count, created_at"
            " FROM parse_memory ORDER BY use_count DESC, created_at DESC").fetchall()
    return {
        "patterns": n,
        "total_reuse": total_use,
        "threshold": SIMILARITY_THRESHOLD,
        "items": [dict(r) for r in rows],
    }


def forget(memory_id: str) -> bool:
    """删除一条记忆（复用过失败时可人工清除）。"""
    init_db()
    with _conn() as c:
        cur = c.execute("DELETE FROM parse_memory WHERE id=?", (memory_id,))
        return cur.rowcount > 0
