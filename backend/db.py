"""SQLite 存储：数据集元数据与清洗历史。"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "nn_studio.db"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  source TEXT NOT NULL,                -- upload | builtin
  task_hint TEXT,                      -- tabular_classification | tabular_regression | ...
  target_col TEXT,
  file_path TEXT NOT NULL,             -- 当前生效版本 CSV 路径
  version INTEGER NOT NULL DEFAULT 1,
  rows INTEGER NOT NULL,
  cols INTEGER NOT NULL,
  size_bytes INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'ready',-- ready | processing | failed
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}' -- schema / stats 等扩展信息
);

CREATE TABLE IF NOT EXISTS cleaning_runs (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version INTEGER NOT NULL,            -- 清洗后产生的版本号
  recipe_json TEXT NOT NULL,           -- 清洗配方
  report_json TEXT NOT NULL,           -- 清洗报告
  created_at TEXT NOT NULL,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  account TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT '成员',
  org TEXT,
  password_hash TEXT NOT NULL,
  salt TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS train_jobs (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'queued', -- queued | running | done | failed
  progress REAL NOT NULL DEFAULT 0,
  epoch INTEGER NOT NULL DEFAULT 0,
  total_epochs INTEGER NOT NULL DEFAULT 0,
  logs_json TEXT NOT NULL DEFAULT '[]',
  metrics_json TEXT NOT NULL DEFAULT '{}',
  model_path TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  task TEXT,                       -- classification | regression
  task_hint TEXT,                  -- tabular_classification | image_classification | ...
  architecture TEXT,               -- mlp | cnn | emb_mlp
  dataset_id TEXT,
  train_job_id TEXT,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  stage TEXT NOT NULL DEFAULT 'candidate', -- candidate | staging | production | archived
  model_path TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS services (
  id TEXT PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,       -- URL slug：/api/svc/{name}/predict
  model_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'running', -- running | stopped
  requests_total INTEGER NOT NULL DEFAULT 0,
  created_by TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE TABLE IF NOT EXISTS providers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  provider_type TEXT NOT NULL DEFAULT 'openai_compat',
  base_url TEXT NOT NULL,
  api_key TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL,
  is_default INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'unknown',  -- unknown | active | auth_failed | unreachable
  latency_ms REAL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- 上传指纹：同一份内容只允许进入一次，避免重复上传把存储撑爆（曾出现同一份 DR 存 4 份）
CREATE TABLE IF NOT EXISTS upload_index (
  sha256 TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  filename TEXT NOT NULL DEFAULT '',
  size_bytes INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

-- 解析失败暂存会话：AI 分析出候选方案后等待用户选择
CREATE TABLE IF NOT EXISTS pending_parses (
  id TEXT PRIMARY KEY,
  staged_path TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  name TEXT NOT NULL,
  filename TEXT NOT NULL,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  target_col TEXT,
  task_hint TEXT,
  error TEXT NOT NULL DEFAULT '',
  suggestions TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
"""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def now_iso() -> str:
    """与 now() 同格式，语义化别名（供外部模块按意图调用）。"""
    return now()


def _commit_with_retry(c: sqlite3.Connection, tries: int = 6) -> None:
    """提交时遇到写锁就退避重试。

    训练线程逐 epoch 写库、前端轮询读库、外部脚本读库可能同时发生，
    SQLite 的 busy_timeout 一旦到点就直接抛 "database is locked"。
    这个异常如果落到上层的「自动建模失败 → 退回基础训练」分支，
    会把跑了 45 分钟的迁移学习结果整个丢掉（实测发生过）。
    写库这种事不能因为一次锁竞争就放弃，退避重试几秒即可。
    """
    for i in range(tries):
        try:
            c.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or i == tries - 1:
                raise
            time.sleep(min(2 ** i, 10))


@contextmanager
def conn():
    # timeout/busy_timeout：训练线程逐 epoch 写库与前端轮询并发时，避免 database is locked
    c = sqlite3.connect(DB_PATH, timeout=60.0)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA busy_timeout=60000")
        yield c
        _commit_with_retry(c)
    finally:
        c.close()


def init_db() -> None:
    # 日志模式选 DELETE 而不是 WAL：
    # WAL 依赖同目录下的 -wal / -shm 共享内存文件，在部分磁盘卷上
    # （本机 E: 实测）会在并发写入时抛 "attempt to write a readonly database"，
    # 直接把训练任务打断。DELETE 模式配合 60s busy_timeout 与提交重试，
    # 单进程 + 多线程轮询的场景足够稳，且不依赖共享内存。
    with conn() as c:
        try:
            c.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.OperationalError:
            pass
    with conn() as c:
        c.executescript(SCHEMA)


# ---------- datasets ----------

def insert_dataset(d: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO datasets
               (id,name,source,task_hint,target_col,file_path,version,rows,cols,
                size_bytes,status,created_at,updated_at,meta_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d["id"], d["name"], d["source"], d.get("task_hint"), d.get("target_col"),
                d["file_path"], d.get("version", 1), d["rows"], d["cols"],
                d["size_bytes"], d.get("status", "ready"),
                d.get("created_at") or now(), d.get("updated_at") or now(),
                json.dumps(d.get("meta", {}), ensure_ascii=False),
            ),
        )


def get_dataset(did: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    return _row_to_dict(row) if row else None


def list_datasets() -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


def update_dataset(did: str, **fields) -> None:
    if not fields:
        return
    if "meta" in fields:
        fields["meta_json"] = json.dumps(fields.pop("meta"), ensure_ascii=False)
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE datasets SET {cols} WHERE id=?", (*fields.values(), did))


def delete_dataset(did: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM cleaning_runs WHERE dataset_id=?", (did,))
        c.execute("DELETE FROM datasets WHERE id=?", (did,))


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["meta"] = json.loads(d.pop("meta_json") or "{}")
    return d


# ---------- users ----------

def insert_user(u: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO users (id,account,name,role,org,password_hash,salt,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (u["id"], u["account"], u["name"], u.get("role", "成员"), u.get("org"),
             u["password_hash"], u["salt"], now()),
        )


def get_user(uid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def get_user_by_account(account: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM users WHERE account=?", (account,)).fetchone()
    return dict(row) if row else None


# ---------- cleaning_runs ----------

def insert_cleaning_run(r: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO cleaning_runs (id,dataset_id,version,recipe_json,report_json,created_at)
               VALUES (?,?,?,?,?,?)""",
            (r["id"], r["dataset_id"], r["version"],
             json.dumps(r["recipe"], ensure_ascii=False),
             json.dumps(r["report"], ensure_ascii=False),
             r.get("created_at") or now()),
        )


def list_cleaning_runs(did: str) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM cleaning_runs WHERE dataset_id=? ORDER BY created_at DESC", (did,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["recipe"] = json.loads(d.pop("recipe_json"))
        d["report"] = json.loads(d.pop("report_json"))
        out.append(d)
    return out


# ---------- train_jobs ----------

def insert_job(j: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO train_jobs
               (id,dataset_id,config_json,status,progress,epoch,total_epochs,
                logs_json,metrics_json,model_path,error,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                j["id"], j["dataset_id"], json.dumps(j.get("config", {}), ensure_ascii=False),
                j.get("status", "queued"), j.get("progress", 0), j.get("epoch", 0),
                j.get("total_epochs", 0), json.dumps(j.get("logs", []), ensure_ascii=False),
                json.dumps(j.get("metrics", {}), ensure_ascii=False), j.get("model_path"),
                j.get("error"), j.get("created_at") or now(), j.get("updated_at") or now(),
            ),
        )


def _job_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["config"] = json.loads(d.pop("config_json") or "{}")
    d["logs"] = json.loads(d.pop("logs_json") or "[]")
    d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
    return d


def get_job(jid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM train_jobs WHERE id=?", (jid,)).fetchone()
    return _job_to_dict(row) if row else None


def list_jobs() -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """SELECT j.*, d.name AS dataset_name, d.task_hint AS task_hint
               FROM train_jobs j LEFT JOIN datasets d ON d.id = j.dataset_id
               ORDER BY j.created_at DESC"""
        ).fetchall()
    return [_job_to_dict(r) for r in rows]


def update_job(jid: str, **fields) -> None:
    if not fields:
        return
    if "config" in fields:
        fields["config_json"] = json.dumps(fields.pop("config"), ensure_ascii=False)
    if "logs" in fields:
        fields["logs_json"] = json.dumps(fields.pop("logs"), ensure_ascii=False)
    if "metrics" in fields:
        fields["metrics_json"] = json.dumps(fields.pop("metrics"), ensure_ascii=False)
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE train_jobs SET {cols} WHERE id=?", (*fields.values(), jid))


# ---------- models ----------

def insert_model(m: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO models
               (id,name,version,task,task_hint,architecture,dataset_id,train_job_id,
                metrics_json,stage,model_path,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["id"], m["name"], m.get("version", 1), m.get("task"), m.get("task_hint"),
                m.get("architecture"), m.get("dataset_id"), m.get("train_job_id"),
                json.dumps(m.get("metrics", {}), ensure_ascii=False), m.get("stage", "candidate"),
                m.get("model_path"), m.get("created_at") or now(), m.get("updated_at") or now(),
            ),
        )


def _model_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
    return d


def get_model(mid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    return _model_to_dict(row) if row else None


def list_models(include_archived: bool = False) -> list[dict]:
    """默认不含已归档模型。

    归档不是删除：权重和指标都留着，只是不再出现在模型列表和网关目录里。
    这样修复前的旧模型（比如 PTBXL 那版 20 个组合标签的）不会在演示时被误选，
    但历史记录仍然可查。
    """
    sql = """SELECT m.*, d.name AS dataset_name
             FROM models m LEFT JOIN datasets d ON d.id = m.dataset_id"""
    if not include_archived:
        # archived / rejected 都退出目录：前者是历史版本，后者是没通过发布门。
        # 它们仍可凭 id 查到（审计需要），但不该出现在"可选模型"里。
        sql += " WHERE COALESCE(m.stage, 'candidate') NOT IN ('archived', 'rejected')"
    sql += " ORDER BY m.created_at DESC"
    with conn() as c:
        rows = c.execute(sql).fetchall()
    return [_model_to_dict(r) for r in rows]


def update_model(mid: str, **fields) -> None:
    if not fields:
        return
    if "metrics" in fields:
        fields["metrics_json"] = json.dumps(fields.pop("metrics"), ensure_ascii=False)
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE models SET {cols} WHERE id=?", (*fields.values(), mid))


def delete_model(mid: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM models WHERE id=?", (mid,))


# ---------- services ----------

def insert_service(s: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO services (id,name,model_id,status,requests_total,created_by,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (s["id"], s["name"], s["model_id"], s.get("status", "running"),
             s.get("requests_total", 0), s.get("created_by"),
             s.get("created_at") or now(), s.get("updated_at") or now()),
        )


def get_service(sid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM services WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


def get_service_by_name(name: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM services WHERE name=?", (name,)).fetchone()
    return dict(row) if row else None


def list_services() -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """SELECT s.*, m.name AS model_name, m.version AS model_version, m.task_hint AS task_hint
               FROM services s LEFT JOIN models m ON m.id = s.model_id
               ORDER BY s.created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_service(sid: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE services SET {cols} WHERE id=?", (*fields.values(), sid))


def delete_service(sid: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM services WHERE id=?", (sid,))


# ---------- providers ----------

def insert_provider(p: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO providers (id,name,provider_type,base_url,api_key,model,
                                      is_default,status,latency_ms,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (p["id"], p["name"], p.get("provider_type", "openai_compat"), p["base_url"],
             p.get("api_key", ""), p["model"], 1 if p.get("is_default") else 0,
             p.get("status", "unknown"), p.get("latency_ms"),
             p.get("created_at") or now(), p.get("updated_at") or now()),
        )


def get_provider(pid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM providers WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def list_providers() -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM providers ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]


def update_provider(pid: str, **fields) -> None:
    if not fields:
        return
    if "is_default" in fields:
        fields["is_default"] = 1 if fields["is_default"] else 0
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE providers SET {cols} WHERE id=?", (*fields.values(), pid))


def delete_provider(pid: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM providers WHERE id=?", (pid,))


def clear_default_provider() -> None:
    with conn() as c:
        c.execute("UPDATE providers SET is_default=0")


def get_default_provider() -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT * FROM providers WHERE is_default=1 AND api_key<>'' LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


# ---------- 解析失败暂存会话 ----------

def insert_pending_parse(p: dict) -> None:
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO pending_parses"
            " (id, staged_path, dataset_id, name, filename, size_bytes, target_col,"
            "  task_hint, error, suggestions, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (p["id"], p["staged_path"], p["dataset_id"], p["name"], p["filename"],
             int(p.get("size_bytes") or 0), p.get("target_col"), p.get("task_hint"),
             p.get("error") or "", json.dumps(p.get("suggestions") or {}, ensure_ascii=False),
             p.get("created_at") or now()),
        )


def get_pending_parse(pid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM pending_parses WHERE id=?", (pid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["suggestions"] = json.loads(d.get("suggestions") or "{}")
    except json.JSONDecodeError:
        d["suggestions"] = {}
    return d


def delete_pending_parse(pid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM pending_parses WHERE id=?", (pid,)).fetchone()
        c.execute("DELETE FROM pending_parses WHERE id=?", (pid,))
    return dict(row) if row else None


# ---------- 上传指纹 ----------

def find_upload_by_hash(sha256: str) -> dict | None:
    """同一份内容是否已经进过平台。命中则直接复用，不再重复落盘。"""
    with conn() as c:
        row = c.execute("SELECT * FROM upload_index WHERE sha256=?", (sha256,)).fetchone()
    return dict(row) if row else None


def record_upload_hash(sha256: str, dataset_id: str, filename: str, size_bytes: int) -> None:
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO upload_index (sha256,dataset_id,filename,size_bytes,created_at)"
            " VALUES (?,?,?,?,?)",
            (sha256, dataset_id, filename, size_bytes, now()))

