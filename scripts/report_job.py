"""把训练任务的结果整理成可读摘要。

平台结果散在三处：datasets（数据画像）、train_jobs（指标与日志）、models（登记产物）。
写实验报告时要反复拼这些字段，集中到一个脚本里，保证每天的报告口径一致。

用法：
  python scripts/report_job.py              # 最近一次任务
  python scripts/report_job.py job_xxx      # 指定任务
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "backend" / "nn_studio.db"


def load(job_id: str | None) -> tuple[dict, dict]:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    if job_id in (None, "", "latest"):
        row = c.execute("SELECT * FROM train_jobs ORDER BY created_at DESC LIMIT 1").fetchone()
    else:
        row = c.execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise SystemExit(f"找不到任务 {job_id}")
    job = dict(row)
    ds = c.execute("SELECT * FROM datasets WHERE id=?", (job.get("dataset_id"),)).fetchone()
    return job, dict(ds) if ds else {}


def metrics_of(job: dict) -> dict:
    try:
        return json.loads(job.get("metrics_json") or "{}")
    except Exception:  # noqa: BLE001
        return {}


def block(title: str, ev: dict) -> None:
    if not ev:
        return
    keys = [k for k in ("acc", "test_acc", "macro_f1", "qwk", "macro_auc",
                        "rmse", "test_rmse", "r2", "samples", "test_samples")
            if ev.get(k) is not None]
    print(f"### {title}")
    print("  " + " · ".join(f"{k}={ev[k]}" for k in keys))
    for row in (ev.get("per_class") or []):
        print("  {class}: P={precision} R={recall} F1={f1} n={support}".format(**row))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", nargs="?", default="latest")
    a = ap.parse_args()

    job, ds = load(a.job_id)
    m = metrics_of(job)

    print(f"## 任务 {job['id']}")
    print(f"- 数据集：{ds.get('name')} · {ds.get('task_hint')} · "
          f"{ds.get('rows')} 行 × {ds.get('cols')} 列")
    print(f"- 状态：{job.get('status')} · 进度 {round(float(job.get('progress') or 0) * 100, 1)}%")
    print(f"- 方案：{m.get('arch') or m.get('model') or '-'} · "
          f"mode={m.get('mode') or '-'} · 轮次数 {len(m.get('rounds') or [])}")
    if m.get("best_val") is not None:
        print(f"- 最佳验证分：{m.get('best_val')}")
    si = m.get("split_info") or {}
    if si:
        print(f"- 划分：{si.get('source')} · {si.get('note') or ''}")

    block("验证集", m.get("val_metrics") or {})
    block("测试集", m.get("test_metrics") or {})

    if job.get("status") == "failed":
        print("### 失败原因")
        print("  " + str(job.get("error"))[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
