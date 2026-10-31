"""在已入库的数据集上直接起训练任务（跳过重复上传）。

为什么要这个脚本：
一条 600MB 的宽表上传 + 解析要两三分钟，而调一次训练超参不该重复付这个成本。
实验迭代（改路由、改指标、改切分）时，直接用 dataset_id 起任务即可。

用法：
  python scripts/retrain.py --dataset ds_38423c076a --epochs 20 --out experiments/day4_ptbxl.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"


def login() -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"account": "admin", "password": "Admin@123"}, timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def wait_train(tok: str, jid: str, limit: int = 14400) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    t0 = time.time()
    last = -1
    errs = 0
    while time.time() - t0 < limit:
        try:
            r = requests.get(f"{BASE}/api/train/{jid}", headers=h, timeout=60)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            j = r.json()
            errs = 0
        except (requests.RequestException, ValueError) as e:
            errs += 1
            print(f"    [轮询异常 {errs}/10] {e}", flush=True)
            if errs >= 10:
                raise
            time.sleep(min(5 * errs, 30))
            continue
        pct = int(float(j.get("progress") or 0) * 100)
        m = j.get("metrics") or {}
        if pct != last and (pct % 5 == 0 or pct > 95):
            print(f"    [{int(time.time()-t0)}s] {pct}% · {m.get('stage') or ''}", flush=True)
            last = pct
        if j.get("status") in ("done", "failed"):
            j["seconds"] = round(time.time() - t0, 1)
            return j
        time.sleep(2)
    raise TimeoutError("训练超时")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = login()
    body: dict = {"dataset_id": a.dataset, "transfer": True}
    if a.epochs:
        body["epochs"] = a.epochs
    if a.auto:
        body["auto"] = True
    r = requests.post(f"{BASE}/api/train", headers={"Authorization": f"Bearer {tok}"},
                      json=body, timeout=120)
    r.raise_for_status()
    job = r.json()
    print(f"[1/2] 任务已创建 {job['id']} · config={job.get('config')}")
    job = wait_train(tok, job["id"])
    print(f"[2/2] 训练 {job['status']} · {job.get('seconds')}s")
    m = job.get("metrics") or {}
    print("   best_val:", json.dumps(m.get("best_val"), ensure_ascii=False)[:300])
    print("   test    :", json.dumps(m.get("test_metrics"), ensure_ascii=False)[:300])
    if job.get("error"):
        print("   error:", str(job["error"])[:400])
    if a.out:
        Path(a.out).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        print("   已写入", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
