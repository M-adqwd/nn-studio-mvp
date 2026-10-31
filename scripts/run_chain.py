"""平台全链路驱动脚本：上传 → AI 解析 → 训练 → 模型登记 → 推理。

用法：
  python scripts/run_chain.py --csv <path> --name "名称" --target label \
      --out experiments/day0.json

设计目的：把「验证平台能否承接某类真实任务」固化成可重复执行的脚本，
而不是靠手点界面。每天的结果落到 experiments/ 下，作为对客户的实测证据。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"


def login() -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"account": "admin", "password": "Admin@123"}, timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def upload(tok: str, path: Path, name: str, target: str | None, hint: str | None) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    data = {"name": name}
    if target:
        data["target_col"] = target
    if hint:
        data["task_hint"] = hint
    t0 = time.time()
    suffix = path.suffix.lower()
    mime = {".zip": "application/zip", ".csv": "text/csv", ".tsv": "text/tab-separated-values",
            ".json": "application/json", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".parquet": "application/octet-stream"}.get(suffix, "application/octet-stream")
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/datasets/upload", headers=h, data=data,
                          files={"file": (path.name, f, mime)}, timeout=3600)
    r.raise_for_status()
    out = r.json()
    out["upload_seconds"] = round(time.time() - t0, 1)
    return out


def wait_parse(tok: str, task_id: str, limit: int = 3600) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    t0 = time.time()
    last = None
    while time.time() - t0 < limit:
        r = requests.get(f"{BASE}/api/datasets/parse-status/{task_id}", headers=h, timeout=60)
        r.raise_for_status()
        j = r.json()
        msg = (j.get("message") or "")[:60]
        if msg != last:
            print(f"    [{int(time.time()-t0)}s] {j.get('status')} · {msg}", flush=True)
            last = msg
        if j.get("status") in ("done", "failed"):
            j["parse_seconds"] = round(time.time() - t0, 1)
            return j
        time.sleep(3)
    raise TimeoutError("解析超时")


def train(tok: str, did: str, epochs: int, auto: bool, extra: dict | None = None) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    body = {"dataset_id": did, "task": "classification", "epochs": epochs,
            "batch_size": 64, "auto": auto, "val_split": 0.2, "transfer": True}
    body.update(extra or {})
    r = requests.post(f"{BASE}/api/train", headers=h, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def wait_train(tok: str, jid: str, limit: int = 5400) -> dict:
    """轮询训练任务。

    轮询期间偶发 5xx 不代表训练失败（多半是 SQLite 写锁或瞬时负载），
    直接 raise 会把整轮实验的结果全丢掉——所以这里对 5xx 与网络错误做退避重试，
    只有连续多次拿不到状态才判定失败。
    """
    h = {"Authorization": f"Bearer {tok}"}
    t0 = time.time()
    last_pct = -1
    errs = 0
    while time.time() - t0 < limit:
        try:
            r = requests.get(f"{BASE}/api/train/{jid}", headers=h, timeout=60)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} 服务端错误")
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
        if pct != last_pct and (pct % 10 == 0 or pct > 95):
            m = j.get("metrics") or {}
            print(f"    [{int(time.time()-t0)}s] {pct}% · {m.get('stage') or ''}", flush=True)
            last_pct = pct
        if j.get("status") in ("done", "failed"):
            j["train_seconds"] = round(time.time() - t0, 1)
            return j
        time.sleep(2)
    raise TimeoutError("训练超时")


def register(tok: str, jid: str, name: str) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{BASE}/api/models", headers=h, json={"train_job_id": jid, "name": name}, timeout=120)
    r.raise_for_status()
    return r.json()


def predict_tabular(tok: str, mid: str, rows: list[dict]) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{BASE}/api/models/{mid}/predict", headers=h, json={"rows": rows}, timeout=120)
    r.raise_for_status()
    return r.json()


def predict_images(tok: str, mid: str, paths: list[Path]) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    files = [("file", (p.name, open(p, "rb"), "image/jpeg")) for p in paths]
    try:
        r = requests.post(f"{BASE}/api/models/{mid}/predict", headers=h, files=files, timeout=180)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--hint", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--predict-csv", default=None, help="用于推理验证的 CSV（默认取输入前 5 行）")
    ap.add_argument("--predict-images", default=None, help="逗号分隔的图像文件，走 multipart")
    a = ap.parse_args()

    path = Path(a.file)
    tok = login()
    print(f"[1/6] 登录成功")

    up = upload(tok, path, a.name, a.target, a.hint)
    print(f"[2/6] 上传完成 {up['upload_seconds']}s → dataset_id={up['dataset_id']}")

    pj = wait_parse(tok, up["task_id"])
    print(f"[3/6] 解析 {pj['status']} · {pj.get('parse_seconds')}s")
    if pj["status"] != "done":
        print(json.dumps(pj, ensure_ascii=False, indent=2)[:3000])
        return 1
    ds = pj.get("dataset") or {}
    rep = pj.get("report") or {}
    print(f"      结构判定: task={ds.get('task_hint')} target={ds.get('target_col')} rows={ds.get('rows')}")
    print(f"      解析来源: {rep.get('source') or rep.get('mode') or '-'}")

    jid = train(tok, ds.get("id") or up["dataset_id"], a.epochs, a.auto)["id"]
    print(f"[4/6] 训练任务 {jid}")
    job = wait_train(tok, jid)
    m = job.get("metrics") or {}
    print(f"[5/6] 训练 {job['status']} · {job.get('train_seconds')}s"
          f" · best_val={(job.get('best_val') or {})} · test={m.get('test_metrics')}")
    if job["status"] != "done":
        print("      错误:", job.get("error"))

    result = {"upload": up, "parse": {k: v for k, v in pj.items() if k != "dataset"},
              "dataset": {k: v for k, v in ds.items() if k != "schema"}, "train": job}

    mid = None
    if job["status"] == "done":
        try:
            mdl = register(tok, jid, a.name)
            mid = mdl["id"]
            print(f"[6/6] 已登记模型 {mdl['name']} v{mdl['version']} ({mid})")
        except Exception as e:  # noqa: BLE001
            print("      模型登记失败:", e)

        # 推理验证
        try:
            if a.predict_images:
                paths = [Path(p) for p in a.predict_images.split(",") if p.strip()]
                pr = predict_images(tok, mid, paths)
                print("      图像推理:", json.dumps(pr.get("predictions", [])[:5], ensure_ascii=False))
                result["predict"] = pr
            else:
                import pandas as pd
                src = pd.read_csv(a.predict_csv or path)
                feat_cols = [c for c in src.columns if c != (ds.get("target_col") or "label")]
                rows = src[feat_cols].head(5).to_dict("records")
                truth = src[ds.get("target_col") or "label"].head(5).tolist()
                pr = predict_tabular(tok, mid, rows)
                preds = pr.get("predictions", [])[:5]
                print("      推理:", [(p.get("label"), p.get("confidence")) for p in preds],
                      "真值:", truth)
                result["predict"] = {"predictions": preds, "truth": truth}
        except Exception as e:  # noqa: BLE001
            print("      推理失败:", e)
            result["predict_error"] = str(e)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
