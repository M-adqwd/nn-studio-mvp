"""上传链路体检：构造典型坏输入，看现有接口如何失败。

目的不是证明"有 bug"，而是把失败模式固化成可复现的证据，
新 SOP 的错误码与预检规则必须逐条覆盖这里暴露出来的情况。

用法：
  python scripts/audit_upload.py            # 生成样本并逐个上传
  python scripts/audit_upload.py --keep     # 保留生成的样本目录
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
HDR = {}
WORK = Path(__file__).resolve().parent.parent / "_testdata" / "badcases"


def login() -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"account": "admin", "password": "Admin@123"}, timeout=10)
    r.raise_for_status()
    return r.json()["token"]


def build_cases() -> list[Path]:
    WORK.mkdir(parents=True, exist_ok=True)
    cases: list[Path] = []

    (WORK / "empty.csv").write_bytes(b"")
    cases.append(WORK / "empty.csv")

    (WORK / "header_only.csv").write_text("a,b,label\n", encoding="utf-8")
    cases.append(WORK / "header_only.csv")

    rows = "\n".join(f"{i},{i * 2},same" for i in range(50))
    (WORK / "single_class.csv").write_text(f"x,y,label\n{rows}\n", encoding="utf-8")
    cases.append(WORK / "single_class.csv")

    # 目标列被直接泄漏成特征（label 同时以一列形式出现）
    leak = "\n".join(f"{i},c{i % 2},c{i % 2}" for i in range(60))
    (WORK / "target_leak.csv").write_text(f"id,label,label_copy\n{leak}\n", encoding="utf-8")
    cases.append(WORK / "target_leak.csv")

    # UTF-16 + 分号分隔，中文表头
    body = "编号;年龄;诊断\n" + "\n".join(f"{i};{20 + i % 50};{'有病' if i % 2 else '正常'}" for i in range(40))
    (WORK / "utf16_semicolon.csv").write_bytes(body.encode("utf-16"))
    cases.append(WORK / "utf16_semicolon.csv")

    # 伪装成 zip 的文本
    (WORK / "fake.zip").write_bytes(b"this is definitely not a zip file")
    cases.append(WORK / "fake.zip")

    # 路径穿越
    zp = WORK / "traversal.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../../evil.txt", "pwned")
        z.writestr("data/a.csv", "x,label\n1,a\n2,b\n")
    cases.append(zp)

    # 空 zip
    zp = WORK / "empty.zip"
    with zipfile.ZipFile(zp, "w"):
        pass
    cases.append(zp)

    # 一堆图但没有标签目录结构
    zp = WORK / "images_no_labels.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for i in range(5):
            z.writestr(f"img_{i}.jpg", b"\xff\xd8\xff\xdb" + b"0" * 64)
    cases.append(zp)

    # 嵌套压缩包
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("a.csv", "x,label\n1,a\n2,b\n")
    zp = WORK / "nested.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("inner.zip", inner.getvalue())
    cases.append(zp)

    # 不支持的后缀
    (WORK / "archive.rar").write_bytes(b"Rar!\x1a\x07\x00")
    cases.append(WORK / "archive.rar")

    # 超宽表：12000 列信号，无目标列
    cols = ",".join(f"s{i}" for i in range(300))
    wide = cols + "\n" + "\n".join(",".join(str((i * j) % 7) for j in range(300)) for i in range(50))
    (WORK / "wide_no_target.csv").write_text(wide, encoding="utf-8")
    cases.append(WORK / "wide_no_target.csv")

    return cases


def upload(token: str, path: Path) -> dict:
    """上传并等到解析终态，返回可读结论。"""
    t0 = time.time()
    with open(path, "rb") as fh:
        r = requests.post(f"{BASE}/api/datasets/upload",
                          headers={"Authorization": f"Bearer {token}"},
                          files={"file": (path.name, fh, "application/octet-stream")},
                          timeout=120)
    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            body = {"detail": r.text[:200]}
        err = body.get("error") or {}
        return {"case": path.name, "http": r.status_code, "result": "REJECTED",
                "code": err.get("code") or "-",
                "detail": (err.get("message") or body.get("detail") or "")[:160],
                "hint": (err.get("hint") or "")[:120],
                "sec": round(time.time() - t0, 1)}

    task_id = r.json()["task_id"]
    deadline = time.time() + 180
    last = {}
    while time.time() < deadline:
        s = requests.get(f"{BASE}/api/datasets/parse-status/{task_id}",
                         headers={"Authorization": f"Bearer {token}"}, timeout=30).json()
        last = s
        if s.get("status") in ("done", "failed", "error"):
            break
        time.sleep(2)
    status = last.get("status")
    did = last.get("dataset_id")

    # 解析成功后立刻查画像门：这才是「能不能训练」的结论
    gate = {}
    if status == "done" and did:
        try:
            gate = requests.get(f"{BASE}/api/datasets/{did}/profile",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=60).json()
        except Exception:  # noqa: BLE001
            gate = {}
    g = (gate.get("contract") or {}).get("gate") or {}
    codes = [c for c in (g.get("codes") or []) if c.startswith("E")]
    return {
        "case": path.name,
        "http": 200,
        "result": "PARSED" if status == "done" else f"PARSE_{status}",
        "code": (last.get("error_code") or (codes[0] if codes else ("OK" if status == "done" else "-"))),
        "detail": (last.get("error") or last.get("message") or "")[:160],
        "dataset_status": gate.get("status"),
        "gate": g.get("status"),
        "gate_codes": codes,
        "task": task_id,
        "dataset_id": did,
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    token = login()
    cases = build_cases()
    out = []
    for c in cases:
        try:
            row = upload(token, c)
        except Exception as e:  # noqa: BLE001
            row = {"case": c.name, "http": "-", "result": "EXCEPTION",
                   "detail": f"{type(e).__name__}: {e}"[:180], "sec": 0}
        out.append(row)
        print(f"{row['case']:<24} {str(row['http']):<5} {row['result']:<14} "
              f"{row.get('code') or '-':<7} {str(row.get('gate') or '-'):<10} "
              f"{row['detail'][:70]}")

    (WORK / "audit_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细写入 {WORK / 'audit_result.json'}")
    if not args.keep:
        pass  # 样本保留，便于失败复现；如需清理手动删除


if __name__ == "__main__":
    main()
