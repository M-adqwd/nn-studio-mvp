"""Day 6：训练平台 ↔ 阅片平台 真联调。

做的是打通，不是画图：把 ModelForge 训出来的眼底分级模型，通过通用 AI 网关
接进 MIIS 的病例，最终产出一份可追溯的 **draft 报告**（带模型名/版本/概率引用）。

链路：
  MIIS 建病例 → POST /api/v1/ai/classify（multipart 眼底图）
    → MIIS 适配器调 ModelForge /api/gateway/infer
      → 网关统一信封（label/概率/结论文本）
    → MIIS 落 AiJob(done) + draft Report（refs: ai_job）
  → 回读报告，核对 state=draft、refs 完整性、结论文本

用法：
  python scripts/e2e_integration.py --image _data/fundus/dr_split/test/3_severe/xxx.png \
      --model 眼底DR五级-DAY2
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import requests

MIIS = "http://127.0.0.1:8100"
FORGE = "http://127.0.0.1:8000"
HDR = {"X-User-Id": "u-attending", "X-User-Roles": "attending,admin"}


def idem(tag: str) -> dict:
    """写操作必须带幂等键（契约要求），每次联调用新键避免命中旧响应。"""
    import uuid
    return {"Idempotency-Key": f"e2e-{tag}-{uuid.uuid4().hex[:8]}"}


def pick_image(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"没找到图像：{pattern}")
    return Path(hits[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="眼底图像路径或 glob（心电模式可省略）")
    ap.add_argument("--model", default="眼底DR五级-DAY2")
    ap.add_argument("--label", default="", help="该图真实分级（仅用于展示对照）")
    ap.add_argument("--ecg", default="", help="改为心电模式：传入 CSV，取第一行做波形判读")
    a = ap.parse_args()

    if not a.image and not a.ecg:
        raise SystemExit("需要 --image 或 --ecg")
    img = pick_image(a.image) if a.image else Path("(ecg)")
    print(f"[1/5] 检查两侧服务")
    f = requests.get(f"{FORGE}/api/health", timeout=10).json()
    print("      ModelForge:", f.get("ok"))
    cases = requests.get(f"{MIIS}/api/v1/cases", headers=HDR, timeout=10).json()
    print(f"      MIIS 病例数: {cases.get('total')}")

    print(f"[2/5] 在 MIIS 建病例（眼底）")
    is_ecg = bool(a.ecg)
    r = requests.post(f"{MIIS}/api/v1/cases", headers={**HDR, **idem("case")}, timeout=20, json={
        "patient_ref": "MRN:E2E-ECG-001" if is_ecg else "MRN:E2E-DR-001",
        "studies": ["1.2.840.10008.5.1.4.1.1.30.1.9002" if is_ecg
                    else "1.2.840.10008.5.1.4.1.1.77.1.4.1.9001"],
        "source": "pacs",
        "modality": "ECG" if is_ecg else "OP",
        "title": "心电判读联调（ModelForge）" if is_ecg else "眼底 DR 分级联调（ModelForge）",
    })
    r.raise_for_status()
    case = r.json()
    case_id = case["case_id"]
    print(f"      case_id={case_id}")

    print(f"[3/5] 调 MIIS /ai/classify → ModelForge 模型 {a.model}")
    if a.ecg:
        # 心电：走 JSON rows（波形不是图像，必须走另一条入参形态）
        import csv as _csv
        with open(a.ecg, newline="", encoding="utf-8") as fh:
            row = next(_csv.DictReader(fh))
        row.pop("label", None)
        row.pop("patient", None)
        r = requests.post(
            f"{MIIS}/api/v1/ai/classify",
            headers={**HDR, **idem("cls")},
            json={"case_id": case_id, "model": a.model, "rows": [row]},
            timeout=180,
        )
    else:
        with img.open("rb") as fh:
            r = requests.post(
                f"{MIIS}/api/v1/ai/classify",
                headers={**HDR, **idem("cls")},
                data={"case_id": case_id, "model": a.model,
                      "study_instance_uid": "1.2.840.10008.5.1.4.1.1.77.1.4.1.9001"},
                files={"file": (img.name, fh, "image/png")},
                timeout=180,
            )
    if r.status_code >= 400:
        print("      失败:", r.status_code, r.text[:400])
        return 1
    out = r.json()
    print(f"      job_id={out['job_id']} report_id={out['report_id']} state={out['state']}")
    print(f"      模型: {out['model']}")
    print(f"      结论: {out['findings']}")
    print(f"      印象: {out['impression']}")
    if a.label:
        print(f"      真实分级（对照）: {a.label}")

    print(f"[4/5] 回读报告，核对证据链")
    r = requests.get(f"{MIIS}/api/v1/reports/{out['report_id']}", headers=HDR, timeout=20)
    if r.status_code >= 400:
        # 有些实现只提供按病例列表，退一步从这里找
        r = requests.get(f"{MIIS}/api/v1/cases/{case_id}/reports", headers=HDR, timeout=20)
    r.raise_for_status()
    rep = r.json()
    if isinstance(rep, dict) and "items" in rep:
        rep = next(x for x in rep["items"] if x.get("report_id") == out["report_id"])
    refs = rep.get("refs") or []
    print(f"      state={rep.get('state')} refs={refs}")
    ok_state = rep.get("state") == "draft"
    ok_ref = any(x.get("kind") == "ai_job" and x.get("uid") == out["job_id"] for x in refs)
    print(f"      草稿态：{ok_state} · 引用 AI 任务：{ok_ref}")

    print(f"[5/5] 核对 AI 任务留痕")
    r = requests.get(f"{MIIS}/api/v1/ai/jobs/{out['job_id']}", headers=HDR, timeout=20)
    if r.status_code < 400:
        jb = r.json()
        print(f"      job={jb['job_id']} status={jb['status']} model={jb['model']} v{jb['model_version']}")
        st = jb.get("stats") or {}
        print(f"      top={st.get('top')} disclaimer={'有' if st.get('disclaimer') else '无'}")

    ok = ok_state and ok_ref
    print("\n结论：", "打通成功（报告为 draft，等医师复核）" if ok else "未完全打通，需检查")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
