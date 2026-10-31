"""G8 回归门：SOP v2 的门禁规则必须始终成立。

设计成**离线纯函数测试**，不连服务、不写库、不产生垃圾数据集：
门禁逻辑本身是纯判定，就应当能用这种方式被钉住。
任何一次 SOP 改动都要先过这里，再谈上线。

用法：
  cd backend && ../.venv/Scripts/python.exe ../scripts/sop_regression.py
"""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import preflight  # noqa: E402
import spec  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {name}")
    else:
        FAILS.append(f"{name} {detail}")
        print(f"  ✗ {name} {detail}")


def codes(rep) -> list[str]:
    return [i.code for i in rep.issues]


def build_fixtures(root: Path) -> dict[str, Path]:
    f: dict[str, Path] = {}
    f["empty"] = root / "empty.csv"; f["empty"].write_bytes(b"")
    f["header"] = root / "header.csv"; f["header"].write_text("a,b,label\n", encoding="utf-8")
    f["fakezip"] = root / "fake.zip"; f["fakezip"].write_bytes(b"not a zip")
    f["rar"] = root / "x.rar"; f["rar"].write_bytes(b"Rar!\x1a\x07\x00")
    f["utf16"] = root / "u16.csv"
    f["utf16"].write_bytes(("编号;年龄;诊断\n" + "\n".join(
        f"{i};{i % 40};{'A' if i % 2 else 'B'}" for i in range(20))).encode("utf-16"))

    z = root / "empty.zip"
    with zipfile.ZipFile(z, "w"):
        pass
    f["emptyzip"] = z

    z = root / "nested.zip"
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as iz:
        iz.writestr("a.csv", "x,label\n1,a\n2,b\n")
    with zipfile.ZipFile(z, "w") as oz:
        oz.writestr("inner.zip", inner.getvalue())
    f["nested"] = z

    z = root / "traversal.zip"
    with zipfile.ZipFile(z, "w") as oz:
        oz.writestr("../../evil.txt", "x")
    f["traversal"] = z

    z = root / "images.zip"
    with zipfile.ZipFile(z, "w") as oz:
        for i in range(3):
            oz.writestr(f"img{i}.jpg", b"\xff\xd8\xff\xdb" + b"0" * 32)
    f["images"] = z

    f["good"] = root / "good.csv"
    f["good"].write_text("a,b,label\n" + "\n".join(f"{i},{i * 2},{'X' if i % 2 else 'Y'}"
                                                   for i in range(150)), encoding="utf-8")
    return f


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f = build_fixtures(root)

        print("\n[G0] 接收预检")
        check("空文件 → E1003",
              "E1003" in codes(preflight.inspect_upload(f["empty"], "empty.csv")))
        check("只有表头 → E1003",
              "E1003" in codes(preflight.inspect_upload(f["header"], "header.csv")))
        check("假 zip → E1002",
              "E1002" in codes(preflight.inspect_upload(f["fakezip"], "fake.zip")))
        check("rar → E1001",
              "E1001" in codes(preflight.inspect_upload(f["rar"], "x.rar")))
        check("空 zip → E1007",
              "E1007" in codes(preflight.inspect_upload(f["emptyzip"], "empty.zip")))
        check("嵌套 zip → E1008",
              "E1008" in codes(preflight.inspect_upload(f["nested"], "nested.zip")))
        check("路径穿越 → E1006",
              "E1006" in codes(preflight.inspect_upload(f["traversal"], "traversal.zip")))
        check("UTF-16 表格放行",
              preflight.inspect_upload(f["utf16"], "u16.csv").ok)
        check("正常 CSV 放行",
              preflight.inspect_upload(f["good"], "good.csv").ok)

        print("\n[G0.5] 压缩包结构诊断（不依赖大模型）")
        import ai_pipeline
        with tempfile.TemporaryDirectory() as wd:
            with zipfile.ZipFile(f["images"]) as z:
                z.extractall(wd)
            msg = ai_pipeline._diagnose_archive(Path(wd))
            check("有图无标签 → 本地诊断出原因", bool(msg), str(msg)[:40])

        print("\n[G1] 数据集画像门")
        base = {"rows": 2000, "cols": 10, "classes": ["A", "B"],
                "class_counts": {"A": 1200, "B": 800}, "missing_pct": 0.01,
                "duplicate_rows": 0, "constant_cols": [], "leak_suspects": [],
                "target_source": "user"}
        ok = spec.gate_dataset("tabular_classification", dict(base), target="label")
        check("正常表格通过", ok.ok, ok.summary())

        bad = dict(base, classes=["A"], class_counts={"A": 2000})
        check("单类别 → E3003",
              "E3003" in codes(spec.gate_dataset("tabular_classification", bad, "label")))

        leak = dict(base, leak_suspects=["label_copy"])
        check("标签泄漏 → E3004",
              "E3004" in codes(spec.gate_dataset("tabular_classification", leak, "label")))

        wide = dict(base, cols=12000, target_source="fallback")
        check("宽表兜底目标 → E3001",
              "E3001" in codes(spec.gate_dataset("tabular_classification", wide, "s2999")))

        few = dict(base, rows=30)
        check("样本不足 → E3002",
              "E3002" in codes(spec.gate_dataset("tabular_classification", few, "label")))

        img_bad = {"rows": 300, "cols": 2, "classes": ["A", "B"], "decode_fail_pct": 0.2,
                   "target_source": "file", "class_counts": {"A": 150, "B": 150}}
        check("图像损坏率过高 → E3002",
              "E3002" in codes(spec.gate_dataset("image_classification", img_bad, "label")))

        txt_short = {"rows": 500, "cols": 2, "classes": ["A", "B"], "text_len_mean": 3,
                     "target_source": "ai", "class_counts": {"A": 250, "B": 250}}
        check("文本过短 → E3002",
              "E3002" in codes(spec.gate_dataset("text_classification", txt_short, "label")))

        print("\n[G3] 训练准入")
        check("不支持的任务 → E5001",
              "E5001" in codes(spec.gate_training("video_classification")))
        check("needs_fix 数据 → E5002",
              "E5002" in codes(spec.gate_training(
                  "tabular_classification", "needs_fix",
                  {"codes": ["E3003"]})))
        check("未开启降级 → E5003",
              "E5003" in codes(spec.gate_training(
                  "tabular_classification", "ready", degraded=True)))
        check("显式降级放行",
              spec.gate_training("tabular_classification", "ready",
                                 degraded=True, allow_degraded=True).ok)

        print("\n[G4] 发布门")
        good_m = {"test_metrics": {"macro_f1": 0.72, "qwk": 0.8,
                                   "per_class": [{"support": 10}, {"support": 10}]}}
        check("指标达标可发布",
              spec.gate_release("tabular_classification", good_m, has_test=True).ok)
        low = {"test_metrics": {"macro_f1": 0.41}}
        check("指标不达标 → E6002",
              "E6002" in codes(spec.gate_release("tabular_classification", low, True)))
        check("无测试集 → E6003",
              "E6003" in codes(spec.gate_release("tabular_classification", {}, False)))
        check("实验能力不可发布 → E6004",
              "E6004" in codes(spec.gate_release(
                  "object_detection", {"test_metrics": {"mAP": 0.9}}, True)))

        print("\n[G8] 真实数据集不得误伤")
        import db
        import preflight as pf
        expect_pass = {"眼底DR五级-DAY2", "Chest X-Ray 肺炎筛查集"}
        seen = set()
        for ds in db.list_datasets():
            if ds["name"] in expect_pass:
                c = pf.ensure_contract(ds, persist=False)
                g = c.get("gate") or {}
                seen.add(ds["name"])
                check(f"{ds['name']} 通过画像门",
                      g.get("status") in ("pass", "warn"),
                      f"实际 {g.get('status')} {[x for x in (g.get('codes') or []) if x.startswith('E')]}")
        for name in expect_pass - seen:
            check(f"{name} 存在", False, "数据集中找不到")

    print("\n" + ("全部通过" if not FAILS else f"失败 {len(FAILS)} 项："))
    for x in FAILS:
        print("  -", x)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
