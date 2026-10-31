"""目标检测训练：基于 ultralytics（YOLO）。

与分类/表格路径的差别
----------------------
检测任务的产物是**边界框**，不是单一标签。所以：
* 训练输入是 YOLO 标准目录结构（images/ 与 labels/ 分开）
* 指标是 mAP50 / mAP50-95 / precision / recall，不是 accuracy
* 测试集仍全程不参与训练，只用于最终评估

CPU 上可跑：YOLOv8n 在 CPU 上小数据集通常几分钟内完成一轮。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd


def _yolo_env() -> dict:
    """ultralytics 需要的环境变量：禁掉它的联网检查与遥测。"""
    env = dict(os.environ)
    env["YOLO_OFFLINE"] = "true"
    env["YOLO_VERBOSE"] = "false"
    env["YOLO_AUTOINSTALL"] = "false"
    env["WANDB_DISABLED"] = "true"
    env["ULTRALYTICS_OFFLINE"] = "true"
    return env


# 预训练权重统一放在 _hf_models/ 下，避免每次训练都联网重新下载
_WEIGHTS_DIR = Path(__file__).parent / "_hf_models"


def resolve_weight(name: str) -> str:
    """把模型名解析成本地权重路径；本地没有就回落到原名（由 ultralytics 下载）。"""
    p = _WEIGHTS_DIR / name
    if p.exists() and p.stat().st_size > 0:
        return str(p)
    return name


def prepare_yolo_dirs(manifest: pd.DataFrame, root: Path, work: Path,
                      class_names: list[str]) -> dict:
    """把 manifest 转成 ultralytics 要的目录结构。

    返回 {"images": ..., "labels": ..., "data_yaml": ..., "counts": {...}}
    """
    work = Path(work)
    # 清理旧结构，避免残留文件混进训练
    for sub in ("images", "labels"):
        d = work / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        (d / "train").mkdir(parents=True, exist_ok=True)
        (d / "val").mkdir(parents=True, exist_ok=True)
        (d / "test").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    boxes_per_split = {"train": 0, "val": 0, "test": 0}

    # ultralytics 要求 val 目录非空，但用户的数据常常只有 train/test。
    # 这里先把所有样本按 split 落位，若 val 为空则从 train 里划出一份
    # （分层的：按每个样本的首个类别轮流分配，避免某个类别全被划走）。
    rows = []
    for _, row in manifest.iterrows():
        split = str(row.get("split") or "train")
        if split not in counts:
            split = "train"
        src = Path(str(row["filepath"]))
        if not src.is_absolute():
            src = Path(root) / src
        if not src.exists():
            continue
        rows.append((split, src, str(row.get("boxes") or "")))

    if not any(s == "val" for s, _, _ in rows):
        train_rows = [r for r in rows if r[0] == "train"]
        if len(train_rows) >= 4:
            # 按类别轮流取，每 5 个里取 1 个做验证
            by_cls: dict[str, list] = {}
            for r in train_rows:
                first = r[2].split(";")[0].split(",")[0] if r[2] else "_"
                by_cls.setdefault(first, []).append(r)
            moved = []
            for lst in by_cls.values():
                step = 5
                moved.extend(lst[::step])
            moved_set = {id(r) for r in moved}
            new_rows = []
            for r in rows:
                if id(r) in moved_set:
                    new_rows.append(("val", r[1], r[2]))
                else:
                    new_rows.append(r)
            rows = new_rows

    for split, src, boxes_raw in rows:
        stem = src.stem
        ext = src.suffix or ".jpg"
        img_dst = work / "images" / split / f"{stem}{ext}"
        lbl_dst = work / "labels" / split / f"{stem}.txt"
        try:
            shutil.copy2(src, img_dst)
        except OSError:
            continue
        # 写 YOLO 标签：class_id x y w h
        lines = []
        for item in boxes_raw.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(",")
            if len(parts) < 5:
                continue
            name = parts[0]
            cid = class_names.index(name) if name in class_names else 0
            try:
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        try:
            lbl_dst.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            continue
        counts[split] += 1
        boxes_per_split[split] += len(lines)

    # data.yaml
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_text = (
        f"path: {work.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + names_block + "\n"
    )
    yaml_path = work / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    return {"images": work / "images", "labels": work / "labels",
            "data_yaml": str(yaml_path), "counts": counts,
            "boxes": boxes_per_split}


def train_detection(data_yaml: str, work: Path, epochs: int = 30,
                    model_name: str = "yolov8n.pt", imgsz: int = 320,
                    batch: int = 8, device: str = "cpu",
                    log=None, on_epoch=None) -> dict:
    """跑 YOLO 训练，返回 ultralytics 的结果对象与指标。

    on_epoch(epoch, total, history, record) —— 逐轮回传，
    否则 300 秒的训练全程看不到任何变化（用户反馈：轮数跑完了还在转）。
    """
    from ultralytics import YOLO

    old_env = dict(os.environ)
    os.environ.update(_yolo_env())
    history: list[dict] = []
    try:
        model = YOLO(resolve_weight(model_name))

        # ultralytics 原生回调：每轮结束把指标取出来
        def _on_epoch_end(trainer):
            try:
                ep = int(getattr(trainer, "epoch", 0)) + 1
                total = int(getattr(trainer, "epochs", epochs))
                metrics = {}
                for k, v in (getattr(trainer, "metrics", {}) or {}).items():
                    if isinstance(v, (int, float)):
                        metrics[k] = round(float(v), 5)
                rec = {"epoch": ep, **metrics}
                # 统一字段名，前端曲线/指标卡才能直接用
                if "metrics/mAP50(B)" in rec:
                    rec["val_acc"] = rec["metrics/mAP50(B)"]
                if "metrics/precision(B)" in rec:
                    rec["train_acc"] = rec["metrics/precision(B)"]
                history.append(rec)
                if log:
                    mp = metrics.get("metrics/mAP50(B)")
                    extra = f" · mAP50 {mp:.4f}" if isinstance(mp, float) else ""
                    log(f"epoch {ep}/{total}{extra}")
                if on_epoch:
                    on_epoch(ep, total, history, rec)
            except Exception:  # noqa: BLE001
                pass

        model.add_callback("on_train_epoch_end", _on_epoch_end)

        t0 = time.time()
        res = model.train(
            data=data_yaml,
            epochs=max(1, int(epochs)),
            imgsz=int(imgsz),
            batch=int(batch),
            device=device,
            project=str(work / "runs"),
            name="detect",
            exist_ok=True,
            verbose=False,
            plots=False,
            val=True,
            seed=42,
        )
        dt = time.time() - t0
        metrics = {}
        try:
            box = res.results_dict
            metrics = {k: float(v) for k, v in box.items()
                       if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            metrics = {}
        if log:
            log(f"YOLO 训练完成 {dt:.1f}s")
            for k, v in metrics.items():
                log(f"  {k}: {v:.4f}")
        return {"metrics": metrics, "seconds": round(dt, 1),
                "model": model, "results": res, "history": history}
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def evaluate_detection(model, data_yaml: str, split: str = "test",
                       imgsz: int = 320, device: str = "cpu", log=None) -> dict:
    """在指定划分上评估检测器。"""
    old_env = dict(os.environ)
    os.environ.update(_yolo_env())
    try:
        res = model.val(data=data_yaml, split=split, imgsz=int(imgsz),
                        device=device, verbose=False, plots=False)
        out = {}
        try:
            for k, v in res.results_dict.items():
                if isinstance(v, (int, float)):
                    out[k] = round(float(v), 5)
        except Exception:  # noqa: BLE001
            pass
        if log:
            log(f"检测评估（{split}）：" +
                " · ".join(f"{k}={v}" for k, v in out.items()))
        return out
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def detection_plan(card: dict, n_images: int) -> dict:
    """检测任务的方案基线。"""
    if n_images < 100:
        epochs, imgsz, batch = 60, 320, 8
    elif n_images < 1000:
        epochs, imgsz, batch = 40, 320, 16
    else:
        epochs, imgsz, batch = 25, 416, 16
    return {
        "model": "yolov8n.pt",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": "cpu",
        "source": "rule",
        "reason": f"{n_images} 张标注图 · YOLOv8n 预训练权重 · CPU 训练",
    }
