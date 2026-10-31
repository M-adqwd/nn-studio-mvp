"""目标检测（YOLO / COCO 格式）的独立识别与训练。

为什么要平台自己识别
--------------------
YOLO 数据集长得非常明确：images/ 与 labels/ 成对出现，标签是「每行一个
class_id x y w h」的 .txt。这类数据如果被当成图像分类处理，
每张图的多目标会被压成一个标签、**边界框信息全部丢失** ——
用户以为在训检测器，实际训的是分类器，结果还看不出来。

所以平台必须能独立认出「这是检测数据」，不能只依赖 AI 的声明。

支持两种标注格式
----------------
* YOLO 归一化格式：labels/xxx.txt，每行 `class_id x_center y_center w h`
* COCO JSON：annotations/*.json，含 images / annotations / categories
另外还支持 YOLOv8 的**分割标注**（每行 `class_id` 后跟 2N 个归一化多边形顶点）。
这种格式不能直接当检测框用，平台会算出多边形的最小外接矩形转成检测框，
这样同一套检测训练流程照样能跑 —— 否则这类数据会因「一个标签都认不出」
而彻底解析失败。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# YOLO 标签行：class_id 后跟 4 个浮点数（可含空格/制表符分隔）
_YOLO_LINE = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                        r"([\d.eE+-]+)\s+([\d.eE+-]+)\s*$")

# YOLOv8 分割行：class_id 后跟偶数个浮点数（多边形顶点 x1 y1 x2 y2 ...）
_SEG_LINE = re.compile(r"^\s*(\d+)\s+((?:[\d.eE+-]+\s+){3,}[\d.eE+-]+)\s*$")


def _seg_to_box(values: list[float]) -> tuple[float, float, float, float] | None:
    """把归一化多边形顶点转成 YOLO 检测框 (xc, yc, w, h)。

    YOLOv8 分割标注：class_id x1 y1 x2 y2 ...（顶点成对，交替 x/y）。
    """
    if len(values) < 6 or len(values) % 2 != 0:
        return None
    xs = values[0::2]
    ys = values[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    return (x0 + w / 2.0, y0 + h / 2.0, w, h)


def _parse_label_line(ln: str) -> tuple[int, float, float, float, float] | None:
    """解析一行 YOLO 标注，检测框与分割多边形都支持。"""
    m = _YOLO_LINE.match(ln)
    if m:
        return (int(m.group(1)), float(m.group(2)), float(m.group(3)),
                float(m.group(4)), float(m.group(5)))
    m = _SEG_LINE.match(ln)
    if m:
        try:
            vals = [float(x) for x in m.group(2).split()]
        except ValueError:
            return None
        box = _seg_to_box(vals)
        if box:
            return (int(m.group(1)), *box)
    return None


def find_yolo_dataset(root: Path) -> dict | None:
    """在数据集根目录里找 YOLO 结构。

    返回 {"labels_dir", "images_dir", "classes", "class_map", "label_files"} 或 None。

    支持两种常见布局：
      * 标准 YOLO：images/train|val|test/ 与 labels/train|val|test/ 分离
      * Darknet 老格式：图片与标签同级放在一个目录（ts/ts/00027.jpg + .txt）
    """
    root = Path(root)
    if not root.exists():
        return None

    classes, class_map = _read_class_names(root, root)

    # 布局一：标准 YOLO 的 labels/ 目录
    # 注意：同一个数据集里可能有多套并存的目录（如 Roboflow 导出的
    # BoneFractureYolo8/ 与 bone fracture detection.v4-v4.yolov8/），
    # 也可能同一个 labels/ 下有 train/valid/test 多个子目录。
    # 都要收集，否则只会取到其中一个划分，val/test 直接丢失。
    groups: dict[str, dict] = {}     # 按 images_dir 所在的上层分组
    for ld in [p for p in root.rglob("*")
               if p.is_dir() and p.name.lower() == "labels"]:
        label_files = [t for t in ld.rglob("*.txt") if _is_yolo_label_file(t)]
        if not label_files:
            continue
        images_dir = _guess_images_dir(ld, root)
        if not images_dir:
            continue
        # 分组键：labels/ 的上级目录（train/valid/test 的共同父目录），
        # 这样同一数据集的三个划分会归到一组，不会各自分组。
        # 注意不能用 images_dir.parent —— 那正好是划分名（train/valid/test），
        # 会导致每个划分自成一组，最后只取到一个划分。
        key = str(ld.parent.parent)
        g = groups.setdefault(key, {"images_dirs": [], "labels_dirs": [],
                                    "label_files": []})
        g["images_dirs"].append(images_dir)
        g["labels_dirs"].append(ld)
        g["label_files"].extend(label_files)

    if groups:
        # 选标签最多的一套（同一数据集里可能重复导出了多份）
        key = max(groups, key=lambda k: len(groups[k]["label_files"]))
        g = groups[key]
        return {"labels_dir": g["labels_dirs"][0],
                "labels_dirs": g["labels_dirs"],
                "images_dirs": g["images_dirs"],
                "images_dir": g["images_dirs"][0],
                "classes": classes, "class_map": class_map,
                "label_files": g["label_files"], "layout": "yolo"}

    # 布局二：Darknet 式「图片与标签同级」
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        label_files = [t for t in d.glob("*.txt") if _is_yolo_label_file(t)]
        if len(label_files) < 3:
            continue
        # 同目录必须有对应图片
        has_img = any(
            (t.with_suffix(e)).exists()
            for t in label_files[:5]
            for e in (".jpg", ".jpeg", ".png", ".bmp"))
        if not has_img:
            continue
        return {"labels_dir": d, "images_dir": d,
                "classes": classes, "class_map": class_map,
                "label_files": label_files, "layout": "darknet"}
    return None


def _is_yolo_label_file(p: Path) -> bool:
    """判断一个 .txt 是 YOLO 标签文件还是类别名清单。

    类别名清单（如 classes.names）：每行是名字，没有数字坐标。
    YOLO 标签：至少一行能解析出 class_id + 坐标
    （检测框 4 个数，或分割多边形 6 个以上的数）。
    """
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except OSError:
        return False
    if not lines:
        return False
    hit = 0
    for ln in lines[:20]:
        if _parse_label_line(ln):
            hit += 1
    # 至少一半的行是坐标格式才算标签文件
    return hit >= max(1, len(lines[:20]) // 2)


def _guess_images_dir(labels_dir: Path, root: Path) -> Path | None:
    """从 labels 目录反推 images 目录。"""
    cands = []
    # 同级：labels/xxx -> images/xxx
    for parent in labels_dir.parents:
        for name in ("images", "Images", "IMG"):
            p = parent / name
            if p.is_dir():
                cands.append(p)
    # 上两级：dataset/images/train -> dataset/labels/train
    for parent in labels_dir.parents:
        if parent == root:
            break
    if labels_dir.parent.name.lower() in ("train", "val", "valid", "test"):
        p = labels_dir.parent.parent / "images" / labels_dir.parent.name
        if p.is_dir():
            cands.append(p)
    # 兜底：全盘找 images 目录
    if not cands:
        for p in root.rglob("*"):
            if p.is_dir() and p.name.lower() == "images":
                cands.append(p)
    return cands[0] if cands else None


def _read_class_names(root: Path, _unused: Path | None = None) -> tuple[list[str], dict]:
    """读类别名列表，返回 (classes, {class_id: name})。

    优先 data.yaml 的 names，其次 classes.names / labels.names 等清单文件。
    """
    classes: list[str] = []
    # 1. data.yaml
    for y in root.rglob("*.yaml"):
        try:
            txt = y.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(r"^\s*names\s*:\s*\[(.*?)\]", txt, re.M | re.S)
        if m:
            names = [x.strip().strip("'\"") for x in m.group(1).split(",")]
            names = [n for n in names if n]
            if names:
                classes = names
                break
    # 2. names 清单文件（classes.names 每行一个类别名，不是坐标）
    if not classes:
        cands = (list(root.rglob("classes.names")) + list(root.rglob("*.names"))
                 + list(root.rglob("*labels*.txt")))
        for p in cands:
            try:
                lines = [ln.strip() for ln in
                         p.read_text(encoding="utf-8", errors="ignore").splitlines()
                         if ln.strip()]
            except OSError:
                continue
            if lines and not _is_yolo_label_file(p):
                classes = lines
                break
    if not classes:
        return [], {}
    return classes, {i: n for i, n in enumerate(classes)}


def _split_of(rel: str) -> str | None:
    """从相对路径判断划分。"""
    low = rel.lower()
    for key, name in (("val", "val"), ("valid", "val"), ("train", "train"),
                      ("test", "test"), ("eval", "test"), ("dev", "val")):
        for seg in low.replace("\\", "/").split("/"):
            s = seg.strip()
            if s == key or s.startswith(key) or s.endswith(key):
                return name
    return None


def _read_split_lists(root: Path) -> dict[str, str]:
    """读 train.txt / val.txt / test.txt 清单，返回 {图片名或路径: split}。

    Darknet 老数据集常用这种方式划分：清单里每行一个图片路径。
    注意清单里的路径可能是别人机器上的绝对路径（如 /home/my_name/...），
    这里只取文件名做匹配。
    """
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for name in (f"{split}.txt", f"{split}_list.txt", f"{split}.list"):
            for p in root.rglob(name):
                try:
                    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                hit = 0
                for ln in lines:
                    ln = ln.strip()
                    if not ln:
                        continue
                    key = Path(ln.replace("\\", "/")).name
                    if not key:
                        continue
                    mapping[key] = split
                    mapping[Path(key).stem] = split
                    hit += 1
                if hit:
                    break
    return mapping


def build_manifest(root: Path, progress=None) -> pd.DataFrame | None:
    """扫描 YOLO 数据集，产出带边界框的 manifest。

    列：filepath / label / split / boxes
    boxes 格式 '类别名,x,y,w,h;类别名,x,y,w,h'

    progress(done, total, label) —— 扫描阶段的进度回调，
    大数据集要扫几万个文件，没有进度用户只会觉得卡死了。
    """
    info = find_yolo_dataset(root)
    if not info:
        return None
    labels_dir: Path = info["labels_dir"]
    labels_dirs: list[Path] = info.get("labels_dirs") or [labels_dir]
    images_dir: Path = info["images_dir"]
    # 每个 labels 目录对应的 images 目录（不同划分的 images 是分开的）
    images_of: dict[str, Path] = {}
    for ld in labels_dirs:
        g = _guess_images_dir(ld, root)
        if g:
            images_of[str(ld)] = g
    classes: list[str] = info["classes"]
    cmap: dict = info["class_map"]
    split_map = _read_split_lists(root)

    # 建立 图片文件名 → 路径 的索引（图片可能在子目录里）
    img_index: dict[str, Path] = {}
    all_image_dirs = [images_of.get(str(ld), images_dir) for ld in labels_dirs]
    for idir in all_image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"):
            for p in idir.rglob(ext):
                img_index.setdefault(p.stem, p)
                img_index.setdefault(p.name, p)

    # 逐个划分目录处理：同一个 labels/ 下的 train/valid/test 要各自算路径，
    # 只处理单个目录会让 val/test 划分直接丢失。
    rows = []
    all_label_files: list[tuple[Path, Path]] = []
    for ld in labels_dirs:
        lfiles = [t for t in ld.rglob("*.txt") if _is_yolo_label_file(t)]
        all_label_files.extend((ld, t) for t in sorted(lfiles))
    total_files = len(all_label_files)
    done_files = 0
    for ld, lf in all_label_files:
            try:
                lines = [ln.strip() for ln in
                         lf.read_text(encoding="utf-8", errors="ignore").splitlines()
                         if ln.strip()]
            except OSError:
                continue
            boxes = []
            names = []
            for ln in lines:
                parsed = _parse_label_line(ln)
                if not parsed:
                    continue
                cid, x, y, w, h = parsed
                name = cmap.get(cid, f"class_{cid}")
                boxes.append(f"{name},{x:.6f},{y:.6f},{w:.6f},{h:.6f}")
                if name not in names:
                    names.append(name)
            if not boxes:
                continue
            # 图片路径：优先同目录的 images/ 对应位置，再按文件名索引
            ld_images = images_of.get(str(ld), images_dir)
            rel_img = _find_image_for(lf, ld, ld_images, img_index)
            if not rel_img:
                continue
            # 划分优先级：清单文件 > 路径目录名
            split = (split_map.get(rel_img.name)
                     or split_map.get(rel_img.stem)
                     or _split_of(str(rel_img))
                     or _split_of(str(lf.relative_to(root)))
                     or "train")
            rows.append({
                "filepath": str(rel_img).replace("\\", "/"),
                "label": names[0] if names else "object",
                "split": split,
                "boxes": ";".join(boxes),
            })
            done_files += 1
            if progress and done_files % 200 == 0:
                try:
                    progress(done_files, total_files, "正在读取标注")
                except Exception:  # noqa: BLE001
                    pass

    if not rows:
        return None
    df = pd.DataFrame(rows)
    # 一个图像可能有多个标签文件被合并到同一图片
    df = df.groupby(["filepath", "split"], as_index=False).agg({
        "label": "first",
        "boxes": lambda s: ";".join(
            [b for part in s for b in str(part).split(";") if b]),
    })
    df["label"] = df["boxes"].str.split(";").str[0].str.split(",").str[0]
    return df


def _find_image_for(label_file: Path, labels_dir: Path, images_dir: Path,
                    index: dict[str, Path]) -> Path | None:
    """为标签文件找到对应的图片。"""
    # 1. 同名同目录（labels/ 与 images/ 平级时）
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = label_file.with_suffix(ext)
        if cand.exists():
            return cand
    # 2. 镜像目录：labels/train/x.txt -> images/train/x.jpg
    try:
        rel = label_file.relative_to(labels_dir)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = images_dir / rel.with_suffix(ext)
            if cand.exists():
                return cand
    except ValueError:
        pass
    # 3. 按文件名索引
    return index.get(label_file.stem)


def detect_coco(root: Path) -> pd.DataFrame | None:
    """解析 COCO 格式标注。"""
    root = Path(root)
    ann_files = [p for p in root.rglob("*.json")
                 if p.name.lower() in ("instances_*.json", "annotations.json",
                                       "_annotations.coco.json", "instances.json")]
    if not ann_files:
        return None
    af = ann_files[0]
    try:
        data = json.loads(af.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
        return None

    cats = {int(c["id"]): str(c.get("name", c["id"]))
            for c in data.get("categories", [])}
    img_by_id = {int(i["id"]): i for i in data.get("images", [])}

    # 找到图片所在目录
    img_dir = None
    for cand in (af.parent / "images", af.parent, root):
        if cand.exists() and any(cand.rglob("*.jpg")):
            img_dir = cand
            break
    if img_dir is None:
        return None

    by_image: dict[int, list] = {}
    for a in data["annotations"]:
        by_image.setdefault(int(a["image_id"]), []).append(a)

    rows = []
    for img_id, anns in by_image.items():
        info = img_by_id.get(img_id)
        if not info:
            continue
        fn = str(info.get("file_name", ""))
        # 找实际图片
        cand = img_dir / fn
        if not cand.exists():
            hits = list(img_dir.rglob(Path(fn).name))
            if not hits:
                continue
            cand = hits[0]
        boxes, names = [], []
        for a in anns:
            name = cats.get(int(a.get("category_id", 0)), f"class_{a.get('category_id')}")
            x, y, w, h = [float(v) for v in a.get("bbox", [0, 0, 0, 0])]
            iw = float(info.get("width") or 1) or 1.0
            ih = float(info.get("height") or 1) or 1.0
            # COCO 是绝对像素 → 转成 YOLO 归一化
            boxes.append(f"{name},{x/iw:.6f},{y/ih:.6f},{w/iw:.6f},{h/ih:.6f}")
            if name not in names:
                names.append(name)
        if not boxes:
            continue
        rows.append({"filepath": str(cand.relative_to(root)).replace("\\", "/"),
                     "label": names[0], "split": _split_of(fn) or "train",
                     "boxes": ";".join(boxes)})
    if not rows:
        return None
    return pd.DataFrame(rows)


def is_detection_manifest(df: pd.DataFrame) -> bool:
    """判断 manifest 是否是检测格式（含 boxes 列且有内容）。"""
    if "boxes" not in df.columns:
        return False
    try:
        return bool(df["boxes"].astype(str).str.strip().ne("").any())
    except Exception:  # noqa: BLE001
        return False
