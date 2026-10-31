"""数据集结构指纹：识别「结构相同」的数据集，复用已验证的 AI 解析脚本。

核心思路：把文件名中的随机部分（哈希、时间戳、序号、样本名）抹掉，
只保留结构骨架与文件类型分布。骨架一致即视为同一种结构。
"""
from __future__ import annotations

import re
from collections import Counter

# 目录/文件名中需要泛化的噪声段
_NOISE_PATTERNS = [
    (re.compile(r"\.[0-9a-f]{16,}$", re.I), ".<hash>"),          # rf.abc123... (roboflow)
    (re.compile(r"\.[0-9a-f]{8,}$", re.I), ".<hash>"),
    (re.compile(r"^[0-9a-f]{16,}$", re.I), "<hash>"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "<hash>"),
    (re.compile(r"(?<=[_\-.])\d{5,}(?=([_\-.]|$))"), "<num>"),   # 长数字串
    (re.compile(r"^(img|image|img_|sample|img\d+|dsc|photo|pic|pix|file|picture|sample)"
                r"[_\-]?\d*$", re.I), "<sample>"),
    # 通用：任意单词前缀 + 纯数字（如 photo_0000、shot12、img-042）→ 样本文件
    (re.compile(r"^[a-z][a-z0-9]*[_\-]\d{2,}$", re.I), "<sample>"),
    (re.compile(r"[_\-]\d{4,}[_\-]", re.I), "_<num>_"),
    (re.compile(r"20\d{2}[-_]?\d{2}[-_]?\d{2}", re.I), "<date>"),  # 日期
    (re.compile(r"\d{4}[-_]?\d{2}[-_]?\d{2}[T_]\d{2}", re.I), "<datetime>"),
    (re.compile(r"[\(\[](copy|副本)\d*[\)\]]", re.I), ""),
]

# 结构性目录名归一（保留 train/val/class 这类语义，抹掉数据集专有名）
_STRUCT_SEGMENTS = {
    "train", "training", "trainset", "val", "valid", "validation", "test", "testing",
    "images", "image", "img", "imgs", "labels", "label", "annotations", "annotation",
    "masks", "mask", "data", "dataset", "datasets", "raw", "all", "samples",
}


_EXT_ALIASES = {
    "jpg": "img", "jpeg": "img", "png": "img", "bmp": "img", "webp": "img",
    "gif": "img", "tif": "img", "tiff": "img",
    "txt": "ann", "ts": "ann", "label": "ann", "anno": "ann", "names": "ann",
    "yaml": "cfg", "yml": "cfg", "json": "meta", "xml": "meta", "cfg": "meta",
    "csv": "table", "tsv": "table", "parquet": "table", "pq": "table",
    "xlsx": "table", "xls": "table", "jsonl": "table",
}


def _normalize_segment(seg: str) -> str:
    """把单个路径段泛化：抹掉哈希、序号、日期等噪声，但保留归一后的扩展名。"""
    ext = ""
    base = seg
    # 只在末段是「名.扩展名」且扩展名较短时才拆，避免把 a.b.c 拆错
    if "." in seg:
        head, _, tail = seg.rpartition(".")
        if head and len(tail) <= 6 and tail.isalnum():
            base = head
            ext = "." + _EXT_ALIASES.get(tail.lower(), tail.lower())
        else:
            base = seg
    s = base
    for pat, rep in _NOISE_PATTERNS:
        s = pat.sub(rep, s)
    s = s.strip("._-")
    low = s.lower()
    if low in _STRUCT_SEGMENTS:
        return low + ext
    # 纯数字或纯噪声 → 统一占位
    if not s or re.fullmatch(r"[\W_]*|\d+|<hash>|<num>|<sample>|<date>|<datetime>", s, re.I):
        return "<n>" + ext
    return "<name>" + ext


def _normalize_path(path: str) -> str:
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    return "/".join(_normalize_segment(p) for p in parts)


def _column_signature(profile: dict) -> str:
    """表格类数据集的列结构签名。

    单文件 CSV / Parquet 若只看目录骨架，所有文件的指纹会完全相同，
    导致记忆库把「A 列结构的表」误判成「B 列结构的表」并复用错脚本。
    因此把列名（归一化后）与列数纳入签名。
    """
    cols = profile.get("columns") or []
    if not cols:
        return ""
    norm = []
    for c in cols[:40]:
        name = str(c.get("name", "")) if isinstance(c, dict) else str(c)
        # 列名也做噪声泛化，避免 a_0001 / a_0002 造成签名抖动
        s = name
        for pat, rep in _NOISE_PATTERNS:
            s = pat.sub(rep, s)
        norm.append(s.strip("._- ").lower())
    # 保持列顺序（列顺序不同通常意味着语义不同），但忽略大小写噪声
    return "C" + ",".join(norm) + f"#{len(cols)}"


def dataset_fingerprint(profile: dict) -> dict:
    """由结构画像计算结构指纹（与具体文件名、样本数量无关）。

    返回 {"signature": str, "layout": [归一化路径...], "ext_mix": {...},
           "shape": [每层目录的去重段数]}
    """
    tree = profile.get("tree") or []
    layout = [_normalize_path(p) for p in tree[:400]]

    # 层级形状：只看目录骨架（去掉末段文件名），统计每层出现过的不同目录数，
    # 刻画"几层、每层几个分支"。这样多套一层外壳不会改变骨架。
    dirs = [p.rsplit("/", 1)[0] for p in layout if "/" in p]
    depth_max = 0
    per_level: dict[int, set] = {}
    for p in dirs:
        segs = p.split("/")
        depth_max = max(depth_max, len(segs))
        for d, s in enumerate(segs):
            per_level.setdefault(d, set()).add(s)
    # 顶层单文件包（如 cifar-10-batches-py/*.pickle）没有目录骨架，用文件层兜底
    shape = [len(per_level.get(d, set())) for d in range(depth_max)]
    if not shape:
        shape = [1, len({p.split("/")[0] for p in layout})]

    # 叶子类型分布（扩展名已在 _normalize_segment 里归一，这里直接统计）
    ext_mix = Counter(p.split("/")[-1] for p in layout)

    # 签名里用「类型占比」而不是「绝对数量」：
    # 同一采集流程的第 1 批和第 2 批数据集文件数通常不同（如 50 张 vs 40 张），
    # 若把数量写进签名，同结构不同体量的数据集就永远命中不了记忆库。
    total = sum(ext_mix.values()) or 1
    ext_ratio = {k: round(v / total, 3) for k, v in ext_mix.items()}

    sig_parts = [
        "L" + "-".join(str(x) for x in shape),
        "E" + ",".join(f"{k}:{ext_ratio[k]}" for k, v in
                       sorted(ext_ratio.items(), key=lambda kv: -kv[1])[:8]),
    ]
    col_sig = _column_signature(profile)
    if col_sig:
        sig_parts.append(col_sig)
    return {
        "signature": "|".join(sig_parts),
        "layout_sample": layout[:12],
        "shape": shape,
        "ext_mix": dict(ext_mix),
    }


def similarity(fp_a: dict, fp_b: dict) -> float:
    """两个指纹的相似度 0~1。层级形状权重最高，其次叶子类型分布。"""
    sa, sb = fp_a.get("shape") or [], fp_b.get("shape") or []
    if not sa or not sb:
        return 0.0
    n = min(len(sa), len(sb))
    shape_sim = sum(1 for i in range(n) if sa[i] == sb[i]) / max(len(sa), len(sb))

    ea, eb = fp_a.get("ext_mix") or {}, fp_b.get("ext_mix") or {}
    keys = set(ea) | set(eb)
    if keys:
        # 按占比比较而非绝对数量：同结构不同体量（50 张 vs 40 张）应视为高相似
        ta = sum(ea.values()) or 1
        tb = sum(eb.values()) or 1
        ext_sim = sum(min(ea.get(k, 0) / ta, eb.get(k, 0) / tb) for k in keys)
    else:
        ext_sim = 0.0

    return round(0.6 * shape_sim + 0.4 * ext_sim, 4)