"""智能体自动建模：上传数据集后由智能体接管建模全流程。

为什么需要这个模块
------------------
同一份数据，直接交给 AI 训练准确率高，传到软件里训练却差很多。差距不在玄学，
而是软件侧为了能在 CPU 上「跑得完」做过几处保守取舍：

    1. 从零训练 SmallCNN，**没有预训练权重**          <- 最大头
    2. 输入压到 64x64，细节全丢
    3. **完全没有数据增强**，小数据必然过拟合
    4. 单次随机划分，没有交叉验证与多种子平均

本模块把「AI 直训」的套路固化进软件，同时适配 CPU 约束：

    * 冻结预训练 backbone **预提取特征**：前向次数从 epochs x N 降到 K x N
      （K 为增强副本数），CPU 上快一个数量级
    * 只在特征上训练分类头，秒级收敛
    * 输入 160-224px，使用 ImageNet 标准化
    * 增强做**多视角平均**后注入特征，保留增强带来的不变性
    * 结果不达标时，智能体看着真实指标自动换方案重训（上限 MAX_ROUNDS）

三条红线（不可协商）
--------------------
    * 划分以 AI 解析阶段锁定的 split 列为准，本模块**只消费、绝不重切**
    * 测试集全程不参与训练与调参
    * 解析脚本在沙箱内执行，训练在沙箱外执行
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

import db

MAX_ROUNDS = int(os.environ.get("NN_MAX_ROUNDS", "3"))

# 增强副本数：对每张图做 K 次随机增强后特征取平均。
# K=1 即不增强（最快），K 越大正则效果越强、耗时线性增加。
AUG_COPIES = {"none": 1, "light": 2, "medium": 4, "strong": 8}

# 单轮特征提取的前向次数上限，避免 CPU 上跑太久
_MAX_FORWARDS = 6000

# 各 backbone 的特征维度与推荐输入尺寸
_BACKBONE_SPECS = {
    "mobilenet_v3_small": {"feat_dim": 576, "sizes": {"cpu": 160, "cuda": 224}},
    "resnet18": {"feat_dim": 512, "sizes": {"cpu": 192, "cuda": 224}},
}

# 输入尺寸上限。CPU 上默认 224 是为了控制耗时，但眼底、病理这类
# 「病灶只有几个像素」的影像在 224 下会直接丢掉判据，
# 需要时可设 NN_MAX_IMG_SIZE=320 放宽（耗时约翻倍）。
MAX_IMG_SIZE = int(os.environ.get("NN_MAX_IMG_SIZE", "224"))


# ---------- 设备与权重 ----------

def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_backbone(name: str) -> tuple[nn.Module, int, object]:
    """构建去掉分类头的预训练 backbone，返回 (网络, 特征维度, 权重枚举)。

    权重枚举里带着官方推荐的 mean/std，标准化必须与预训练时一致，
    否则迁移学习的收益会被吃掉一大半。
    """
    from torchvision import models as tvm

    if name == "mobilenet_v3_small":
        w = tvm.MobileNet_V3_Small_Weights.DEFAULT
        net = tvm.mobilenet_v3_small(weights=w)
        feat_dim = int(net.classifier[0].in_features)
        net.classifier = nn.Identity()
        return net, feat_dim, w
    if name == "resnet18":
        w = tvm.ResNet18_Weights.DEFAULT
        net = tvm.resnet18(weights=w)
        feat_dim = int(net.fc.in_features)
        net.fc = nn.Identity()
        return net, feat_dim, w
    raise ValueError(f"未知 backbone：{name}")


def ensure_weights(name: str = "mobilenet_v3_small") -> dict:
    """确保预训练权重可用；首次使用会下载到本地缓存（沙箱禁网也能读缓存）。"""
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cached = cache_dir.exists() and any(cache_dir.glob("*.pth"))
    try:
        _build_backbone(name)
        return {"ok": True, "cached": bool(cached), "detail": "权重已就绪"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "cached": False, "detail": str(e)[:300]}


# ---------- 数据体检 ----------

def inspect(df: pd.DataFrame, task_hint: str, meta: dict) -> dict:
    """产出数据卡：后续所有决策都基于它，而不是凭空猜。"""
    card: dict = {"rows": int(len(df)), "task_hint": task_hint}

    if "split" in df.columns:
        vc = df["split"].astype(str).str.strip().str.lower().value_counts()
        card["splits"] = {str(k): int(v) for k, v in vc.items()}
    else:
        card["splits"] = {}

    is_cls = task_hint in ("image_classification", "text_classification",
                           "tabular_classification")
    if is_cls:
        lcol = meta.get("target_col") or ("label" if "label" in df.columns else None)
        if lcol and lcol in df.columns:
            vc = df[lcol].astype(str).value_counts()
            card["n_classes"] = int(len(vc))
            card["class_counts"] = {str(k): int(v) for k, v in vc.items()}
            card["min_class"] = int(vc.min())
            card["max_class"] = int(vc.max())
            card["imbalance"] = round(float(vc.max()) / max(float(vc.min()), 1), 2)
        else:
            card["n_classes"] = 0

    if task_hint == "object_detection":
        # 检测任务：类别来自所有框，不是每张图的主标签
        names: list[str] = []
        n_boxes = 0
        per_img: list[int] = []
        if "boxes" in df.columns:
            for b in df["boxes"].astype(str):
                items = [x.strip() for x in b.split(";") if x.strip()]
                per_img.append(len(items))
                n_boxes += len(items)
                for it in items:
                    nm = it.split(",")[0]
                    if nm and nm not in names:
                        names.append(nm)
        card["n_classes"] = len(names)
        card["classes_list"] = names
        card["n_boxes"] = n_boxes
        card["avg_objects"] = round(n_boxes / max(len(df), 1), 2)
        card["max_objects"] = max(per_img) if per_img else 0
        card["image_root"] = meta.get("image_root")
        return card

    if task_hint == "image_classification":
        card["image_root"] = meta.get("image_root")
        bad, probed = _probe_images(meta.get("image_root"), df, k=40)
        card["bad_images"] = bad
        card["probed_images"] = probed
        card["bad_ratio"] = round(bad / probed, 3) if probed else 0.0
    return card


def _probe_images(root, df: pd.DataFrame, k: int = 40) -> tuple[int, int]:
    """抽查若干张图能否正常解码，返回 (坏图数, 抽查数)。

    不扫全量，够快即可——目的是在开训前就发现「图像文件整批丢失/损坏」，
    而不是等训练完才发现指标是随机水平。
    """
    if not root or "filepath" not in df.columns:
        return 0, 0
    root_p = Path(root)
    paths = df["filepath"].astype(str).tolist()
    if not paths:
        return 0, 0
    step = max(1, len(paths) // k)
    sample = paths[::step][:k]
    bad = 0
    for rel in sample:
        try:
            p = root_p / rel
            if not p.exists():
                bad += 1
                continue
            with Image.open(p) as im:
                im.convert("RGB")
        except Exception:  # noqa: BLE001
            bad += 1
    return bad, len(sample)


# ---------- 智能体决策 ----------

def _aug_copies(aug: str, n: int) -> int:
    """增强副本数：数据越少越需要多视角，但总量太大时收敛。"""
    base = AUG_COPIES.get(str(aug), 1)
    if base <= 1:
        return 1
    return max(1, min(base, max(1, _MAX_FORWARDS // max(n, 1))))


def baseline_plan(card: dict) -> dict:
    """规则基线：不依赖 AI，任何情况下都能给出可用方案。

    样本越少 -> 越依赖预训练特征、越强增强、越多轮次；
    类别越多 -> 越大 backbone。

    训练方式默认用微调（解冻主干末段）：实测同一份数据上，
    微调比「冻结只训分类头」的测试集准确率高 20+ 个百分点
    （交通标志数据集：0.4144 → 0.6396），而耗时只多 20 秒左右。
    冻结模式保留为超大样本时的省时选项。
    """
    n = int(card.get("rows") or 0)
    nc = int(card.get("n_classes") or 0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if n < 300 or nc <= 3:
        backbone = "mobilenet_v3_small"
        aug = "strong" if n < 200 else "medium"
        epochs, lr, batch, wd = 40, 3e-3, 16, 1e-3
    elif n < 5000:
        backbone = "mobilenet_v3_small"
        aug, epochs, lr, batch, wd = "medium", 25, 2e-3, 32, 5e-4
    else:
        backbone = "resnet18"
        aug, epochs, lr, batch, wd = "light", 15, 1e-3, 32, 1e-4

    if nc >= 20:
        backbone = "resnet18"

    # 验证集太小时，指标差异可能只差几个样本，选方案等于抛硬币。
    # 这时偏向更保守的选择（更大输入尺寸对小目标更稳）。
    splits = card.get("splits") or {}
    val_n = int(splits.get("val") or 0)
    img_size = _BACKBONE_SPECS[backbone]["sizes"][dev]
    if 0 < val_n < 150:
        img_size = min(MAX_IMG_SIZE, img_size + 32)
    # 显式指定输入尺寸（如眼底图需要更大视野看清微血管瘤）
    _force = int(os.environ.get("NN_IMG_SIZE", "0") or 0)
    if _force:
        img_size = min(MAX_IMG_SIZE, _force)

    imb = float(card.get("imbalance") or 1.0)
    # 超大样本：冻结提特征可缓存复用，多轮迭代几乎零成本，更划算
    mode = "frozen" if n > 20000 else "finetune"
    if mode == "finetune":
        # 微调时主干的学习率要比分类头小两个数量级，轮次也相应减少
        epochs = max(8, min(epochs, 20))
        batch = max(8, min(batch, 16))
    return {
        "backbone": backbone,
        "img_size": img_size,
        "augment": aug,
        "aug_copies": _aug_copies(aug, n),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch,
        "weight_decay": wd,
        "dropout": 0.3 if n < 1000 else 0.15,
        "label_smoothing": 0.05 if imb > 3 else 0.0,
        "class_weight": imb > 3,
        # 训练模式：frozen = 冻结 backbone 只训分类头（快，特征可缓存）
        #          finetune = 解冻 backbone 末段一起训（更准）
        "mode": mode,
        "finetune_lr": 3e-4,
        "source": "rule",
        "reason": f"样本 {n} 条 · 类别 {nc} 个 · 最大/最小类比 {imb}"
                  + (" · 超大样本用快速模式" if mode == "frozen" else ""),
    }


_PLAN_SYSTEM = (
    "你是自动建模专家。根据数据卡选择图像分类方案，"
    "只输出 JSON，不要解释，不要 markdown 代码块。"
)


_AUG_ORDER = ["none", "light", "medium", "strong"]


def _revise_by_rule(plan: dict, verdict: str) -> dict:
    """AI 不可用时的兜底迭代：按诊断结论对方案做确定性调整。

    缺了这一步，AI 调用失败会让每轮都跑完全相同的方案，
    迭代次数再多也只是重复劳动，拿不到任何改进。
    """
    p = dict(plan)
    lvl = int(p.get("_level") or 0) + 1
    p["_level"] = lvl
    if verdict == "overfit":
        p["dropout"] = min(0.6, float(p.get("dropout") or 0.2) + 0.15)
        p["weight_decay"] = min(0.1, float(p.get("weight_decay") or 1e-4) * 3)
        p["epochs"] = max(5, int(p.get("epochs") or 25) // 2)
        i = _AUG_ORDER.index(str(p.get("augment") or "medium"))
        if i + 1 < len(_AUG_ORDER):
            p["augment"] = _AUG_ORDER[i + 1]
    elif verdict == "underfit":
        p["backbone"] = "resnet18"
        p["img_size"] = min(MAX_IMG_SIZE, int(p.get("img_size") or 160) + 32)
        p["lr"] = float(p.get("lr") or 2e-3) * 1.5
        p["epochs"] = min(100, int(p.get("epochs") or 25) + 20)
        p["dropout"] = max(0.0, float(p.get("dropout") or 0.2) - 0.1)
        # 欠拟合说明特征表达不够，转微调让主干一起适应任务
        p["mode"] = "finetune"
        p["finetune_lr"] = float(p.get("finetune_lr") or 3e-4)
    else:  # plateau：逐级加大容量
        # 关键点：这里必须「在上一轮方案基础上往前走一步」，而不是每次都从
        # 规则基线重算一遍。否则第 2、3 轮会收敛到完全相同的方案
        # （实测第 3 轮原样重跑了第 2 轮的 resnet18@192，白烧 25 分钟）。
        p["backbone"] = "resnet18"
        p["mode"] = "finetune"
        p["finetune_lr"] = float(p.get("finetune_lr") or 3e-4)
        base_size = int(p.get("img_size") or 160)
        if lvl <= 1:
            p["img_size"] = min(MAX_IMG_SIZE, base_size + 32)
        elif lvl == 2:
            p["img_size"] = min(MAX_IMG_SIZE, base_size + 32)
            p["epochs"] = min(100, int(p.get("epochs") or 20) + 10)
            p["label_smoothing"] = max(float(p.get("label_smoothing") or 0), 0.1)
            i = _AUG_ORDER.index(str(p.get("augment") or "medium"))
            p["augment"] = _AUG_ORDER[min(i + 1, len(_AUG_ORDER) - 1)]
        else:
            p["img_size"] = MAX_IMG_SIZE
            p["epochs"] = min(100, int(p.get("epochs") or 25) + 5)
            p["label_smoothing"] = max(float(p.get("label_smoothing") or 0), 0.15)
            p["class_weight"] = True
    p["aug_copies"] = _aug_copies(str(p.get("augment") or "medium"), 0)
    p["source"] = "rule-revised"
    p["reason"] = f"AI 不可用，按诊断（{verdict}）自动调整（第 {lvl} 次调整）"
    return p


def decide(card: dict, prev_feedback: str | None = None,
           verdict: str | None = None, prev_plan: dict | None = None) -> dict:
    """智能体决策：在规则基线基础上用 AI 微调；AI 不可用时退回规则迭代。"""
    plan = baseline_plan(card)
    try:
        import llm
        payload = {
            "数据卡": {
                "样本量": card.get("rows"),
                "任务类型": card.get("task_hint"),
                "类别数": card.get("n_classes"),
                "每类样本数": card.get("class_counts"),
                "最大最小类比": card.get("imbalance"),
                "坏图数量": card.get("bad_images"),
                "划分情况": card.get("splits"),
            },
            "规则基线方案": plan,
        }
        if prev_feedback:
            payload["上一轮结果反馈"] = prev_feedback
        prompt = (
            json.dumps(payload, ensure_ascii=False)
            + "\n\n可调整字段：backbone(mobilenet_v3_small|resnet18)、"
            "img_size(128|160|192|224|256|320)、augment(none|light|medium|strong)、"
            "epochs(1-100)、lr、batch_size(8-128)、dropout(0-0.6)、"
            "weight_decay(0-0.1)、label_smoothing(0-0.2)、class_weight(bool)、"
            "mode(frozen|finetune)。\n"
            f"约束：运行在 CPU 上，img_size 不要超过 {MAX_IMG_SIZE}；样本少于 200 时加强增强、"
            "减小 batch、增大 epochs；类别严重不平衡时启用 class_weight。\n"
            "mode 说明：frozen = 冻结主干网络、只训练分类头（快，特征可缓存复用）；"
            "finetune = 解冻主干网络末段一起训练（更准，耗时略增）。"
            "样本量少于 3000 且类别不多时用 finetune 通常能拿到更高准确率；"
            "样本很多时用 frozen 更划算。\n"
            '输出格式：{"backbone":..,"img_size":..,"augment":..,"epochs":..,'
            '"lr":..,"batch_size":..,"dropout":..,"weight_decay":..,'
            '"label_smoothing":..,"class_weight":..,"mode":"..","reason":"一句话理由"}'
        )
        raw = llm.chat([{"role": "user", "content": prompt}],
                       system=_PLAN_SYSTEM, timeout=120)
        obj = llm._parse_json_loose(raw)
        if isinstance(obj, dict) and obj.get("backbone") in _BACKBONE_SPECS:
            merged = dict(plan)
            for k in ("backbone", "img_size", "augment", "epochs", "lr",
                      "batch_size", "dropout", "weight_decay",
                      "label_smoothing", "class_weight", "mode"):
                if k in obj and obj[k] is not None:
                    merged[k] = obj[k]
            merged["img_size"] = int(merged["img_size"])
            merged["epochs"] = max(1, min(int(merged["epochs"]), 100))
            merged["batch_size"] = max(8, min(int(merged["batch_size"]), 128))
            merged["lr"] = float(merged["lr"])
            if merged.get("mode") not in ("frozen", "finetune"):
                merged["mode"] = "frozen"
            # 微调模式用 backbone 末段的学习率，比分类头大两个数量级
            merged["finetune_lr"] = float(merged.get("finetune_lr") or 3e-4)
            merged["aug_copies"] = _aug_copies(
                str(merged.get("augment") or "medium"), int(card.get("rows") or 0))
            merged["source"] = "ai"
            merged["reason"] = str(obj.get("reason") or merged.get("reason"))[:200]
            # 保护：AI 不允许把输入尺寸降到规则基线以下。
            # 实测降尺寸会明显掉准确率（小目标尤其吃亏），
            # 而 AI 在这一步的判断并不比规则可靠。
            base_size = int(plan.get("img_size") or merged["img_size"])
            if merged["img_size"] < base_size:
                merged["img_size"] = base_size
                merged["reason"] += "（已保持规则基线的输入尺寸）"
            # 同理：规则基线判定为微调时，不允许 AI 退回冻结。
            # 实测同一份数据上微调的测试集准确率高 20+ 个百分点。
            if (plan.get("mode") or "frozen") == "finetune" \
                    and merged.get("mode") != "finetune":
                merged["mode"] = "finetune"
                merged["reason"] += "（已采用更准的微调方式）"
            return merged
    except Exception:  # noqa: BLE001
        pass
    # AI 不可用：首轮用规则基线，后续轮次按诊断结论确定性调整
    if verdict in ("overfit", "underfit", "plateau"):
        # 基于上一轮的方案继续调整，而不是回到基线重算
        return _revise_by_rule(prev_plan or plan, verdict)
    return plan


# ---------- 图像变换 ----------

def _norm_of(weights) -> tuple[list[float], list[float]]:
    tf = weights.transforms()
    return list(tf.mean), list(tf.std)


def _train_tf(img_size: int, weights):
    from torchvision import transforms as T
    mean, std = _norm_of(weights)
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.64, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        T.RandomRotation(12),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def _val_tf(img_size: int, weights):
    from torchvision import transforms as T
    mean, std = _norm_of(weights)
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def _read_batch(root: Path, rels: list[str], tf) -> tuple[list, int]:
    """解码一批图像，返回 (张量列表, 坏图数量)。

    坏图位置留 None，由调用方用同批有效样本的特征均值填充——
    直接填零会让该样本落在特征分布之外，反而干扰分类头。
    """
    items: list = []
    bad = 0
    for rel in rels:
        try:
            with Image.open(root / rel) as im:
                items.append(tf(im.convert("RGB")))
        except Exception:  # noqa: BLE001
            items.append(None)
            bad += 1
    return items, bad


# ---------- 特征提取 ----------

@torch.no_grad()
def extract_features(root: Path, paths: list[str], plan: dict,
                     cache_path: Path | None = None, log=None,
                     on_progress=None) -> np.ndarray:
    """冻结 backbone 预提取特征；K 次增强取平均，兼得正则与速度。

    缓存文件名带 (backbone, img_size, aug_copies, 样本数)，换方案自动失效。
    on_progress(done, total, elapsed) —— 提特征是最耗时的环节，必须实时回传，
    否则用户看到的就是「日志不动、进度条不动、最后突然全出来」。
    """
    if cache_path and cache_path.exists():
        if log:
            log(f"复用特征缓存：{cache_path.name}")
        return np.load(cache_path)["f"]

    dev = device_of()
    net, feat_dim, w = _build_backbone(plan["backbone"])
    net.to(dev).eval()
    size = int(plan.get("img_size") or 160)
    k = int(plan.get("aug_copies") or 1)
    bsz = int(plan.get("batch_size") or 32)
    tf = _train_tf(size, w) if k > 1 else _val_tf(size, w)

    out = np.zeros((len(paths), feat_dim), dtype=np.float32)
    t0 = time.time()
    n_bad = 0
    total = len(paths)
    for start in range(0, len(paths), bsz):
        chunk = paths[start:start + bsz]
        acc = np.zeros((len(chunk), feat_dim), dtype=np.float32)
        for _ in range(k):
            items, bad = _read_batch(root, chunk, tf)
            n_bad += bad
            valid = [x for x in items if x is not None]
            fill = torch.stack(valid).mean(0) if valid else torch.zeros(3, size, size)
            batch = torch.stack([x if x is not None else fill for x in items]).to(dev)
            acc += net(batch).float().cpu().numpy()
        out[start:start + len(chunk)] = acc / k
        if on_progress:
            try:
                on_progress(min(start + len(chunk), total), total, time.time() - t0)
            except Exception:  # noqa: BLE001
                pass

    if log and n_bad:
        log(f"跳过 {n_bad} 次损坏图像解码，已用同批均值填充")
    if log:
        log(f"特征提取完成：{len(paths)} 张 · {feat_dim} 维 · 增强 x{k} · {time.time()-t0:.1f}s")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, f=out)
    return out


# ---------- 进度推送 ----------

class Progress:
    """把训练进度实时写回任务记录。

    之前智能体路径只在训练结束时写一次，导致用户看到：
    日志在滚但曲线空白、进度条不动、状态一直卡「训练中」，最后突然全蹦出来。
    这里统一处理：特征提取（最耗时）→ 逐轮训练 → 评估，三段都实时回传。
    """

    def __init__(self, job_id: str, logs: list[str], stage: str = "",
                 weights: tuple[float, float] = (0.0, 1.0),
                 total_epochs: int = 0):
        self.job_id = job_id
        self.logs = logs
        self.stage = stage
        self.w0, self.w1 = weights
        self.total_epochs = total_epochs
        self._last_write = 0.0
        self._history: list[dict] = []
        # 跨轮累计的曲线：每轮训练时以 round 标签接续，
        # 否则第 2 轮一开始就把第 1 轮的曲线冲掉，用户看到曲线「断掉重画」
        self._all_history: list[dict] = []
        # 本轮 history 已消费到的位置：on_epoch 传进来的是「本轮到目前」的
        # 完整 history，不做去重会导致点数按 1+2+3+… 爆炸增长
        self._round_seen = 0
        # 已写出的最大进度：保证进度条只增不减
        self._max_progress = 0.0

    @property
    def all_history(self) -> list[dict]:
        """跨轮累计的训练曲线。"""
        return list(self._all_history)

    def _write(self, progress: float, metrics: dict | None = None) -> None:
        try:
            p = max(0.0, min(1.0, progress))
            # 进度必须单调不减：多轮迭代时 w0/w1 会重设，
            # 阶段切换会把进度拉回本轮区间起点，用户看到进度条倒退。
            p = max(p, self._max_progress)
            self._max_progress = p
            fields = {"progress": p}
            # 注意：db.update_job 里 metrics=None 会被写成 null 覆盖掉已有曲线，
            # 所以只在确实有内容时才传。
            if metrics:
                fields["metrics"] = metrics
            if self.total_epochs:
                fields["total_epochs"] = self.total_epochs
            db.update_job(self.job_id, **fields)
        except Exception:  # noqa: BLE001
            pass

    def stage_begin(self, text: str, frac: float = 0.0) -> None:
        """进入一个新阶段（如「提取特征」），写一条日志。"""
        self.stage = text
        self._write(self.w0 + (self.w1 - self.w0) * frac,
                    {"stage": self.stage, "stage_progress": 0.0,
                     "history": self._all_history})

    def feature_progress(self, done: int, total: int, elapsed: float) -> None:
        """特征提取进度：这是用户等待最久的阶段，必须有反馈。"""
        frac = (done / total) if total else 0.0
        # 特征提取占整个流程的 70%（经验值），映射到进度条的 0→0.7
        p = self.w0 + (self.w1 - self.w0) * frac * 0.7
        now = time.time()
        if now - self._last_write < 1.0 and done < total:
            return
        self._last_write = now
        eta = (elapsed / done * (total - done)) if done else 0
        self._write(p, {"stage": self.stage, "stage_progress": round(frac, 4),
                        "eta_seconds": int(eta), "history": self._all_history})

    def epoch(self, ep: int, total: int, history: list[dict], record: dict) -> None:
        """每轮训练结束：更新曲线与进度。"""
        self._history = list(history)
        frac = ep / total if total else 0.0
        # 训练占 0.7→0.95
        p = self.w0 + (self.w1 - self.w0) * (0.7 + 0.25 * frac)
        self.total_epochs = total

        # 跨轮累计曲线：保留前面轮次的数据，用全局序号接续。
        # 注意 on_epoch 传进来的是「本轮到目前为止的完整 history」，
        # 若每次都整体追加，第 2 轮会把第 1 轮的点重复写一遍，
        # 点数按 1+2+3+… 爆炸增长。这里只追加尚未消费过的新点。
        if len(history) <= self._round_seen:
            # 长度没有增长 → 说明进入了新一轮（history 重新从头开始）
            self._round_seen = 0
        for rec in history[self._round_seen:]:
            item = dict(rec)
            # 全局连续序号：前端 x 轴按此标注，跨轮也不会跳号
            item["epoch"] = len(self._all_history) + 1
            item["round_epoch"] = rec.get("epoch")
            self._all_history.append(item)
        self._round_seen = len(history)
        try:
            # epoch/total_epochs 表示「当前这一轮」的训练进度。
            # 多轮迭代时每轮方案不同，跨轮累计会让轮次数字忽大忽小。
            db.update_job(self.job_id, epoch=ep, total_epochs=total)
        except Exception:  # noqa: BLE001
            pass
        self._write(p, {"stage": self.stage, "stage_progress": round(frac, 4),
                        "history": self._all_history, "last_record": record,
                        "round_epoch": ep, "round_total": total})


# ---------- 分类头 ----------

def _iter_batches(n: int, bsz: int):
    """按 bsz 切批，保证每批至少 2 条样本。

    BatchNorm 在训练态下要求每批通道维长度 > 1，样本数不是 batch 的整数倍时
    最后一批常只剩 1 条，会直接抛
    「Expected more than 1 value per channel when training, got input size torch.Size([1, 256])」，
    整个训练任务就此失败（Day 2 实测）。余 1 条时并回上一批即可。
    """
    bsz = max(2, int(bsz))
    n = int(n)
    if n <= 0:
        return
    starts = list(range(0, n, bsz))
    if len(starts) > 1 and n - starts[-1] == 1:
        starts.pop()
    for k, st in enumerate(starts):
        en = starts[k + 1] if k + 1 < len(starts) else n
        yield st, en


class Head(nn.Module):
    """作用在冻结特征上的分类头。"""

    def __init__(self, din: int, hidden: int, dout: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dout),
        )

    def forward(self, x):
        return self.net(x)


class TransferNet(nn.Module):
    """冻结的预训练 backbone + 分类头。

    训练期只训 Head（输入是预提取好的特征），推理期需要 backbone + head 串联，
    所以这里提供一个与训练期结构完全一致的完整网络，供推理端原样重建。
    """

    def __init__(self, backbone: str, nclass: int, hidden: int,
                 dropout: float = 0.2, img_size: int | None = None):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim, w = _build_backbone(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.head = Head(feat_dim, hidden, nclass, dropout)
        self.weights = w
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = int(img_size or _BACKBONE_SPECS[backbone]["sizes"][dev])

    def infer_transform(self):
        """推理期预处理：必须与预训练一致（中心裁剪 + ImageNet 标准化）。"""
        return _val_tf(self.img_size, self.weights)

    def forward(self, x):
        return self.head(self.backbone(x))


def train_head(fts_tr, y_tr, fts_va, y_va, plan: dict, log=None,
               on_epoch=None) -> tuple[nn.Module, list[dict], str]:
    """在预提取特征上训练分类头，含余弦调度、早停与最佳权重回滚。

    on_epoch(epoch, total, history, record) —— 每轮结束回调，
    用于把进度与曲线实时推给前端（否则训练全程看不到任何变化）。
    """
    nclass = int(max(int(np.max(y_tr)), int(np.max(y_va)))) + 1
    din = int(np.asarray(fts_tr).shape[1])
    hidden = max(32, min(256, din // 2))
    model = Head(din, hidden, nclass, float(plan.get("dropout") or 0.2))

    epochs = max(1, min(int(plan.get("epochs") or 25), 100))
    bsz = max(8, min(int(plan.get("batch_size") or 32), 128))
    lr = float(plan.get("lr") or 2e-3)
    patience = int(plan.get("patience") or 8)
    ls = float(plan.get("label_smoothing") or 0.0)

    weight = None
    if plan.get("class_weight"):
        cnt = np.bincount(np.asarray(y_tr), minlength=nclass).astype(np.float32)
        weight = torch.from_numpy(cnt.sum() / np.maximum(cnt, 1.0) / nclass)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=float(plan.get("weight_decay") or 0.0))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    Xtr = torch.from_numpy(np.asarray(fts_tr, dtype=np.float32))
    Xva = torch.from_numpy(np.asarray(fts_va, dtype=np.float32))
    Ytr = torch.as_tensor(np.asarray(y_tr), dtype=torch.long)
    Yva = torch.as_tensor(np.asarray(y_va), dtype=torch.long)

    history: list[dict] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    best_ep, since = 0, 0
    stop = ""

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, seen = 0.0, 0
        for st, en in _iter_batches(len(perm), bsz):
            bi = perm[st:en]
            opt.zero_grad()
            loss = criterion(model(Xtr[bi]), Ytr[bi])
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
            seen += len(bi)
        tr_loss = tot / max(seen, 1)

        model.eval()
        with torch.no_grad():
            out_va = model(Xva)
            va_loss = float(criterion(out_va, Yva))
            va_acc = float((out_va.argmax(1) == Yva).float().mean())
            out_tr = model(Xtr)
            tr_acc = float((out_tr.argmax(1) == Ytr).float().mean())

        if va_loss < best - 1e-6:
            best, best_ep, since = va_loss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        sch.step()

        history.append({"epoch": ep, "train_loss": round(tr_loss, 5),
                        "train_acc": round(tr_acc, 5),
                        "val_loss": round(va_loss, 5),
                        "val_acc": round(va_acc, 5)})
        if log:
            log(f"epoch {ep}/{epochs} · loss {tr_loss:.4f} · "
                f"val_loss {va_loss:.4f} · val_acc {va_acc:.4f}")
        if on_epoch:
            try:
                on_epoch(ep, epochs, history, history[-1])
            except Exception:  # noqa: BLE001
                pass   # 进度回传失败不能影响训练本身

        if patience and since >= patience:
            stop = f"早停：验证损失连续 {patience} 轮未改善，于第 {ep} 轮终止"
            break
        if not np.isfinite(tr_loss) or tr_loss > 1e6:
            stop = f"训练中止：损失发散（{tr_loss:.4f}），请降低学习率"
            break

    if best_ep and best_ep < len(history):
        model.load_state_dict(best_state)
        if log:
            log(f"已回滚至第 {best_ep} 轮最佳权重")
    return model, history, stop


def _freeze_bn(module) -> None:
    """把 BatchNorm 切到 eval 模式：固定均值方差，只学仿射参数。

    小样本微调时 batch 统计噪声极大，BN 层必须冻结，否则预训练特征会被破坏。
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def finetune_backbone(root: Path, paths: list[str], y_all: np.ndarray,
                      ti: np.ndarray, vi: np.ndarray, plan: dict,
                      nclass: int, log=None, on_epoch=None,
                      on_batch=None):
    """解冻 backbone 末段做端到端微调。

    这是「直接把数据交给资深工程师」时最常用的做法，也是当前软件
    与之准确率差距的主要来源（实测同数据 +4pp）。

    与冻结提特征的取舍：
    - 冻结：特征可缓存，多次迭代几乎零成本
    - 微调：每轮都要重新前向，慢一些但更准
    """
    from PIL import Image
    from torchvision import models as tvm

    backbone = str(plan.get("backbone") or "mobilenet_v3_small")
    w = (tvm.MobileNet_V3_Small_Weights.DEFAULT
         if backbone == "mobilenet_v3_small" else tvm.ResNet18_Weights.DEFAULT)
    net, feat_dim, _ = _build_backbone(backbone)
    size = int(plan.get("img_size") or 160)
    epochs = max(1, min(int(plan.get("epochs") or 12), 60))
    bsz = max(4, min(int(plan.get("batch_size") or 16), 64))
    # backbone 末段的学习率要比分类头小两个数量级
    head_lr = float(plan.get("lr") or 2e-3)
    ft_lr = float(plan.get("finetune_lr") or 3e-4)
    patience = int(plan.get("patience") or 6)
    ls = float(plan.get("label_smoothing") or 0.0)
    drop = float(plan.get("dropout") or 0.2)

    for p in net.parameters():
        p.requires_grad_(False)
    if backbone == "mobilenet_v3_small":
        tail = net.features[-1]
    else:
        tail = net.layer4
    for p in tail.parameters():
        p.requires_grad_(True)

    head = Head(feat_dim, max(32, min(256, feat_dim // 2)), nclass, drop)
    model = nn.ModuleList([net, head])   # 便于统一取 state_dict / forward

    weight = None
    if plan.get("class_weight"):
        cnt = np.bincount(np.asarray(y_all[ti]), minlength=nclass).astype(np.float32)
        weight = torch.from_numpy(cnt.sum() / np.maximum(cnt, 1.0) / nclass)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)

    ft_params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(
        [{"params": ft_params, "lr": ft_lr},
         {"params": head.parameters(), "lr": head_lr}],
        weight_decay=float(plan.get("weight_decay") or 0.0))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    train_tf = _train_tf(size, w)
    val_tf = _val_tf(size, w)

    # 一次性把图片读进内存，避免每个 epoch 重复读盘
    vi_set = set(int(i) for i in vi)
    tr_items: list[tuple] = []
    va_items: list[tuple] = []
    for i, rel in enumerate(paths):
        try:
            with Image.open(root / rel) as im:
                arr = im.convert("RGB").copy()
        except Exception:  # noqa: BLE001
            continue
        if int(i) in vi_set:
            va_items.append((arr, int(y_all[i])))
        else:
            tr_items.append((arr, int(y_all[i])))

    if not tr_items or not va_items:
        raise ValueError("微调需要至少一条训练样本和一条验证样本")

    history: list[dict] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    best_ep, since = 0, 0
    stop = ""

    for ep in range(1, epochs + 1):
        net.train(); head.train()
        net.apply(_freeze_bn)     # BN 统计量冻结
        perm = np.random.permutation(len(tr_items))
        tot, seen, train_hit = 0.0, 0, 0
        for start, end in _iter_batches(len(perm), bsz):
            chunk = perm[start:end]
            xs = torch.stack([train_tf(tr_items[int(j)][0]) for j in chunk])
            ys = torch.tensor([tr_items[int(j)][1] for j in chunk], dtype=torch.long)
            opt.zero_grad()
            out = head(net(xs))
            loss = criterion(out, ys)
            loss.backward()
            opt.step()
            tot += float(loss) * len(chunk)
            # 顺便统计训练命中率（训练模式下有 dropout，是有偏估计，
            # 但能反映拟合趋势；留着 0 会让曲线图少一条线）
            train_hit += int((out.argmax(1) == ys).sum())
            seen += len(chunk)
            if on_batch:
                try:
                    on_batch(end, len(tr_items))
                except Exception:  # noqa: BLE001
                    pass
        tr_loss = tot / max(seen, 1)
        tr_acc = train_hit / max(seen, 1)

        net.eval(); head.eval()
        correct, vloss, vn = 0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(va_items), bsz):
                chunk = va_items[start:start + bsz]
                xs = torch.stack([val_tf(c[0]) for c in chunk])
                ys = torch.tensor([c[1] for c in chunk], dtype=torch.long)
                out = head(net(xs))
                vloss += float(criterion(out, ys)) * len(chunk)
                correct += int((out.argmax(1) == ys).sum())
                vn += len(chunk)
        va_loss = vloss / max(vn, 1)
        va_acc = correct / max(vn, 1)

        if va_loss < best - 1e-6:
            best, best_ep, since = va_loss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        sch.step()

        history.append({"epoch": ep, "train_loss": round(tr_loss, 5),
                        "train_acc": round(tr_acc, 5),
                        "val_loss": round(va_loss, 5),
                        "val_acc": round(va_acc, 5)})
        if log:
            log(f"epoch {ep}/{epochs} · loss {tr_loss:.4f} · "
                f"val_loss {va_loss:.4f} · val_acc {va_acc:.4f}")
        if on_epoch:
            try:
                on_epoch(ep, epochs, history, history[-1])
            except Exception:  # noqa: BLE001
                pass

        if patience and since >= patience:
            stop = f"早停：验证损失连续 {patience} 轮未改善，于第 {ep} 轮终止"
            break
        if not np.isfinite(tr_loss) or tr_loss > 1e6:
            stop = f"训练中止：损失发散（{tr_loss:.4f}），请降低学习率"
            break

    if best_ep and best_ep < len(history):
        model.load_state_dict(best_state)
        if log:
            log(f"已回滚至第 {best_ep} 轮最佳权重")
    return model, head, net, history, stop


# ---------- 评估 ----------

@torch.no_grad()
def _eval_transfer(net, head, root: Path, paths: list[str], idx: np.ndarray,
                   y_all: np.ndarray, nclass: int, plan: dict) -> dict:
    """对「已微调的 backbone + head」做评估。

    与 evaluate() 的区别：evaluate 吃的是已缓存的特征向量，
    这里要现场跑一遍 backbone 前向。
    """
    from PIL import Image
    from torchvision import models as tvm
    backbone = str(plan.get("backbone") or "mobilenet_v3_small")
    w = (tvm.MobileNet_V3_Small_Weights.DEFAULT
         if backbone == "mobilenet_v3_small" else tvm.ResNet18_Weights.DEFAULT)
    size = int(plan.get("img_size") or 160)
    tf = _val_tf(size, w)
    net.eval(); head.eval()

    feats = []
    ys = []
    for i in idx:
        i = int(i)
        try:
            with Image.open(root / paths[i]) as im:
                x = tf(im.convert("RGB")).unsqueeze(0)
        except Exception:  # noqa: BLE001
            continue
        feats.append(net(x).float().numpy()[0])
        ys.append(int(y_all[i]))
    if not feats:
        raise ValueError("评估集为空或全部图像无法读取")
    X = np.stack(feats).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return evaluate(head, X, y, nclass)


def evaluate(model, feats: np.ndarray, y: np.ndarray, nclass: int,
             classes: list[str] | None = None) -> dict:
    """输出准确率、宏 F1、QWK、每类 P/R/F1 与混淆矩阵。"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(feats, dtype=np.float32)))
        pred = logits.argmax(1).numpy()
    y = np.asarray(y)
    acc = float((pred == y).mean()) if len(y) else 0.0
    cm = np.zeros((nclass, nclass), dtype=int)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t][p] += 1
    per_class, f1s = [], []
    for c in range(nclass):
        tp = int(cm[c][c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
        per_class.append({"class": c, "support": int(cm[c, :].sum()),
                          "precision": round(prec, 4),
                          "recall": round(rec, 4),
                          "f1": round(f1, 4)})
    # 有序分级（眼底 DR、糖网分期）要看二次加权 kappa，
    # macro-F1 会把「判成相邻一级」与「判成最重一级」同等惩罚，与临床风险不符。
    import metrics_extra as mx
    return {"acc": round(acc, 5),
            "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 5),
            "qwk": mx.quadratic_kappa(y, pred, nclass),
            "per_class": mx.name_per_class(per_class, classes),
            "confusion_matrix": cm.tolist()}


# ---------- 诊断 ----------

def diagnose(history: list[dict], n: int) -> dict:
    """根据训练曲线判断过拟合 / 欠拟合，产出下一轮调整建议。"""
    if not history:
        return {"verdict": "unknown", "hint": ""}
    last = history[-1]
    tr_acc = float(last.get("train_acc") or 0)
    va_acc = float(last.get("val_acc") or 0)
    gap = tr_acc - va_acc

    if va_acc >= 0.92 and gap < 0.08:
        return {"verdict": "good", "gap": round(gap, 4),
                "hint": "验证指标已达标且无明显过拟合，无需继续迭代"}
    if gap >= 0.15:
        return {"verdict": "overfit", "gap": round(gap, 4),
                "hint": "训练集远高于验证集（过拟合）：增强加强一档、增大 dropout "
                        "与 weight_decay、减少 epochs、换更小的 backbone"}
    if tr_acc < 0.6:
        return {"verdict": "underfit", "gap": round(gap, 4),
                "hint": "训练集本身就没收敛（欠拟合）：换更大的 backbone、"
                        "提高输入尺寸、增大 lr 与 epochs、减小正则"}
    return {"verdict": "plateau", "gap": round(gap, 4),
            "hint": "指标尚可但有提升空间：换更强 backbone 或提高输入尺寸再试一轮"}


# ---------- 主流程 ----------

def auto_train(job_id: str) -> None:
    """智能体自动建模总入口：按任务类型分派到对应的建模器。

    划分全程使用解析阶段锁定的 split 列，测试集不参与任何训练与调参。
    """
    job = db.get_job(job_id)
    d = db.get_dataset(job["dataset_id"])
    if not d:
        db.update_job(job_id, status="failed", error="数据集不存在")
        return
    task_hint = d.get("task_hint") or "tabular_classification"

    if task_hint == "image_classification":
        _auto_train_image(job_id, job, d)
    elif task_hint == "object_detection":
        _auto_train_detection(job_id, job, d)
    elif task_hint == "text_classification":
        _auto_train_text(job_id, job, d)
    elif task_hint in ("tabular_classification", "tabular_regression"):
        _auto_train_tabular(job_id, job, d)
    elif task_hint == "signal_classification":
        import automl_signal as asig
        asig.auto_train_signal(job_id, job, d)
    else:
        db.update_job(job_id, status="failed",
                      error=f"暂不支持的任务类型：{task_hint}")


def _auto_train_image(job_id: str, job: dict, d: dict) -> None:
    """图像分类：冻结预训练 backbone 提特征 + 训分类头 + 自动迭代。"""
    import train as trainer

    cfg = job["config"]
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        db.update_job(job_id, logs=logs)

    try:
        db.update_job(job_id, status="running")
        # 立刻给一个非零进度与阶段说明 —— 否则前几秒前端显示 0%，
        # 用户会以为卡住了
        prog = Progress(job_id, logs, total_epochs=0)
        prog.stage_begin("正在读取数据并做体检…", frac=0.02)
        log("智能体接管：开始数据体检…")

        task_hint = d.get("task_hint") or "tabular_classification"
        df = pd.read_csv(d["file_path"])
        meta = d.get("meta") or {}

        card = inspect(df, task_hint, meta)
        log(f"数据卡：{card['rows']} 条样本 · {card.get('n_classes', 0)} 个类别 · "
            f"划分 {card['splits']}")
        # 开训前拦截整批坏图：否则会用均值特征硬训出一个随机水平的模型，
        # 表面「训练完成」实则完全无效，比直接报错更糟。
        if (card.get("bad_ratio") or 0) > 0.2:
            raise ValueError(
                f"图像文件不可用：抽查 {card.get('probed_images', 0)} 张中 "
                f"{card.get('bad_images', 0)} 张无法读取，请重新上传数据集")
        if card.get("bad_images"):
            log(f"提示：抽查发现 {card.get('bad_images')} 张图像无法解码，已跳过")

        splits = trainer.resolve_splits(df, float(cfg.get("val_split") or 0.2))
        ti, vi, te = splits["train"], splits["val"], splits["test"]
        log(f"数据划分：{splits['note']} —— 训练 {len(ti)} · 验证 {len(vi)}"
            + (f" · 测试 {len(te)}（不参与训练）" if len(te) else ""))
        if len(ti) == 0:
            raise ValueError("划分后训练集为空，请检查数据集的 split 标注")
        if len(vi) == 0:
            raise ValueError("划分后验证集为空，请检查数据集的 split 标注")
        if len(te):
            leak = (set(ti.tolist()) | set(vi.tolist())) & set(te.tolist())
            if leak:
                raise ValueError(
                    f"划分异常：{len(leak)} 个测试集样本同时出现在训练/验证集中，已中止训练")
            log(f"已校验：{len(te)} 个测试集样本与训练/验证集零交集")

        classes = sorted(df["label"].astype(str).unique().tolist())
        cidx = {v: i for i, v in enumerate(classes)}
        y_all = df["label"].astype(str).map(cidx).values.astype(np.int64)
        root = Path(meta["image_root"])
        all_paths = [str(p) for p in df["filepath"]]

        wdir = Path(db.STORAGE_DIR) / f"_work_{d['id']}"
        wdir.mkdir(parents=True, exist_ok=True)

        rounds: list[dict] = []
        best: dict | None = None
        feedback: str | None = None
        prev_verdict: str | None = None
        last_plan: dict | None = None

        for r in range(1, MAX_ROUNDS + 1):
            plan = decide(card, feedback, verdict=prev_verdict, prev_plan=last_plan)
            # 用户给的轮次是「上限」而不是建议值：
            # 图像分类路径原先直接采用方案的 20/25 轮，用户在配置里写 8 轮
            # 也会被无视，成本和等待时间完全对不上。这里按上限截断。
            if cfg.get("epochs"):
                plan["epochs"] = max(1, min(int(plan.get("epochs") or 20),
                                            int(cfg["epochs"])))
            mode = plan.get("mode") or "frozen"
            mode_txt = "微调主干网络" if mode == "finetune" else "冻结主干只训分类头"
            log(f"第 {r}/{MAX_ROUNDS} 轮方案：{plan['backbone']} · "
                f"输入 {plan['img_size']}px · 增强 {plan['augment']}×{plan['aug_copies']} · "
                f"{plan['epochs']} 轮 · {mode_txt}"
                + (f" · {plan['reason']}" if plan.get("reason") else ""))

            # 本轮在总进度条上的区间
            prog.w0 = (r - 1) / MAX_ROUNDS
            prog.w1 = r / MAX_ROUNDS

            if mode == "finetune":
                # 微调：端到端训练，主干末段一起更新
                prog.stage_begin(f"第 {r}/{MAX_ROUNDS} 轮：准备微调数据", frac=0.0)
                prog.stage_begin(f"第 {r}/{MAX_ROUNDS} 轮：微调训练中", frac=0.25)
                model, head, net, history, stop = finetune_backbone(
                    root, all_paths, y_all, ti, vi, plan, len(classes),
                    log=log, on_epoch=prog.epoch)
                # 统一用 head 作为「模型主体」，与冻结路径保持一致，
                # backbone 权重单独存（微调时 backbone 也被更新了）
                model = head
                # 评估要用完整网络（backbone + head）
                ev = _eval_transfer(net, head, root, all_paths, vi, y_all,
                                    len(classes), plan)
                diag = diagnose(history, len(y_all))
                test_ev = None
                if len(te):
                    te_ev = _eval_transfer(net, head, root, all_paths, te, y_all,
                                           len(classes), plan)
                    test_ev = {"test_acc": te_ev["acc"],
                               "test_samples": int(len(te)),
                               "macro_f1": te_ev["macro_f1"],
                               # QWK 必须一并带出：分级任务的行业标准就是它，
                               # 只在验证集算、测试集不算，等于对外报不出可信指标（Day 2 实测）
                               "qwk": te_ev.get("qwk"),
                               "per_class": te_ev["per_class"],
                               "confusion_matrix": te_ev["confusion_matrix"]}
                    log(f"测试集评估（全程未参与训练）："
                        f"acc {te_ev['acc']} · macro_f1 {te_ev['macro_f1']}"
                        + (f" · qwk {te_ev['qwk']}" if te_ev.get("qwk") is not None else ""))
            else:
                # 冻结：提特征后只训分类头，特征可缓存复用
                prog.stage_begin(f"第 {r}/{MAX_ROUNDS} 轮：提取图像特征", frac=0.0)

                cache = wdir / ("feat_%s_%d_%d_%d.npz" % (
                    plan["backbone"], int(plan["img_size"]),
                    int(plan["aug_copies"]), len(all_paths)))
                feats = extract_features(root, all_paths, plan, cache, log=log,
                                         on_progress=prog.feature_progress)

                ftr, ytr = feats[ti], y_all[ti]
                fvs, yvs = feats[vi], y_all[vi]

                prog.stage_begin(f"第 {r}/{MAX_ROUNDS} 轮：训练中", frac=0.7)
                model, history, stop = train_head(ftr, ytr, fvs, yvs, plan, log=log,
                                                 on_epoch=prog.epoch)
                ev = evaluate(model, fvs, yvs, len(classes), classes=classes)
                diag = diagnose(history, len(y_all))

                test_ev = None
                if len(te):
                    test_ev = evaluate(model, feats[te], y_all[te], len(classes), classes=classes)
                    # 键名与常规训练路径一致（test_acc / test_samples）
                    test_ev = {"test_acc": test_ev["acc"],
                               "test_samples": int(len(te)),
                               "macro_f1": test_ev["macro_f1"],
                           "qwk": test_ev.get("qwk"),
                           "per_class": test_ev.get("per_class"),
                           "confusion_matrix": test_ev.get("confusion_matrix")}
                log("测试集评估（全程未参与训练）："
                    + f"acc {test_ev['test_acc']} · macro_f1 {test_ev['macro_f1']}"
                    + (f" · qwk {test_ev['qwk']}" if test_ev.get("qwk") is not None else ""))

            rec = {"round": r, "plan": plan, "val": ev, "test": test_ev,
                   "history": history, "verdict": diag["verdict"],
                   "hint": diag.get("hint", ""), "stop_reason": stop,
                   "model": model}
            # 微调模式下 backbone 权重也被更新了，必须连同 head 一起保存，
            # 否则推理时会用回预训练权重，准确率凭空掉一截。
            if (plan.get("mode") or "frozen") == "finetune":
                rec["backbone_state"] = {k: v.detach().clone()
                                         for k, v in net.state_dict().items()}
            rounds.append(rec)

            # 选优：只看验证集准确率会被小验证集的波动带偏
            #（126 张的验证集上，差 0.8% 只差 1 个样本）。
            # 用「准确率 + macro_F1」的综合分来选，避免选到只对多数类好的方案。
            def _score(e: dict) -> float:
                if not e:
                    return -1.0
                return 0.7 * float(e.get("acc") or 0) + 0.3 * float(e.get("macro_f1") or 0)

            if best is None or _score(ev) > _score(best["val"]):
                best = rec

            if diag["verdict"] == "good" or r >= MAX_ROUNDS:
                break
            feedback = (
                f"上一轮验证准确率 {ev['acc']}，训练集准确率 "
                f"{history[-1].get('train_acc')}，判定为{diag['verdict']}。{diag['hint']}"
            )
            prev_verdict = diag["verdict"]
            last_plan = plan
            log(f"诊断：{diag['verdict']} —— 调整方案后重训")

        if best is None:
            raise ValueError("自动建模未产出可用模型")

        # 迁移学习产物 = 冻结 backbone（按名从 torchvision 取权重）+ 训练好的分类头。
        # 只存分类头权重即可，backbone 权重由 torchvision 缓存提供，模型文件小得多。
        m = best["model"]
        p = best["plan"]
        split_info = {"source": splits["source"], "note": splits["note"],
                      "train": int(len(ti)), "val": int(len(vi)),
                      "test": int(len(te))}
        # 迭代记录里挂着模型对象与 Tensor 权重，落 JSON 前先剔掉
        # （backbone_state 是 Tensor 字典，JSON 序列化会直接报错）
        rounds_flat = [{k: v for k, v in x.items()
                        if k not in ("model", "backbone_state", "_key")}
                       for x in rounds]

        # best_val 的键名必须与常规训练路径保持一致（含 val_acc）——
        # 前端统一按 val_acc 取值，键名不一致会取到 None。
        hb = dict(best["history"][-1]) if best.get("history") else {}
        # 准确率必须用「回滚到最佳权重后」的评估值，而不是最后一轮的 history：
        # 早停会让最后一轮变差，用它展示会让用户看到的数字低于实际模型水平。
        bval = best.get("val") or {}
        hb["acc"] = bval.get("acc", hb.get("val_acc"))
        hb["val_acc"] = bval.get("acc", hb.get("val_acc"))
        hb["macro_f1"] = bval.get("macro_f1")
        hb["per_class"] = bval.get("per_class")
        hb["confusion_matrix"] = bval.get("confusion_matrix")
        hb["epoch"] = (best.get("history") or [{}])[-1].get("epoch", 1)

        # 曲线用跨轮累计的版本：用户看到的是一条完整的训练轨迹，
        # 而不是最后一轮孤零零的十几根线
        metrics = {"history": prog.all_history or best["history"],
                   "task": "classification",
                   "arch": "transfer_head", "best_val": hb,
                   "test_metrics": best["test"] or {}, "config": p,
                   "datacard": card, "rounds": rounds_flat,
                   "advice": _plain_advice(
                       {"verdict": best.get("verdict", "unknown"),
                        "hint": best.get("hint", "")}, len(y_all)),
                   "split_info": split_info}

        model_path = trainer.MODELS_DIR / f"{job_id}.pt"
        is_ft = (p.get("mode") or "frozen") == "finetune"
        ckpt = {
            "state_dict": m.state_dict(),
            "meta": {
                "task": "classification",
                "arch": "transfer_head",
                "task_hint": task_hint,
                "backbone": p["backbone"],
                "mode": p.get("mode") or "frozen",
                "prep": {"classes": classes, "img_size": int(p["img_size"])},
                "target_col": d.get("target_col") or "label",
                "dataset_id": d["id"],
                "dataset_name": d["name"],
                "arch_params": {"hidden": int(m.net[0].out_features),
                                "dropout": float(p.get("dropout") or 0.2),
                                "img_size": int(p["img_size"])},
                "config": p,
                "datacard": card,
                "split_info": split_info,
                "test_metrics": best["test"] or {},
                "rounds": rounds_flat,
            },
        }
        if is_ft and best.get("backbone_state"):
            # 微调过的主干权重必须一起存，否则推理时会退回预训练权重，
            # 准确率会凭空掉一大截（实测差 4 个百分点以上）。
            ckpt["backbone_state"] = best["backbone_state"]
        torch.save(ckpt, model_path)
        log(f"自动建模完成：共 {len(rounds)} 轮，采用第 {best['round']} 轮方案"
            f"（{p['backbone']} · {p['img_size']}px · "
            f"{'微调主干' if is_ft else '冻结主干'}），模型已保存：{model_path.name}")
        db.update_job(job_id, status="done", progress=1.0,
                      model_path=str(model_path), metrics=metrics)
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=str(e)[:500],
                      logs=logs + [f"自动建模失败：{e}"])


# ---------- 目标检测 ----------

def _auto_train_detection(job_id: str, job: dict, d: dict) -> None:
    """目标检测：YOLO 格式目录 + ultralytics 训练 + mAP 评估。

    划分同样以解析阶段锁定的 split 列为准，测试集不参与训练。
    """
    import automl_detect as ad
    import train as trainer

    cfg = job["config"]
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        db.update_job(job_id, logs=logs)

    try:
        db.update_job(job_id, status="running")
        prog = Progress(job_id, logs, total_epochs=0)
        prog.stage_begin("正在读取标注数据…", frac=0.02)
        log("智能体接管：目标检测任务，开始准备 YOLO 数据结构…")

        df = pd.read_csv(d["file_path"])
        meta = d.get("meta") or {}
        classes = list(meta.get("classes") or [])
        if not classes:
            raise ValueError("数据集未记录检测类别，无法训练检测模型")
        if "boxes" not in df.columns:
            raise ValueError("数据集缺少边界框标注（boxes 列），无法用于目标检测训练")

        splits = trainer.resolve_splits(df, float(cfg.get("val_split") or 0.2))
        ti, vi, te = splits["train"], splits["val"], splits["test"]
        log(f"数据划分：{splits['note']} —— 训练 {len(ti)} · 验证 {len(vi)}"
            + (f" · 测试 {len(te)}（不参与训练）" if len(te) else ""))
        if len(ti) == 0:
            raise ValueError("划分后训练集为空，请检查数据集的 split 标注")
        if len(vi) == 0:
            raise ValueError("划分后验证集为空，请检查数据集的 split 标注")
        if len(te):
            leak = (set(ti.tolist()) | set(vi.tolist())) & set(te.tolist())
            if leak:
                raise ValueError(
                    f"划分异常：{len(leak)} 个测试集样本同时出现在训练/验证集中，已中止训练")
            log(f"已校验：{len(te)} 个测试集样本与训练/验证集零交集")

        root = Path(meta["image_root"])
        wdir = Path(db.STORAGE_DIR) / f"_work_{d['id']}"
        wdir.mkdir(parents=True, exist_ok=True)
        ydir = wdir / "yolo"
        prep = ad.prepare_yolo_dirs(df, root, ydir, classes)
        log(f"YOLO 目录就绪：train {prep['counts']['train']} 张"
            f"（{prep['boxes']['train']} 个框） · val {prep['counts']['val']} 张"
            f"（{prep['boxes']['val']} 个框）"
            + (f" · test {prep['counts']['test']} 张" if prep["counts"]["test"] else ""))

        plan = ad.detection_plan({}, len(df))
        # 前端可指定检测模型与轮次（auto 模式下由 auto_params 决定）
        if cfg.get("det_model"):
            plan["model"] = str(cfg["det_model"])
        if cfg.get("auto"):
            from train import auto_params
            ap = auto_params("object_detection", len(df))
            plan["epochs"] = int(ap["epochs"])
            plan["batch"] = int(ap["batch_size"])
        elif cfg.get("epochs"):
            plan["epochs"] = int(cfg["epochs"])
        if cfg.get("batch_size"):
            plan["batch"] = int(cfg["batch_size"])
        log(f"方案：{plan['model']} · {plan['epochs']} 轮 · {plan['imgsz']}px "
            f"· batch {plan['batch']} · {plan['reason']}")

        prog.stage_begin("准备 YOLO 训练数据", frac=0.05)
        prog.stage_begin("YOLO 训练中", frac=0.1)

        res = ad.train_detection(prep["data_yaml"], ydir,
                                 epochs=int(plan["epochs"]),
                                 model_name=str(plan["model"]),
                                 imgsz=int(plan["imgsz"]),
                                 batch=int(plan["batch"]),
                                 device=str(plan["device"]), log=log,
                                 on_epoch=prog.epoch)
        model = res["model"]
        metrics = res["metrics"]

        test_metrics = {}
        if len(te) and prep["counts"]["test"]:
            test_metrics = ad.evaluate_detection(
                model, prep["data_yaml"], split="test",
                imgsz=int(plan["imgsz"]), device=str(plan["device"]), log=log)

        split_info = {"source": splits["source"], "note": splits["note"],
                      "train": int(len(ti)), "val": int(len(vi)),
                      "test": int(len(te))}
        hb = {"epoch": plan["epochs"]}
        hb.update({k: round(float(v), 5) for k, v in metrics.items()
                   if isinstance(v, (int, float))})
        # 键名与其它路径统一（前端按 val_acc 取），检测任务用 mAP50 充当
        if "metrics/mAP50(B)" in hb:
            hb["val_acc"] = hb["metrics/mAP50(B)"]
        if "metrics/precision(B)" in hb:
            hb["train_acc"] = hb["metrics/precision(B)"]
        det_history = res.get("history") or [{"epoch": plan["epochs"], **hb}]
        job_metrics = {"history": det_history,
                       "task": "detection", "arch": "yolo",
                       "best_val": hb, "test_metrics": test_metrics,
                       "advice": _detection_advice(metrics, len(df),
                                                  int(prep["boxes"]["train"])),
                       "config": plan, "datacard": {"rows": len(df),
                                                    "n_classes": len(classes)},
                       "split_info": split_info,
                       "rounds": [{"round": 1, "plan": plan, "val": hb,
                                   "test": test_metrics}]}

        model_path = trainer.MODELS_DIR / f"{job_id}.pt"
        try:
            model.save(str(model_path))
        except Exception:  # noqa: BLE001
            model_path = trainer.MODELS_DIR / f"{job_id}_yolo.pt"
            model.save(str(model_path))
        log(f"检测训练完成：模型已保存 {model_path.name}")
        db.update_job(job_id, status="done", progress=1.0,
                      model_path=str(model_path), metrics=job_metrics)
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=str(e)[:500],
                      logs=logs + [f"检测训练失败：{e}"])


def _tabular_advice(task: str, val: dict, n: int) -> dict:
    """表格任务的诊断建议（分类看准确率，回归看 R²）。"""
    if task == "classification":
        acc = val.get("acc")
        ncls = len(val.get("per_class") or []) or 1
        if acc is None:
            return {"level": "ok", "summary": "训练完成。",
                    "plain_summary": "训练完成。", "plain_tips": []}
        base = 100.0 / max(ncls, 1)
        pct = acc * 100
        if pct >= 90:
            return {"level": "ok",
                    "summary": f"准确率 {pct:.2f}%，效果良好。",
                    "plain_summary": f"准确率 {pct:.1f}%，在你的数据上判断得很准。",
                    "plain_tips": []}
        if pct > base + 20:
            return {"level": "ok",
                    "summary": f"准确率 {pct:.2f}%，明显优于随机基线 {base:.1f}%。",
                    "plain_summary": f"准确率 {pct:.1f}%，比瞎猜（{base:.0f}%）好很多，"
                                     "可以用来做实际判断。",
                    "plain_tips": []}
        if pct > base + 5:
            return {"level": "warn",
                    "summary": f"准确率 {pct:.2f}%，仅略高于随机基线 {base:.1f}%。",
                    "plain_summary": f"准确率 {pct:.1f}%，只比瞎猜（{base:.0f}%）好一点。"
                                     "多半是数据里缺少可学的规律，或者某个特征没有真正用上。",
                    "plain_tips": ["检查各列与答案之间是否真的有关系",
                                   "确认答案列没选错",
                                   "可以再补一些数据"]}
        return {"level": "bad",
                "summary": f"准确率 {pct:.2f}%，与随机基线 {base:.1f}% 相当。",
                "plain_summary": f"准确率只有 {pct:.1f}%，跟瞎猜（{base:.0f}%）差不多——"
                                 "模型基本没学到东西。多半是数据本身缺少规律。",
                "plain_tips": ["确认答案列填对了",
                               "检查数据里各列与答案是否真的相关",
                               "如果数据是人工构造的，可能本身就没有规律"]}

    r2 = val.get("r2")
    if r2 is None:
        return {"level": "ok", "summary": "训练完成。",
                "plain_summary": "训练完成。", "plain_tips": []}
    if r2 >= 0.8:
        return {"level": "ok", "summary": f"R²={r2:.4f}，拟合效果良好。",
                "plain_summary": f"预测效果很好（R² = {r2:.2f}），数值预测相当准。",
                "plain_tips": []}
    if r2 >= 0.4:
        return {"level": "ok", "summary": f"R²={r2:.4f}，拟合效果尚可。",
                "plain_summary": f"预测效果还行（R² = {r2:.2f}），可以参考使用。",
                "plain_tips": []}
    if r2 >= 0:
        return {"level": "warn", "summary": f"R²={r2:.4f}，预测能力有限。",
                "plain_summary": f"预测能力有限（R² = {r2:.2f}），预测值只能当参考。",
                "plain_tips": ["看看是不是漏掉了重要的特征列",
                               "确认答案列本身是否适合用数值回归"]}
    return {"level": "bad", "summary": f"R²={r2:.4f}，模型未学到有效规律。",
            "plain_summary": f"模型基本没学到东西（R² = {r2:.2f}，比直接猜平均值还差）。"
                             "很可能数据本身就没有可学的规律。",
            "plain_tips": ["确认答案列与特征列之间确实有数学关系",
                           "检查数据是不是随机生成的"]}


def _plain_advice(diag: dict, n: int) -> dict:
    """把自动建模的诊断结论转成面向小白的两套文案。

    automl 路径绕过了 train.diagnose（那套产出的是技术口径），
    这里补一份同样格式的「plain_*」，前端优先展示它。
    """
    verdict = diag.get("verdict", "unknown")
    hint = diag.get("hint", "")
    table = {
        "good": {"level": "ok",
                 "summary": "模型效果达标，且没有明显过拟合。",
                 "plain_summary": "模型效果不错，也没有「背答案」的迹象，可以放心用。",
                 "plain_tips": []},
        "overfit": {"level": "warn",
                    "summary": "训练集表现远好于验证集，存在过拟合。",
                    "plain_summary": "模型把训练数据记住了，但换新数据就认不出——"
                                     "这种「死记硬背」在数据少时最常见。",
                    "plain_tips": ["增加一些数据是最有效的办法",
                                   "在「高级调优」里把防过拟合的强度调高",
                                   "训练轮次可以减半"]},
        "underfit": {"level": "warn",
                     "summary": "训练集本身尚未收敛，模型容量不足。",
                     "plain_summary": "模型还没学够——连见过的训练数据都认不全。"
                                      "可以给它更多时间，或换个更能干的模型。",
                     "plain_tips": ["把训练轮次调多一些",
                                    "在「高级调优」里换个更大的模型"]},
        "plateau": {"level": "warn",
                    "summary": "指标尚可但已趋于平台期。",
                    "plain_summary": "效果还可以，但再练下去也不太会变好了。"
                                     "换个大点的模型或把数据补一补，可能还有提升空间。",
                    "plain_tips": ["在「高级调优」里换个更���的模型试试",
                                   "补充更多数据"]},
    }
    out = dict(table.get(verdict, {
        "level": "ok", "summary": "", "plain_summary": "训练完成。",
        "plain_tips": []}))
    if hint:
        out["hint"] = hint
    if n < 50 and verdict not in ("good",):
        out["plain_tips"] = list(out.get("plain_tips") or []) + [
            f"目前只有 {n} 条数据，结果波动会比较大，尽量多准备一些"]
    return out


def _detection_advice(metrics: dict, n_images: int, n_boxes: int) -> dict:
    """检测任务的诊断建议（同样给两套文案：技术版 + 大白话版）。"""
    map50 = metrics.get("metrics/mAP50(B)") or 0.0
    rec = metrics.get("metrics/recall(B)") or 0.0
    prec = metrics.get("metrics/precision(B)") or 0.0

    if map50 >= 0.7:
        return {"level": "ok",
                "summary": f"检测效果良好：mAP50={map50:.4f}，P={prec:.4f}，R={rec:.4f}。",
                "plain_summary": f"检测效果不错，mAP50 达到 {map50 * 100:.1f}%。"
                                 "目标基本都能认准、框也够准。",
                "tips": ["可以尝试 YOLOv8s 进一步提升精度"],
                "plain_tips": ["想再准一点，可以在「高级调优」里换成更���的 YOLOv8s"]}
    if map50 >= 0.45:
        weak = ("漏掉的目标偏多" if rec < prec else "认出来的目标不够准")
        return {"level": "warn",
                "summary": f"检测效果一般：mAP50={map50:.4f}，召回率 {rec:.4f} 偏低，存在漏检。",
                "plain_summary": f"检测效果一般，mAP50 是 {map50 * 100:.1f}%。"
                                 f"主要问题是{weak}。",
                "tips": ["增加训练轮次", "确认标注框没有明显漏标",
                         "目标很小的话可提高 imgsz"],
                "plain_tips": ["多训练一些轮次",
                               "检查一下标注：有没有框漏标或标错位置",
                               "如果目标在图里很小，可以把输入尺寸调大"]}
    if n_images < 200:
        return {"level": "bad",
                "summary": f"检测效果偏低：mAP50={map50:.4f}，且标注图仅 {n_images} 张。",
                "plain_summary": f"检测效果偏差（mAP50 只有 {map50 * 100:.1f}%），"
                                 f"而且标注图只有 {n_images} 张 —— 训练数据太少是主要原因。",
                "tips": [f"当前仅 {n_images} 张标注图，建议补充到 500 张以上",
                         "确认标注质量：框的位置与类别是否正确"],
                "plain_tips": [f"现在只有 {n_images} 张标注图，建议补到 500 张以上",
                               "确认框画得准、类别标得对"]}
    return {"level": "bad",
            "summary": f"检测效果偏低：mAP50={map50:.4f}。",
            "plain_summary": f"检测效果偏差（mAP50 只有 {map50 * 100:.1f}%）。"
                             "可能是标注数据太少，或者标注本身有问题。",
            "tips": ["增加标注数据量", "检查标注框的准确度",
                     "尝试提高输入尺寸（imgsz）"],
            "plain_tips": ["多准备一些标注数据",
                           "检查框画得准不准",
                           "可以把输入尺寸调大再试"]}


# ---------- 文本分类 ----------

def _auto_train_text(job_id: str, job: dict, d: dict) -> None:
    """文本分类：预训练语言模型提句向量 + 训分类头 + 自动迭代。"""
    import automl_text as at
    import train as trainer

    cfg = job["config"]
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        db.update_job(job_id, logs=logs)

    try:
        db.update_job(job_id, status="running")
        # 立刻给一个非零进度与阶段说明 —— 否则前几秒前端显示 0%，
        # 用户会以为卡住了
        prog = Progress(job_id, logs, total_epochs=0)
        prog.stage_begin("正在读取数据并做体检…", frac=0.02)
        log("智能体接管：开始数据体检…")

        df = pd.read_csv(d["file_path"])
        meta = d.get("meta") or {}
        text_col = meta.get("text_col")
        if not text_col or text_col not in df.columns:
            # 历史数据可能没记 text_col —— 现场识别一次，别让整批数据白费
            text_col = _guess_text_col(df, d.get("target_col"))
            if not text_col:
                raise ValueError("未找到文本正文列，无法训练文本分类模型")
            log(f"提示：数据集未记录正文列，已自动识别为「{text_col}」")

        card = inspect(df, "text_classification", meta)
        card["text_col"] = text_col
        log(f"数据卡：{card['rows']} 条样本 · {card.get('n_classes', 0)} 个类别 · "
            f"正文列「{text_col}」")

        splits = trainer.resolve_splits(df, float(cfg.get("val_split") or 0.2))
        ti, vi, te = splits["train"], splits["val"], splits["test"]
        log(f"数据划分：{splits['note']} —— 训练 {len(ti)} · 验证 {len(vi)}"
            + (f" · 测试 {len(te)}（不参与训练）" if len(te) else ""))
        if len(ti) == 0:
            raise ValueError("划分后训练集为空，请检查数据集的 split 标注")
        if len(vi) == 0:
            raise ValueError("划分后验证集为空，请检查数据集的 split 标注")
        if len(te):
            leak = (set(ti.tolist()) | set(vi.tolist())) & set(te.tolist())
            if leak:
                raise ValueError(
                    f"划分异常：{len(leak)} 个测试集样本同时出现在训练/验证集中，已中止训练")
            log(f"已校验：{len(te)} 个测试集样本与训练/验证集零交集")

        classes = sorted(df[card.get("target_col") or d.get("target_col") or "label"]
                         .astype(str).unique().tolist())
        target = card.get("target_col") or d.get("target_col") or "label"
        classes = sorted(df[target].astype(str).unique().tolist())
        cidx = {v: i for i, v in enumerate(classes)}
        y_all = df[target].astype(str).map(cidx).values.astype(np.int64)
        texts = df[text_col].astype(str).tolist()

        plan = at.text_baseline_plan(card, texts)
        st = at.ensure_encoder(plan["encoder"])
        if not st["ok"]:
            raise ValueError(f"文本编码器不可用：{st['detail']}")
        log(f"方案：{at.TEXT_ENCODERS[plan['encoder']]['label']} · "
            f"{plan['epochs']} 轮 · lr {plan['lr']}")

        wdir = Path(db.STORAGE_DIR) / f"_work_{d['id']}"
        wdir.mkdir(parents=True, exist_ok=True)
        cache = wdir / f"txt_{plan['encoder']}_{len(texts)}.npz"

        prog.stage_begin("读取文本并提取语义特征", frac=0.0)

        feats = None
        if cache.exists():
            feats = np.load(cache)["f"]
            log(f"复用文本特征缓存：{cache.name}")
        else:
            feats = at.encode_texts(texts, plan["encoder"],
                                    batch_size=int(plan.get("batch_size") or 32),
                                    log=log,
                                    on_progress=prog.feature_progress)
            np.savez_compressed(cache, f=feats)

        prog.stage_begin("训练分类头", frac=0.7)
        model, history, stop = train_head(feats[ti], y_all[ti],
                                          feats[vi], y_all[vi], plan, log=log,
                                          on_epoch=prog.epoch)
        ev = evaluate(model, feats[vi], y_all[vi], len(classes), classes=classes)
        diag = diagnose(history, len(y_all))
        test_ev = None
        if len(te):
            test_ev = evaluate(model, feats[te], y_all[te], len(classes), classes=classes)
            test_ev = {"test_acc": test_ev["acc"], "test_samples": int(len(te)),
                       "macro_f1": test_ev["macro_f1"],
                       "qwk": test_ev.get("qwk"),
                       "per_class": test_ev.get("per_class"),
                       "confusion_matrix": test_ev.get("confusion_matrix")}
            log("测试集评估（全程未参与训练）："
                + f"acc {test_ev['test_acc']} · macro_f1 {test_ev['macro_f1']}")

        split_info = {"source": splits["source"], "note": splits["note"],
                      "train": int(len(ti)), "val": int(len(vi)),
                      "test": int(len(te))}
        hb = dict(history[-1]) if history else {}
        hb["acc"] = hb.get("val_acc")
        hb["macro_f1"] = ev.get("macro_f1")
        hb["per_class"] = ev.get("per_class")
        hb["confusion_matrix"] = ev.get("confusion_matrix")
        metrics = {"history": prog.all_history or history,
                   "task": "classification",
                   "arch": "text_transfer_head", "best_val": hb,
                   "test_metrics": test_ev or {}, "config": plan,
                   "datacard": card, "split_info": split_info,
                   "advice": _plain_advice(diag, len(y_all)),
                   "rounds": [{"round": 1, "plan": plan, "val": ev,
                               "test": test_ev, "history": history,
                               "verdict": diag["verdict"],
                               "hint": diag.get("hint", "")}]}

        model_path = trainer.MODELS_DIR / f"{job_id}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "meta": {
                "task": "classification", "arch": "text_transfer_head",
                "task_hint": "text_classification",
                "encoder": plan["encoder"], "text_col": text_col,
                "prep": {"classes": classes},
                "target_col": target, "dataset_id": d["id"],
                "dataset_name": d["name"],
                "arch_params": {"hidden": int(model.net[0].out_features),
                                "dropout": float(plan.get("dropout") or 0.2),
                                "feat_dim": int(feats.shape[1]),
                                "encoder": plan["encoder"]},
                "config": plan, "datacard": card,
                "split_info": split_info,
                "test_metrics": test_ev or {},
            },
        }, model_path)
        log(f"自动建模完成：验证准确率 {ev['acc']} · macro_f1 {ev['macro_f1']}"
            f"（诊断：{diag['verdict']}），模型已保存：{model_path.name}")
        db.update_job(job_id, status="done", progress=1.0,
                      model_path=str(model_path), metrics=metrics)
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=str(e)[:500],
                      logs=logs + [f"自动建模失败：{e}"])


def _guess_text_col(df: pd.DataFrame, target: str | None) -> str | None:
    """现场识别正文列（历史数据缺 text_col 时的兜底）。"""
    try:
        import ai_pipeline as ap
        marker = set(ap._SPLIT_MARKER_COLS)
        return ap._pick_text_column(df, target, marker)
    except Exception:  # noqa: BLE001
        return None


# ---------- 表格分类 / 回归 ----------

def _auto_train_tabular(job_id: str, job: dict, d: dict) -> None:
    """表格任务：HistGradientBoosting + 自动迭代（比纯 MLP 强得多）。

    表格数据的强基线是梯度提升树 —— 它对非线性和特征交互的建模能力
    远超 MLP，且不需要 GPU。MLP 作为备选保留。
    """
    import train as trainer

    cfg = job["config"]
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        db.update_job(job_id, logs=logs)

    try:
        db.update_job(job_id, status="running")
        # 立刻给一个非零进度与阶段说明 —— 否则前几秒前端显示 0%，
        # 用户会以为卡住了
        prog = Progress(job_id, logs, total_epochs=0)
        prog.stage_begin("正在读取数据并做体检…", frac=0.02)
        log("智能体接管：开始数据体检…")

        df = pd.read_csv(d["file_path"])
        meta = d.get("meta") or {}
        target = d.get("target_col")
        if not target or target not in df.columns:
            raise ValueError("未找到目标列，无法训练表格模型")
        task = "regression" if d.get("task_hint") == "tabular_regression" \
            else "classification"

        card = inspect(df, d.get("task_hint") or "", meta)
        log(f"数据卡：{card['rows']} 条样本 · "
            f"{card.get('n_classes', 0) if task == 'classification' else '回归'} · "
            f"划分 {card['splits']}")

        splits = trainer.resolve_splits(df, float(cfg.get("val_split") or 0.2))
        ti, vi, te = splits["train"], splits["val"], splits["test"]
        log(f"数据划分：{splits['note']} —— 训练 {len(ti)} · 验证 {len(vi)}"
            + (f" · 测试 {len(te)}（不参与训练）" if len(te) else ""))
        if len(ti) == 0:
            raise ValueError("划分后训练集为空，请检查数据集的 split 标注")
        if len(vi) == 0:
            raise ValueError("划分后验证集为空，请检查数据集的 split 标注")
        if len(te):
            leak = (set(ti.tolist()) | set(vi.tolist())) & set(te.tolist())
            if leak:
                raise ValueError(
                    f"划分异常：{len(leak)} 个测试集样本同时出现在训练/验证集中，已中止训练")
            log(f"已校验：{len(te)} 个测试集样本与训练/验证集零交集")

        X_np, y_np, prep = trainer.prep_tabular(df, target, task)
        log(f"特征矩阵：{X_np.shape[0]} 行 × {X_np.shape[1]} 列"
            f"（数值 {len(prep['num_cols'])} · 类别 {len(prep['cat_cols'])}）")

        rounds: list[dict] = []
        best: dict | None = None
        for r in range(1, MAX_ROUNDS + 1):
            plan = _tabular_plan(card, r)
            log(f"第 {r}/{MAX_ROUNDS} 轮方案：{plan['model']} · {plan['reason']}")
            prog.w0 = (r - 1) / MAX_ROUNDS
            prog.w1 = r / MAX_ROUNDS
            prog.stage_begin(f"第 {r}/{MAX_ROUNDS} 轮：训练中", frac=0.3)
            t0 = time.time()
            model, ev, hist = _fit_tabular(X_np, y_np, ti, vi, plan, task,
                                           len(prep["classes"] or []),
                                           classes=prep["classes"])
            # 表格任务没有逐 epoch 曲线，用轮次进度 + 已完成的指标做占位
            prog.epoch(r, MAX_ROUNDS, hist or [{"epoch": r}], hist[0] if hist else {})
            log(f"  {plan['model']} 训练完成 {time.time()-t0:.1f}s · "
                f"{'val_acc ' + str(ev.get('acc')) if task == 'classification' else 'val_rmse ' + str(ev.get('rmse'))}")

            test_ev = None
            if len(te):
                _cls = prep["classes"] or []
                test_ev = _eval_tabular(model, X_np[te], y_np[te], task,
                                        len(_cls), classes=_cls)
                if task == "regression":
                    std = (prep.get("y_stats") or {}).get("std", 1.0)
                    test_ev = {"test_rmse": round(test_ev.get("rmse", 0.0) * std, 5),
                               "test_samples": int(len(te)),
                               "r2": test_ev.get("r2")}
                else:
                    test_ev = {"test_acc": test_ev.get("acc"),
                               "test_samples": int(len(te)),
                               "macro_f1": test_ev.get("macro_f1"),
                               "qwk": test_ev.get("qwk"),
                               "per_class": test_ev.get("per_class"),
                               "confusion_matrix": test_ev.get("confusion_matrix")}
                log("测试集评估（全程未参与训练）："
                    + (f"acc {test_ev.get('test_acc')}" if task == "classification"
                       else f"rmse {test_ev.get('test_rmse')}"))

            rec = {"round": r, "plan": plan, "val": ev, "test": test_ev,
                   "history": hist, "model": model}
            rounds.append(rec)
            key = ev.get("acc") if task == "classification" else -ev.get("rmse", 1e9)
            if best is None or key > best["_key"]:
                rec["_key"] = key
                best = rec
            else:
                rec["_key"] = key
            if r >= MAX_ROUNDS:
                break
            plan = dict(plan)
            plan["reason"] = "换一组超参再试"

        if best is None:
            raise ValueError("自动建模未产出可用模型")

        split_info = {"source": splits["source"], "note": splits["note"],
                      "train": int(len(ti)), "val": int(len(vi)),
                      "test": int(len(te))}
        rounds_flat = [{k: v for k, v in x.items() if k not in ("model", "_key")}
                       for x in rounds]
        best_round = next((x for x in rounds_flat if x["round"] == best["round"]), {})
        # 指标键名与其它路径保持一致：分类用 val_acc / macro_f1，
        # 回归用 val_rmse —— 前端按固定键名取值，键名不统一会取到 None。
        bv = best_round.get("val") or {}
        hb = {"epoch": best_round.get("round", 1)}
        if task == "classification":
            hb["val_acc"] = bv.get("acc")
            hb["acc"] = bv.get("acc")
            hb["macro_f1"] = bv.get("macro_f1")
            hb["per_class"] = bv.get("per_class")
            hb["confusion_matrix"] = bv.get("confusion_matrix")
        else:
            # RMSE 在标准化量纲上算，需要按 y_stats 还原成原始量纲
            std = (prep.get("y_stats") or {}).get("std", 1.0)
            rmse = (bv.get("rmse") or 0.0) * std
            hb["val_rmse"] = round(rmse, 5)
            hb["rmse"] = round(rmse, 5)
            hb["r2"] = bv.get("r2")
        metrics = {"history": prog.all_history
                   or best_round.get("history") or [],
                   "task": task, "arch": best["plan"]["model"],
                   "best_val": hb, "test_metrics": best["test"] or {},
                   "config": best["plan"], "datacard": card,
                   "advice": _tabular_advice(task, best["val"], len(y_np)),
                   "rounds": rounds_flat, "split_info": split_info}

        import pickle
        model_path = trainer.MODELS_DIR / f"{job_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": best["model"], "prep": prep,
                         "plan": best["plan"], "task": task}, f)
        log(f"自动建模完成：共 {len(rounds)} 轮，采用第 {best['round']} 轮"
            f"（{best['plan']['model']}），模型已保存：{model_path.name}")
        db.update_job(job_id, status="done", progress=1.0,
                      model_path=str(model_path), metrics=metrics)
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=str(e)[:500],
                      logs=logs + [f"自动建模失败：{e}"])


def _tabular_plan(card: dict, rnd: int) -> dict:
    """表格任务的候选方案序列：梯度提升树为主，MLP 为辅。"""
    n = int(card.get("rows") or 0)
    if rnd == 1:
        return {"model": "hgb", "max_iter": 300, "learning_rate": 0.08,
                "max_leaf_nodes": 31, "l2_regularization": 1.0,
                "early_stopping": True,
                "reason": f"梯度提升树 · 样本 {n} 条"}
    if rnd == 2:
        return {"model": "hgb", "max_iter": 600, "learning_rate": 0.05,
                "max_leaf_nodes": 63, "l2_regularization": 0.5,
                "early_stopping": True,
                "reason": "加深加长，学习率减半"}
    return {"model": "hgb", "max_iter": 900, "learning_rate": 0.03,
            "max_leaf_nodes": 127, "l2_regularization": 0.1,
            "early_stopping": True,
            "reason": "更大容量 + 更强正则"}


def _fit_tabular(X, y, ti, vi, plan: dict, task: str, nclass: int,
                 classes: list[str] | None = None):
    """在预处理好的特征矩阵上拟合表格模型。"""
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    Xtr, Xva = X[ti], X[vi]
    ytr, yva = y[ti], y[vi]
    hist: list[dict] = []
    if task == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=int(plan.get("max_iter") or 300),
            learning_rate=float(plan.get("learning_rate") or 0.08),
            max_leaf_nodes=int(plan.get("max_leaf_nodes") or 31),
            l2_regularization=float(plan.get("l2_regularization") or 1.0),
            early_stopping=bool(plan.get("early_stopping", True)),
            random_state=42)
        model.fit(Xtr, ytr)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=int(plan.get("max_iter") or 300),
            learning_rate=float(plan.get("learning_rate") or 0.08),
            max_leaf_nodes=int(plan.get("max_leaf_nodes") or 31),
            l2_regularization=float(plan.get("l2_regularization") or 1.0),
            early_stopping=bool(plan.get("early_stopping", True)),
            random_state=42)
        model.fit(Xtr, ytr)
    ev = _eval_tabular(model, Xva, yva, task, nclass, classes=classes)
    hist.append({"epoch": 1, "val_acc": ev.get("acc"),
                  "val_rmse": ev.get("rmse")})
    return model, ev, hist


def _eval_tabular(model, X, y, task: str, nclass: int,
                  classes: list[str] | None = None) -> dict:
    if task == "classification":
        pred = model.predict(X)
        y = np.asarray(y)
        acc = float((pred == y).mean()) if len(y) else 0.0
        cm = np.zeros((nclass, nclass), dtype=int)
        for t, p in zip(y.tolist(), pred.tolist()):
            if 0 <= t < nclass and 0 <= p < nclass:
                cm[t][p] += 1
        f1s = []
        per_class = []
        for c in range(nclass):
            tp = int(cm[c][c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            f1s.append(f1)
            per_class.append({"class": c, "support": int(cm[c, :].sum()),
                              "precision": round(prec, 4), "recall": round(rec, 4),
                              "f1": round(f1, 4)})
        import metrics_extra as mx
        return {"acc": round(acc, 5),
                "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 5),
                "qwk": mx.quadratic_kappa(y, pred, nclass),
                "per_class": mx.name_per_class(per_class, classes),
                "confusion_matrix": cm.tolist()}
    pred = model.predict(X)
    y = np.asarray(y, dtype=np.float64)
    rmse = float(np.sqrt(((pred - y) ** 2).mean())) if len(y) else 0.0
    return {"rmse": round(rmse, 5), "r2": round(float(model.score(X, y)), 5)}
