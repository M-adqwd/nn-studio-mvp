"""训练引擎：后台线程执行 PyTorch 训练，进度与指标落 train_jobs 表。

支持三类任务：
- 表格 MLP（分类 CrossEntropy / 回归 MSE；类别列 one-hot、数值列标准化）
- 图像 CNN（小卷积网，PIL 读图缩放 64x64，输入为 zip 解压的 ImageFolder manifest）
- 文本 EmbeddingBag+MLP（字符 bigram crc32 哈希分词，无需外部分词器）
"""
from __future__ import annotations

import threading
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Body, Depends, HTTPException
from PIL import Image
from torch import nn

import analysis
import auth
import db
import preflight
import spec

torch.manual_seed(42)
np.random.seed(42)

# 划分用的独立随机源：训练过程会消耗全局随机数，
# 若共用 np.random，同一数据集两次训练会得到不同划分，
# 用户看到「同样的数据结果不一样」会非常困惑。
# 用固定种子 + 每次调用重建，保证同一份数据永远得到同一份划分。
_SPLIT_SEED = 42


def _split_rng() -> np.random.Generator:
    return np.random.default_rng(_SPLIT_SEED)

MODELS_DIR = db.STORAGE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
TEXT_VOCAB = 4096
TEXT_DIM = 64
TEXT_MAXLEN = 256

router = APIRouter()


# ---------- 模型定义 ----------

class MLP(nn.Module):
    def __init__(self, din: int, hidden: int, dout: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(din, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, nclass: int, img_size: int = IMG_SIZE, dropout: float = 0.0):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        s = max(8, img_size // 8)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64 * s * s, 128), nn.ReLU(),
            nn.Linear(128, nclass),
        )

    def forward(self, x):
        return self.head(self.feat(x))


class TextClf(nn.Module):
    def __init__(self, vocab: int, dim: int, nclass: int, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")
        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, nclass))
        self.fc = nn.Sequential(*layers)

    def forward(self, ids, offsets):
        return self.fc(self.emb(ids, offsets))


# ---------- 数据预处理 ----------

def hash_tokens(text: str, vocab: int = TEXT_VOCAB, ngram: int = 2) -> list[int]:
    """稳定哈希分词（crc32 与进程无关，推理时可复现）。英文按空格切词，其余取字符 bigram。"""
    t = str(text).strip()
    if not t:
        return [0]
    parts = t.split()
    if len(parts) > 1 and all(len(p) > 1 for p in parts[:5]):
        grams = parts  # 类英文文本：按词
    else:
        grams = [t[i:i + ngram] for i in range(max(1, len(t) - ngram + 1))]
    return [zlib.crc32(g.encode("utf-8")) % vocab for g in grams][:TEXT_MAXLEN]


def prep_tabular(df: pd.DataFrame, target_col: str, task: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (X, y, prep_meta)。数值列标准化、类别列 one-hot（Top 50）。"""
    # split 是划分标记，不是特征 —— 必须排除。
    # 若把它当特征，模型能从 split 列直接读出答案（训练集/验证集标签分布不同），
    # 会造成严重的数据泄漏，验证指标虚高、上线后立刻崩。
    # 清单刻意不含 "set"/"group" 这类宽泛名字 —— 真实数据里确有此类业务列。
    _marker = {"split", "subset", "partition", "fold", "dataset_split",
               "data_split", "split_type", "set_split"}
    # 患者/病例号同理：它是分组键，不是特征。当特征用会让模型记住「哪个病人」，
    # 在训练集上表现极好、换一批病人立刻失效。
    _grp = detect_group_col(df)
    drop_cols = [target_col] + [c for c in df.columns
                                if str(c).strip().lower() in _marker]
    if _grp:
        drop_cols.append(_grp)
    feats = df.drop(columns=[c for c in drop_cols if c in df.columns])
    num_cols, cat_cols, stats = [], [], {}
    parts = []
    for c in feats.columns:
        s = feats[c]
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
            mean = float(s.mean()) if not np.isnan(s.mean()) else 0.0
            std = float(s.std()) or 1.0
            parts.append(((s.fillna(mean) - mean) / std).values.reshape(-1, 1))
            num_cols.append(str(c))
            stats[str(c)] = {"kind": "num", "mean": mean, "std": std}
        elif analysis.infer_col_type(s) in ("category", "bool"):
            vc = s.astype(str).fillna("(空)")
            cats = sorted(vc.unique().tolist())[:50]
            idx = {v: i for i, v in enumerate(cats)}
            onehot = np.zeros((len(df), len(cats)), dtype=np.float32)
            for r, v in enumerate(vc):
                j = idx.get(v)
                if j is not None:
                    onehot[r, j] = 1.0
            parts.append(onehot)
            cat_cols.append(str(c))
            stats[str(c)] = {"kind": "cat", "cats": cats}
    if not parts:
        raise ValueError("没有可用的特征列（仅支持数值与类别列参与训练）")
    X = np.concatenate(parts, axis=1).astype(np.float32)

    if task == "classification":
        classes = sorted(df[target_col].astype(str).unique().tolist())
        cidx = {v: i for i, v in enumerate(classes)}
        y = df[target_col].astype(str).map(cidx).values.astype(np.int64)
        y_stats = None
    else:
        classes = None
        y_raw = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values.astype(np.float32)
        y_mean, y_std = float(y_raw.mean()), float(y_raw.std()) or 1.0
        y = ((y_raw - y_mean) / y_std).astype(np.float32)  # 标准化目标，稳定 MSE 训练
        y_stats = {"mean": y_mean, "std": y_std}

    prep = {"num_cols": num_cols, "cat_cols": cat_cols, "stats": stats,
            "classes": classes, "input_dim": int(X.shape[1]), "y_stats": y_stats}
    return X, y, prep


def load_image_array(root: Path, rel: str) -> np.ndarray:
    with Image.open(root / rel) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(im, dtype=np.float32) / 255.0


class ImagePathDataset(torch.utils.data.Dataset):
    """按需读取图像的 Dataset。

    图像任务的数据量往往很大（全量入内存需数 GB），
    这里只保存路径，逐 batch 解码，读完即释放。
    """

    def __init__(self, root: Path, paths: list[str]):
        self.root = Path(root)
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        arr = load_image_array(self.root, self.paths[i])
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _stratified_take(pool: np.ndarray, df: pd.DataFrame, existing: np.ndarray,
                     need: int) -> np.ndarray:
    """从候选池里按类别分层取 n 条，尽量覆盖各个类别。

    验证集只补少数类的话，指标会被多数类主导而失真。
    """
    rng = _split_rng()
    if need <= 0 or len(pool) == 0:
        return np.array([], dtype=np.int64)
    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        return rng.permutation(pool)[:need]

    # 现有验证集里各类的占比，作为分配的参考
    have = df.iloc[existing][label_col].astype(str).value_counts()
    total = max(int(have.sum()), 1)
    want = (have / total * need).round().astype(int).to_dict()

    taken: list[int] = []
    by_class = {}
    for idx in pool.tolist():
        by_class.setdefault(str(df.iloc[idx][label_col]), []).append(idx)

    for cls, items in by_class.items():
        k = int(want.get(cls, 0))
        if k <= 0 or not items:
            continue
        arr = np.array(items)
        picked = rng.permutation(arr)[:k]
        taken.extend(int(x) for x in picked)

    # 分层没取够时，用剩余样本随机补齐
    if len(taken) < need:
        rest = [i for i in pool.tolist() if i not in set(taken)]
        extra = rng.permutation(np.array(rest, dtype=np.int64))[:need - len(taken)]
        taken.extend(int(x) for x in extra)
    return np.array(taken[:need], dtype=np.int64)


def _legacy_splits(df: pd.DataFrame, val_split: float,
                   test_split: float = 0.15,
                   group_col: str | None = None) -> dict:
    """旧切分实现（v2 之前的逻辑），仅在无法走 G2 切分门时兜底。

    保留原因：回归任务与没有标签列的场景不适用「按类别分层」，
    硬套分类切分会让几千个连续取值被当成类别，反而更糟。
    """
    n = len(df)
    rng = _split_rng()
    empty = np.array([], dtype=np.int64)
    gcol = group_col or detect_group_col(df)
    label_col = "label" if "label" in df.columns else None

    def _holdout(pool: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """按患者分组或按类别分层，从 pool 里抽出 frac 比例的 holdout。"""
        if gcol:
            return _group_holdout(df, pool, frac, gcol, seed)
        return _stratified_holdout(df, pool, frac, label_col, seed)

    if "split" in df.columns:
        sp = df["split"].astype(str).str.strip().str.lower()
        # 兼容 train/validation/test 等别名
        sp = sp.replace({"validation": "val", "valid": "val", "dev": "val",
                         "eval": "val", "training": "train", "testing": "test"})
        ti = np.where(sp.values == "train")[0]
        vi = np.where(sp.values == "val")[0]
        te = np.where(sp.values == "test")[0]

        if len(ti) > 0:
            # 按数据实情描述，不笼统地说「自带 train/test」——
            # 有的数据集只有 train，有的有 train+test，文案必须如实反映。
            has_native_test = len(te) > 0
            native = ["train"]
            if len(vi) > 0:
                native.append("val")
            if has_native_test:
                native.append("test")
            note = f"沿用数据集自带的 {'/'.join(native)} 划分"
            if has_native_test:
                # 自带 test 时也检查是不是同一患者跨划分（常见于医院导出的数据）
                dup = _group_leak(df, te, np.concatenate([ti, vi]), gcol)
                if dup:
                    note += f"；检测到 {dup} 个患者同时出现在测试集与训练集，指标需按患者复核"
            if len(vi) == 0:
                # 数据集没有验证集：从训练集内部切出一份，仅用于早停与权重选择
                vi, ti = _holdout(ti, val_split / max(1 - val_split, 1e-6), 7)
                note = ("数据集未自带验证集，已从训练集切出一份用于早停"
                        + ("；测试集全程保留" if has_native_test else ""))
            # ---- 硬隔离：测试集绝不进入训练与验证 ----
            # 即使上游数据出现同一行既标 train 又标 test 的异常，
            # 也在这里强制把测试集样本从训练/验证集中剔除，
            # 保证测试集对模型完全不可见，指标才可信。
            if len(te):
                test_set = set(te.tolist())
                ti = np.array([i for i in ti.tolist() if i not in test_set], dtype=np.int64)
                vi = np.array([i for i in vi.tolist() if i not in test_set], dtype=np.int64)
            else:
                # 没有自带测试集：从训练集里切出独立测试集
                te, ti = _holdout(ti, test_split, 13)
                if len(te):
                    note += (f"；数据集未自带测试集，已从训练集切出 {len(te)} 条"
                             f"作为独立测试集（训练与早停全程不可见）")

            # ---- 验证集过小时从训练集补足 ----
            # 自带验证集只有几十条时，指标波动会大到没法选模型
            # （32 条验证集里 1 条样本就是 3.1% 的差异）。
            # 这里从训练集里借一部分补到合理规模，
            # 测试集仍然完全隔离，不受影响。
            min_val = 100
            if len(vi) > 0 and len(vi) < min_val and len(ti) > min_val * 2:
                need = min_val - len(vi)
                # 按类别分层补：每个类别都补一点，避免只补到多数类
                extra = _stratified_take(ti, df, vi, need)
                if len(extra):
                    vi = np.concatenate([vi, extra])
                    ti = np.array([i for i in ti.tolist() if i not in set(extra.tolist())],
                                  dtype=np.int64)
                    note += (f"；自带验证集只有 {len(vi) - len(extra)} 条，"
                             f"已从训练集分层补足到 {len(vi)} 条，"
                             f"以免指标波动过大导致选错模型")
            return {"train": ti, "val": vi, "test": te,
                    "source": "dataset_split", "note": note,
                    "group_col": gcol}

    # 无自带划分：切出 test 与 val 两份 holdout
    all_idx = np.arange(n, dtype=np.int64)
    te, rest = _holdout(all_idx, test_split, 13)
    vi, ti = _holdout(rest, val_split / max(1 - test_split, 1e-6), 7)
    if len(ti) == 0:
        ti, vi = rest, vi[:1]
    how = f"按患者分组（{gcol}）" if gcol else "按类别分层"
    return {"train": ti, "val": vi, "test": te,
            "source": "random_split", "group_col": gcol,
            "note": (f"数据集未自带划分，已{how}切出 训练/验证/测试 = "
                     f"{len(ti)}/{len(vi)}/{len(te)}")}


def resolve_splits(df: pd.DataFrame, val_split: float,
                   test_split: float = 0.15,
                   group_col: str | None = None,
                   label_col: str | None = None) -> dict:
    """解析出本次训练实际使用的样本索引。

    v2 起委托给 G2 切分门（`split.py`）：稀有类优先分配、分组隔离、
    切完必检。旧实现对稀有类的处理是「随机分组」，会把稀有类整组划进测试集
    —— MIT-BIH 的 F 类就这样被抽走 93%，训练侧只剩 54 条，F1 归零。

    分类任务走新路径；回归任务或找不到标签列时退回 `_legacy_splits`，
    因为连续目标的分层切分没有意义。
    """
    import split as _split

    target = label_col or ("label" if "label" in df.columns else None)
    usable = False
    if target and target in df.columns:
        try:
            nun = int(df[target].nunique(dropna=True))
            usable = 2 <= nun <= 200
        except Exception:  # noqa: BLE001
            usable = False

    if usable:
        try:
            plan = _split.plan_splits(
                df, label_col=target, val_frac=val_split, test_frac=test_split,
                group_col=group_col)
            note = "；".join(plan.notes)
            if not plan.ok:
                note += "；切分门未通过：" + plan.report.summary()
            return {
                "train": plan.splits["train"], "val": plan.splits["val"],
                "test": plan.splits["test"], "source": plan.source,
                "note": note, "group_col": plan.group_col,
                "plan": plan, "gate": plan.report.as_dict(),
                "index_hash": plan.index_hash(),
            }
        except Exception as e:  # noqa: BLE001
            # 切分门内部异常不能把整个训练拖垮，但必须留痕
            fallback = _legacy_splits(df, val_split, test_split, group_col)
            fallback["note"] = (f"G2 切分门不可用（{str(e)[:120]}），"
                                f"已退回旧切分：{fallback.get('note', '')}")
            fallback["fallback"] = True
            return fallback

    return _legacy_splits(df, val_split, test_split, group_col)


# 常见「同一患者/同一病例」列名。命中则按组切分，避免同一患者的
# 多次检查同时落在训练与测试，造成指标虚高。
GROUP_MARKER = {"patient", "patient_id", "patientid", "pid", "subject_id",
                "subjectid", "record_id", "record", "person_id", "case_id",
                "study_id", "patient_no", "mrn"}


def detect_group_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() in GROUP_MARKER and df[c].nunique() < len(df) * 0.9:
            return str(c)
    return None


def _group_leak(df: pd.DataFrame, a: np.ndarray, b: np.ndarray,
                gcol: str | None) -> int:
    if not gcol or not len(a) or not len(b):
        return 0
    ga = set(df.iloc[a][gcol].astype(str))
    gb = set(df.iloc[b][gcol].astype(str))
    return len(ga & gb)


def _group_holdout(df: pd.DataFrame, pool: np.ndarray, frac: float,
                   gcol: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """以「患者」为最小单位抽 holdout，同一患者不跨划分。

    分组切分不做类别分层——按组分层会把稀有类患者整组塞进测试集，
    训练侧反而学不到该类别。这里只在分组内随机，并在切分说明里如实标注。
    """
    if frac <= 0 or len(pool) == 0:
        return np.array([], dtype=np.int64), pool
    rng = np.random.default_rng(seed)
    sub = df.iloc[pool]
    groups: dict[str, list[int]] = {}
    for idx, g in zip(pool.tolist(), sub[gcol].astype(str).tolist()):
        groups.setdefault(g, []).append(idx)
    keys = list(groups.keys())
    rng.shuffle(keys)
    target = int(round(len(pool) * min(frac, 0.9)))
    held: list[int] = []
    for k in keys:
        if len(held) >= target:
            break
        held.extend(groups[k])
    hs = set(held)
    rest = [i for i in pool.tolist() if i not in hs]
    return (np.array(sorted(held), dtype=np.int64),
            np.array(sorted(rest), dtype=np.int64))


def _stratified_holdout(df: pd.DataFrame, pool: np.ndarray, frac: float,
                        label_col: str | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按类别分层抽 holdout；稀有类至少留 1 条，否则该类指标算不出来。"""
    if frac <= 0 or len(pool) == 0:
        return np.array([], dtype=np.int64), pool
    rng = np.random.default_rng(seed)
    if not label_col:
        perm = rng.permutation(pool)
        k = max(1, int(round(len(pool) * frac)))
        return np.sort(perm[:k]).astype(np.int64), np.sort(perm[k:]).astype(np.int64)

    buckets: dict[str, list[int]] = {}
    labs = df.iloc[pool][label_col].astype(str).tolist()
    for idx, lab in zip(pool.tolist(), labs):
        buckets.setdefault(lab, []).append(idx)

    held: list[int] = []
    for lab, items in buckets.items():
        arr = rng.permutation(np.array(items, dtype=np.int64))
        k = int(round(len(arr) * frac))
        if k == 0 and len(arr) > 1:
            k = 1
        held.extend(arr[:k].tolist())
    hs = set(held)
    rest = [i for i in pool.tolist() if i not in hs]
    return (np.array(sorted(held), dtype=np.int64),
            np.array(sorted(rest), dtype=np.int64))


# ---------- 训练主流程 ----------

def _log(logs: list[str], msg: str) -> list[str]:
    logs = (logs + [f"[{time.strftime('%H:%M:%S')}] {msg}"])[-200:]
    return logs


def diagnose(history: list[dict], task: str, mname: str, n: int,
             patience: int, since_best: int) -> dict:
    """根据训练曲线生成诊断结论与改进建议（供前端展示）。

    输出两套文案：
      summary / tips  —— 面向懂技术的用户，保留精确术语与参数
      plain_summary / plain_tips —— 面向不懂 AI 的用户，说人话
    前端默认展示 plain_*，技术文案折叠起来。
    """
    out = {"level": "ok", "summary": "", "tips": [],
           "plain_summary": "", "plain_tips": []}
    if len(history) < 2:
        out["plain_summary"] = "训练刚刚开始。"
        return out

    first, last = history[0], history[-1]
    t0, t1 = first["train_loss"], last["train_loss"]
    v0, v1 = first["val_loss"], last["val_loss"]
    cls = task == "classification"
    m1 = last.get(mname)

    # 学习率过大 / 没有学到东西
    if t1 > t0 * 1.05 and t1 > 0.3:
        out.update(level="bad",
                   summary="训练损失没有下降，模型基本没有学到有效特征。",
                   tips=["学习率偏高，建议降到当前的 1/2（如 0.005 → 0.002）",
                         "检查特征是否存在极端值或量纲差异，必要时先做标准化",
                         "确认目标列没有大量噪声或错标样本"],
                   plain_summary="模型没能从数据里学到东西——训练过程中错误一点没减少。"
                                 "多半是学习步子迈得太大，或者数据本身缺少规律。",
                   plain_tips=["把学习率降到当前的一半再试",
                               "检查数据里是否有量级差得离谱的列（比如年龄是几十、收入是几百万）",
                               "确认「答案列」本身填对了、且没有大量标错的数据"])
        return out

    # 过拟合
    if v1 > v0 * 1.08 and t1 < t0 * 0.7:
        out.update(level="bad",
                   summary="出现明显过拟合：训练集效果持续变好，但验证集指标反而变差。",
                   tips=[f"减少训练轮次，当前 {len(history)} 轮已足够，可降到 {max(5, len(history) // 2)} 轮",
                         "开启 Dropout（0.1~0.3）抑制过拟合",
                         "启用早停（如 patience=8）自动在最佳点终止",
                         "增加样本量，或对训练集做数据增强"],
                   plain_summary="模型把训练数据背下来了，但换一批新数据就认不出了——"
                                 "这叫做「死记硬背」。数据太少时最容易出现这种情况。",
                   plain_tips=[f"训练轮次减半试试（现在 {len(history)} 轮，改为 {max(5, len(history) // 2)} 轮就够）",
                               "增加防死记硬背的设置（Dropout 调到 0.1~0.3）",
                               "补充更多样本是最有效的办法"])
        return out

    # 欠拟合
    if t1 > 0.3 * max(t0, 1e-9):
        out.update(level="warn",
                   summary="欠拟合：训练集损失仍偏高，模型容量可能不足。",
                   tips=["增加隐藏层宽度或训练轮次",
                         "提高学习率加快收敛（如 0.003 → 0.01）",
                         "检查特征是否存在大量缺失或噪声"],
                   plain_summary="模型还没学够——连它见过的训练数据都认不全。"
                                 "可以给它更多时间，或者换个更能干的模型。",
                   plain_tips=["把训练轮次调多一些",
                               "把学习速度调快一点",
                               "检查数据里是不是缺了不少内容"])
        return out

    # 样本过少
    if n < 50:
        out.update(level="warn",
                   summary=f"样本量偏少（仅 {n} 条），指标波动大，可信度有限。",
                   tips=["建议至少准备 200 条以上样本",
                         "开启类别权重缓解标签不平衡",
                         "适当放宽验证集比例至 30%，让验证集更有代表性"],
                   plain_summary=f"数据有点少（只有 {n} 条），得出的分数波动会比较大，"
                                 "不太能代表真实表现。",
                   plain_tips=["尽量准备 200 条以上，效果会稳定很多",
                               "各类别数据量差太多时，勾选「启用类别权重」",
                               "验证集比例调到 30%，让检验更充分"])
        return out

    # 分类准确率偏低
    if cls and isinstance(m1, (int, float)) and m1 < 0.6:
        out.update(level="warn",
                   summary=f"准确率 {m1 * 100:.1f}% 偏低，可能尚未学到有效模式。",
                   tips=["检查标签分布是否严重不平衡，必要时开启类别权重",
                         "增加训练轮次或提高模型容量（加宽隐藏层）",
                         "确认特征与标签之间确实存在可学习的相关性"],
                   plain_summary=f"准确率只有 {m1 * 100:.1f}%，模型基本还在「猜」。"
                                 "要么数据量不够，要么数据里本来就没有可学的规律。",
                   plain_tips=["检查各类别的数据量是不是差很多，差太多就勾选「启用类别权重」",
                               "多训练一些轮次试试",
                               "确认「答案列」和「线索列」之间确实有规律可循"])
        return out

    # 收敛正常：给出可选优化项
    tips = []
    plain_tips = []
    if since_best >= 3:
        tips.append(f"验证集已连续 {since_best} 轮未改善，建议开启早停（patience={max(5, since_best)}）避免无效训练")
        plain_tips.append("可以打开「自动停止」，效果不再提升时就停下来，省时间")
    if not patience:
        tips.append("开启早停可在指标停滞时自动终止，节省时间并避免过拟合")
        plain_tips.append("打开「自动停止」，能省下不少等待时间")
    if not cls and isinstance(m1, (int, float)) and m1 > 0:
        vals = [h[mname] for h in history if h.get(mname) is not None]
        if vals and min(vals) > 0 and abs(min(vals) - m1) / min(vals) > 0.3:
            tips.append("末轮 RMSE 明显劣于历史最佳，建议开启早停以保留最佳权重")
            plain_tips.append("最后一轮反而不如中间某一轮好，打开「自动停止」会保留最好的那一版")
    tips.append("可尝试更换优化器（Adam → SGD）或微调学习率进一步提升效果")
    plain_tips.append("想再榨一点效果，可以在「高级调优」里微调学习率")

    best_v = min((h[mname] for h in history if h.get(mname) is not None), default=None)
    out.update(level="ok",
               summary=f"训练正常收敛，最佳 {mname} {best_v:.4f}。" if best_v is not None else "训练正常收敛。",
               plain_summary="训练过程正常，模型已经收敛。" if best_v is not None else "训练过程正常。",
               tips=tips, plain_tips=plain_tips)
    return out


def run_job(job_id: str) -> None:
    job = db.get_job(job_id)
    cfg = job["config"]
    # 优先交给智能体自动建模（预训练迁移学习 / 梯度提升树 + 自动迭代）。
    # 前端传 transfer=false 可退回原来的基础训练路径。
    _ds = db.get_dataset(job["dataset_id"])
    _hint = (_ds or {}).get("task_hint")
    if (_ds and _hint in ("image_classification", "object_detection",
                          "text_classification", "signal_classification",
                          "tabular_classification", "tabular_regression")
            and cfg.get("transfer", True)):
        try:
            import automl
            automl.auto_train(job_id)
            return
        except Exception as e:  # noqa: BLE001
            # SOP P5：不做静默降级。
            # 以前这里会直接退回基础训练，前端看到的是一份"成功"的任务和一份差得多的指标
            # （实测 60% vs 72%），客户无从判断该不该用。现在必须由调用方显式选择降级。
            reason = str(e)[:300]
            if not cfg.get("allow_degraded"):
                db.update_job(
                    job_id, status="failed",
                    error=f"自动建模失败：{reason}。未开启降级训练（allow_degraded=true），"
                          f"任务终止——避免产出指标明显更差却被当作成功的结果。")
                return
            db.update_job(job_id, status="running",
                          error=f"自动建模不可用，已按显式要求降级为基础训练：{reason}")
    # 回退时保留自动建模阶段的日志：把 logs 清空等于抹掉前面 45 分钟发生过什么
    _prev_job = db.get_job(job_id) or {}
    logs: list[str] = list(_prev_job.get("logs") or [])
    _fallback_reason = locals().get("_fallback_reason")
    try:
        db.update_job(job_id, status="running", logs=_log(logs, "任务启动，加载数据集…"))
        d = db.get_dataset(job["dataset_id"])
        if not d:
            raise ValueError("数据集不存在")
        task_hint = d.get("task_hint") or "tabular_classification"
        if task_hint == "signal_classification":
            # 信号任务只能走 1D 卷积路径。基础路径会把这 12000 列当普通特征
            # 喂进 MLP，既跑不出可用精度，也是对客户的误导。
            raise ValueError("信号任务需要 1D 卷积建模，请开启自动建模（transfer）后重试")
        df = pd.read_csv(d["file_path"])

        epochs = max(1, min(int(cfg.get("epochs", 10)), 100))
        lr = float(cfg.get("lr", 1e-3))
        batch_size = max(8, min(int(cfg.get("batch_size", 32)), 256))
        hidden = max(8, min(int(cfg.get("hidden", 64)), 512))
        val_split = min(max(float(cfg.get("val_split", 0.2)), 0.1), 0.4)
        use_cw = bool(cfg.get("class_weight"))
        dropout = min(max(float(cfg.get("dropout", 0.0)), 0.0), 0.6)
        weight_decay = min(max(float(cfg.get("weight_decay", 0.0)), 0.0), 0.1)
        patience = max(0, min(int(cfg.get("patience", 0)), 50))  # 0 = 不启用早停
        optimizer = str(cfg.get("optimizer") or "adam").lower()
        sched_name = str(cfg.get("scheduler") or "none").lower()

        # ---- 构建张量 ----
        if task_hint == "image_classification":
            task, arch = "classification", "cnn"
            root = Path(d["meta"]["image_root"])
            classes = sorted(df["label"].astype(str).unique().tolist())
            cidx = {v: i for i, v in enumerate(classes)}
            y_all = torch.from_numpy(df["label"].astype(str).map(cidx).values.astype(np.int64))
            # 大体量关键：图像不一次性载入内存，改为按需读取（Dataset + DataLoader）。
            # 数万张 64×64 图像全量入内存需数 GB，会直接 OOM。
            filepaths = [str(p) for p in df["filepath"]]
            X_all = ImagePathDataset(root, filepaths)
            prep = {"classes": classes, "img_size": IMG_SIZE}
        elif task_hint == "text_classification":
            task, arch = "classification", "emb_mlp"
            text_col = (d.get("meta") or {}).get("text_col")
            if not text_col or text_col not in df.columns:
                raise ValueError("未找到文本列，无法训练文本分类模型")
            classes = sorted(df[d["target_col"]].astype(str).unique().tolist())
            cidx = {v: i for i, v in enumerate(classes)}
            tok_lists = [hash_tokens(t) for t in df[text_col].astype(str)]
            X_all, y_all = tok_lists, torch.from_numpy(df[d["target_col"]].astype(str).map(cidx).values.astype(np.int64))
            prep = {"classes": classes, "vocab": TEXT_VOCAB, "text_col": text_col}
        else:
            task = "regression" if task_hint == "tabular_regression" else "classification"
            arch = "mlp"
            X_np, y_np, prep = prep_tabular(df, d["target_col"], task)
            X_all = torch.from_numpy(X_np)
            y_all = torch.from_numpy(y_np)
            if task == "classification":
                classes = prep["classes"]
                cidx = {v: i for i, v in enumerate(classes)}

        n = len(y_all)
        if n < 8:
            raise ValueError(f"样本量过少（{n} 条），至少需要 8 条")
        db.update_job(job_id, total_epochs=epochs,
                      logs=_log(logs, f"数据就绪：{n} 条样本 · 任务 {task_hint} · 架构 {arch.upper()}"))

        # ---- 锁定训练/验证/测试划分 ----
        # AI 解析阶段已识别出哪些数据属于训练集/验证集/测试集，这里直接消费。
        splits = resolve_splits(df, val_split)
        ti, vi, te = splits["train"], splits["val"], splits["test"]
        msg = (f"数据划分：{splits['note']} —— "
               f"训练 {len(ti)} 条 · 验证 {len(vi)} 条")
        if len(te):
            msg += f" · 测试 {len(te)} 条（不参与训练，用于最终评估）"
        logs = _log(logs, msg)
        db.update_job(job_id, logs=logs)
        if len(ti) == 0:
            raise ValueError("划分后训练集为空，请检查数据集的 split 标注")
        if len(vi) == 0:
            raise ValueError("划分后验证集为空，请检查数据集的 split 标注")

        # 硬校验：测试集样本不得出现在训练集或验证集中。
        # 这是测试集可信度的最后一道防线，宁可训练失败也不能静默污染。
        if len(te):
            leak = (set(ti.tolist()) | set(vi.tolist())) & set(te.tolist())
            if leak:
                raise ValueError(
                    f"划分异常：{len(leak)} 个测试集样本同时出现在训练/验证集中，已中止训练")
            logs = _log(logs, f"已校验：{len(te)} 个测试集样本与训练/验证集零交集")
            db.update_job(job_id, logs=logs)

        # ---- 模型与优化器 ----
        if task == "classification":
            nclass = len(classes)
            if arch == "cnn":
                model = SmallCNN(nclass, IMG_SIZE, dropout)
            elif arch == "emb_mlp":
                model = TextClf(TEXT_VOCAB, TEXT_DIM, nclass, dropout)
            else:
                model = MLP(int(X_all.shape[1]), hidden, nclass, dropout)
            weight = None
            if use_cw:
                cnt = np.bincount(y_all[ti].numpy(), minlength=nclass).astype(np.float32)
                weight = torch.from_numpy((cnt.sum() / np.maximum(cnt, 1) / nclass))
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            model = MLP(int(X_all.shape[1]), hidden, 1, dropout)
            criterion = nn.MSELoss()

        if optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif optimizer == "rmsprop":
            opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = None
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)

        def batches(idx):
            # BatchNorm 在训练态不接受单样本批：样本数不是 batch_size 整数倍时
            # 最后一批可能只剩 1 条，直接抛「Expected more than 1 value per channel」。
            # 余 1 条时并回上一批（batch_size 只是大致的目标值，多一条无碍）。
            bs = max(2, int(batch_size))
            starts = list(range(0, len(idx), bs))
            if len(starts) > 1 and len(idx) - starts[-1] == 1:
                starts.pop()
            for k, st in enumerate(starts):
                en = starts[k + 1] if k + 1 < len(starts) else len(idx)
                yield idx[st:en]

        def forward_batch(bi):
            if arch == "emb_mlp":
                seqs = [X_all[i] for i in bi]
                ids = torch.tensor([t for s in seqs for t in s], dtype=torch.long)
                offs = torch.tensor([0] + list(np.cumsum([len(s) for s in seqs])[:-1]), dtype=torch.long)
                return model(ids, offs)
            if arch == "cnn":
                # 图像按需读取：逐样本解码，避免全量入内存
                batch = torch.stack([X_all[int(i)] for i in bi])
                return model(batch)
            return model(X_all[bi])

        # ---- 训练循环（早停 / 调度器 / 最佳权重回滚） ----
        history: list[dict] = []
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_score = float("inf")
        best_ep, since_best = 0, 0
        mname = "val_acc" if task == "classification" else "val_rmse"
        stop_reason = ""
        for ep in range(1, epochs + 1):
            model.train()
            tot_loss, tot_n = 0.0, 0
            for bi in batches(ti):
                opt.zero_grad()
                out = forward_batch(bi)
                yb = y_all[bi]
                loss = criterion(out, yb) if task == "classification" else criterion(out.squeeze(-1), yb)
                loss.backward()
                opt.step()
                tot_loss += float(loss) * len(bi)
                tot_n += len(bi)
            train_loss = tot_loss / max(tot_n, 1)

            model.eval()
            with torch.no_grad():
                out = forward_batch(vi)
                yv = y_all[vi]
                val_loss = float(criterion(out, yv) if task == "classification" else criterion(out.squeeze(-1), yv))
                if task == "classification":
                    val_metric = float((out.argmax(1) == yv).float().mean())
                else:
                    rmse_std = float(torch.sqrt(((out.squeeze(-1) - yv) ** 2).mean()))
                    val_metric = rmse_std * ((prep.get("y_stats") or {}).get("std", 1.0))  # 还原为原始量纲

            # 监控量：分类看验证损失（比准确率平滑），回归看 RMSE
            monitor = val_loss if task == "classification" else val_metric
            if monitor < best_score - 1e-6:
                best_score, best_ep, since_best = monitor, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1

            if scheduler is not None:
                scheduler.step()

            rec = {"epoch": ep, "train_loss": round(train_loss, 5),
                   "val_loss": round(val_loss, 5), mname: round(val_metric, 5)}
            history.append(rec)
            logs = _log(logs, f"epoch {ep}/{epochs} · loss {train_loss:.4f} · val_loss {val_loss:.4f} · {mname} {val_metric:.4f}")
            advice = diagnose(history, task, mname, n, patience, since_best)
            db.update_job(job_id, epoch=ep, progress=ep / epochs, logs=logs,
                          metrics={"history": history, "task": task, "arch": arch, "advice": advice})

            if patience and since_best >= patience:
                stop_reason = f"早停：验证集指标连续 {patience} 轮未改善，于第 {ep} 轮提前终止"
                break
            if not np.isfinite(train_loss) or train_loss > 1e6:
                stop_reason = f"训练异常中止：损失值发散（{train_loss:.4f}），请降低学习率后重试"
                break

        # 回滚至最佳权重，避免用已过拟合的末轮权重交付
        if best_ep and best_ep < len(history):
            model.load_state_dict(best_state)
            logs = _log(logs, f"已回滚至第 {best_ep} 轮的最佳权重（而非末轮）")

        # ---- 测试集最终评估（AI 锁定的 test 划分，全程不参与训练与调参）----
        test_metrics: dict = {}
        if len(te) > 0:
            model.eval()
            with torch.no_grad():
                te_correct, te_total = 0, 0
                te_sq_err = 0.0
                for bi in batches(te):
                    out = forward_batch(bi)
                    yt = y_all[bi]
                    if task == "classification":
                        te_correct += int((out.argmax(1) == yt).sum())
                        te_total += len(yt)
                    else:
                        te_sq_err += float(((out.squeeze(-1) - yt) ** 2).sum())
                        te_total += len(yt)
                if te_total:
                    if task == "classification":
                        test_metrics = {"test_acc": round(te_correct / te_total, 5),
                                        "test_samples": te_total}
                    else:
                        rmse = float(np.sqrt(te_sq_err / te_total))
                        test_metrics = {"test_rmse": round(
                            rmse * ((prep.get("y_stats") or {}).get("std", 1.0)), 5),
                            "test_samples": te_total}
            logs = _log(logs, "测试集评估（全程未参与训练）：" + " · ".join(
                f"{k}={v}" for k, v in test_metrics.items()))
            db.update_job(job_id, logs=logs)

        # ---- 保存模型 ----
        model_path = MODELS_DIR / f"{job_id}.pt"
        arch_params = {"hidden": hidden, "dropout": dropout, "img_size": IMG_SIZE, "text_dim": TEXT_DIM}
        tcfg = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "val_split": val_split,
                "optimizer": optimizer, "weight_decay": weight_decay, "scheduler": sched_name,
                "dropout": dropout, "hidden": hidden, "class_weight": use_cw, "patience": patience}
        torch.save({
            "state_dict": model.state_dict(),
            "meta": {"task": task, "arch": arch, "task_hint": task_hint, "prep": prep,
                     "target_col": d.get("target_col"), "dataset_id": d["id"], "dataset_name": d["name"],
                     # 结构参数必须随模型保存：推理端需按同样结构重建网络
                     "arch_params": arch_params, "config": tcfg},
        }, model_path)
        final = {"history": history, "task": task, "arch": arch, "best_val": history[-1],
                 "best_epoch": best_ep, "stopped_early": bool(stop_reason), "stop_reason": stop_reason,
                 "advice": diagnose(history, task, mname, n, patience, since_best),
                 "config": tcfg, "arch_params": arch_params,
                 # 划分锁定信息：让前端能明确展示「哪些数据参与了训练」
                 "split_info": {
                     "source": splits["source"],
                     "note": splits["note"],
                     "train": int(len(ti)), "val": int(len(vi)), "test": int(len(te)),
                 },
                 "test_metrics": test_metrics}
        # 标注这是回退路线产出的模型，前端与报告必须能看见
        if _fallback_reason:
            final["fallback"] = True
            final["fallback_reason"] = _fallback_reason
            logs = _log(logs, f"注意：自动建模失败（{_fallback_reason}），"
                              "本结果是基础训练路线产出，指标不可与自动建模结果直接比较")
        msg = stop_reason or f"训练完成，共 {len(history)} 轮"
        db.update_job(job_id, status="done", progress=1.0, metrics=final,
                      model_path=str(model_path), logs=_log(logs, f"{msg}，模型已保存：{model_path.name}"))
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=str(e)[:500], logs=_log(logs, f"训练失败：{e}"))


# ---------- API ----------

def auto_params(task: str | None, rows: int) -> dict:
    """按任务类型与样本量推荐训练参数（auto 模式）。

    规则与前端向导保持一致：样本越少训练轮次越多、批越小、学习率越大；
    图像任务（CNN）整体收敛慢，轮次相对保守。
    """
    rows = int(rows or 0)
    if task == "object_detection":
        # 检测器收敛需要更多轮次；CPU 上 batch 要小
        if rows < 100:
            return {"epochs": 60, "lr": 0.01, "batch_size": 8}
        if rows < 1000:
            return {"epochs": 40, "lr": 0.01, "batch_size": 16}
        return {"epochs": 25, "lr": 0.01, "batch_size": 16}
    if task == "image_classification":
        if rows < 500:
            return {"epochs": 15, "lr": 0.005, "batch_size": 16}
        if rows < 5000:
            return {"epochs": 8, "lr": 0.003, "batch_size": 32}
        return {"epochs": 5, "lr": 0.001, "batch_size": 64}
    if rows < 500:
        return {"epochs": 30, "lr": 0.01, "batch_size": 16}
    if rows < 5000:
        return {"epochs": 20, "lr": 0.005, "batch_size": 32}
    if rows < 50000:
        return {"epochs": 10, "lr": 0.003, "batch_size": 64}
    return {"epochs": 5, "lr": 0.001, "batch_size": 128}


@router.post("/api/train")
def create_job(cfg: dict = Body(...), user: dict = Depends(auth.current_user)) -> dict:
    did = cfg.get("dataset_id")
    ds = db.get_dataset(did) if did else None
    if not ds:
        raise HTTPException(404, "数据集不存在")

    # ---- G3 训练准入：能力矩阵 + 数据集就绪度 ----
    # 以前这里零校验，单类别、标签泄漏、把信号列当标签的数据都能建任务。
    contract = preflight.ensure_contract(ds)
    gate = spec.gate_training(
        ds.get("task_hint") or "",
        dataset_status=ds.get("status") or "ready",
        dataset_gate=contract.get("gate"),
        degraded=bool(cfg.get("degraded")),
        allow_degraded=bool(cfg.get("allow_degraded")),
    )
    if not gate.ok:
        raise spec.GateBlocked(gate, http_status=422)

    config = {k: v for k, v in cfg.items() if k != "dataset_id"}
    if config.get("auto"):
        # 自动调参：epochs/lr/batch_size 由样本量决定，覆盖手填值
        config.update(auto_params(ds.get("task_hint"), ds.get("rows") or 0))
    job = {
        "id": f"job_{uuid.uuid4().hex[:10]}",
        "dataset_id": did,
        "config": config,
    }
    db.insert_job(job)
    threading.Thread(target=run_job, args=(job["id"],), daemon=True).start()
    out = db.get_job(job["id"])
    out["gate"] = gate.as_dict()
    return out


@router.get("/api/train")
def list_jobs(user: dict = Depends(auth.current_user)) -> list[dict]:
    return db.list_jobs()


@router.get("/api/train/{jid}")
def get_job(jid: str, user: dict = Depends(auth.current_user)) -> dict:
    j = db.get_job(jid)
    if not j:
        raise HTTPException(404, "训练任务不存在")
    d = db.get_dataset(j["dataset_id"])
    j["dataset_name"] = d["name"] if d else None
    j["task_hint"] = d.get("task_hint") if d else None
    return j
