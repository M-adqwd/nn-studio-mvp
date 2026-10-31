"""G2 切分门：把「怎么分训练/验证/测试」从训练代码里独立出来，变成可预览、可校验、可复现的一道门。

为什么需要独立成门
------------------
之前的切分逻辑内联在 `train.py:resolve_splits`，只用「按患者分组后随机分配」。
它在稀有类上会出致命问题：MIT-BIH 的 F 类 802 条里，208 号患者 372 条、213 号 362 条，
两者占 91.5%。随机分配把这两个患者一起划进测试集，训练侧只剩 54 条 F 类，
模型压根没学会，F1 = 0.0011，而 acc 0.804 只是「多数类 76.7%」的假象。

所以这里的核心不是「随机」，而是：
1. 稀有类优先 —— 先分配含稀有类的分组，且必须分散到不同折；
2. 切完必检 —— 每折类别覆盖、组间隔离、索引一致性、分布偏移，全部量化；
3. 检不过就重试或降级，并把代价写清楚，绝不静默放过。

错误码（SOP §6 E4xxx）
-----------------------
E4001 某折（通常是测试集）缺类，无法计算该类指标
E4002 分组隔离失败：同一患者跨折出现，测试集被污染
E4003 训练集某类样本不足，模型无法学到该类
E4004 划分之间样本重叠或索引不完整
E4005 稀有类被少数分组垄断，任意切分都无法兼顾训练与评估
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import spec

FOLDS = ("train", "val", "test")

# 常见「同一患者 / 同一病例」列名。命中则按组切分，避免同一患者的多条记录
# 同时出现在训练与测试，造成指标虚高。
GROUP_MARKER = {
    "patient", "patient_id", "patientid", "pid", "subject_id", "subjectid",
    "record_id", "record", "person_id", "case_id", "study_id", "patient_no", "mrn",
}


# ---------------------------------------------------------------- 标签矩阵

def detect_group_col(df: pd.DataFrame) -> str | None:
    """找患者/病例列。要求取值有重复（nunique 明显小于行数），否则不是分组列。"""
    for c in df.columns:
        name = str(c).strip().lower()
        if name in GROUP_MARKER and df[c].nunique() < len(df) * 0.9:
            return str(c)
    return None


def labels_matrix(df: pd.DataFrame, label_col: str | None = None,
                  classes: list[str] | None = None,
                  multilabel: bool = False) -> tuple[np.ndarray, list[str]]:
    """把标签列转成统一的矩阵。

    单标签 -> (n,) int；多标签 -> (n, C) 0/1。
    多标签支持两种形态：`MI|STTC` 这类竖线分隔的字符串，或每个类别一列 0/1。
    """
    if label_col and label_col in df.columns:
        s = df[label_col]
        if multilabel:
            parts = s.astype(str).str.split(r"[|;,]", regex=True)
            if classes is None:
                found: set[str] = set()
                for p in parts:
                    found.update(x.strip() for x in p if x.strip())
                classes = sorted(found)
            idx = {c: i for i, c in enumerate(classes)}
            m = np.zeros((len(df), len(classes)), dtype=np.float32)
            for r, p in enumerate(parts):
                for x in p:
                    j = idx.get(x.strip())
                    if j is not None:
                        m[r, j] = 1.0
            return m, list(classes)
        codes, uniques = pd.factorize(s.astype(str), sort=True)
        return codes.astype(np.int64), [str(u) for u in uniques]

    # 没有单一 label 列：尝试「每类一列 0/1」的多标签布局
    if classes:
        cols = [c for c in classes if c in df.columns]
        if len(cols) == len(classes):
            return df[cols].to_numpy(dtype=np.float32), list(classes)

    raise ValueError("找不到可用的标签列，请在数据集契约里指定 target_col")


# ---------------------------------------------------------------- 计划对象

@dataclass
class SplitPlan:
    strategy: str
    splits: dict[str, np.ndarray]
    classes: list[str]
    group_col: str | None
    fracs: dict[str, float]
    report: spec.GateReport
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    attempts: int = 1
    source: str = "generated"

    @property
    def ok(self) -> bool:
        return self.report.ok

    def index_hash(self) -> str:
        """切分指纹：同样的索引集合、同样的顺序 => 同一个指纹。用于溯源与复现比对。"""
        h = hashlib.sha256()
        for f in FOLDS:
            h.update(f.encode())
            h.update(np.sort(self.splits[f]).astype(np.int64).tobytes())
        return h.hexdigest()[:16]

    def as_dict(self, with_index: bool = False) -> dict:
        out = {
            "ok": self.ok,
            "strategy": self.strategy,
            "source": self.source,
            "group_col": self.group_col,
            "fracs": self.fracs,
            "counts": {f: int(len(self.splits[f])) for f in FOLDS},
            "per_class": self.per_class,
            "attempts": self.attempts,
            "index_hash": self.index_hash(),
            "notes": self.notes,
            "gate": self.report.as_dict(),
        }
        if with_index:
            out["splits"] = {f: self.splits[f].tolist() for f in FOLDS}
        return out


# ---------------------------------------------------------------- 稀有类优先贪心

def _rarity_score(gid: str, gcounts: dict[str, np.ndarray],
                  global_counts: np.ndarray) -> float:
    """分组含越稀有的类，分数越高，越应该先分配。

    用「该类在分组里的数量 / 该类全局数量」求和：稀有类分母小，权重自然被放大。
    这样 208 号患者（372 条 F）会排在最前面先被安排。
    """
    cnt = gcounts[gid]
    score = 0.0
    for c in np.nonzero(cnt)[0]:
        score += float(cnt[c]) / max(float(global_counts[c]), 1.0)
    return score


def _greedy_assign(gcounts: dict[str, np.ndarray], global_counts: np.ndarray,
                   fracs: dict[str, float], rare_threshold: float = 0.05,
                   min_rare_per_fold: int = 30,
                   rng: np.random.Generator | None = None) -> dict[str, str]:
    """把每个分组整体分配到一折。

    分两阶段，因为「按比例分配」在极度失衡的数据上必然失败：
    MIT-BIH 的 N 类占 90%，任何以绝对数量为代价的函数都会被 N 完全主导，
    结果就是把所有小分组堆进同一折（实测曾切出 train 4856 / val 50271 / test 45506）。

    阶段 1（稀有类下限）：对每个稀有类，按「该类数量」降序把含它的分组分配给
    当前该类样本最少的折，直到每折都达到下限。大分组因此必然被拆到不同折——
    208 号患者（372 条 F）进训练，213 号（362 条 F）进测试，两边都能见到 F。

    阶段 2（比例填充）：剩下不含稀有类的分组，按「相对偏差」填充。
    相对偏差 = |实际-目标|/目标，让每个类的权重相当，不被多数类淹没。
    """
    total = float(global_counts.sum())
    n_c = len(global_counts)
    cur = {f: np.zeros(n_c, dtype=float) for f in FOLDS}
    assign: dict[str, str] = {}
    unassigned = set(gcounts.keys())

    rare = [c for c in range(n_c)
            if global_counts[c] > 0 and global_counts[c] / total < rare_threshold]
    # 下限随数据规模自适应：小数据集别硬要 30 条
    floor = int(min(min_rare_per_fold,
                    max(3, min(global_counts[c] for c in rare) / (len(FOLDS) * 2)))) \
        if rare else 0
    floor = max(floor, 2) if rare else 0

    # 平局顺序：train 先（模型要学），其次 test（评估必须缺一不可），最后 val
    tie = {"train": 0.0, "test": 1e-9, "val": 2e-9}

    for c in sorted(rare, key=lambda x: global_counts[x]):
        cand = [g for g in unassigned if gcounts[g][c] > 0]
        if not cand:
            continue
        cand.sort(key=lambda g: (-gcounts[g][c], g))
        for g in cand:
            if all(cur[f][c] >= floor for f in FOLDS):
                break
            deficits = {f: floor - cur[f][c] for f in FOLDS}
            best_f = max(FOLDS, key=lambda f: (round(deficits[f], 6), -tie[f]))
            assign[g] = best_f
            cur[best_f] += gcounts[g]
            unassigned.discard(g)

    # 阶段 2：其余按「折完成率优先 + 类别偏差」填充
    target = {f: global_counts * fracs[f] for f in FOLDS}
    target_size = {f: total * fracs[f] for f in FOLDS}
    rest = sorted(unassigned, key=lambda g: -gcounts[g].sum())
    if rng is not None:
        # 同规模分组打散，避免每次切出同一份
        rest = sorted(unassigned, key=lambda g: (-gcounts[g].sum(), rng.random()))
    for g in rest:
        cnt = gcounts[g]
        nz = np.nonzero(cnt)[0]
        best_f, best_cost = None, None
        for f in FOLDS:
            after = cur[f] + cnt
            # 完成率：当前样本量相对目标还差多少。必须用它做主判据——
            # 只用「与目标的绝对差」时，目标最大的折永远显得欠账最多，
            # 结果所有小分组都往小折堆，训练集反而最空（实测 train 44% vs 目标 65%）。
            fill = float(cur[f].sum()) / max(target_size[f], 1.0)
            dev = 0.0
            for c in nz:
                t = target[f][c]
                if t <= 1e-9:
                    # 该折本不该有这个类，塞进去就是污染分布
                    dev += after[c] / max(total, 1.0) * 10.0
                else:
                    dev += abs(after[c] - t) / t
            dev /= max(len(nz), 1)
            cost = fill + 0.3 * dev
            if best_cost is None or cost < best_cost:
                best_f, best_cost = f, cost
        assign[g] = best_f  # type: ignore[assignment]
        cur[best_f] += cnt  # type: ignore[index]

    return assign


# ---------------------------------------------------------------- 切分执行

def _split_by_groups(labels: np.ndarray, groups: np.ndarray, fracs: dict[str, float],
                     n_classes: int, seed: int,
                     rare_threshold: float = 0.05) -> tuple[dict[str, np.ndarray], dict]:
    """分组切分：以分组为最小单位，稀有类优先分配到不同折。"""
    uniq = pd.unique(groups)
    gcounts: dict[str, np.ndarray] = {}
    gindex: dict[str, list[int]] = {}
    multi = labels.ndim == 2
    for i, g in enumerate(groups.tolist()):
        gid = str(g)
        if gid not in gcounts:
            gcounts[gid] = np.zeros(n_classes, dtype=float)
            gindex[gid] = []
        if multi:
            gcounts[gid] += labels[i]
        else:
            gcounts[gid][labels[i]] += 1
        gindex[gid].append(i)

    global_counts = np.zeros(n_classes, dtype=float)
    for v in gcounts.values():
        global_counts += v

    rng = np.random.default_rng(seed)
    assign = _greedy_assign(gcounts, global_counts, fracs,
                            rare_threshold=rare_threshold, rng=rng)

    splits = {f: [] for f in FOLDS}
    for gid, f in assign.items():
        splits[f].extend(gindex[gid])
    out = {f: np.array(sorted(v), dtype=np.int64) for f, v in splits.items()}
    meta = {"n_groups": int(len(uniq)), "groups_per_fold":
            {f: int(sum(1 for x in assign.values() if x == f)) for f in FOLDS}}
    return out, meta


def _split_stratified(labels: np.ndarray, n_classes: int,
                      fracs: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """无分组时的分层切分：按类别比例分到三折，稀有类先保证每折都有。"""
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx_by_class: dict[int, list[int]] = {}
    for i in range(n):
        c = int(labels[i]) if labels.ndim == 1 else int(np.argmax(labels[i]))
        idx_by_class.setdefault(c, []).append(i)

    splits = {f: [] for f in FOLDS}
    for c in sorted(idx_by_class, key=lambda k: len(idx_by_class[k])):
        idxs = idx_by_class[c].copy()
        rng.shuffle(idxs)
        n_c = len(idxs)
        n_test = int(round(n_c * fracs["test"]))
        n_val = int(round(n_c * fracs["val"]))
        if n_c >= 3:
            # 每折至少留 1 条，避免稀有类在某折消失
            n_test = max(1, min(n_test, n_c - 2))
            n_val = max(1, min(n_val, n_c - n_test - 1))
        splits["test"].extend(idxs[:n_test])
        splits["val"].extend(idxs[n_test:n_test + n_val])
        splits["train"].extend(idxs[n_test + n_val:])
    return {f: np.array(sorted(v), dtype=np.int64) for f, v in splits.items()}


def _split_predefined(df: pd.DataFrame, labels: np.ndarray) -> dict[str, np.ndarray] | None:
    """沿用数据集自带的 split 列。"""
    if "split" not in df.columns:
        return None
    sp = (df["split"].astype(str).str.strip().str.lower()
          .replace({"validation": "val", "valid": "val", "dev": "val", "eval": "val",
                    "training": "train", "testing": "test"}))
    out = {f: np.where(sp.values == f)[0].astype(np.int64) for f in FOLDS}
    if len(out["train"]) == 0:
        return None
    return out


# ---------------------------------------------------------------- 校验

def _counts_per_class(labels: np.ndarray, idx: np.ndarray, n_classes: int) -> np.ndarray:
    if len(idx) == 0:
        return np.zeros(n_classes, dtype=int)
    sub = labels[idx]
    if sub.ndim == 2:
        return sub.sum(axis=0).astype(int)
    return np.bincount(sub, minlength=n_classes).astype(int)


def _holdout_idx(labels: np.ndarray, pool: np.ndarray, frac: float,
                 groups: np.ndarray | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """从给定索引池里抽出一份 holdout，返回 (held, rest)。

    有分组时以组为单位整体抽（不拆散患者）；无分组时按类别分层。
    自带 split 列只给了 train 时，用它补出独立的 val / test。
    """
    if frac <= 0 or len(pool) == 0:
        return np.array([], dtype=np.int64), np.sort(pool)
    frac = min(frac, 0.9)
    rng = np.random.default_rng(seed)

    if groups is not None:
        gvals = groups[pool]
        uniq = pd.unique(gvals)
        order = rng.permutation(len(uniq))
        target = int(round(len(pool) * frac))
        held_g: list = []
        n = 0
        for k in order:
            if n >= target:
                break
            g = uniq[k]
            mask = gvals == g
            held_g.append(g)
            n += int(mask.sum())
        mask = np.isin(gvals, held_g)
        return np.sort(pool[mask]), np.sort(pool[~mask])

    held: list[int] = []
    rest: list[int] = []
    sub = labels[pool]
    cls = sub if sub.ndim == 1 else np.argmax(sub, axis=1)
    for c in np.unique(cls):
        idx_c = pool[cls == c]
        k = int(round(len(idx_c) * frac))
        if len(idx_c) >= 3:
            k = max(1, min(k, len(idx_c) - 2))   # 每侧至少留 1 条
        elif len(idx_c) < 2:
            k = 0
        perm = rng.permutation(len(idx_c))
        held.extend(idx_c[perm[:k]].tolist())
        rest.extend(idx_c[perm[k:]].tolist())
    return np.array(sorted(held), dtype=np.int64), np.array(sorted(rest), dtype=np.int64)


def _topup_val(sp: dict[str, np.ndarray], labels: np.ndarray, n_classes: int,
               target: int = 100, seed: int = 0,
               allow_grouped: bool = False) -> dict[str, np.ndarray]:
    """验证集过小时从训练集分层补足。

    自带验证集常常只有几十条，指标波动大到没法选模型（32 条验证集里
    1 条样本就代表 3.1% 的差异）。从训练集借一部分补到合理规模，
    测试集完全不动。
    有分组约束时默认不补——从训练集抽单条会破坏「同一患者不跨折」。
    """
    if len(sp["val"]) >= target or len(sp["train"]) < target * 3:
        return sp
    need = target - len(sp["val"])
    rng = np.random.default_rng(seed)
    train_idx = sp["train"]
    if labels.ndim == 1:
        cls = labels[train_idx]
    else:
        cls = np.argmax(labels[train_idx], axis=1)

    picked: list[int] = []
    per = max(1, need // max(n_classes, 1))
    for c in range(n_classes):
        pool_c = train_idx[cls == c]
        if len(pool_c) <= 1:
            continue
        k = min(per, len(pool_c) - 1)
        picked.extend(rng.choice(pool_c, size=k, replace=False).tolist())
        if len(picked) >= need:
            break
    if len(picked) < need:
        rest = np.setdiff1d(train_idx, np.array(picked, dtype=np.int64))
        if len(rest):
            extra = rng.choice(rest, size=min(need - len(picked), len(rest)), replace=False)
            picked.extend(extra.tolist())
    arr = np.array(sorted(set(picked)), dtype=np.int64)
    if len(arr) == 0:
        return sp
    return {"train": np.setdiff1d(train_idx, arr),
            "val": np.sort(np.concatenate([sp["val"], arr])),
            "test": sp["test"]}


def validate(labels: np.ndarray, splits: dict[str, np.ndarray], classes: list[str],
             groups: np.ndarray | None = None,
             min_train_per_class: int = 20,
             min_test_per_class: int = 10,
             max_ratio: float = 20.0) -> spec.GateReport:
    """G2 判定。四类检查：索引一致性、分组隔离、类别覆盖、分布偏移。"""
    n_classes = len(classes)
    rep = spec.GateReport(gate="G2-split")
    n = len(labels)

    # ---- 1. 索引一致性：不重叠、不越界、并集为全集
    seen = np.zeros(n, dtype=int)
    for f in FOLDS:
        idx = splits[f]
        if len(idx) and (idx.min() < 0 or idx.max() >= n):
            rep.issues.append(spec.Issue("E4004", f"{f} 折包含越界索引", "切分实现异常，请重算"))
            continue
        seen[idx] += 1
    dup = int((seen > 1).sum())
    missing = int((seen == 0).sum())
    if dup:
        rep.issues.append(spec.Issue(
            "E4004", f"有 {dup} 条样本同时落在多个划分里",
            "检查切分逻辑；重叠样本会让测试指标虚高"))
    if missing:
        rep.issues.append(spec.Issue(
            "W4001", f"有 {missing} 条样本未被任何划分使用", "若为有意剔除可忽略", spec.WARN))

    # ---- 2. 分组隔离：同一患者不得跨折
    if groups is not None:
        gsets = {}
        for f in FOLDS:
            if len(splits[f]):
                gsets[f] = set(pd.unique(groups[splits[f]]).tolist())
            else:
                gsets[f] = set()
        cross = (gsets["train"] & gsets["test"]) | (gsets["train"] & gsets["val"]) \
            | (gsets["val"] & gsets["test"])
        if cross:
            rep.issues.append(spec.Issue(
                "E4002", f"有 {len(cross)} 个分组（患者）同时出现在多个划分中",
                "同一患者的多条记录跨训练/测试会导致指标虚高；请按分组切分"))

    # ---- 3. 类别覆盖
    tc = _counts_per_class(labels, splits["train"], n_classes)
    vc = _counts_per_class(labels, splits["val"], n_classes)
    ec = _counts_per_class(labels, splits["test"], n_classes)
    tot = _counts_per_class(labels, np.arange(n), n_classes)

    absent_train = [classes[i] for i in range(n_classes) if tc[i] == 0 and tot[i] > 0]
    if absent_train:
        rep.issues.append(spec.Issue(
            "E4003", f"训练集完全缺少这些类别：{'、'.join(absent_train)}",
            "模型不可能学会没见过的类别；调整切分或补充数据"))
    for i in range(n_classes):
        if tc[i] == 0 and tot[i] == 0:
            continue
        if 0 < tc[i] < min_train_per_class and tot[i] >= min_train_per_class:
            rep.issues.append(spec.Issue(
                "W4002", f"训练集中「{classes[i]}」只有 {tc[i]} 条（建议 ≥{min_train_per_class}）",
                "该类指标会很不稳定，建议补充样本或合并到相邻类", spec.WARN))

    absent_test = [classes[i] for i in range(n_classes) if ec[i] == 0 and tot[i] >= min_test_per_class]
    if absent_test:
        rep.issues.append(spec.Issue(
            "E4001", f"测试集完全缺少这些类别：{'、'.join(absent_test)}",
            "这些类别的指标无法计算，发布的模型不能声称支持它们"))

    # ---- 4. 分布偏移：训练与测试的类别占比相差过大
    if len(splits["train"]) and len(splits["test"]):
        ttr = tc / max(tc.sum(), 1)
        ter = ec / max(ec.sum(), 1)
        for i in range(n_classes):
            if tot[i] < min_test_per_class:
                continue
            a, b = ttr[i], ter[i]
            if a <= 0:
                continue
            ratio = max(a, b) / min(a, b)
            if ratio > max_ratio:
                rep.issues.append(spec.Issue(
                    "W4003",
                    f"「{classes[i]}」在训练集占 {a:.2%}、测试集占 {b:.2%}（相差 {ratio:.0f} 倍）",
                    "分布偏移会让测试指标失真；检查切分是否被少数分组垄断", spec.WARN))

    return rep


# ---------------------------------------------------------------- 主入口

def plan_splits(df: pd.DataFrame, *, label_col: str | None = None,
                classes: list[str] | None = None,
                multilabel: bool = False,
                val_frac: float = 0.2, test_frac: float = 0.15,
                group_col: str | None = None,
                min_train_per_class: int | None = None,
                min_test_per_class: int = 10,
                seed: int | None = None,
                max_attempts: int = 8) -> SplitPlan:
    """算出一份切分方案并给出 G2 判定。

    流程：选定策略 → 切分 → 校验 → 不过关则换种子重试 → 仍不过关就如实报告。
    重试只调种子，不会偷偷改变策略——策略变更必须让人看见。
    """
    labels, class_names = labels_matrix(df, label_col, classes, multilabel)
    if labels.ndim == 1 and class_names:
        n_classes = len(class_names)
    else:
        n_classes = labels.shape[1] if labels.ndim == 2 else len(class_names)
    if min_train_per_class is None:
        # 默认：每个类至少见 20 条，类多时略放宽，免得大数据集被卡
        min_train_per_class = max(20, min(50, int(len(df) / max(n_classes, 1) * 0.02)))

    gcol = group_col if group_col else detect_group_col(df)
    groups = df[gcol].astype(str).to_numpy() if gcol else None
    base_seed = 20240101 if seed is None else int(seed)
    fracs = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}

    # 稀有类阈值随类数自适应：类别少的任务里，10% 的类已经需要保护了。
    # MIT-BIH 只有 4 类，V 占 6.96% 用固定 5% 会漏掉，导致测试集只剩 221 条 V。
    rare_threshold = 0.15 if n_classes <= 5 else 0.05

    # 自带 split 列：优先沿用。自带测试集无论如何都要原样保留——
    # 重新切会让「测试指标」与数据提供方的口径对不上，也破坏可比性。
    preset = _split_predefined(df, labels)
    if preset is not None and len(preset["train"]) > 0:
        sp = {f: np.sort(preset[f]) for f in FOLDS}
        notes: list[str] = []
        native = [f for f in FOLDS if len(sp[f])]

        if len(sp["test"]) == 0:
            te, ti = _holdout_idx(labels, sp["train"], fracs["test"], groups, base_seed + 13)
            sp["test"], sp["train"] = te, ti
            notes.append(f"数据集未自带测试集，已从训练集切出 {len(te)} 条作为独立测试集"
                         "（训练与早停全程不可见）")
        else:
            # 硬隔离：即使上游同一行既标 train 又标 test，也强制从训练侧剔除
            ts = set(sp["test"].tolist())
            for f in ("train", "val"):
                if len(sp[f]):
                    kept = [i for i in sp[f].tolist() if i not in ts]
                    sp[f] = np.array(kept, dtype=np.int64)
            notes.append(f"沿用数据集自带的 {'/'.join(native)} 划分，测试集原样保留")

        if len(sp["val"]) == 0:
            vi, ti = _holdout_idx(labels, sp["train"], fracs["val"], groups, base_seed + 7)
            sp["val"], sp["train"] = vi, ti
            notes.append(f"数据集未自带验证集，已从训练集切出 {len(vi)} 条用于早停与选模型")
        elif not gcol:
            sp = _topup_val(sp, labels, n_classes, seed=base_seed)
            notes.append("验证集已按类别分层补足到合理规模，避免指标波动导致选错模型")

        rep = validate(labels, sp, class_names, groups,
                       min_train_per_class, min_test_per_class)
        return SplitPlan(
            strategy="predefined", splits=sp, classes=class_names, group_col=gcol,
            fracs=fracs, report=rep, source="dataset_split",
            per_class=_per_class_dict(labels, sp, class_names), notes=notes)

    strategy = "group_strat" if gcol else "stratified"
    best: SplitPlan | None = None
    attempts = 0
    for k in range(max_attempts):
        attempts = k + 1
        s = base_seed + k * 7919
        if gcol:
            sp, gmeta = _split_by_groups(labels, groups, fracs, n_classes, s,
                                         rare_threshold=rare_threshold)  # type: ignore[arg-type]
        else:
            sp = _split_stratified(labels, n_classes, fracs, s)
            sp = _topup_val(sp, labels, n_classes, seed=s)
            gmeta = {}
        rep = validate(labels, sp, class_names, groups,
                       min_train_per_class, min_test_per_class)
        plan = SplitPlan(
            strategy=strategy, splits=sp, classes=class_names, group_col=gcol,
            fracs=fracs, report=rep, attempts=attempts,
            per_class=_per_class_dict(labels, sp, class_names),
            notes=_describe(sp, class_names, gcol, gmeta, fracs))
        if best is None or len(rep.blocking) < len(best.report.blocking):
            best = plan
        if rep.ok:
            break
    assert best is not None

    # 仍不过关：把「为什么做不到」讲清楚，而不是抛一句失败
    if not best.report.ok:
        best.notes.append(
            "已尝试 %d 种随机分配仍未通过；稀有类可能被极少数分组垄断，"
            "继续重试不会改善，需要补充样本、合并稀有类，或改用分层抽样并接受患者级泄漏风险"
            % attempts)
        if gcol and any(i.code == "E4003" for i in best.report.blocking):
            rare_hint = _rare_concentration(labels, groups, class_names)
            if rare_hint:
                best.notes.append(rare_hint)
    return best


def _per_class_dict(labels: np.ndarray, splits: dict[str, np.ndarray],
                    classes: list[str]) -> dict[str, dict[str, int]]:
    n_classes = len(classes)
    out: dict[str, dict[str, int]] = {}
    for f in FOLDS:
        c = _counts_per_class(labels, splits[f], n_classes)
        out[f] = {classes[i]: int(c[i]) for i in range(n_classes)}
    return out


def _describe(sp: dict[str, np.ndarray], classes: list[str], gcol: str | None,
              gmeta: dict, fracs: dict[str, float]) -> list[str]:
    notes = []
    if gcol:
        notes.append(
            f"按「{gcol}」分组切分（{gmeta.get('n_groups', '?')} 个分组），"
            f"同一分组不跨折；稀有类优先分散到各折")
    else:
        notes.append("无患者/病例列，按类别分层切分")
    notes.append("训练/验证/测试 = %d/%d/%d（目标比例 %.0f%%/%.0f%%/%.0f%%）" % (
        len(sp["train"]), len(sp["val"]), len(sp["test"]),
        fracs["train"] * 100, fracs["val"] * 100, fracs["test"] * 100))
    return notes


def _rare_concentration(labels: np.ndarray, groups: np.ndarray | None,
                        classes: list[str]) -> str | None:
    """诊断「稀有类被少数分组垄断」：列出把 80% 以上稀有类占掉的分组。"""
    if groups is None or labels.ndim != 1:
        return None
    n_c = len(classes)
    tot = np.bincount(labels, minlength=n_c)
    if tot.sum() == 0:
        return None
    rare_idx = [c for c in range(n_c) if 0 < tot[c] / tot.sum() < 0.05]
    if not rare_idx:
        return None
    msgs = []
    for c in rare_idx:
        mask = labels == c
        vc = pd.Series(groups[mask]).value_counts()
        if len(vc) == 0:
            continue
        head = vc.head(2)
        share = head.sum() / tot[c]
        if share >= 0.8:
            detail = "、".join(f"{g} 号 {n} 条" for g, n in head.items())
            msgs.append(f"「{classes[c]}」共 {tot[c]} 条，其中 {share:.0%} 集中在 {detail}")
    if not msgs:
        return None
    return "稀有类集中度诊断：" + "；".join(msgs) + \
        "——任意分组切分都无法让训练集与测试集都见到足量该类"


def apply_plan(plan: SplitPlan, df: pd.DataFrame) -> pd.DataFrame:
    """把切分结果写回 `split` 列，便于下游按列读取与复现。"""
    out = df.copy()
    col = np.full(len(df), "", dtype=object)
    for f in FOLDS:
        col[plan.splits[f]] = f
    out["split"] = col
    return out
