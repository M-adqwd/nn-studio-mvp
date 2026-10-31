"""补充指标：分级任务的 QWK、类别名回填、多标签宏平均。

平台原有的 acc / macro-F1 对「有序分级」和「多标签判读」都不够用：
- 视网膜病变、糖网分级是 0~4 级有序标签，临床与竞赛口径都用二次加权 kappa（QWK）；
  用 macro-F1 会把「判成相邻一级」和「判成最重一级」同等惩罚，与临床风险不符。
- 心电图一份报告常有多个诊断码，需要一对其余 + 宏平均。

这些指标集中在这里，避免 automl 四条建模路径各写一遍。
"""
from __future__ import annotations

import numpy as np


def quadratic_kappa(y_true, y_pred, n_class: int | None = None) -> float | None:
    """二次加权 kappa。二分类没有「相邻一级」的概念，返回 None。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or (n_class is not None and n_class < 3):
        return None
    try:
        from sklearn.metrics import cohen_kappa_score

        labels = list(range(n_class)) if n_class else None
        v = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return round(float(v), 5)
    except Exception:  # noqa: BLE001
        # 某一类在 y 里完全缺失时 sklearn 会抛，指标缺失不影响主流程
        return None


def name_per_class(per_class: list[dict], classes: list[str] | None) -> list[dict]:
    """把 per_class 里的 class 序号换成真实类别名。

    客户拿到报告时看到的必须是 "Moderate DR" 而不是 "2"。
    """
    if not classes:
        return per_class
    out = []
    for item in per_class:
        it = dict(item)
        idx = it.get("class")
        if isinstance(idx, int) and 0 <= idx < len(classes):
            it["class_index"] = idx
            it["class"] = str(classes[idx])
        out.append(it)
    return out


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str],
                       thresholds: np.ndarray | float | None = None) -> dict:
    """多标签：宏平均 AUC / F1 / 每类阈值。

    PTB-XL 这类数据一份心电图有多个 SCP 诊断码，softmax 单标签会把
    「同时有 MI 和 STTC」的病人强行压成一个类，指标和临床都对不上。
    """
    from sklearn.metrics import f1_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = y_true.shape[1]
    if isinstance(thresholds, (int, float)) or thresholds is None:
        thr = np.full(n, float(thresholds if thresholds is not None else 0.5))
    else:
        thr = np.asarray(thresholds, dtype=np.float64)
    y_pred = (y_prob >= thr.reshape(1, -1)).astype(np.int8)

    aucs, per_class = [], []
    for i, name in enumerate(classes):
        col = y_true[:, i]
        auc = None
        if col.min() != col.max():          # 该类在测试集里全是 0 或全是 1 时 AUC 无定义
            try:
                auc = float(roc_auc_score(col, y_prob[:, i]))
            except Exception:  # noqa: BLE001
                auc = None
        if auc is not None:
            aucs.append(auc)
        tp = int(((y_pred[:, i] == 1) & (col == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (col == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (col == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_class.append({
            "class": name, "support": int(col.sum()),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "auc": round(auc, 4) if auc is not None else None,
            "threshold": round(float(thr[i]), 3),
        })
    return {
        "macro_auc": round(float(np.mean(aucs)), 5) if aucs else None,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 5),
        "per_class": per_class,
        "exact_match": round(float((y_pred == y_true).all(axis=1).mean()), 5),
    }
