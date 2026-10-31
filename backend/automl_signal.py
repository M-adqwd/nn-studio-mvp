"""1D 生理信号建模（心电 / 脑电 / 脉搏波等）。

平台原有五条建模路径都是「图像 / 文本 / 表格 / 检测」，没有信号通道：
心电图 12 导联 10 秒的原始波形既不是图像也不是表格特征，
用 GBDT 吃 1000 个采样点只能学到极粗糙的形态，用图像路径要先把波形画成图，都不正经。

本模块补上这条通道，要点：

1. **两种数据形态**
   - 宽表多导联：列名 `<lead>_<序号>`（如 `II_0 … II_999`），按前缀还原成 (导联, 采样点)
   - 单导联平铺：`x0 … x186`，视为 1 导联

2. **多标签**：一份心电图常同时有多个诊断码（PTB-XL 的 SCP 语句），
   `label` 形如 `MI|STTC` 时走 BCE + 逐类阈值，而不是 softmax 互斥。

3. **按患者切分**：`trainer.resolve_splits` 已支持按 patient 列分组，
   同一患者的所有记录只落在一个划分里——心电尤其致命，
   同一病人多次复查的波形高度相似，随机切分能让 AUC 虚高十几个点。

4. **指标**：单标签给出 acc / macro-F1 / QWK（有序分级），
   多标签给出 macro-AUC / macro-F1 / 每类阈值。
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import db
import metrics_extra as mx
import train as trainer

MAX_EPOCHS = 40
PATIENCE = 6


# ---------- 数据形态识别 ----------

_SPLIT_LIKE = {"split", "subset", "partition", "fold", "dataset_split",
               "data_split", "split_type", "set_split", "filepath"}


def signal_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    drop = {target_col, *[c for c in df.columns
                          if str(c).strip().lower() in _SPLIT_LIKE]}
    gcol = trainer.detect_group_col(df)
    if gcol:
        drop.add(gcol)
    return [str(c) for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def detect_layout(df: pd.DataFrame, target_col: str) -> dict:
    """识别导联与采样长度。

    返回 {"leads": [[列名…], …], "length": int, "shape": "multi"|"single"}
    """
    cols = signal_columns(df, target_col)
    if not cols:
        raise ValueError("未找到数值型信号列，无法按 1D 信号建模")

    groups: dict[str, list[tuple[int, str]]] = {}
    for c in cols:
        m = re.match(r"^(.*?)[_\-]?(\d+)$", c)
        if m:
            groups.setdefault(m.group(1) or "lead", []).append((int(m.group(2)), c))
        else:
            groups.setdefault("lead", []).append((-1, c))

    real = {k: v for k, v in groups.items() if len(v) >= 32}
    if len(real) >= 2:
        # 多导联：每个导联的点数应一致
        lengths = {len(v) for v in real.values()}
        if len(lengths) == 1:
            leads = []
            for k in sorted(real, key=lambda x: (len(x), x)):
                leads.append([c for _, c in sorted(real[k])])
            return {"leads": leads, "length": lengths.pop(), "shape": "multi"}

    # 单导联：按尾部序号排序，没有序号就按原始列序
    ordered = sorted(groups.get("lead", []), key=lambda t: t[0])
    if len(ordered) < len(cols):          # 有多组但长度不一致，退化成按列序平铺
        ordered = [(-i, c) for i, c in enumerate(cols)]
    return {"leads": [[c for _, c in ordered]], "length": len(ordered), "shape": "single"}


def build_tensor(df: pd.DataFrame, layout: dict) -> np.ndarray:
    """(N, 导联数, 采样点)，并按每条记录做 z-score。

    每条记录单独标准化比全局标准化更稳：不同设备的增益、电极阻抗差异
    会整体缩放波形，逐条归一把这部分差异先消掉，模型只看形态。
    """
    arr = np.stack([df[cols].to_numpy(dtype=np.float32) for cols in layout["leads"]], axis=1)
    mu = arr.mean(axis=2, keepdims=True)
    sd = arr.std(axis=2, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((arr - mu) / sd).astype(np.float32)


# ---------- 标签 ----------

_SEPS = ("|", ";", ",")


def parse_labels(series: pd.Series) -> tuple[np.ndarray, list[str], bool]:
    """返回 (y, classes, multilabel)。

    多标签的判定看「有没有分隔符」而不是看唯一值个数：
    单标签数据也可能有很多类别，但不会出现 `MI|STTC` 这种组合写法。
    """
    vals = series.astype(str).str.strip().tolist()
    multilabel = any(any(s in v for s in _SEPS) for v in vals)
    if not multilabel:
        classes = sorted(set(vals))
        idx = {c: i for i, c in enumerate(classes)}
        return np.array([idx[v] for v in vals], dtype=np.int64), classes, False

    codes: list[set[str]] = []
    vocab: list[str] = []
    for v in vals:
        parts = {p.strip() for p in re.split(r"[|;,]", v) if p.strip()}
        codes.append(parts)
        for p in parts:
            if p not in vocab:
                vocab.append(p)
    vocab = sorted(vocab)
    vi = {c: i for i, c in enumerate(vocab)}
    y = np.zeros((len(vals), len(vocab)), dtype=np.float32)
    for r, parts in enumerate(codes):
        for p in parts:
            y[r, vi[p]] = 1.0
    return y, vocab, True


# ---------- 网络 ----------

class ResBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 2, k: int = 7):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=p, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.act = nn.GELU()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout))

    def forward(self, x):
        s = x if self.short is None else self.short(x)
        x = self.act(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.act(x + s)


class SignalNet(nn.Module):
    """轻量 1D ResNet：CPU 上单条推理毫秒级。"""

    def __init__(self, n_leads: int, n_out: int, width: int = 48, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, width, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.Sequential(
            ResBlock1d(width, width, stride=1),
            ResBlock1d(width, width * 2, stride=2),
            ResBlock1d(width * 2, width * 4, stride=2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 4, n_out)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.drop(x))


def build_net(meta: dict) -> SignalNet:
    net = SignalNet(int(meta["n_leads"]), int(meta["n_out"]),
                    int(meta.get("width") or 48), float(meta.get("dropout") or 0.2))
    return net


# ---------- 训练 ----------

def _evaluate(net, X, y, multilabel: bool, classes: list[str],
              thresholds: np.ndarray | None = None) -> dict:
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            outs.append(net(torch.from_numpy(X[i:i + 256])))
    logits = torch.cat(outs).numpy()
    if multilabel:
        prob = 1.0 / (1.0 + np.exp(-logits))
        m = mx.multilabel_metrics(y, prob, classes, thresholds)
        m["acc"] = m.get("exact_match")
        return m
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = prob.argmax(1)
    yt = np.asarray(y, dtype=np.int64)
    acc = float((pred == yt).mean()) if len(yt) else 0.0
    per, f1s = [], []
    for c in range(len(classes)):
        tp = int(((pred == c) & (yt == c)).sum())
        fp = int(((pred == c) & (yt != c)).sum())
        fn = int(((pred != c) & (yt == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
        per.append({"class": classes[c], "class_index": c,
                    "support": int((yt == c).sum()), "precision": round(prec, 4),
                    "recall": round(rec, 4), "f1": round(f1, 4)})
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(yt.tolist(), pred.tolist()):
        cm[t][p] += 1
    return {"acc": round(acc, 5),
            "macro_f1": round(float(np.mean(f1s)), 5) if f1s else 0.0,
            "qwk": mx.quadratic_kappa(yt, pred, len(classes)),
            "per_class": per, "confusion_matrix": cm.tolist()}


def _pos_weight(y: np.ndarray) -> torch.Tensor | None:
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 0.2, 20.0), dtype=torch.float32)


def tune_thresholds(y_val: np.ndarray, prob_val: np.ndarray) -> np.ndarray:
    """逐类在验证集上找 F1 最优阈值。

    多标签统一 0.5 阈值几乎必然让稀有类全判 0，
    临床上「漏诊」比「多报一个待排查」代价高得多，所以按类调阈值。
    """
    ths = []
    for i in range(y_val.shape[1]):
        col = y_val[:, i]
        best_t, best_f1 = 0.5, -1.0
        if col.sum() == 0:
            ths.append(0.5)
            continue
        for t in np.arange(0.05, 0.951, 0.05):
            pred = (prob_val[:, i] >= t).astype(int)
            tp = int(((pred == 1) & (col == 1)).sum())
            fp = int(((pred == 1) & (col == 0)).sum())
            fn = int(((pred == 0) & (col == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        ths.append(best_t)
    return np.array(ths, dtype=np.float32)


def auto_train_signal(job_id: str, job: dict, d: dict) -> None:
    from automl import Progress

    cfg = job.get("config") or {}
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        db.update_job(job_id, logs=logs)

    prog = Progress(job_id, logs, weights=(0.0, 1.0))
    try:
        db.update_job(job_id, status="running", progress=0.02)
        prog.stage_begin("读取信号数据", 0.02)

        df = pd.read_csv(d["file_path"])
        target = d.get("target_col") or "label"
        layout = detect_layout(df, target)
        X = build_tensor(df, layout)
        y, classes, multilabel = parse_labels(df[target])
        log(f"信号形态：{layout['shape']} · {X.shape[1]} 导联 × {X.shape[2]} 采样点 · "
            f"{len(df)} 条记录")
        log(f"标签：{'多标签' if multilabel else '单标签'} · {len(classes)} 类"
            + ("" if not multilabel else f" · 平均每例 {float(y.sum(1).mean()):.2f} 个诊断码"))

        val_split = min(max(float(cfg.get("val_split", 0.2)), 0.1), 0.4)
        splits = trainer.resolve_splits(df, val_split)
        ti, vi, te = splits["train"], splits["val"], splits["test"]
        if len(ti) == 0 or len(vi) == 0:
            raise ValueError("划分后训练集或验证集为空")
        if len(te):
            leak = set(ti.tolist()) & set(te.tolist())
            if leak:
                raise ValueError(f"划分异常：{len(leak)} 条测试样本同时出现在训练集，已中止")
        log(f"数据划分：{splits['note']} —— 训练 {len(ti)} · 验证 {len(vi)}"
            + (f" · 测试 {len(te)}（不参与训练）" if len(te) else ""))

        torch.manual_seed(42)
        net = build_net({"n_leads": X.shape[1], "n_out": len(classes),
                         "width": int(cfg.get("width") or 48),
                         "dropout": float(cfg.get("dropout") or 0.2)})
        opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.get("lr") or 1e-3),
                                weight_decay=1e-4)
        epochs = int(cfg.get("epochs") or 24)
        epochs = max(4, min(epochs, MAX_EPOCHS))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=float(cfg.get("lr") or 1e-3), total_steps=epochs,
            pct_start=0.25)
        bs = int(cfg.get("batch_size") or 64)

        Xtr, ytr = X[ti], y[ti]
        Xva, yva = X[vi], y[vi]

        if multilabel:
            lossf = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(ytr))
            ytr_t = torch.from_numpy(ytr)
            yva_t = torch.from_numpy(yva)
        else:
            cnt = np.bincount(ytr, minlength=len(classes)).astype(np.float64)
            inv = cnt.sum() / (len(classes) * np.maximum(cnt, 1))
            # 轻度不平衡用平方根软化（避免过拟少数类）；
            # 但极端不平衡时软化等于没加 —— MIT-BIH 四分类里 F 类只占 5%，
            # 平方根权重下 F1 只有 0.001（748 条全部被判成别的类，Day 5 实测）。
            # 失衡比超过 8 倍就改用完整逆频率。
            ratio = float(cnt.max() / max(cnt.min(), 1.0))
            w = inv if ratio >= 8 else np.sqrt(inv)
            lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
            ytr_t = torch.from_numpy(ytr)
            yva_t = torch.from_numpy(yva)

        history: list[dict] = []
        best = {"score": -1e9, "state": None, "epoch": 0}
        bad = 0
        for ep in range(1, epochs + 1):
            net.train()
            perm = np.random.default_rng(ep).permutation(len(Xtr))
            tot, nseen = 0.0, 0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                xb = torch.from_numpy(Xtr[idx])
                yb = ytr_t[idx]
                opt.zero_grad()
                out = net(xb)
                loss = lossf(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                tot += float(loss) * len(idx)
                nseen += len(idx)
            try:
                sched.step()
            except Exception:  # noqa: BLE001
                pass

            ev = _evaluate(net, Xva, yva, multilabel, classes)
            score = float(ev["macro_auc"]) if (multilabel and ev.get("macro_auc")) else float(ev["macro_f1"])
            rec = {"epoch": ep, "train_loss": round(tot / max(nseen, 1), 4),
                   "val_loss": None, "val_acc": ev.get("acc"),
                   "val_f1": ev.get("macro_f1"), "val_auc": ev.get("macro_auc")}
            history.append(rec)
            log(f"  epoch {ep:02d} loss {rec['train_loss']} · "
                f"{'auc' if multilabel else 'f1'} {score:.4f}")
            prog.epoch(ep, epochs, history, rec)

            if score > best["score"] + 1e-5:
                best = {"score": score, "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    log(f"  验证指标 {PATIENCE} 轮无提升，提前停止（最佳 epoch {best['epoch']}）")
                    break

        if best["state"] is None:
            raise ValueError("训练未产出可用权重")
        net.load_state_dict(best["state"])

        # 多标签阈值必须在验证集上定，测试集只用来报数
        thresholds = None
        if multilabel:
            net.eval()
            with torch.no_grad():
                pv = 1.0 / (1.0 + torch.exp(-net(torch.from_numpy(Xva)))).numpy()
            thresholds = tune_thresholds(yva, pv)
            log("逐类阈值（验证集标定）：" + ", ".join(
                f"{c}={t:.2f}" for c, t in zip(classes, thresholds)))

        val_ev = _evaluate(net, Xva, yva, multilabel, classes, thresholds)
        test_ev = {}
        if len(te):
            test_ev = _evaluate(net, X[te], y[te], multilabel, classes, thresholds)
            if multilabel:
                log(f"测试集：macro-AUC {test_ev.get('macro_auc')} · "
                    f"macro-F1 {test_ev.get('macro_f1')} · 样本 {len(te)}")
            else:
                log(f"测试集：acc {test_ev.get('acc')} · macro-F1 {test_ev.get('macro_f1')}"
                    + (f" · QWK {test_ev.get('qwk')}" if test_ev.get("qwk") is not None else "")
                    + f" · 样本 {len(te)}")

        split_info = {"source": splits["source"], "note": splits["note"],
                      "train": int(len(ti)), "val": int(len(vi)), "test": int(len(te)),
                      "group_col": splits.get("group_col")}
        meta = {
            "task": "classification",
            "arch": "cnn1d",
            "task_hint": "signal_classification",
            "n_leads": int(X.shape[1]),
            "n_out": int(len(classes)),
            "width": int(cfg.get("width") or 48),
            "dropout": float(cfg.get("dropout") or 0.2),
            "multilabel": bool(multilabel),
            "normalize": "per_record",
            "prep": {"classes": classes, "multilabel": bool(multilabel),
                     "columns": layout["leads"], "length": int(X.shape[2]),
                     "thresholds": (thresholds.tolist() if thresholds is not None else None)},
            "target_col": target,
            "dataset_id": d["id"],
            "dataset_name": d["name"],
            "config": {"epochs": epochs, "batch_size": bs,
                       "lr": float(cfg.get("lr") or 1e-3),
                       "best_epoch": best["epoch"]},
            "split_info": split_info,
            "test_metrics": test_ev,
        }
        model_path = trainer.MODELS_DIR / f"{job_id}.pt"
        torch.save({"state_dict": net.state_dict(), "meta": meta}, model_path)

        hb = {"epoch": best["epoch"], "acc": val_ev.get("acc"),
              "val_acc": val_ev.get("acc"), "macro_f1": val_ev.get("macro_f1"),
              "qwk": val_ev.get("qwk"), "macro_auc": val_ev.get("macro_auc"),
              "per_class": val_ev.get("per_class"),
              "confusion_matrix": val_ev.get("confusion_matrix")}
        metrics = {"history": history, "task": "classification", "arch": "cnn1d",
                   "best_val": hb, "test_metrics": test_ev, "config": meta["config"],
                   "split_info": split_info,
                   "advice": _advice(val_ev, multilabel, len(df), len(classes))}
        db.update_job(job_id, status="done", progress=1.0,
                      model_path=str(model_path), metrics=metrics)
        log(f"信号模型已保存：{model_path.name}")
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=str(e)[:500],
                      logs=logs + [f"信号建模失败：{e}"])


def _advice(ev: dict, multilabel: bool, n: int, nclass: int) -> dict:
    tips, plain = [], []
    if n < 500:
        tips.append("样本量偏小，建议补充记录数或做更强的数据增强（时间平移、导联随机丢弃）")
    if multilabel:
        auc = ev.get("macro_auc")
        if auc is not None and auc < 0.85:
            tips.append("宏平均 AUC 偏低，优先检查稀有诊断码的样本量是否够学")
        plain.append("已按类分别设定判定阈值，稀有类不会因为阈值过高被整体漏掉")
    else:
        f1 = ev.get("macro_f1") or 0
        if f1 < 0.7:
            tips.append("宏 F1 偏低，检查是否为类别严重不平衡，或采样率/导联数与训练数据不一致")
        qwk = ev.get("qwk")
        if qwk is not None:
            plain.append(f"二次加权 kappa {qwk}，反映分级判断与真实等级的接近程度")
    return {"summary": "信号模型训练完成", "tips": tips, "plain_tips": plain}
