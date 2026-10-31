"""1D 信号路径自测：形态识别 / 多标签解析 / 前向与反传 / 指标。

不需要真实数据，用合成的 12 导联波形验证代码通路是否成立，
真实数据（PTB-XL）跑之前先在这里把结构性错误挡掉。
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import automl_signal as sig  # noqa: E402


def synth(n: int = 240, leads: int = 12, length: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    # 三类形态：正弦 / 双峰 / 尖峰 + 噪声，模拟不同节律
    cls = rng.integers(0, 3, size=n)
    data = {}
    for li in range(leads):
        wave = np.zeros((n, length), dtype=np.float32)
        for i, c in enumerate(cls):
            base = np.sin(t * (1 + c) + li * 0.3)
            if c == 1:
                base = base + 0.6 * np.sin(t * 3 + li)
            elif c == 2:
                base = np.sign(base) * np.abs(base) ** 0.4
            wave[i] = base + rng.normal(0, 0.15, length)
        for k in range(length):
            data[f"L{li:02d}_{k}"] = wave[:, k]
    df = pd.DataFrame(data)
    return df, cls


def main() -> int:
    df, cls = synth()
    # 多标签：把类别 2 的记录再叠加一个 "STTC" 诊断码
    labels = ["NORM" if c == 0 else "MI" if c == 1 else "MI|STTC" for c in cls]
    df["label"] = labels
    df["patient_id"] = [f"p{i // 3}" for i in range(len(df))]   # 每组 3 条同病人
    df["split"] = np.where(np.arange(len(df)) % 5 == 0, "test", "train")

    layout = sig.detect_layout(df, "label")
    assert layout["shape"] == "multi", f"应识别为多导联，实际 {layout['shape']}"
    assert len(layout["leads"]) == 12, f"导联数应为 12，实际 {len(layout['leads'])}"
    assert layout["length"] == 250, f"采样点应为 250，实际 {layout['length']}"

    X = sig.build_tensor(df, layout)
    assert X.shape == (240, 12, 250), X.shape
    # 逐条标准化后每个导联均值应接近 0
    assert np.abs(X.mean(axis=2)).max() < 1e-4, "逐条标准化未生效"

    y, classes, multilabel = sig.parse_labels(df["label"])
    assert multilabel, "带 | 的标签应判为多标签"
    assert set(classes) == {"MI", "NORM", "STTC"}, classes
    assert y.shape == (240, 3), y.shape

    # 单标签也要能解析
    y1, c1, ml1 = sig.parse_labels(pd.Series(["a", "b", "a"]))
    assert not ml1 and c1 == ["a", "b"] and y1.tolist() == [0, 1, 0]

    net = sig.build_net({"n_leads": 12, "n_out": len(classes), "width": 24, "dropout": 0.1})
    out = net(torch.from_numpy(X[:4]))
    assert out.shape == (4, 3), out.shape

    # 前向 + 反传 + 阈值标定整条链路
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(2):
        opt.zero_grad()
        loss = lossf(net(torch.from_numpy(X[:32])), torch.from_numpy(y[:32]))
        loss.backward()
        opt.step()

    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(X))).numpy()
    thr = sig.tune_thresholds(y, prob)
    assert thr.shape == (3,), thr.shape

    ev = sig._evaluate(net, X, y, True, classes, thr)
    for k in ("macro_auc", "macro_f1", "per_class"):
        assert k in ev, f"评估结果缺 {k}"

    # 单标签分支也要能算
    net2 = sig.build_net({"n_leads": 12, "n_out": 3, "width": 16, "dropout": 0.0})
    ev2 = sig._evaluate(net2, X, np.array(cls), False, ["NORM", "MI", "STTC"])
    assert "macro_f1" in ev2

    print("信号路径自测通过")
    print(f"  形态 {layout['shape']} · {X.shape[1]}×{X.shape[2]} · 多标签 {multilabel} · 类别 {classes}")
    print(f"  多标签指标 macro_auc={ev.get('macro_auc')} macro_f1={ev.get('macro_f1')}")
    print(f"  逐类阈值 {np.round(thr, 2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
