"""切分与指标的回归验证（不需要起服务）。

覆盖三件事：
1. 无自带划分时必须切出独立 test，且 test 与 train/val 零交集；
2. 有 patient 列时按患者分组，同一患者不跨划分；
3. QWK 在有序分级上能算出来，且「判成相邻一级」比「判成最远一级」得分更高。

跑法：cd backend && python tests/test_split_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import metrics_extra as mx  # noqa: E402
import train as trainer  # noqa: E402


def _df(n=600, with_patient=True, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])
    d = {"label": labels, "x": rng.normal(size=n)}
    if with_patient:
        # 每个患者 3 条记录，模拟同一病人多次检查
        d["patient_id"] = [f"p{i // 3}" for i in range(n)]
    return pd.DataFrame(d)


def test_no_native_split():
    df = _df(with_patient=False)
    s = trainer.resolve_splits(df, 0.2)
    assert len(s["test"]) > 0, "无自带划分时必须切出测试集"
    assert len(s["val"]) > 0
    tri = set(s["train"].tolist())
    vai = set(s["val"].tolist())
    tei = set(s["test"].tolist())
    assert not (tri & tei) and not (vai & tei) and not (tri & vai), "划分之间必须有交集检查"
    assert len(tri) + len(vai) + len(tei) == len(df), "划分应覆盖全部样本"
    print(f"  ✓ 无自带划分：train/val/test = {len(tri)}/{len(vai)}/{len(tei)}")
    print(f"    说明：{s['note']}")


def test_group_split():
    df = _df(with_patient=True)
    s = trainer.resolve_splits(df, 0.2)
    g = df["patient_id"].astype(str)
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        ga = set(g.iloc[s[a]].tolist())
        gb = set(g.iloc[s[b]].tolist())
        assert not (ga & gb), f"{a} 与 {b} 存在同一患者：{list(ga & gb)[:3]}"
    print(f"  ✓ 按患者分组：train/val/test = "
          f"{len(s['train'])}/{len(s['val'])}/{len(s['test'])}，"
          f"无患者跨划分（group_col={s.get('group_col')}）")


def test_native_split_keeps_test():
    df = _df(n=300, with_patient=False)
    df["split"] = ["train"] * 200 + ["test"] * 100
    s = trainer.resolve_splits(df, 0.2)
    assert len(s["test"]) == 100, "自带测试集必须原样保留，不能被重新切"
    assert not (set(s["train"].tolist()) & set(s["test"].tolist()))
    print(f"  ✓ 自带测试集保留：test={len(s['test'])}，val 从 train 切出 {len(s['val'])}")


def test_native_train_only_gets_test():
    """只有 train 列的数据集（Kaggle 常见形态）必须自动补出测试集。"""
    df = _df(n=400, with_patient=False)
    df["split"] = "train"
    s = trainer.resolve_splits(df, 0.2)
    assert len(s["test"]) > 0, "只有 train 时必须切出测试集"
    assert not (set(s["train"].tolist()) & set(s["test"].tolist()))
    print(f"  ✓ 只有 train：自动补出 test={len(s['test'])}  val={len(s['val'])}")


def test_qwk():
    y = np.array([0, 1, 2, 3, 4] * 40)
    near = np.where(y > 0, y - 1, y)          # 都判低一级
    far = np.where(y < 2, 4, 0)               # 判到相反一端
    q_near = mx.quadratic_kappa(y, near, 5)
    q_far = mx.quadratic_kappa(y, far, 5)
    assert q_near is not None and q_far is not None
    assert q_near > q_far, f"相邻一级的 QWK({q_near}) 应高于最远一级({q_far})"
    print(f"  ✓ QWK：相邻误判 {q_near} > 反向误判 {q_far}")
    print(f"  ✓ 二分类不返回 QWK：{mx.quadratic_kappa(np.array([0, 1, 0]), np.array([0, 1, 1]), 2)}")


def test_class_names():
    per = [{"class": 0, "support": 5}, {"class": 2, "support": 3}]
    out = mx.name_per_class(per, ["Healthy", "Mild", "Severe"])
    assert out[0]["class"] == "Healthy" and out[1]["class"] == "Severe"
    print(f"  ✓ 类别名回填：{[o['class'] for o in out]}")


def test_multilabel():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=np.int8)
    p = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3], [0.8, 0.6, 0.2], [0.1, 0.9, 0.7]])
    m = mx.multilabel_metrics(y, p, ["MI", "STTC", "CD"])
    assert m["macro_auc"] is not None and m["macro_f1"] > 0.9
    print(f"  ✓ 多标签：macro_auc={m['macro_auc']} macro_f1={m['macro_f1']} "
          f"exact_match={m['exact_match']}")


if __name__ == "__main__":
    print("切分与指标验证")
    test_no_native_split()
    test_group_split()
    test_native_split_keeps_test()
    test_native_train_only_gets_test()
    test_qwk()
    test_class_names()
    test_multilabel()
    print("\n全部通过")
