"""数据分析与清洗的纯函数实现。所有函数都接受 pandas DataFrame 并返回可 JSON 序列化的 dict。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------- 基础探测 ----------

def infer_col_type(s: pd.Series) -> str:
    """识别列类型：number | category | text | datetime | bool"""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    # object：看唯一值比例
    nunique = s.nunique(dropna=True)
    total = max(len(s), 1)
    if nunique <= 1:
        return "category"
    if nunique / total < 0.05 or nunique <= 20:
        return "category"
    return "text"


def build_schema(df: pd.DataFrame, target_col: str | None = None) -> list[dict]:
    """生成字段元数据，与前端 ds.schema 结构对齐。"""
    out = []
    for col in df.columns:
        s = df[col]
        t = infer_col_type(s)
        item: dict[str, Any] = {
            "name": str(col),
            "type": t,
            "role": "target" if (target_col and col == target_col) else "feature",
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if t == "number":
            desc = s.describe()
            item.update(
                min=float(desc["min"]), max=float(desc["max"]),
                mean=float(desc["mean"]), std=float(desc["std"] if not pd.isna(desc["std"]) else 0),
            )
        out.append(item)
    return out


def detect_task(df: pd.DataFrame, target_col: str | None) -> str:
    """根据目标列形态猜测任务类型。"""
    if not target_col or target_col not in df.columns:
        return "tabular_classification"
    s = df[target_col]
    t = infer_col_type(s)
    if t == "number" and s.nunique() > 20:
        return "tabular_regression"
    return "tabular_classification"


def build_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float(df.isna().sum().sum() / max(df.size, 1)),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def preview_rows(df: pd.DataFrame, n: int = 20) -> dict:
    sub = df.head(n)
    return {
        "columns": [str(c) for c in sub.columns],
        "rows": [[None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row]
                 for row in sub.itertuples(index=False, name=None)],
    }


# ---------- EDA ----------

def numeric_distribution(s: pd.Series, bins: int = 10) -> dict:
    s = s.dropna()
    if s.empty:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(s.values, bins=bins)
    return {
        "bins": [float((edges[i] + edges[i + 1]) / 2) for i in range(len(counts))],
        "counts": counts.tolist(),
        "min": float(s.min()), "max": float(s.max()),
        "mean": float(s.mean()), "median": float(s.median()),
        "std": float(s.std() if not pd.isna(s.std()) else 0),
    }


def category_distribution(s: pd.Series, top: int = 10) -> dict:
    vc = s.value_counts(dropna=False).head(top)
    return {
        "labels": [str(k) for k in vc.index],
        "counts": [int(v) for v in vc.values],
    }


def eda_payload(df: pd.DataFrame, target_col: str | None) -> dict:
    schema = build_schema(df, target_col)
    num_cols = [c["name"] for c in schema if c["type"] == "number" and c["role"] != "target"]
    cat_cols = [c["name"] for c in schema if c["type"] in ("category", "bool") and c["role"] != "target"]

    # 数值分布
    numeric = {col: numeric_distribution(df[col]) for col in num_cols[:12]}
    # 类别分布
    category = {col: category_distribution(df[col]) for col in cat_cols[:12]}
    # 文本列：字符长度分布
    text_cols = [c["name"] for c in schema if c["type"] == "text" and c["role"] != "target"]
    text = {col: numeric_distribution(df[col].astype(str).str.len()) for col in text_cols[:6]}

    # 相关性（Pearson）
    corr = {"labels": [], "matrix": []}
    if num_cols:
        sub = df[num_cols].corr(method="pearson").fillna(0)
        corr = {
            "labels": [str(c) for c in sub.columns],
            "matrix": [[float(round(v, 3)) for v in row] for row in sub.values],
        }

    # 目标列均衡
    target = None
    if target_col and target_col in df.columns:
        s = df[target_col]
        if infer_col_type(s) == "number" and s.nunique() > 20:
            target = {"kind": "regression", **numeric_distribution(s)}
        else:
            target = {"kind": "classification", **category_distribution(s, top=20)}

    return {
        "schema": schema,
        "stats": build_stats(df),
        "numeric": numeric,
        "category": category,
        "text": text,
        "correlation": corr,
        "target": target,
    }


# ---------- 清洗 ----------

def apply_recipe(df: pd.DataFrame, recipe: dict) -> tuple[pd.DataFrame, dict]:
    """按清洗配方执行，返回 (新 DataFrame, 清洗报告)。

    recipe 结构：
    {
      "drop_duplicates": true,
      "missing": { "<col>": "drop_row" | "mean" | "median" | "mode" | "zero" | "constant:<v>", ... },
      "outliers": { "<col>": "iqr" | "zscore" | null, ... },
      "drop_columns": ["<col>", ...],
      "type_casts": { "<col>": "number" | "category" | "text", ... }
    }
    """
    before_rows = len(df)
    report: dict[str, Any] = {"before_rows": before_rows, "steps": []}

    out = df.copy()

    # 1. 删除不需要的列
    drop_cols = [c for c in recipe.get("drop_columns", []) if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
        report["steps"].append({"op": "drop_columns", "cols": drop_cols})

    # 2. 类型转换
    for col, target in (recipe.get("type_casts") or {}).items():
        if col not in out.columns:
            continue
        try:
            if target == "number":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif target in ("category", "text"):
                out[col] = out[col].astype("string")
            report["steps"].append({"op": "type_cast", "col": col, "to": target})
        except Exception as e:  # noqa: BLE001
            report["steps"].append({"op": "type_cast", "col": col, "to": target, "error": str(e)})

    # 3. 缺失值
    for col, strategy in (recipe.get("missing") or {}).items():
        if col not in out.columns or not strategy:
            continue
        n_before = int(out[col].isna().sum())
        if n_before == 0:
            continue
        if strategy == "drop_row":
            out = out[out[col].notna()]
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        elif strategy == "mode":
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])
        elif strategy == "zero" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        elif strategy.startswith("constant:"):
            out[col] = out[col].fillna(strategy.split(":", 1)[1])
        report["steps"].append({"op": "missing", "col": col, "strategy": strategy, "filled_or_dropped": n_before})

    # 4. 异常值
    for col, method in (recipe.get("outliers") or {}).items():
        if col not in out.columns or not method or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif method == "zscore":
            mu, sigma = s.mean(), s.std(ddof=0) or 1.0
            z = (s - mu).abs() / sigma
            lo, hi = None, None
            mask = z <= 3
            n_drop = int((~mask).sum())
            out = out[mask]
            report["steps"].append({"op": "outliers", "col": col, "method": "zscore", "dropped": n_drop})
            continue
        else:
            continue
        mask = (s >= lo) & (s <= hi)
        n_drop = int((~mask).sum())
        out = out[mask]
        report["steps"].append({"op": "outliers", "col": col, "method": method, "dropped": n_drop})

    # 5. 去重
    if recipe.get("drop_duplicates"):
        n_dup = int(out.duplicated().sum())
        out = out.drop_duplicates()
        if n_dup:
            report["steps"].append({"op": "drop_duplicates", "dropped": n_dup})

    out = out.reset_index(drop=True)
    report["after_rows"] = int(len(out))
    report["rows_removed"] = before_rows - report["after_rows"]
    return out, report
