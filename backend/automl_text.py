"""文本分类的智能体建模：预训练语言模型提特征 + 分类头。

为什么不用原来的 hash_tokens + EmbeddingBag
------------------------------------------
原方案把文本按 crc32 哈希成固定词表里的 id，哈希会抹掉语义：
「good」和「bad」只是两个毫不相关的数字，模型学不到任何语言学知识。
在真实语料（新闻标题、评论）上，这种表示的准确率明显低于语义模型。

本模块用 transformers 的预训练编码器（如 distilbert）提句向量，
再在上面训练分类头 —— 与图像路径的「冻结 backbone 提特征」是同一套思路。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 可用的文本编码器，按「效果 / CPU 速度」排序。
# hf_pretrained 为 None 时需要本地目录；否则从 HF_ENDPOINT 指定的镜像拉取。
TEXT_ENCODERS = {
    "distilbert": {
        "hf": "distilbert-base-uncased",
        "dim": 768,
        "max_len": 128,
        "multilingual": False,
        "label": "DistilBERT（英文，速度快）",
    },
    "minilm": {
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "max_len": 128,
        "multilingual": False,
        "label": "MiniLM（英文，更快）",
    },
    "chinese_roberta": {
        "hf": "hfl/chinese-roberta-wwm-ext",
        "dim": 768,
        "max_len": 192,
        "multilingual": True,
        "label": "Chinese RoBERTa（中文）",
    },
    "multilingual": {
        "hf": "distilbert-base-multilingual-cased",
        "dim": 768,
        "max_len": 128,
        "multilingual": True,
        "label": "DistilBERT 多语言（中文/英文）",
    },
}

DEFAULT_ENCODER = "multilingual"


def _has_cjk(texts: list[str]) -> bool:
    """判断语料是否含中文，决定选哪个编码器。"""
    for t in texts[:200]:
        if any("\u4e00" <= ch <= "\u9fff" for ch in str(t)):
            return True
    return False


def pick_encoder(texts: list[str]) -> str:
    """按语料语言自动选编码器。"""
    if _has_cjk(texts):
        return "chinese_roberta" if _has_cjk(texts) else DEFAULT_ENCODER
    return "distilbert"


def _local_dir(name: str) -> Path:
    return Path(__file__).parent / "_hf_models" / name


def _resolve_source(name: str) -> str:
    """优先用已下载到本地的目录，避免每次联网。"""
    spec = TEXT_ENCODERS.get(name)
    if not spec:
        raise ValueError(f"未知文本编码器：{name}")
    d = _local_dir(name)
    if (d / "config.json").exists() and (d / "config.json").stat().st_size > 0:
        return str(d)
    return spec["hf"]


def ensure_encoder(name: str = DEFAULT_ENCODER) -> dict:
    """确保编码器可用。返回 {"ok", "detail", "local"}。"""
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"transformers 不可用：{e}"[:200]}
    d = _local_dir(name)
    if (d / "config.json").exists() and (d / "config.json").stat().st_size > 0:
        try:
            AutoTokenizer.from_pretrained(str(d))
            AutoModel.from_pretrained(str(d))
            return {"ok": True, "detail": "本地模型已就绪", "local": True}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "detail": str(e)[:200], "local": True}
    return {"ok": False, "detail": "需要先下载模型文件", "local": False}


@torch.no_grad()
def encode_texts(texts: list[str], name: str = DEFAULT_ENCODER,
                 batch_size: int = 32, log=None, on_progress=None) -> np.ndarray:
    """用预训练编码器把文本转成句向量。

    与图像路径同理：每条文本只过一次 encoder，特征缓存复用，
    之后只训分类头 —— 这是 CPU 上能跑得动语义模型的关键。
    on_progress(done, total, elapsed) —— 编码是耗时环节，必须实时回传进度。
    """
    from transformers import AutoModel, AutoTokenizer

    spec = TEXT_ENCODERS.get(name) or TEXT_ENCODERS[DEFAULT_ENCODER]
    src = _resolve_source(name)
    tok = AutoTokenizer.from_pretrained(src)
    mdl = AutoModel.from_pretrained(src).eval()

    max_len = int(spec["max_len"])
    out = None
    t0 = time.time()
    total = len(texts)
    for i in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        res = mdl(**enc).last_hidden_state
        # last_hidden_state 是 (batch, seq, dim)。有些模型没提供 pooler 输出，
        # 这里统一用 masked mean pooling（忽略 pad 位），比取 CLS 更稳。
        if res.dim() == 3:
            mask = enc["attention_mask"].unsqueeze(-1).to(res.dtype)
            feat = (res * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            feat = res
        if out is None:
            out = np.zeros((len(texts), feat.shape[1]), dtype=np.float32)
        out[i:i + len(chunk)] = feat.float().numpy()
        if on_progress:
            try:
                on_progress(min(i + len(chunk), total), total, time.time() - t0)
            except Exception:  # noqa: BLE001
                pass
        if log and i and i % (batch_size * 20) == 0:
            log(f"  文本编码 {i}/{len(texts)} …")
    if log:
        log(f"文本编码完成：{len(texts)} 条 · {out.shape[1]} 维 · {time.time()-t0:.1f}s")
    return out


def text_baseline_plan(card: dict, texts_sample: list[str]) -> dict:
    """文本任务的规则基线方案。"""
    n = int(card.get("rows") or 0)
    enc = pick_encoder(texts_sample) if texts_sample else DEFAULT_ENCODER
    if n < 500:
        epochs, lr, drop = 40, 3e-3, 0.3
    elif n < 20000:
        epochs, lr, drop = 25, 2e-3, 0.2
    else:
        epochs, lr, drop = 12, 1e-3, 0.1
    imb = float(card.get("imbalance") or 1.0)
    return {
        "encoder": enc,
        "epochs": epochs,
        "lr": lr,
        "batch_size": 32,
        "dropout": drop,
        "weight_decay": 1e-3,
        "label_smoothing": 0.05 if imb > 3 else 0.0,
        "class_weight": imb > 3,
        "source": "rule",
        "reason": f"样本 {n} 条 · 类别 {card.get('n_classes', 0)} 个 · "
                  f"编码器 {TEXT_ENCODERS.get(enc, {}).get('label', enc)}",
    }
