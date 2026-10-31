# 实测日志：Day 0 — ECG 心搏分类（链路验证）

**日期**：2026-10-30
**目的**：不改一行平台代码，验证「上传 → AI 解析 → 训练 → 模型登记 → 推理」在真实医疗数据上是否走得通。
**结论**：**链路通，但暴露 3 个必须在 Day 2–3 修掉的问题。**

---

## 1. 数据

| 项 | 值 |
|---|---|
| 来源 | PhysioNet MIT-BIH Arrhythmia Database（`mitdb`，44 条记录 / 48 段） |
| 构造 | 360Hz → 125Hz 重采样，以 R 峰为中心截 187 点，AAMI 五分类映射 |
| 脚本 | `_data/ecg/build_mitbih.py` |
| 成品 | `mitbih_beats.csv`（100,648 拍 / 5 类）、`mitbih_beats_4cls.csv`（100,633 拍 / 4 类） |
| 类别分布 | N 90,045 · V 7,006 · S 2,780 · F 802 · Q 15 |

> Q 类仅 15 拍（数据源里本就罕见），Day 0 用四分类版；Q 类回归留给 Day 4–6 的信号路径一起处理。

**为什么不用 Kaggle 那份 187 点 CSV**：Kaggle 下载需要账号凭据（本次环境无 `~/.kaggle/kaggle.json`），
而这份数据本身就是从 MIT-BIH 按同样口径构造的，直接从 PhysioNet 原始库生成更可控，
并且**额外保留了 patient 列**（Kaggle 版没有）——这是后面做患者级切分的前提。

## 2. 执行结果

脚本：`scripts/run_chain.py`（可重复执行，结果落 `experiments/day0_mitbih.json`）

| 环节 | 结果 | 耗时 |
|---|---|---|
| 登录 | OK | <1s |
| 上传（144MB CSV） | `ds_1d508eab28` | 1.2s |
| AI 解析 | done，结构记忆命中，**未调用大模型** | 69.8s |
| 训练 | `job_f374e47bdd`，HGB，1 轮达标 | 58.0s |
| 模型登记 | `mdl_4b658e5f32` v1 | <1s |
| 推理（5 条抽样） | 5/5 与真值一致 | <1s |

**指标**：val_acc **0.9877** · macro-F1 **0.9005**

| 类 | 支持数 | precision | recall | F1 |
|---|---|---|---|---|
| F | 171 | 0.890 | 0.661 | 0.758 |
| N | 18,044 | 0.990 | 0.998 | 0.994 |
| S | 537 | 0.973 | 0.810 | 0.884 |
| V | 1,374 | 0.973 | 0.958 | 0.966 |

## 3. 暴露的问题（按严重程度）

### P0：没有独立测试集，报出去的数字是验证集

解析阶段 `split_counts = {train: 100633, val: 0, test: 0}`，训练时验证集从训练集里切，
`test_metrics = {}` 为空。也就是说 **98.77% 这个数字没有经过测试集**。
平台 README 写的「测试集全程零参与训练」只有数据自带划分时才成立；纯 CSV 上传时
根本不存在测试集。对客户报指标，这是硬伤。

**修复**：Day 2 给 `resolve_splits()` 加「无自带划分时切出独立 test」，并在指标里明确标注来源。

### P0：结构记忆误命中

`memory_hit = true`，`memory_source = "Maternal Health Risk Data Set"`，`ai_called = false`。
我们的 188 列 CSV 和那个孕妇健康数据集的列数撞了，指纹只按「形状」匹配就复用了别人的解析脚本。
这次结果凑巧正确（task/target 都对），但机制上会**把 A 数据集的解析结论套到 B 数据集上**。

**修复**：Day 2 把指纹的列名与语义签一块纳入，或对「列数相同但列名完全不同」的情况降级为 AI 复核。

### P1：指标里类别名丢了

`per_class` 里 `"class": 0/1/2/3`，客户拿到报告看不出这是 F/N/S/V 哪一类。
**修复**：Day 2 在 evaluate 里回填类别名。

## 4. 复现方式

```bash
# 数据
python _data/ecg/build_mitbih.py
# 链路（需后端已启动）
python scripts/run_chain.py --file _data/ecg/mitbih_day0.csv \
  --name "MITBIH-心搏四分类" --target label --hint tabular_classification \
  --auto --out experiments/day0_mitbih.json
```

## 5. 对客户的表述

> ECG 心搏级分类在平台上零改动可跑通，端到端 2 分钟出结果（10 万条样本）。
> 但当前版本的指标来自验证集，尚不能作为交付指标；节律级判读（多标签、按患者切分）需要平台扩展。
