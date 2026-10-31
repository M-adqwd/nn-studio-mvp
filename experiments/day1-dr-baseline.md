# 实测日志：Day 1 — 眼底 DR 五级分类（零改动基线）

**日期**：2026-10-30
**目的**：不修改平台代码，用真实眼底图跑通五级分级，拿到一条可对比的基线。
**结论**：**链路跑通，但这一轮的指标作废** —— 平台在这场实验里暴露了两个 P0 缺陷，
一个让第三轮白跑，一个让前两轮的结果被静默覆盖。两处都已修复，Day 2 重跑取数。

---

## 1. 数据

| 项 | 值 |
|---|---|
| 来源 | HuggingFace 镜像 `Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256`（APTOS 子集，已预处理为 256×256） |
| 规模 | 2,750 张 · 5 级有序标签 |
| 类别分布 | Healthy 1,000 · Mild 370 · Moderate 900 · Severe 190 · Proliferate 290 |
| 脚本 | `_data/fundus/export_fundus.py`（`flat` 扁平导出 / `split` 分层切分导出） |
| 本轮形态 | `dr_flat`（扁平目录，不带划分）→ 由平台自行切分 |

> Kaggle 直连需要账号凭据（本机无 `~/.kaggle/kaggle.json`，匿名下载返回 403），
> 改走镜像站；镜像站实测 5.4 MB/s，比 PhysioNet 快约 20 倍。

## 2. 执行过程（以及它在哪里坏了）

平台拿到 2,750 张图后自动排了 3 轮方案：

| 轮 | 方案 | 验证准确率 | 实际耗时 |
|---|---|---|---|
| 1 | mobilenet_v3_small · 160px · 冻结主干 · 20 轮 | 0.6273（第 14 轮最佳） | ~8 min |
| 2 | resnet18 · 192px · 微调主干 · 20 轮 | **0.7236**（第 15 轮最佳） | ~25 min |
| 3 | resnet18 · 192px · 微调主干 · 20 轮（**与第 2 轮完全相同**） | 0.6073 | ~25 min |

第 3 轮结束后任务被标记为完成，但落库的最终指标变成了：

```
arch = cnn, best_val.epoch = 8, val_acc = 0.6073, test_metrics = {}
```

—— 这是一次**全新的 8 轮普通 CNN 训练**，不是上面三轮里的任何一轮。

## 3. 两个 P0 缺陷

### P0-1：第三轮的方案和第二轮一字不差

日志里写着「诊断：plateau —— 调整方案后重训」，但第三轮的方案仍是
`resnet18 · 192px · 微调 · 20 轮`，与第二轮完全相同。

根因在 `_revise_by_rule()`：它每次都从**规则基线**（mobilenet@160）重算一次调整结果，
而不是在**上一轮方案**的基础上往前走一步。同一份数据、同一个诊断结论（plateau），
重算必然得到同一份方案。结果是第三轮完整重跑了第二轮的 25 分钟，指标还更差。

**修复**：`decide()` 增加 `prev_plan` 参数，迭代改为「在上一轮方案上递进」，
并引入调整层级（`_level`）：
160px → 192px → 224px，逐级加大输入尺寸、轮次与增强强度。
已用单元验证：连续三次 plateau 调整分别产出 192px/20轮 → 224px/30轮+strong → 224px/35轮+smoothing。

### P0-2：SQLite 写锁导致长任务失败，且失败被静默掩盖

`run_job()` 里有一段兜底：

```python
try:
    automl.auto_train(job_id)
    return
except Exception as e:
    db.update_job(job_id, status="running", error=f"自动建模不可用，已退回基础训练：{e}")
# 然后继续走常规 CNN 路径，logs = []（清空）
```

三轮迁移学习跑完后，保存结果那一步撞上 `database is locked`，
异常落进这个分支 —— 于是平台**丢掉 45 分钟的成果**，改跑 8 轮普通 CNN，
用 `logs = []` 把前面的日志整段抹掉，最后把任务标成 `done`。
如果没有人盯着日志，看到的就是「训练成功，准确率 60.7%」。

对一个要交付给客户的平台，这比训练失败严重得多：**它会把失败伪装成成功，并且给一个更差的数字。**

**修复**：
1. `db.conn()` 的 `busy_timeout` 15s → 60s，提交时对 `locked` 做指数退避重试；
2. 回退分支不再清空日志，改为在原日志后追加；
3. 最终指标写入 `fallback: true` + `fallback_reason`，前端与报告能直接看到「这是降级路线的结果」。

## 4. 顺带修掉的三处

| 问题 | 影响 | 处理 |
|---|---|---|
| 图像分类路径无视用户指定的轮次 | 配 8 轮仍跑 20 轮，成本失控 | 以用户轮次为**上限**截断 |
| `MAX_ROUNDS` 硬编码为 3 | 无法按预算控制迭代次数 | 改为 `NN_MAX_ROUNDS` 环境变量 |
| 结构记忆误命中（Day 0 遗留） | 不同表复用别人的解析脚本 | 相似度匹配前先做列结构一致性校验 |

## 5. 这一轮能对外说什么

**能说**：眼底五级分级在平台上端到端可跑，2,750 张图从上传到出模型约 1 小时（纯 CPU）；
自动迭代确实在换方案（mobilenet 冻结 → resnet18 微调 + 放大到 192px，验证准确率 0.63 → 0.72）。

**不能说**：0.6073 或 0.7236 都不是可交付指标 ——
前者是降级路线的产物，后者没有独立测试集（本数据集不带划分，验证集从训练集里切出来的），
且没有 QWK。要对外报数，必须走 Day 2 的配置：自带 train/val/test 划分 + QWK + 明确标注数据来源。

## 6. 复现

```bash
# 后端（Day 2 起生效的全部修复）
NN_MAX_IMG_SIZE=320 NN_IMG_SIZE=224 NN_MAX_ROUNDS=2 python -m uvicorn main:app --port 8000

python scripts/run_chain.py --file _data/fundus/dr_split.zip \
  --name "眼底DR五级-DAY2" --hint image_classification --epochs 8 --auto \
  --out experiments/day2_dr.json
```
