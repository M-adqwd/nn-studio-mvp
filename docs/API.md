# ModelForge 接口文档

> 后端基址：`http://127.0.0.1:8000`
> 除 `GET /api/health`、`POST /api/auth/login`、`POST /api/svc/{name}/predict` 外，
> **所有接口都需要携带认证头**：`Authorization: Bearer <token>`

---

## 目录

- [认证](#1-认证)
- [数据集](#2-数据集)
- [解析任务](#3-解析任务)
- [解析记忆库](#4-解析记忆库)
- [训练任务](#5-训练任务)
- [模型仓库](#6-模型仓库)
- [推理](#7-推理)
- [服务管理](#8-服务管理)
- [大模型服务配置](#9-大模型服务配置)
- [通用约定](#10-通用约定)

---

## 1. 认证

### 1.1 登录

```
POST /api/auth/login
```

**请求体**

```json
{
  "account": "admin",
  "password": "Admin@123"
}
```

**响应 200**

```json
{
  "token": "eyJ1aWQiOiJ1Xzk4YWM2NGExZjciLCJleHAiOjE3OTI4NjA5NjF9.xxx",
  "user": {
    "id": "usr_98ac64a1f27",
    "account": "admin",
    "name": "管理员",
    "role": "平台管理员",
    "org": "默认组织"
  }
}
```

**错误**

| 状态码 | 场景 |
|---|---|
| 400 | 账号或密码为空 |
| 401 | 账号或密码不正确 |

### 1.2 获取当前用户

```
GET /api/auth/me
```

**响应 200**

```json
{
  "id": "usr_98ac64a1f27",
  "account": "admin",
  "name": "管理员",
  "role": "平台管理员",
  "org": "默认组织"
}
```

### 1.3 注册账号（仅管理员）

```
POST /api/auth/register
```

**请求体**

```json
{ "account": "zhangsan", "password": "Pass@123", "name": "张三" }
```

| 状态码 | 场景 |
|---|---|
| 403 | 当前用户不是平台管理员 |
| 400 | 账号少于 3 个字符 |

### 1.4 健康检查（无需认证）

```
GET /api/health
```

**响应 200** → `{"status": "ok", ...}`

---

## 2. 数据集

### 2.1 列表

```
GET /api/datasets
```

**响应 200** → 数组，每项包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | `ds_xxxxxxxxxx` |
| `name` | string | 数据集名称 |
| `source` | string | `upload` / `sample` |
| `task_hint` | string | `image_classification` / `object_detection` / `text_classification` / `tabular_classification` / `tabular_regression` |
| `target_col` | string | 目标列名 |
| `file_path` | string | 落盘的 CSV 路径 |
| `version` | int | 数据版本 |
| `rows` / `cols` | int | 样本数 / 列数 |
| `size_bytes` | int | 原始文件大小 |
| `created_at` / `updated_at` | string | 时间戳 |
| `meta` | object | 元信息（含 `classes` / `split_counts` / `image_root` / `ml_plan` / `datacard`） |

### 2.2 上传（multipart）

```
POST /api/datasets/upload
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | ✅ | 数据文件，≤ 3GB |
| `name` | string | — | 数据集名称，默认取文件名 |
| `target_col` | string | — | 目标列名，留空由 AI 判定 |
| `task_hint` | string | — | 任务类型，留空由 AI 判定 |

**支持的格式**

- 归档：`.zip` `.tar` `.tar.gz` `.tgz`
- 表格：`.csv` `.tsv` `.txt` `.xlsx` `.xls` `.json` `.jsonl` `.parquet` `.pq`

**响应 200**（立即返回，不阻塞）

```json
{
  "parsing": true,
  "task_id": "pt_ea22f6003564",
  "dataset_id": "ds_56c6625ed8",
  "name": "我的数据集",
  "message": "文件已接收，AI 正在解析…"
}
```

> 解析在后台线程执行。大数据集解析可能长达十几分钟，
> 同步执行会把整个后端占死（连 `/api/health` 都会无响应）。

**错误**

| 状态码 | 场景 |
|---|---|
| 400 | 文件超过 3GB / 格式不在白名单 / 文件接收失败 |

### 2.3 详情

```
GET /api/datasets/{did}
```

**响应 200** → 单个数据集对象（字段同 2.1），`meta` 额外含 `_parse_report`。

### 2.4 数据预览

```
GET /api/datasets/{did}/preview?n=20
```

**响应 200**

```json
{
  "columns": ["filepath", "label", "split"],
  "rows": [{ "filepath": "images/train/a.jpg", "label": "cat", "split": "train" }],
  "total": 2060
}
```

### 2.5 样本图像

```
GET /api/datasets/{did}/image?path=images/train/a.jpg&token=<token>
```

**响应 200** → 图片二进制流（`Content-Type: image/*`）

> `token` 作为查询参数是给 `<img src>` 用的——浏览器标签无法带认证头。

### 2.6 删除

```
DELETE /api/datasets/{did}
```

**响应 200** → `{"ok": true, ...}`

> 同时清理该数据集的解析工作目录、模型文件与训练任务。

### 2.7 修改目标列

```
PATCH /api/datasets/{did}/target
```

**请求体**

```json
{ "target_col": "label" }
```

**响应 200**

```json
{
  "ok": true,
  "target_col": "label",
  "task_hint": "text_classification",
  "old_target_col": "text",
  "message": "已把答案列改为「label」"
}
```

**副作用**：任务类型会随之重判（分类 ↔ 回归 ↔ 文本分类）。

> 这是给「AI 把答案列认错了」的场景用的——不用重传数据。

### 2.8 探索性分析

```
GET /api/datasets/{did}/eda
```

**响应 200** → 列结构、类型分布、缺失统计、相关性等。

### 2.9 AI 数据洞察

```
GET /api/datasets/{did}/ai-inspect
```

**响应 200** → 大模型对当前数据的分析结论（自然语言）。

### 2.10 清洗

```
POST /api/datasets/{did}/clean
```

**请求体**

```json
{
  "recipe": {
    "drop_duplicates": true,
    "fill_missing": "median",
    "remove_outliers": true
  }
}
```

**响应 200** → `{"ok": true, "version": 2, "report": {...}}`

清洗生成新版本，原始数据只读保留，可回溯。

### 2.11 清洗历史

```
GET /api/datasets/{did}/cleaning-runs
```

**响应 200** → 数组，含 `id` / `version` / `recipe` / `report` / `created_at`。

---

## 3. 解析任务

### 3.1 查询进度

```
GET /api/datasets/parse-status/{task_id}
```

**响应 200**

```json
{
  "id": "pt_ea22f6003564",
  "status": "running",
  "message": "正在读取标注… 1400/2060（68%）",
  "progress": 0.71,
  "stage": "scan",
  "files_done": 1400,
  "files_total": 2060,
  "elapsed": 74,
  "eta_seconds": 11,
  "round": 2,
  "max_rounds": 5,
  "dataset_id": "ds_56c6625ed8",
  "name": "BoneFracture",
  "created_at": 1730000000.0
}
```

| 字段 | 说明 |
|---|---|
| `status` | `running` / `done` / `failed` |
| `progress` | 0~1 |
| `stage` | `extract`（解压）/ `scan`（识别结构）/ `ai_exec`（AI 执行脚本） |
| `files_done` / `files_total` | 已处理 / 总文件数 |
| `elapsed` | 当前阶段已用秒数 |
| `eta_seconds` | 预计剩余秒数 |
| `round` / `max_rounds` | 当前迭代轮次 / 上限（5） |

**状态为 `failed` 时** `message` 携带具体原因。

### 3.2 取解析结果

```
GET /api/datasets/parse-status/{task_id}/result
```

**响应 200** → 数据集对象，附带 `_parse_report`：

```json
{
  "dataset": {
    "id": "ds_56c6625ed8",
    "name": "BoneFracture",
    "task_hint": "object_detection",
    "rows": 2060,
    "meta": { "...": "..." }
  },
  "report": {
    "fingerprint": "L3-1|images:0.8|labels:0.2",
    "memory_hit": false,
    "memory_score": 0.0,
    "ai_called": false,
    "ai_fix_calls": 0,
    "detection_direct": true,
    "ai_reason": "平台直接识别出目标检测标注结构，跳过 AI 解析",
    "split_counts": { "train": 1804, "val": 173, "test": 83 },
    "split_source": "platform_detection",
    "extract_seconds": 150.2,
    "seconds": 236.8,
    "structure": "..."
  }
}
```

| `report` 字段 | 说明 |
|---|---|
| `memory_hit` | 是否命中结构记忆（命中则 0 次 AI 调用） |
| `memory_score` | 相似度 0~1 |
| `ai_called` | 是否为了生成主解析脚本调用了 AI |
| `ai_fix_calls` | 因校验不通过触发的修正次数 |
| `detection_direct` | 是否由平台直接识别出检测结构 |
| `split_counts` | 各划分的样本数 |
| `split_source` | `platform_detection` / `source_path` / `ai_product` / `dataset_split` |

---

## 4. 解析记忆库

### 4.1 列出

```
GET /api/parse-memory
```

**响应 200**

```json
{
  "items": [
    {
      "id": "mem_xxx",
      "signature": "L3-1|images:0.8|labels:0.2",
      "kind": "object_detection",
      "dataset_name": "BoneFracture",
      "use_count": 3,
      "fail_count": 0,
      "created_at": "..."
    }
  ]
}
```

### 4.2 删除

```
DELETE /api/parse-memory/{mid}
```

**响应 200** → `{"ok": true}`

> 删除后同结构数据会重新调用 AI 解析一次。

---

## 5. 训练任务

### 5.1 创建

```
POST /api/train
```

**请求体**

```json
{
  "dataset_id": "ds_56c6625ed8",
  "task": "classification",
  "epochs": 20,
  "lr": 0.002,
  "batch_size": 16,
  "val_split": 0.2,
  "auto": false,
  "transfer": true,
  "det_model": "yolov8n.pt",
  "class_weight": true,
  "hidden": 64,
  "optimizer": "adam",
  "scheduler": "none",
  "weight_decay": 0.0005,
  "dropout": 0.2,
  "patience": 8
}
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dataset_id` | string | — | **必填** |
| `auto` | bool | true | true 时 `epochs`/`lr`/`batch_size` 由样本量自动决定 |
| `transfer` | bool | true | true 走智能体路径（迁移学习）；false 退回从零训练 |
| `epochs` | int | 自动 | 1~100（检测任务可到 300） |
| `lr` | float | 自动 | 学习率 |
| `batch_size` | int | 自动 | 8~128 |
| `val_split` | float | 0.2 | 无自带验证集时从训练集切出的比例 |
| `det_model` | string | `yolov8n.pt` | 检测任务模型（`yolov8n.pt` / `yolov8s.pt`） |
| `class_weight` | bool | — | 类别不平衡时启用 |
| `hidden` | int | 64 | 表格任务隐藏层宽度 |

**响应 200** → 训练任务对象（同 5.3）。

### 5.2 列表

```
GET /api/train
```

**响应 200** → 数组，任务对象列表。

### 5.3 详情

```
GET /api/train/{jid}
```

**响应 200**

```json
{
  "id": "job_2e95710575",
  "dataset_id": "ds_56c6625ed8",
  "status": "running",
  "progress": 0.62,
  "epoch": 12,
  "total_epochs": 20,
  "logs": ["第 1/3 轮方案：mobilenet_v3_small · 输入 192px · 微调主干"],
  "error": null,
  "model_path": null,
  "created_at": "...",
  "metrics": {
    "stage": "微调训练中",
    "stage_progress": 0.6,
    "eta_seconds": 42,
    "history": [
      { "epoch": 1, "train_loss": 1.46, "train_acc": 0.32, "val_loss": 1.34, "val_acc": 0.30 },
      { "epoch": 2, "train_loss": 1.19, "train_acc": 0.41, "val_loss": 1.33, "val_acc": 0.42 }
    ],
    "last_record": { "epoch": 2, "val_acc": 0.42 },
    "round_epoch": 12,
    "round_total": 20
  }
}
```

**训练中** `metrics.history` 会随每轮训练增长（前端据此画曲线）。

**训练完成后** `metrics` 额外包含：

```json
{
  "task": "classification",
  "arch": "transfer_head",
  "best_val": { "val_acc": 0.4048, "acc": 0.4048, "macro_f1": 0.384 },
  "test_metrics": {
    "test_acc": 0.6306,
    "test_samples": 111,
    "macro_f1": 0.6267,
    "per_class": [{ "class": 0, "support": 21, "precision": 0.57, "recall": 0.57, "f1": 0.57 }],
    "confusion_matrix": [[12, 2], [1, 14]]
  },
  "config": { "backbone": "mobilenet_v3_small", "mode": "finetune", "img_size": 192 },
  "datacard": { "rows": 2060, "n_classes": 7, "imbalance": 2.6 },
  "rounds": [
    { "round": 1, "plan": {}, "val": {}, "history": [], "verdict": "underfit", "hint": "..." }
  ],
  "split_info": { "source": "platform_detection", "train": 1804, "val": 173, "test": 83 },
  "advice": {
    "level": "ok",
    "summary": "验证准确率 40.5%，低于随机基线 14.3%。",
    "plain_summary": "准确率 40.5%，比瞎猜（14%）好不少。",
    "tips": ["..."],
    "plain_tips": ["多准备一些样本"]
  }
}
```

> `advice` 同时提供技术版（`summary`/`tips`）与大白话版（`plain_summary`/`plain_tips`）。

**状态取值**：`running` / `done` / `failed`

---

## 6. 模型仓库

### 6.1 登记（从训练任务）

```
POST /api/models
```

**请求体**

```json
{ "train_job_id": "job_2e95710575", "name": "骨折检测模型" }
```

**响应 200**

```json
{
  "id": "mdl_c4f083df93",
  "name": "骨折检测模型",
  "version": 1,
  "task": "detection",
  "task_hint": "object_detection",
  "architecture": "yolo",
  "dataset_id": "ds_56c6625ed8",
  "train_job_id": "job_2e95710575",
  "stage": "candidate",
  "model_path": "E:\\...\\models\\job_2e95710575.pt",
  "metrics": { "...": "..." }
}
```

| 状态码 | 场景 |
|---|---|
| 404 | 训练任务不存在 |
| 400 | 训练任务尚未完成 / 未产出模型文件 |

> 同名模型自动递增版本号。

### 6.2 列表 / 详情

```
GET /api/models
GET /api/models/{mid}
```

### 6.3 更新

```
PUT /api/models/{mid}
```

**请求体**

```json
{ "name": "新名字", "stage": "production" }
```

`stage` 取值：`candidate` → `staging` → `production`

### 6.4 删除

```
DELETE /api/models/{mid}
```

---

## 7. 推理

### 7.1 模型推理

```
POST /api/models/{mid}/predict
```

**图像模型**（`multipart/form-data`，字段名 `file`，可重复提交多张）

```bash
curl -X POST http://127.0.0.1:8000/api/models/mdl_xxx/predict \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@cat.jpg"
```

**响应 200**

```json
{
  "task": "classification",
  "arch": "transfer_head",
  "predictions": [
    { "label": "cat", "confidence": 0.87, "probs": { "cat": 0.87, "dog": 0.13 } }
  ]
}
```

**表格模型**（JSON）

```json
{ "rows": [{ "age": 25, "bmi": 22.5 }] }
```

**响应 200** → `{"task": "classification", "arch": "hgb", "predictions": [...]}`

**文本模型**（JSON）

```json
{ "texts": ["这家餐厅的服务态度非常好"] }
```

---

## 8. 服务管理

### 8.1 创建

```
POST /api/services
```

**请求体**

```json
{ "name": "fracture-detector", "model_id": "mdl_c4f083df93" }
```

**响应 200**

```json
{
  "id": "svc_xxx",
  "name": "fracture-detector",
  "model_id": "mdl_c4f083df93",
  "status": "running",
  "requests_total": 0,
  "model": { "...": "..." }
}
```

| 状态码 | 场景 |
|---|---|
| 400 | 服务名不是 slug（小写字母开头，`a-z0-9-`，2~41 字符） |
| 404 | 模型不存在 |
| 409 | 服务名已被占用 |

### 8.2 列表 / 启停 / 删除

```
GET    /api/services
POST   /api/services/{sid}/start
POST   /api/services/{sid}/stop
DELETE /api/services/{sid}
```

### 8.3 服务调用

```
POST /api/svc/{name}/predict
```

**无需认证**（供外部系统调用）

**请求体**：与 [7.1](#71-模型推理) 相同

**响应 200** → 同 7.1

---

## 9. 大模型服务配置

### 9.1 列出

```
GET /api/providers
```

### 9.2 新增

```
POST /api/providers
```

**请求体**

```json
{
  "name": "MiniMax",
  "provider_type": "openai_compatible",
  "base_url": "https://api.minimaxi.com/v1",
  "api_key": "sk-xxx",
  "model": "MiniMax-M2.5",
  "is_default": true
}
```

| 字段 | 说明 |
|---|---|
| `provider_type` | `openai_compatible` / `anthropic` / `local` |
| `is_default` | 只能有一个默认服务；AI 解析与建模决策都用它 |

**注意**：`model` 可填多个候选，用逗号分隔。调用失败时会自动试下一个。

### 9.3 更新 / 删除

```
PUT    /api/providers/{pid}
DELETE /api/providers/{pid}
```

### 9.4 连通性测试

```
POST /api/providers/{pid}/test
```

**响应 200**

```json
{ "ok": true, "latency_ms": 342, "message": "连接正常" }
```

---

## 10. 通用约定

### 10.1 认证

```
Authorization: Bearer <token>
```

Token 由 `POST /api/auth/login` 签发。

### 10.2 错误响应

FastAPI 标准格式：

```json
{ "detail": "文件超过 3GB 上限" }
```

| 状态码 | 含义 |
|---|---|
| 400 | 请求参数错误 / 业务前置条件不满足 |
| 401 | 未认证或凭证失效 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 409 | 资源冲突（如服务名重复） |
| 500 | 服务端错误 |

### 10.3 ID 命名规范

| 前缀 | 对象 |
|---|---|
| `ds_` | 数据集 |
| `pt_` | 解析任务 |
| `job_` | 训练任务 |
| `mdl_` | 模型 |
| `svc_` | 服务 |
| `usr_` | 用户 |
| `mem_` | 记忆条目 |

### 10.4 轮询建议

| 场景 | 间隔 | 超时 |
|---|---|---|
| 解析进度 | 1.2 秒 | 12 分钟 |
| 训练进度 | 1.5 秒 | 按数据量，检测任务可达 30 分钟 |

### 10.5 任务类型枚举

| 值 | 说明 |
|---|---|
| `image_classification` | 图像分类 |
| `object_detection` | 目标检测 |
| `text_classification` | 文本分类 |
| `tabular_classification` | 表格分类 |
| `tabular_regression` | 表格回归 |

### 10.6 训练架构枚举（`metrics.arch`）

| 值 | 说明 |
|---|---|
| `transfer_head` | 预训练主干 + 分类头（图像） |
| `text_transfer_head` | 预训练语言模型 + 分类头（文本） |
| `hgb` | 梯度提升树（表格） |
| `yolo` | YOLOv8（检测） |
| `cnn` / `emb_mlp` / `mlp` | 从零训练路径 |
