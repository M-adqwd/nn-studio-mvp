# 平台 ↔ AI 契约

> 本文定义 ModelForge 平台与「负责写数据准备脚本的大模型」之间的约定。
> 平台与 AI 的所有交互都必须符合本契约；违反契约的产物会被平台拒绝。

代码位置：`backend/llm.py`（提示词定义）、`backend/ai_pipeline.py`（执行与校验）、
`backend/detection.py`（平台独立识别路径）。

---

## 1. 整体契约

```
用户上传压缩包
     ↓
平台生成结构画像（profile）+ 指纹（fingerprint）
     ↓
平台查记忆库：命中 → 直接复用已验证脚本（不调用 AI）
     ↓ 未命中
平台把画像 + 契约 + 划分要求 + 检测要求 发给 AI
     ↓
AI 返回一段 Python 脚本（数据准备脚本）
     ↓
平台在沙箱里执行脚本 → 平台独立校验产物
     ↓ 不合格
平台把 stdout / stderr / 校验结论回喂给 AI，AI 重写（最多 5 轮）
     ↓
平台读取产物 → 以「源路径」为准锁定划分 → 落库
     ↓
平台把该脚本与结构指纹写入记忆库
```

**关键设计**：AI 只负责「写脚本」，**不负责判定划分归属**。
划分是用户数据里既成的事实，平台以源路径/源文件名为准，AI 的标注只作复述与兜底。

---

## 2. 产物契约

AI 生成的脚本**只需产出以下文件**，其余（清洗、切分、训练）由平台接管。

### 2.1 `manifest.csv` —— 文件型数据集（图像、音频等）

每个样本一行，必须包含且仅需这三列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `filepath` | string | 样本文件相对于「数据集根目录」的**相对路径**，用正斜杠 `/` 分隔 |
| `label` | string | 该样本的类别名 |
| `split` | string | 所属划分，取值 `train` / `val` / `test` |

**目标检测（YOLO 格式）额外需要两列**：

| 列名 | 类型 | 说明 |
|---|---|---|
| `boxes` | string | 该图里**所有**目标的边界框，多个用 `;` 分隔，每个框为 `类别名,x_center,y_center,width,height`，坐标 0~1 归一化 |

示例：

```csv
filepath,label,split,boxes
images/train/cat_01.jpg,cat,train,cat,0.5,0.5,0.3,0.4;dog,0.2,0.3,0.1,0.2
images/train/dog_02.jpg,dog,train,dog,0.7,0.6,0.2,0.3
images/test/cat_03.jpg,cat,test,cat,0.4,0.4,0.25,0.35
```

> **多目标必须全部写出来。** 只取第一个目标、或只写类别丢掉框，
> 等于把目标检测降级成图像分类——用户以为在训检测器，实际训的是分类器，
> 而且界面上完全看不出来。

### 2.2 `dataset.csv` —— 表格型数据集

把整理好的整张表写成 CSV（表头 + 数据行），其中必须包含一列 `split`
（取值 `train` / `val` / `test`）标记该行属于哪个划分。

### 2.3 `meta.json` —— 元信息

一份 JSON，至少包含这些键：

| 键 | 类型 | 说明 |
|---|---|---|
| `kind` | string | `image_classification` / `object_detection` / `text_classification` / `tabular_classification` / `tabular_regression` |
| `target_col` | string | 预测目标列名。图像任务固定为 `label`；表格任务填真实列名 |
| `text_col` | string | 文本分类的正文列名；非文本任务填空字符串 |
| `classes` | string[] | 类别名列表。分类任务必填 |
| `summary` | string | 一句话说明：这是什么数据集、怎么组织的、划分怎么来的 |

目标检测额外需要 `"detection": true`，且 `classes` 填**全部类别名（按 class_id 顺序，不是字典序）**。

---

## 3. 划分识别契约（最重要）

用户的训练集 / 验证集 / 测试集**通常已经划分好了**。AI 的核心任务是准确识别哪些数据属于哪个划分。

### 3.1 识别优先级

1. **目录名**：`train` / `training` / `val` / `valid` / `validation` / `test` / `eval` / `dev`
   及其变体（`train2017`、`val_set`、`TEST` 等）
2. **清单文件**：`train.txt` / `val.txt` / `test.txt`（每行一个文件路径）
   —— 注意这类清单常含别人机器的绝对路径，只取文件名部分匹配
3. **文件名模式**：`train_001.jpg` / `image_val_02.png`
4. **列名**：表格数据里的 `split` / `subset` / `partition` / `fold` / `dataset_split` 等列

### 3.2 平台侧的执行规则

即使 AI 标注了划分，平台仍会**独立复核**：

| 平台行为 | 位置 |
|---|---|
| 从源路径重新推导划分，覆盖 AI 的标注 | `_derive_split_from_filepath()` |
| AI 没写有效划分时，用源文件名重建 | `_recover_split_by_source()` |
| 归一化划分列的别名与大小写 | `normalize_split_column()` |
| 校验划分与源目录是否一致 | `_split_consistency()` |

### 3.3 无划分时的处理

**数据没有自带验证集是正常情况。**

- 平台不会为了凑齐 train/val/test 而随机重切——那会破坏用户的原始划分
- 训练时按比例从训练集内部切一份用于早停与权重选择
- 提示词中的原文：「用户没要求造验证集就别造」

### 3.4 验证集下限（平台补充规则）

自带验证集**不足 100 条**时，平台会从**训练集**按类别分层补足到 100 条。

原因：32 条验证集里 1 个样本就是 3.1% 的指标波动，选模型等于抛硬币。
补足时保证各类均衡（如 NORMAL 50 + PNEUMONIA 50）。

> 测试集**完全不受影响**，仍然零参与训练。

---

## 4. 目标检测识别契约

如果数据集带边界框标注（BoundingBox），**它就是目标检测任务，不是图像分类**。

### 4.1 支持的标注格式

| 格式 | 结构 | 平台支持 |
|---|---|---|
| YOLO 检测 | `class_id x_center y_center w h`（5 个数） | ✅ |
| **YOLOv8 分割** | `class_id` + 2N 个归一化多边形顶点 | ✅ 转外接矩形 |
| COCO | `annotations/*.json` + `images/` | ✅ |

> 分割格式说明：平台算出多边形的**最小外接矩形**转成检测框，
> 这样同一套检测训练流程照样能跑。
> 原生分割训练（YOLO-seg）尚未接入。

### 4.2 目录布局

| 布局 | 示例 |
|---|---|
| 标准 YOLO | `images/train|val|test/` 与 `labels/train|val|test/` 分离 |
| YOLOv8 | `dataset/{train,valid,test}/{images,labels}/` + `data.yaml` |
| Darknet 老格式 | 图片与标签同级：`ts/ts/00027.jpg` + `ts/ts/00027.txt` |

### 4.3 类别来源

按优先级：

1. `data.yaml` 的 `names` 列表
2. `classes.names` / `labels.names` 等清单文件（每行一个类别名）
3. 都找不到时用 `class_<id>` 兜底

### 4.4 平台独立识别路径

**检测数据不走 AI 解析**（`detection.find_yolo_dataset()` + `build_manifest()`）。

原因：AI 脚本会把多目标压成单标签、边界框全部丢失。平台自己能可靠认出
「图片旁有坐标格式的 .txt」，所以直接走平台路径——省一次 AI 调用，
也不会被记忆库里的旧分类脚本带偏。

---

## 5. 沙箱契约（AI 脚本的执行环境）

AI 生成的脚本是**不可信代码**，在受限子进程中执行。

### 5.1 隔离方式

```
python -I <guard.py> <script.py>
```

`-I` = 隔离模式（忽略 `PYTHONPATH`、用户 site-packages 会被禁用）

> **不能加 `-S`**：`python -I -S` 会禁用 site-packages，
> 导致 `import pandas` 失败，所有解析脚本 100% 崩。

### 5.2 限制项

| 限制 | 实现 |
|---|---|
| 禁网络 | 覆盖 `socket.socket` 等为「继承真实类但禁止实例化」的实现 |
| 禁子进程 | 同上方式覆盖 `subprocess.Popen` |
| 禁动态代码执行 | 覆盖 `exec` / `eval` |
| 写入限制 | 只允许写在工作目录内 |
| 路径守卫 | `_guarded_open` / `_guarded_os_open` |
| 超时 | 按数据体量自适应，小数据 120s，大包最多 1800s |
| 磁盘护栏 | 写盘超 12GB 立即终止 |

### 5.3 沙箱陷阱（踩过的坑）

**屏蔽标准库符号时，必须用「类替换类」，不能用「函数替换类」。**

标准库里有继承关系：

```python
# ssl.py
class SSLSocket(socket): ...      # ← socket 必须是类
# asyncio
class Popen(subprocess.Popen): ...  # ← 同理
```

如果用函数替换 `socket.socket`，任何间接 `import ssl` 或 `import asyncio`
的脚本都会在导入时抛 `TypeError: function() argument 1 must be socket.socket, not function`。

### 5.4 脚本硬性要求

1. 当前工作目录就是解压后的数据集根目录，所有路径都相对于它，**不要用绝对路径**
2. 只读取数据、只写契约里的产物文件
3. 禁止联网、禁止执行外部命令、禁止删除文件、禁止写到当前目录之外
4. 脚本必须**可复用**：不能硬编码具体的目录名或列名，
   要自适应探测（同一份脚本下次遇到同结构数据仍能工作）

---

## 6. 校验与迭代契约

### 6.1 平台校验项

| 校验 | 不通过的后果 |
|---|---|
| 产物存在且有数据行 | 回喂报错，AI 重写 |
| 分类任务类别数 ≥ 2 | 回喂报错 |
| `filepath` 在数据目录里真实存在（抽查前 50 条） | 回喂报错并给出缺失样例 |
| 划分与源目录一致 | 回喂报错并说明「源目录里的划分必须原样保留」 |

### 6.2 迭代机制

```
第 1 轮：AI 生成 → 执行 → 校验
  合格 → 落库
  不合格 → 把 [stdout + stderr + 平台校验结论] 组成 feedback
         → llm.improve_parser_code(profile, code, feedback, round_no)
第 2 轮 …第 5 轮
  仍不合格 → 报错，明确说明「最后一次的问题是什么」
```

**回喂给 AI 的内容必须包含真实运行输出**，而不是笼统的「失败了」——
这正是「像人一样跑一次看结果再改」的关键。

### 6.3 成功判据

- 产物有数据行
- 分类任务：label 类别数 ≥ 2
- **不强制要求切出 val/test** —— 数据原本什么样就保留什么样

---

## 7. 结构记忆契约

### 7.1 指纹构成

指纹用于判断「是不是同一种结构」，由三部分组成：

| 组成 | 说明 |
|---|---|
| 文件类型**占比** | 用占比而非绝对数量，保证 50 张与 40 张同结构能命中 |
| 目录树结构 | 归一化后的路径模式（`<sample>` 之类） |
| 列签名 | 表格数据的列名与类型组合 |

### 7.2 命中与复用

| 条件 | 行为 |
|---|---|
| 指纹完全相同 | 直接复用该脚本，**0 次 AI 调用** |
| 相似度 ≥ 0.75 | 复用并记录相似度 |
| 低于阈值 | 走 AI 生成 |

### 7.3 记忆条目字段

| 字段 | 说明 |
|---|---|
| `signature` | 结构签名 |
| `kind` | 任务类型 |
| `code` | 已验证的解析脚本 |
| `use_count` / `fail_count` | 使用与失败计数 |

### 7.4 写入时机

**只有在产物通过全部校验后才写入记忆库。** 未通过校验的脚本绝不能进记忆，
否则会污染后续所有同结构数据的解析。

---

## 8. 平台侧的三条红线

以下是平台自己的硬约束，**不在 AI 的控制范围内**，AI 违反也无法绕过。

| 红线 | 实现 | 位置 |
|---|---|---|
| 划分以源路径为准 | 从源路径/源文件名重新推导，覆盖 AI 标注 | `ai_pipeline._derive_split_from_filepath()` |
| 测试集零参与训练 | 强制把测试集样本从 train/val 剔除；开训前断言零交集 | `train.resolve_splits()` |
| 产物契约完整 | 多目标必须全写进 `boxes`；缺列/类别不足直接拒绝 | `ai_pipeline._inspect_products()` |

---

## 9. 变更约定

修改本契约时，需要同步更新：

- `backend/llm.py` 中的 `_OUTPUT_CONTRACT` / `_SPLIT_REQUIREMENT` / `_DETECTION_REQUIREMENT`
- `backend/ai_pipeline.py` 中的校验逻辑
- `backend/detection.py` 中的识别逻辑
- 本文档

**契约与代码不一致时，以代码为准，但必须同步修正文档。**
