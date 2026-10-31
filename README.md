# ModelForge

本地部署的模型生产平台。上传一个数据集，平台自己判断这是什么数据、怎么切分、用什么模型，
真的训练出来，最后登记到模型仓库并发布成 API。

后端 FastAPI + SQLite，前端原生 JS 无构建步骤，训练在本地 CPU 上跑（有 GPU 会自动用）。

![登录页](docs/screenshots/login.png)

![工作台（首次启动，还没有数据）](docs/screenshots/workbench.png)

## 为什么做这个

起因很具体：手上有一堆格式各异的数据集，每次都要重写一遍"读数据 → 切分 → 训练 → 评估"的脚本，
而且十个数据集里有八个的问题是一样的——切分靠随机、测试集被训练污染、指标只报准确率。
后来想，这套流程本身是可以固化的，只有"这批数据长什么样"是变量。

于是把变量交给模型来判断，把流程做成硬约束：

- 结构由模型判断，判断结果和生成的解析脚本进沙箱跑，跑不通就把报错回喂让它改
- 划分以源路径为准，AI 的说法只当复述；开训前断言 train/val/test 零交集，不满足直接拒绝开训
- 同一个结构只解析一次，指纹命中就复用已验证的脚本，零模型调用

## 流水线

```
上传数据集（zip / tar / 目录 / 单表）
   ↓
结构画像 + 指纹           只读文件头，GB 级压缩包秒级出画像
   ↓
分流
   ├─ 认出 YOLO / COCO 标注 ──→ 平台直接处理（0 次模型调用，约 10 秒）
   ├─ 指纹命中历史结构     ──→ 复用已验证脚本（0 次模型调用，约 1 秒）
   └─ 都不满足            ──→ 模型生成解析脚本 → 沙箱执行 → 平台校验
                                              ↑ 不合格，带报错回喂（最多 5 轮）
   ↓
锁定划分（三条红线：以源路径为准 / 测试集硬隔离 / 验证集不足从训练集分层补足）
   ↓
建模智能体
   ├─ 数据体检：坏图、类别分布、不平衡度
   ├─ 出方案卡：主干、输入尺寸、增强策略、是否迁移学习
   └─ 训练 → 诊断 → 不达标换方案重训（最多 3 轮，保留历轮最优）
   ↓
独立测试集评估 → 模型仓库 → 发布为 API
```

## 支持的数据类型

| 数据类型 | 建模方式 | 主指标 |
|---|---|---|
| 图像分类 | 微调 MobileNetV3 / ResNet18 主干 | accuracy、macro-F1 |
| 文本分类 | 预训练句向量 + 训练分类头 | accuracy、macro-F1 |
| 表格分类 | HistGradientBoosting | accuracy、macro-F1 |
| 表格回归 | HistGradientBoosting | RMSE、R² |
| 目标检测 | YOLOv8n | mAP50、mAP50-95、P/R |

支持的输入：`zip` `tar` `tar.gz` `tgz`；表格 `csv` `tsv` `txt` `xlsx` `xls` `json` `jsonl`
`parquet`；图像 `jpg` `png` `bmp` `webp` `tif`；检测标注支持 YOLO 框、YOLOv8 分割多边形、COCO JSON。

## 跑起来

需要 Python 3.11+，以及一个 OpenAI 兼容的大模型服务（用于结构判断与方案生成）。

```bash
pip install -r requirements.txt

# 一键起（自动挑空闲端口、清理旧进程）
python start.py
```

也可以分开起：

```bash
cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000
cd frontend && python -m http.server 8096 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8096/index.html>。

- 首次启动会建管理员账号：`admin`，密码取环境变量 `NN_ADMIN_PASSWORD`，没设则用 `Admin@123`
- 大模型服务（base_url / key / 模型名）在页面「设置 → 模型服务」里填，存本地 SQLite，不落代码
- 停止：`python start.py --stop`

## 目录

```
backend/
  main.py            数据集、解析、清洗、EDA 的 HTTP 入口
  ai_pipeline.py     解析管线：画像 → 记忆 → 生成脚本 → 沙箱 → 迭代修正
  llm.py             大模型调用、结构画像、提示词
  recall.py          结构记忆：指纹 → 已验证脚本
  fingerprint.py     结构指纹
  detection.py       YOLO / COCO 标注识别（含分割格式）
  automl.py          建模智能体：图像分类
  automl_text.py     建模智能体：文本分类
  automl_signal.py   建模智能体：时序/信号
  automl_detect.py   建模智能体：目标检测
  train.py           训练主流程、划分隔离、指标诊断
  models.py          模型仓库与推理
  services.py        API 服务管理
  providers.py       模型服务配置
  auth.py            登录与会话
frontend/            原生 JS 单页，无构建
docs/                API / 契约 / SOP / PRD
experiments/         Day 0–7 实测记录与原始指标 JSON（结论见 项目实测报告.md）
```

## 已知问题

- 结构理解完全依赖外部大模型，模型服务不稳定时解析会卡住；记忆命中率决定了长期使用的手感
- 沙箱只是限制工作目录与超时，不是真正的安全隔离，别拿它跑不可信脚本
- 检测任务只支持 YOLO 系；分割只解析标注，训练未实现
- 表格任务的超参基本是固定的，没做搜索
- 前端原生 JS 写到后面几个文件已经偏长，再加功能该上构建工具了

## License

MIT
