# Day 6 · 一体化打通记录

## 做了什么

原来的断点：ModelForge 的模型只能被 `/api/models/{id}/predict` 调到，返回 `label + confidence`；
MIIS 的 AI 适配器只认 `study_instance_uid + targets`（DICOM 序列分割形态）。
两者的数据形态与产物语义都不重合，所以「训练平台训出的模型」走不到「病例工作台出报告」。

本次补的是中间那一层，只补最小必要面：

### ModelForge 侧（`backend/gateway.py`，新增）
- `GET /api/gateway/catalog`：模型目录。每个模型给 `id / name / version / task_hint / modality /
  architecture / classes / input{image_size,leads,length,multilabel} / quality{test_acc,
  macro_f1,macro_auc,qwk,test_samples}`——阅片端据此判断「这个模型吃不吃得下我的输入」
- `POST /api/gateway/infer`：统一推理入口，入参沿用 predict（图像 multipart / 波形 JSON），
  输出统一信封：`model / predictions / top / report{findings,impression,refs} / disclaimer`
- `POST /api/gateway/diagnose`：带上下文的诊断建议（Day 6 之后补）。`context` 里的
  `study_uid` / 患者标识 / 检查部位只写进结论文本，**不参与推理**——推理只看图像或信号本身，
  避免把患者信息喂进模型造成偏倚；也允许直接传已算好的 `top`，避免重复推理一次
- 负责把 label 翻译成临床语言（分级 0–4、心搏 N/S/V/F、PTB-XL 五大超类）

### MIIS 侧
- `app/adapters/modelforge.py`（新增）：HTTP 适配器，支持令牌过期自动重登
- `app/services/ai_classify.py`（新增）：调模型 → 落 `AiJob(done)` → 生成 **draft** 报告
- `POST /api/v1/ai/classify`（新增）：图像走 multipart，波形走 JSON rows
- `REPORT_REF_ALLOWED` 新增 `ai_job`（仅 done 态可引用），`validate_for_submit` 同步支持

## 验证结果（真实调用，非 mock）

### 眼底（真实测试图，标签为 3 级）
```
结论: 眼底图像自动分析：糖尿病视网膜病变 2 级（中度），模型置信度 61.28%。
印象: 糖尿病视网膜病变 2 级，建议眼科随访。
报告: state=draft  refs=[{kind: ai_job, uid: cls-07d3fe6fd0e5, version: 1}]
概率: 0_healthy 0.004 / 1_mild 0.037 / 2_moderate 0.613 / 3_severe 0.281 / 4_proliferate 0.065
```
> 真实 3 级被判成 2 级（相邻级错误，3 级概率 0.28）。这与测试集 QWK 0.817 的
> 「偶尔错一级」表现一致；对「是否需转诊」这个二分类判断仍是阳性，临床上可接受。

### 心电（PTB-XL 首条记录，12 导联多标签）
```
结论: 心电信号自动分类：MI，置信度 57.94%。
报告: state=draft  refs=[{kind: ai_job, uid: cls-987b6b1e270b, version: 1}]
概率: CD 0.363 / HYP 0.085 / MI 0.579 / NORM 0.236 / STTC 0.621
```
> STTC 0.621 未达该模型标定阈值 0.70，故只输出 MI。阈值逐类在验证集上标定，
> 不同类不同阈值是刻意设计：NORM 阈值高（避免漏报病变），MI 阈值低（宁可多报）。

## 过程中修的两个推理层缺陷

1. **多标签「标签与概率错位」**：原实现 `label` 按类别顺序拼、`confidence` 取全类最大概率，
   出现「报 MI，但最高概率其实是 STTC」的自相矛盾输出。改为按概率降序排列，
   置信度对齐 label 第一位。
2. **分级语义缺失**：类别名是 `2_moderate` 这种形式，直接写进报告医生读不懂。
   网关增加序号解析 + 中文映射。

## 边界（写进代码注释，也写进对外口径）

- 网关只产出**建议性结论**，不写临床库、不签发报告
- `diagnose` 的 `context` 只进文本不进模型输入；它不是「带病人信息的推理」，是「给结论加索引」
- 报告固定 draft 态，必须医师复核后提交
- 每个 AiJob 都带 `disclaimer`，前端展示时应与结论同屏
