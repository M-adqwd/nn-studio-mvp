"""SOP v2 的门禁判定：能力矩阵 + 各道门规则。

设计要点（对应 docs/SOP.md）：

1. **只判定，不做事**。这里不写库、不改数据、不抛业务异常给用户看，
   只产出结构化的 `GateReport`。是否拒绝请求由调用方（路由层）决定，
   这样才能单测、才能在脚本里复用同一套规则。
2. **每个问题都有稳定错误码**。前端、脚本、客户文档都用同一个码沟通，
   不再出现「解析异常：Could not determine delimiter」这种无法处理的文本。
3. **阻塞与警告严格分开**。警告不改变可训练性，只提示风险；
   阻塞则必须有人为动作（改目标列、补数据、显式降级）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

BLOCK = "block"
WARN = "warn"

PASS = "pass"
NEEDS_FIX = "needs_fix"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ---------------------------------------------------------------- 报告结构

@dataclass
class Issue:
    code: str
    message: str
    hint: str = ""
    level: str = BLOCK
    field: str | None = None

    def as_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "message": self.message,
                "hint": self.hint, "field": self.field}


@dataclass
class GateReport:
    gate: str
    issues: list[Issue] = field(default_factory=list)
    checked_at: str = ""

    def __post_init__(self) -> None:
        if not self.checked_at:
            self.checked_at = _now()

    # ---- 派生属性
    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.level == BLOCK]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == WARN]

    @property
    def ok(self) -> bool:
        return not self.blocking

    @property
    def status(self) -> str:
        if self.blocking:
            return NEEDS_FIX
        return "warn" if self.warnings else PASS

    @property
    def codes(self) -> list[str]:
        return [i.code for i in self.issues]

    def as_dict(self) -> dict:
        return {
            "gate": self.gate,
            "status": self.status,
            "checked_at": self.checked_at,
            "codes": self.codes,
            "blocking": [i.as_dict() for i in self.blocking],
            "warnings": [i.as_dict() for i in self.warnings],
        }

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "通过"
        if self.ok:
            return f"通过（{len(self.warnings)} 条警告）"
        return "未通过：" + "；".join(i.message for i in self.blocking[:3])


class GateBlocked(Exception):
    """门禁拒绝。携带报告与建议 HTTP 状态码，供全局异常处理器统一转信封。"""

    def __init__(self, report: GateReport, http_status: int = 422):
        self.report = report
        self.http_status = http_status
        super().__init__(report.summary())

    def envelope(self) -> dict:
        first = self.report.blocking[0] if self.report.blocking else None
        return {
            "error": {
                "code": first.code if first else "E0000",
                "message": first.message if first else "未通过准入校验",
                "hint": first.hint if first else "",
                "details": {"gate": self.report.gate, "issues": self.report.as_dict()},
            }
        }


# ---------------------------------------------------------------- 能力矩阵

@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    modality: str                 # tabular | image | text | signal | detection
    min_rows: int
    min_classes: int | None = 2
    max_classes: int | None = 200
    min_per_class: int | None = 10
    primary_metric: str = "macro_f1"
    metric_floor: float = 0.60
    publishable: bool = True
    ordinal: bool = False         # 有序分级（需额外看 QWK）
    min_text_len: int = 0
    max_cols: int | None = 2000


TASKS: dict[str, TaskSpec] = {
    "image_classification": TaskSpec(
        "image_classification", "图像分类", "image", min_rows=100,
        min_per_class=10, primary_metric="macro_f1", metric_floor=0.60),
    "tabular_classification": TaskSpec(
        "tabular_classification", "表格分类", "tabular", min_rows=100,
        max_classes=200, min_per_class=5, primary_metric="macro_f1",
        metric_floor=0.60, max_cols=2000),
    "tabular_regression": TaskSpec(
        "tabular_regression", "表格回归", "tabular", min_rows=100,
        min_classes=None, max_classes=None, min_per_class=None,
        primary_metric="r2", metric_floor=0.30, max_cols=2000),
    "text_classification": TaskSpec(
        "text_classification", "文本分类", "text", min_rows=200,
        min_per_class=10, primary_metric="macro_f1", metric_floor=0.60,
        min_text_len=10, max_cols=None),
    "signal_classification": TaskSpec(
        "signal_classification", "信号分类", "signal", min_rows=300,
        min_per_class=10, max_classes=50, primary_metric="macro_auc",
        metric_floor=0.70, max_cols=None),
    "object_detection": TaskSpec(
        "object_detection", "目标检测", "detection", min_rows=100,
        min_classes=1, min_per_class=None, primary_metric="mAP",
        metric_floor=0.25, publishable=False),
}

# 分级任务（类别名形如 0_healthy / 2_moderate）：除 macro-F1 外还要看 QWK
ORDINAL_HINTS = ("grade", "severity", "stage", "分级", "等级")


def spec_of(task: str | None) -> TaskSpec | None:
    return TASKS.get(str(task or "").strip())


def is_ordinal(classes: list[str] | None) -> bool:
    """类别名是否带序号前缀，代表有序分级。

    只认 `0_xxx` / `1-xxx` / `2 xxx` 这种「数字开头 + 分隔符」形式，
    避免把 `2020` 这种纯年份类别误判成分级。
    """
    if not classes or len(classes) < 3:
        return False
    hits = 0
    for c in classes:
        s = str(c).strip()
        if len(s) > 1 and s[0].isdigit() and not s[1].isdigit():
            hits += 1
    return hits >= max(3, int(len(classes) * 0.8))


# ---------------------------------------------------------------- G1 数据集画像门

def gate_dataset(task: str, profile: dict, target: str | None = None) -> GateReport:
    """数据集是否具备训练资格。

    profile 约定字段（缺省即视为未知，不做惩罚）：
      rows, cols, classes, class_counts, missing_pct, duplicate_rows,
      constant_cols, leak_suspects, decode_fail_pct, target_var,
      text_len_mean, target_source
    """
    rep = GateReport("dataset")
    spec = spec_of(task)
    if spec is None:
        rep.issues.append(Issue(
            "E5001", f"不支持的任务类型：{task or '(空)'}",
            "请显式指定任务类型：image_classification / tabular_classification / "
            "tabular_regression / text_classification / signal_classification"))
        return rep

    prof = profile or {}
    rows = int(prof.get("rows") or 0)
    cols = int(prof.get("cols") or 0)

    # --- 目标列
    tgt = target if target is not None else prof.get("target")
    src = str(prof.get("target_source") or "")
    if spec.modality != "detection" and not tgt:
        rep.issues.append(Issue(
            "E3001", "未能确定目标列",
            "请在数据集设置里指定用于训练的标签列；系统不再用「最后一列」兜底",
            field="target"))
    elif src == "fallback" and spec.modality == "tabular":
        # 兜底推断出的目标列必须人工确认：宽表里最后一列常常是特征而非标签
        rep.issues.append(Issue(
            "E3001", f"目标列 `{tgt}` 是自动兜底推断的，需要确认",
            "宽表/信号表里最后一列通常是特征。请确认目标列，或声明为 signal_classification",
            field="target"))

    # --- 样本量
    if rows and rows < spec.min_rows:
        rep.issues.append(Issue(
            "E3002", f"样本量不足：{rows} < {spec.min_rows}",
            f"{spec.label}至少需要 {spec.min_rows} 条样本才能给出可信指标"))

    # --- 类别
    classes = prof.get("classes")
    if spec.min_classes is not None:
        n = len(classes) if classes else int(prof.get("n_classes") or 0)
        if n and n < 2:
            rep.issues.append(Issue(
                "E3003", f"只有 1 个类别，无法训练分类模型",
                "确认标签列是否正确；若这是单类别筛查任务，应改为异常检测或补充负样本"))
        elif n and n < spec.min_classes:
            rep.issues.append(Issue(
                "E3003", f"类别数不足：{n} < {spec.min_classes}"))
        elif n and spec.max_classes and n > spec.max_classes:
            rep.issues.append(Issue(
                "E3003", f"类别数过多：{n} > {spec.max_classes}",
                "类别过多通常是标签列选错（例如把 ID、文件名、连续值当成了标签）"))

    # --- 标签泄漏
    leaks = prof.get("leak_suspects") or []
    if leaks:
        rep.issues.append(Issue(
            "E3004", f"疑似标签泄漏：特征 {', '.join(map(str, leaks[:3]))} 与目标列完全同值",
            "删除同值特征列，或换一个目标列；否则指标会虚高且上线后失效",
            field=str(leaks[0])))

    # --- 缺失率
    miss = prof.get("missing_pct")
    if isinstance(miss, (int, float)):
        if miss > 0.5:
            rep.issues.append(Issue(
                "E3005", f"缺失率过高：{miss:.0%}",
                "先跑清洗补齐/剔除，再训练"))
        elif miss > 0.2:
            rep.issues.append(Issue(
                "W3001", f"缺失率偏高：{miss:.0%}", "建议先清洗", WARN))

    # --- 回归目标
    if task == "tabular_regression":
        var = prof.get("target_var")
        if isinstance(var, (int, float)) and var <= 0:
            rep.issues.append(Issue(
                "E3006", "目标列方差为 0（所有值相同）", "回归任务需要目标值有变化"))

    # --- 图像可解码率
    fail = prof.get("decode_fail_pct")
    if isinstance(fail, (int, float)):
        if fail > 0.05:
            rep.issues.append(Issue(
                "E3002", f"图像解码失败率 {fail:.0%}，超过 5% 上限",
                "抽样检查损坏文件；建议重新导出图像"))
        elif fail > 0:
            rep.issues.append(Issue(
                "W3002", f"图像解码失败率 {fail:.0%}", "训练会自动跳过损坏样本", WARN))

    # --- 文本长度
    if spec.min_text_len:
        tl = prof.get("text_len_mean")
        if isinstance(tl, (int, float)) and tl < spec.min_text_len:
            rep.issues.append(Issue(
                "E3002", f"文本平均长度 {tl:.0f} 字符，低于 {spec.min_text_len}",
                "确认正文列是否选对；过短的文本不适合做分类"))

    # --- 宽表
    if spec.max_cols and cols > spec.max_cols:
        rep.issues.append(Issue(
            "E3001", f"特征列过多：{cols} 列（上限 {spec.max_cols}）",
            "宽表通常不是普通表格任务。若是多导联/时序信号，请声明 signal_classification；"
            "否则请先做特征筛选"))

    # --- 类别不平衡 / 小类
    cc = prof.get("class_counts") or {}
    if cc and spec.min_per_class:
        counts = [int(v) for v in cc.values() if int(v) > 0]
        if counts:
            small = [k for k, v in cc.items() if int(v) < spec.min_per_class]
            if small:
                rep.issues.append(Issue(
                    "W3003", f"{len(small)} 个类别样本少于 {spec.min_per_class}（{', '.join(map(str, small[:3]))}）",
                    "这些类别的指标不可信，建议补样本或合并类别", WARN))
            if max(counts) / max(min(counts), 1) > 20:
                rep.issues.append(Issue(
                    "W3004", f"类别极不平衡：最多/最少 = {max(counts) / max(min(counts), 1):.0f} 倍",
                    "开启类别权重，并优先看 macro 指标而非准确率", WARN))

    # --- 常量列 / 重复行
    const = prof.get("constant_cols") or []
    if const:
        rep.issues.append(Issue(
            "W3005", f"{len(const)} 个特征列取值恒定：{', '.join(map(str, const[:3]))}",
            "常量列对训练无贡献，可删除", WARN))
    dup = prof.get("duplicate_rows")
    if isinstance(dup, (int, float)) and rows and dup / max(rows, 1) > 0.3:
        rep.issues.append(Issue(
            "W3006", f"重复行占比 {dup / max(rows, 1):.0%}",
            "重复样本会让验证集虚高，建议去重", WARN))

    return rep


# ---------------------------------------------------------------- G3 训练准入

def gate_training(task: str, dataset_status: str = "ready",
                  dataset_gate: dict | None = None,
                  degraded: bool = False, allow_degraded: bool = False) -> GateReport:
    """能否为这个数据集创建训练任务。"""
    rep = GateReport("training")
    spec = spec_of(task)
    if spec is None:
        rep.issues.append(Issue(
            "E5001", f"不支持的任务类型：{task or '(空)'}",
            "平台只接受能力矩阵内的任务；请先确认数据集任务声明"))
        return rep

    if dataset_status in ("processing", "failed"):
        rep.issues.append(Issue(
            "E5002", f"数据集状态为 {dataset_status}，尚未就绪",
            "等待解析完成，或重新上传"))
    if dataset_status == NEEDS_FIX:
        codes = [c for c in ((dataset_gate or {}).get("codes") or []) if c.startswith("E")]
        rep.issues.append(Issue(
            "E5002", f"数据集未通过画像门（{', '.join(codes[:3]) or '存在阻塞项'}）",
            "先处理数据集画像里的阻塞项，再开始训练"))

    if degraded and not allow_degraded:
        rep.issues.append(Issue(
            "E5003", "本次训练会降级（自动建模不可用时的基础路径），但没有显式开启",
            "如需在降级条件下训练，请在请求里传 allow_degraded=true；"
            "降级模型的指标通常明显更差，不能用于发布",
            field="allow_degraded"))
    return rep


# ---------------------------------------------------------------- G4 评估门

def _pick_metric(metrics: dict, name: str) -> float | None:
    """从嵌套指标里取主指标：优先 test_metrics（实际落库结构），其次 test/eval/val。"""
    if not isinstance(metrics, dict):
        return None
    for scope in ("test_metrics", "test", "eval", "val"):
        sub = metrics.get(scope)
        if isinstance(sub, dict) and sub.get(name) is not None:
            try:
                return float(sub[name])
            except (TypeError, ValueError):
                return None
    if metrics.get(name) is not None:
        try:
            return float(metrics[name])
        except (TypeError, ValueError):
            return None
    return None


def gate_release(task: str, metrics: dict, has_test: bool = True,
                 coverage: float | None = None,
                 ordinal: bool = False) -> GateReport:
    """模型能否从 candidate 升为 production。"""
    rep = GateReport("release")
    spec = spec_of(task)
    if spec is None:
        rep.issues.append(Issue("E5001", f"不支持的任务类型：{task or '(空)'}"))
        return rep

    if not spec.publishable:
        rep.issues.append(Issue(
            "E6004", f"{spec.label}当前为实验能力，不允许发布",
            "实验能力只能在沙箱试用；发布需先补齐评估口径与稳定性验证"))
        return rep

    if not has_test:
        rep.issues.append(Issue(
            "E6003", "没有独立测试集评估结果，不能发布",
            "请带独立测试集重训；用训练集/验证集指标发布会系统性高估效果"))

    val = _pick_metric(metrics, spec.primary_metric)
    metric_name, floor = spec.primary_metric, spec.metric_floor
    if val is None and task == "signal_classification":
        # 单标签心电任务本身没有 AUC 口径（AUC 属于多标签判定），
        # 退回 macro-F1 并相应调整下限，避免「没有这个指标」把合格模型误杀。
        val = _pick_metric(metrics, "macro_f1")
        metric_name, floor = "macro_f1", 0.60
    if val is None:
        rep.issues.append(Issue(
            "E6002", f"缺少主指标 {spec.primary_metric}",
            "评估阶段必须产出该指标"))
    elif val < floor:
        rep.issues.append(Issue(
            "E6002", f"{metric_name} = {val:.4f}，低于发布下限 {floor}",
            "补充数据或调整方案后重训；不达标模型不允许进入生产"))

    if ordinal:
        qwk = _pick_metric(metrics, "qwk")
        if qwk is not None and qwk < 0.60:
            rep.issues.append(Issue(
                "E6002", f"分级任务 QWK = {qwk:.4f}，低于 0.60",
                "分级任务的相邻等级混淆代价高，QWK 不达标不建议发布"))

    if coverage is not None and coverage < 1.0:
        rep.issues.append(Issue(
            "W6001", f"测试集类别覆盖 {coverage:.0%}，存在未评估类别",
            "未覆盖类别在生产中无法保证表现", WARN))
    return rep
