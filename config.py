"""课程域配置 —— 学习单元（Experiment）及其组成部分（教材/学法/考评/记录）。

2026-09-04 拆分：模型架构配置（BrainConfig / PosEmbConfig / ModelConfig）
已迁至 `model/config.py`，本模块不再被模型子系统依赖。

依赖方向（单向无环）：config.py → model.config。
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any

from model.config import BrainConfig, PosEmbConfig

# ==================== 工具函数 ====================

def _int(s, default=None):
    s = (s or "").strip()
    return int(s) if s else default


def _float(s, default=None):
    s = (s or "").strip()
    return float(s) if s else default


def _bool(s):
    return (s or "").strip().lower() in ("true", "1", "yes")


@dataclass
class Material:
    """教材 —— 学什么"""
    type: str                       # "bead" | "expr" | "env"
    ops: str                        # "+-", "+", etc.
    repeat: int
    start: int
    end: int
    allow_negative: bool
    input_fmt: str = "infix"
    parse: str | None = None
    eval: str | None = None
    sample: int | None = None
    forbidden_patterns: str | None = None
    split: float = 0.2
    abacus_bead: bool = False          # 珠算口诀是否附带盘面快照（首尾锚定，~分隔）
    abacus_order: str = "ltr"          # 珠算顺序 ltr(高位优先)/rtl(低位优先)
    abacus_snapshot: bool = False      # 每句口诀后是否附带该步盘面快照（口诀→快照交织）
    abacus_style: str = "song"         # 算盘风格 song(宋制,上1下4,十进制) | ming(明制,上2下5,十六进制)
    env_steps: int = 20                # env 类型：闭环最大交互步数（防死循环）
    source: str = ""                   # dataset 类型：外部数据集路径（如 parse/dataset/dataset_C.jsonl）
    variant: str = ""                  # head_patterns 内层 key（峰值头数，如 "16"）
    # 派生路径（由 load_experiment 填充）
    train_data_path: str = ""
    test_data_path: str = ""
    bead_data_path: str = ""
    # 派生：本课显式头分布（由 load_experiment 经 head_patterns 填充）
    heads_per_layer: list[int] | None = None

    @classmethod
    def from_row(cls, row: dict) -> "Material":
        """从 CSV 原始字符串行构造。"""
        return cls(
            type=row["type"].strip(),
            ops=row["ops"].strip(),
            repeat=_int(row["repeat"]),
            start=_int(row["start"]),
            end=_int(row["end"]),
            allow_negative=_bool(row["allow_negative"]),
            input_fmt=row.get("input_format", "").strip() or "infix",
            parse=row.get("parse", "").strip() or None,
            eval=row.get("eval", "").strip() or None,
            sample=_int(row.get("sample")),
            forbidden_patterns=row.get("forbidden_patterns", "").strip() or None,
            split=_float(row.get("split", ""), 0.2),
            abacus_bead=_bool(row.get("abacus_bead", "false")),
            abacus_order=(row.get("abacus_order") or "").strip() or "ltr",
            abacus_snapshot=_bool(row.get("abacus_snapshot", "false")),
            abacus_style=(row.get("abacus_style") or "").strip() or "song",
            env_steps=_int(row.get("env_steps"), 20),
            source=(row.get("source") or "").strip(),
            variant=(row.get("variant") or "").strip(),
        )

    @property
    def abacus_base(self) -> int:
        """珠态分解进制：song(宋制,上1下4)=10，ming(明制,上2下5)=16。"""
        return 16 if self.abacus_style == "ming" else 10

    def fill_paths(self, trial_id: int, trial_name: str, material_dir: str) -> None:
        """根据 material 参数生成并填充派生数据文件路径。"""
        if self.type == "dataset":
            # dataset 类型：投影自 source，按 (source,input_format,parse,eval) 去重共享路径，
            # 同四元组的 trial 指向同一份 train/test，避免每课重复生成。
            src_basename = os.path.splitext(os.path.basename(self.source))[0]
            proj_key = f"{src_basename}_{self.input_fmt}_{self.parse or 'none'}_{self.eval or 'none'}"
            self.train_data_path = os.path.join(material_dir, f"{proj_key}_train.jsonl")
            self.test_data_path = os.path.join(material_dir, f"{proj_key}_test.jsonl")
            self.bead_data_path = ""
            return
        if self.type == "bead":
            train_path = os.path.join(material_dir, f"L{trial_id}_{trial_name}_bead.jsonl")
        elif self.type == "env":
            train_path = os.path.join(material_dir, f"L{trial_id}_{trial_name}_env.jsonl")
        else:
            ops_tag = self.ops.replace('&', 'and').replace('|', 'or').replace('*', 'mul').replace('/', 'div')
            mode_tag = f"{self.parse or 'none'}_{self.eval or 'none'}"
            train_path = os.path.join(
                material_dir,
                f"L{trial_id}_{trial_name}_expr_{ops_tag}_{mode_tag}_infix_{self.repeat}_{self.start}-{self.end}.jsonl"
            )
        test_path = train_path.replace('.jsonl', '_test.jsonl')
        self.train_data_path = train_path
        self.test_data_path = train_path if self.split <= 0 else test_path
        self.bead_data_path = os.path.join(material_dir, f"L{trial_id}_{trial_name}_bead.jsonl")



@dataclass
class Method:
    """学法 —— 怎么学"""
    repeat_factor: int
    epochs: int
    batch_size: int
    learning_rate: float
    shuffle: bool

    @classmethod
    def from_row(cls, row: dict) -> "Method":
        """从 CSV 原始字符串行构造。"""
        return cls(
            repeat_factor=int(row["repeat_factor"]),
            epochs=int(row["epochs"]),
            batch_size=int(row["batch_size"]),
            learning_rate=float(row["learning_rate"]),
            shuffle=_bool(row["shuffle"]),
        )

    @classmethod
    def default_from_train(cls, train: "TrainConfig") -> "Method":
        """从 exp 级 TrainConfig 合成默认 Method（experiment 型各课统一）。"""
        return cls(
            repeat_factor=train.repeat_factor,
            epochs=train.epochs,
            batch_size=train.batch_size,
            learning_rate=train.learning_rate,
            shuffle=train.shuffle,
        )


@dataclass
class TrainConfig:
    """训练规范 —— 怎么训（学习单元级共享的训练行为控制）"""
    enable_validation: bool        # 无默认值，强制必填
    early_stop_patience: int       # 无默认值，强制必填
    use_cosine_schedule: bool      # 无默认值，强制必填
    lr_decay_per_trial: float
    dropout: float
    train_seed: int = 42           # 训练/编排随机种子（模型初始化、数据 shuffle、复习采样）
    # ---- experiment 型共享 method（course 型以 method.csv 每课为准，下列字段忽略） ----
    epochs: int = 0                # 共享训练轮数
    repeat_factor: int = 1         # 共享数据重复倍数
    batch_size: int = 512          # 共享批大小
    learning_rate: float = 1e-3    # 共享学习率
    shuffle: bool = True           # 共享是否打乱

    @classmethod
    def from_json(cls, data: dict) -> "TrainConfig":
        """从 train.json 构造；可省略字段缺失时用默认值。"""
        return cls(**data)

@dataclass
class EvalConfig:
    """考评规范 —— 怎么考（课程级共享）"""
    eval_batch_size: int        # 无默认值，强制必填
    print_interval: int         # 无默认值，强制必填
    acc_mode: str = "acc"       # 主指标: "acc" | "acc_ans" | "acc_think"（决定 passed/早停用哪个，默认整串）
    pass_threshold: float = 0.95   # 通过阈值（多 trial 共享，判定单课是否通过）

    @classmethod
    def from_json(cls, data: dict) -> "EvalConfig":
        """从 eval.json 构造。"""
        return cls(**data)

@dataclass
class Record:
    """单 epoch 完整训练评估记录 —— csv(report.csv) 存最优一条，log(jsonl) 存全部。

    验证集 = 测试集，每 epoch 结束即做完整评估，
    故最优 Record 的成绩就是最终成绩（与保存 best model 的时机天然对齐）。
    """
    epoch: int | None = None
    timestamp: str | None = None            # Record 产生时间
    train_loss: float | None = None
    lr: float | None = None
    # 该 epoch 结束时在验证集(=测试集)上的评估（无验证时为 None）
    acc: float | None = None
    acc_ans: float | None = None
    acc_think: float | None = None
    correct: int | None = None
    correct_ans: int | None = None
    correct_think: int | None = None
    total: int | None = None
    epoch_time_s: float | None = None
    elapsed_s: float | None = None         # 累计耗时

    def to_dict(self) -> dict:
        """jsonl 序列化（保留原始精度，None → null）。"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        """jsonl 反序列化（容忍缺键，如旧日志无 acc 字段）。"""
        return cls(**{f.name: d.get(f.name) for f in fields(cls)})

    def to_csv_row(self) -> dict:
        """CSV 行（字符串值；acc 系列保留 4 位小数，None → 空串）。"""
        def fmt(v, nd=None):
            if v is None:
                return ""
            return f"{v:.{nd}f}" if nd is not None else str(v)
        return {
            "epoch": fmt(self.epoch),
            "timestamp": self.timestamp or "",
            "train_loss": fmt(self.train_loss),
            "lr": fmt(self.lr),
            "acc": fmt(self.acc, 4),
            "acc_ans": fmt(self.acc_ans, 4),
            "acc_think": fmt(self.acc_think, 4),
            "correct": fmt(self.correct),
            "correct_ans": fmt(self.correct_ans),
            "correct_think": fmt(self.correct_think),
            "total": fmt(self.total),
            "epoch_time_s": fmt(self.epoch_time_s),
            "elapsed_s": fmt(self.elapsed_s),
        }

    @classmethod
    def from_csv_row(cls, row: dict) -> "Record | None":
        """从 CSV 原始字符串行构造；acc 为空（未评估）返回 None。"""
        def _f(k):
            s = (row.get(k) or "").strip()
            return float(s) if s else None

        def _i(k):
            s = (row.get(k) or "").strip()
            return int(s) if s else None

        if _f("acc") is None:
            return None
        return cls(
            epoch=_i("epoch"),
            timestamp=(row.get("timestamp") or "").strip() or None,
            train_loss=_f("train_loss"), lr=_f("lr"),
            acc=_f("acc"), acc_ans=_f("acc_ans"), acc_think=_f("acc_think"),
            correct=_i("correct"), correct_ans=_i("correct_ans"),
            correct_think=_i("correct_think"), total=_i("total"),
            epoch_time_s=_f("epoch_time_s"), elapsed_s=_f("elapsed_s"),
        )


# Record 字段名列表（列顺序，供 csv 表头/序列化共用）
RECORD_FIELDS = [f.name for f in fields(Record)]



@dataclass
class TrialArtifacts:
    """单课产物路径 —— 训练/评测过程中生成的文件位置。

    由 load_experiment 时统一计算填充（依赖 ckpt_dir / log_dir），
    供各层直接引用，避免散落各处的字符串拼接。
    """
    best_model_path: str   # trial_L{id}.pth —— 通过后最终权重（课末评测 + 下一课初始化）
    checkpoint_path: str   # checkpoint_L{id}.pt —— 断点续训（含 optimizer 状态）
    train_log_path: str    # L{id}_train.jsonl
    test_log_path: str     # L{id}_test.jsonl

    @classmethod
    def for_trial(cls, exp: "Experiment", trial_id: int) -> "TrialArtifacts":
        """根据 Experiment 的目录约定，为指定课计算产物路径。"""
        return cls(
            best_model_path=os.path.join(exp.ckpt_dir, f"trial_L{trial_id}.pth"),
            checkpoint_path=os.path.join(exp.ckpt_dir, f"checkpoint_L{trial_id}.pt"),
            train_log_path=os.path.join(exp.log_dir, f"L{trial_id}_train.jsonl"),
            test_log_path=os.path.join(exp.log_dir, f"L{trial_id}_test.jsonl"),
        )


@dataclass
class Trial:
    """单课聚合配置"""
    id: int
    name: str
    material: Material
    method: Method
    record: Record | None = None       # 最优 epoch 的记录（report.csv 中那行）
    passed: bool | None = None         # None = 未评估
    artifacts: TrialArtifacts | None = None   # load_experiment 时由 loader 填充

    @property
    def display_name(self) -> str:
        """展示名：{族名}-{variant}（variant 空则仅 name），供 report/bucket/图标签用。"""
        if self.material.variant:
            return f"{self.name}-{self.material.variant}"
        return self.name

    @classmethod
    def from_row(cls, row: dict, material_dir: str,
                 default_method: "Method | None" = None) -> "Trial":
        """从合并后的 CSV 原始字符串行构造（含派生路径填充）。

        default_method: method 字段缺失时兜底（experiment 型统一 method 用）。
        """
        material = Material.from_row(row)
        method = Method.from_row(row) if "repeat_factor" in row else default_method
        record = Record.from_csv_row(row)
        passed_s = (row.get("passed") or "").strip()
        trial_id = int(row["id"])
        trial_name = row["name"].strip()
        material.fill_paths(trial_id, trial_name, material_dir)
        return cls(id=trial_id, name=trial_name, material=material, method=method,
                   record=record,
                   passed=(passed_s == "1") if passed_s else None)



@dataclass
class Experiment:
    """学习单元聚合根 —— 一项 exp（研究/学习），含共享配置和全部课节。

    所有 exp 统一为 experiment 语义：每课独立训练、无复习、无权重传递、
    词表按数据源去重后共享（见 material_adapter.ensure_experiment_vocab）。
    """
    name: str
    project_root: str
    brain: BrainConfig = field(default_factory=BrainConfig)
    pos_emb: PosEmbConfig = field(default_factory=PosEmbConfig)
    train: TrainConfig | None = None        # 训练规范（默认 None，load_experiment 严格加载）
    eval: EvalConfig | None = None          # 考评规范（默认 None，load_experiment 严格加载）
    data_seed: int = 42                     # 数据生成随机种子（与训练解耦，保证数据集稳定可复现）
    trials: list[Trial] = field(default_factory=list)

    def _dir(self, *parts: str) -> str:
        """学习单元内子目录/文件路径。"""
        return os.path.join(self.root_dir, *parts)

    @property
    def root_dir(self) -> str:
        """学习单元根目录: experiment/studies/<name>/"""
        return os.path.join(self.project_root, "experiment", "studies", self.name)

    @property
    def material_csv_path(self) -> str:
        """学什么表: experiment/studies/<name>/material.csv"""
        return self._dir("material.csv")

    @property
    def method_csv_path(self) -> str:
        """怎么学表: experiment/studies/<name>/method.csv"""
        return self._dir("method.csv")

    @property
    def report_csv_path(self) -> str:
        """学得怎么样表（最优 Record + passed）: experiment/studies/<name>/report.csv"""
        return self._dir("report.csv")

    @property
    def material_dir(self) -> str:
        """学习单元材料目录。"""
        return self._dir("material")

    @property
    def vocab_dir(self) -> str:
        """学习单元词表目录。"""
        return self._dir("vocab")

    @property
    def ckpt_dir(self) -> str:
        """学习单元模型保存目录（记忆/权重）。"""
        return self._dir("memory")

    @property
    def log_dir(self) -> str:
        """学习单元日志目录。"""
        return self._dir("logs")

    @property
    def config_dir(self) -> str:
        """学习单元结构化 JSON 配置目录。"""
        return self._dir("config")

    @property
    def brain_path(self) -> str:
        """学习单元 brain 配置文件路径。"""
        return self._dir("config", "brain.json")

    @property
    def pos_emb_path(self) -> str:
        """学习单元 pos_emb 配置文件路径。"""
        return self._dir("config", "pos_emb.json")

    @property
    def policy_path(self) -> str:
        """学习单元 policy 配置文件路径。"""
        return self._dir("config", "policy.json")

    @property
    def eval_path(self) -> str:
        """学习单元 eval 配置文件路径。"""
        return self._dir("config", "eval.json")

    @property
    def train_path(self) -> str:
        """学习单元 train 配置文件路径。"""
        return self._dir("config", "train.json")

    @property
    def vocab_path(self) -> str:
        """统一词表路径（course 类型用）。"""
        return os.path.join(self.vocab_dir, f"{self.name}_vocab.json")

    def trial_vocab_path(self, trial_name: str) -> str:
        """单课独立词表路径（experiment 类型用）。"""
        return os.path.join(self.vocab_dir, f"{trial_name}_vocab.json")


@dataclass
class ExecutionContext:
    """运行时上下文 —— 封装训练/评估时的运行时状态。

    设计原则：区分「静态配置」与「动态运行时」。
      - Experiment / Trial / BrainConfig 等：静态配置，可序列化，定义"学什么、怎么学"。
      - ExecutionContext：动态运行时，不可序列化，承载"用什么设备、什么词表、
        上一课权重、复习数据，以及本次尝试的种子与轮数"。

    好处：各层函数签名从散列的 15+ 参数收敛为 (trial, exp, ctx) 三个对象，
    新增运行时状态只需加字段，无需改动各层函数签名。
    """
    vocab_data: Any = None              # Vocab 对象（词表，运行时加载）
    device: Any = None                  # torch.device（计算设备）
    # ---- 以下字段随「重试 attempt」动态变化，由编排层（train_and_eval_trial）更新 ----
    epochs: int = 0                     # 本次尝试的训练轮数（随重试递增）
    seed: int = 42                      # 本次尝试的随机种子
    prev_model_path: str | None = None  # 上一课权重（仅首次尝试用于初始化/零样本评估）
    review_files: list[str] | None = None  # 复习数据文件列表（course 类型非空）
