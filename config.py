"""实验域模型 —— Experiment 聚合根及其组成部分（Material / Method / Trial / Record）。

2026-09-04 拆分：模型架构配置（BrainConfig / PosEmbConfig / ModelConfig）
已迁至 `model/config.py`，本模块不再被模型子系统依赖。

依赖方向（单向无环）：config.py → model.config。

本模块是**纯数据定义**：不读文件，也不认识任何 CSV / YAML / JSON 的键名。
    - 文件格式的解析与校验 → `experiment/schema.py`（唯一读 config.yaml 的地方）
    - 磁盘布局与聚合根装配 → `experiment/loader.py`
这样"配置键改名"和"产物目录调整"都不会波到这里。
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any

from model.config import BrainConfig, PosEmbConfig


@dataclass
class Material:
    """教材 —— 学什么（trial 级纯数据）。

    哪些字段有意义由 `type` 决定，必填项由 schema 校验：
        dataset: source 必填（外部数据集投影，多个 trial 共享同一份 train/test）
        expr:    ops / repeat / start / end 必填（按算符与区间枚举生成）
        bead:    start / end 必填
    """
    type: str
    # ---- expr / bead ----
    ops: str = ""
    repeat: int = 1
    start: int = 0
    end: int = 0
    allow_negative: bool = False
    # ---- 投影方式（expr / dataset 共用）----
    input_format: str = "infix"        # infix | prefix | postfix
    parse: str | None = None           # direct | fixed | pre | post | prepost | prepost_stack | none
    eval: str | None = None            # none | digit | abacus | no_ans
    sample: int | None = None          # 子采样条数（None = 全量）
    split: float = 0.2                 # 测试集比例；<=0 时 train/test 指向同一份
    # ---- 珠算投影 ----
    abacus_bead: bool = False          # 珠算口诀是否附带盘面快照（首尾锚定，~分隔）
    abacus_style: str = "song"         # song(宋制,上1下4,十进制) | ming(明制,上2下5,十六进制)
    # ---- dataset ----
    source: str = ""                   # 如 parse/dataset/dataset_C.jsonl

    @property
    def abacus_base(self) -> int:
        """珠态分解进制：song(宋制,上1下4)=10，ming(明制,上2下5)=16。"""
        return 16 if self.abacus_style == "ming" else 10


@dataclass
class Method:
    """学法 —— 怎么学（trial 级）。

    由 schema 合并 `trial_defaults.method` 后构造，因此五个字段必定齐全。
    """
    repeat_factor: int
    epochs: int
    batch_size: int
    learning_rate: float
    shuffle: bool


@dataclass
class TrainConfig:
    """训练规范 —— 怎么训（实验级共享的训练行为控制）。"""
    enable_validation: bool            # 每 epoch 是否做完整评估（关掉则只看 train_loss）
    early_stop_patience: int           # 连续多少个 epoch 无提升即停（0/负数 = 不早停）
    dropout: float
    train_seed: int = 42               # 模型初始化 / 数据 shuffle 的随机种子


@dataclass
class EvalConfig:
    """考评规范 —— 怎么考（实验级共享）。"""
    eval_batch_size: int               # 无默认值，强制必填
    print_interval: int                # 无默认值，强制必填
    acc_mode: str = "acc"              # 主指标: "acc" | "acc_ans" | "acc_think"（决定 passed/早停用哪个）
    pass_threshold: float = 0.95       # 通过阈值（多 trial 共享，判定单 trial 是否通过）


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
class TrialPaths:
    """单个 trial 的全部磁盘位置（数据 / 权重 / 日志）。

    由 loader 在装配 Experiment 时统一计算，供各层直接引用，
    杜绝散落各处的字符串拼接（旧代码里 data 路径挂在 Material、
    产物路径挂在 TrialArtifacts，同一个 trial 的位置被拆在两处）。
    """
    train_data: str        # 训练集 jsonl
    test_data: str         # 测试集 jsonl（split<=0 时与 train_data 同一份）
    bead_data: str         # 珠态数据 jsonl（dataset 类型为空串）
    best_model: str        # trial_{id}.pth —— 通过后最终权重
    checkpoint: str        # trial_{id}_ckpt.pt —— 断点续训（含 optimizer 状态）
    train_log: str         # trial_{id}_train.jsonl
    test_log: str          # trial_{id}_test.jsonl

    @classmethod
    def for_trial(cls, exp: "Experiment", trial_id: int, trial_name: str,
                  material: Material) -> "TrialPaths":
        """按 Experiment 的目录约定 + material 的投影参数计算全部路径。"""
        material_dir = exp.material_dir
        if material.type == "dataset":
            # dataset 类型：投影自 source，按 (source, input_format, parse, eval) 去重共享路径，
            # 同四元组的 trial 指向同一份 train/test，避免重复生成。
            src_basename = os.path.splitext(os.path.basename(material.source))[0]
            proj_key = (f"{src_basename}_{material.input_format}"
                        f"_{material.parse or 'none'}_{material.eval or 'none'}")
            return cls(
                train_data=os.path.join(material_dir, f"{proj_key}_train.jsonl"),
                test_data=os.path.join(material_dir, f"{proj_key}_test.jsonl"),
                bead_data="",
                best_model=os.path.join(exp.ckpt_dir, f"trial_{trial_id}.pth"),
                checkpoint=os.path.join(exp.ckpt_dir, f"trial_{trial_id}_ckpt.pt"),
                train_log=os.path.join(exp.log_dir, f"trial_{trial_id}_train.jsonl"),
                test_log=os.path.join(exp.log_dir, f"trial_{trial_id}_test.jsonl"),
            )

        if material.type == "bead":
            train_data = os.path.join(material_dir, f"trial{trial_id}_{trial_name}_bead.jsonl")
        else:
            ops_tag = (material.ops.replace('&', 'and').replace('|', 'or')
                       .replace('*', 'mul').replace('/', 'div'))
            mode_tag = f"{material.parse or 'none'}_{material.eval or 'none'}"
            train_data = os.path.join(
                material_dir,
                f"trial{trial_id}_{trial_name}_expr_{ops_tag}_{mode_tag}"
                f"_{material.input_format}_{material.repeat}_{material.start}-{material.end}.jsonl"
            )
        return cls(
            train_data=train_data,
            test_data=train_data if material.split <= 0 else train_data.replace('.jsonl', '_test.jsonl'),
            bead_data=os.path.join(material_dir, f"trial{trial_id}_{trial_name}_bead.jsonl"),
            best_model=os.path.join(exp.ckpt_dir, f"trial_{trial_id}.pth"),
            checkpoint=os.path.join(exp.ckpt_dir, f"trial_{trial_id}_ckpt.pt"),
            train_log=os.path.join(exp.log_dir, f"trial_{trial_id}_train.jsonl"),
            test_log=os.path.join(exp.log_dir, f"trial_{trial_id}_test.jsonl"),
        )


@dataclass
class Trial:
    """单个 trial —— 一次独立训练的完整规格与结果。

    `name` 是全局唯一标签，同时是 report.csv 的 name 列，
    因此也是"配置与结果是否对齐"的指纹（见 store.ReportTable）。
    """
    id: int                            # 列表位置派生（0..N-1），与 report.csv 的 id 一致
    name: str                          # 唯一
    material: Material
    method: Method
    heads: list[int]                   # 逐层头数，长度 = num_hidden_layers（已解析，不含 None）
    paths: TrialPaths
    record: Record | None = None       # 最优 epoch 的记录（report.csv 中那行）
    passed: bool | None = None         # None = 未评估


@dataclass
class Experiment:
    """实验聚合根 —— 一项实验，含共享配置和全部 trial。

    所有实验统一为 experiment 语义：每个 trial 独立训练、无复习、无权重传递、
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
        """实验目录内子目录/文件路径。"""
        return os.path.join(self.root_dir, *parts)

    @property
    def root_dir(self) -> str:
        """实验根目录: experiment/studies/<name>/"""
        return os.path.join(self.project_root, "experiment", "studies", self.name)

    @property
    def config_path(self) -> str:
        """实验配置（唯一输入）: experiment/studies/<name>/config.yaml"""
        return self._dir("config.yaml")

    @property
    def report_path(self) -> str:
        """实验进度表（最优 Record + passed）: experiment/studies/<name>/report.csv"""
        return self._dir("report.csv")

    @property
    def material_dir(self) -> str:
        """实验材料目录（生成的数据集）。"""
        return self._dir("material")

    @property
    def vocab_dir(self) -> str:
        """实验词表目录。"""
        return self._dir("vocab")

    @property
    def ckpt_dir(self) -> str:
        """实验模型保存目录（权重）。"""
        return self._dir("memory")

    @property
    def log_dir(self) -> str:
        """实验日志目录。"""
        return self._dir("logs")

    @property
    def vocab_path(self) -> str:
        """共享词表路径（按探测到的数据源去重后生成）。"""
        return os.path.join(self.vocab_dir, f"{self.name}_vocab.json")


@dataclass
class ExecutionContext:
    """运行时上下文 —— 封装训练/评估时的运行时状态。

    设计原则：区分「静态配置」与「动态运行时」。
      - Experiment / Trial / BrainConfig 等：静态配置，可序列化，定义"学什么、怎么学"。
      - ExecutionContext：动态运行时，不可序列化，承载"用什么设备、什么词表，
        以及本次尝试的种子与轮数"。

    好处：各层函数签名从散列的 15+ 参数收敛为 (trial, exp, ctx) 三个对象，
    新增运行时状态只需加字段，无需改动各层函数签名。
    """
    vocab_data: Any = None              # Vocab 对象（词表，运行时加载）
    device: Any = None                  # torch.device（计算设备）
    seed: int = 42                      # 本次尝试的随机种子
    prev_model_path: str | None = None  # 上一 trial 权重（实验语义下恒为 None，见 runner）
