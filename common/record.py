"""训练记录 —— 单 epoch 完整训练评估记录（Record）。

跨子系统共享契约（非架构、非任一 model_* 后端私有）：
由各个 `model_*` 后端的训练代码生产（每 epoch 一条，jsonl 日志），
由 `experiment` 消费（report.csv 存最优一条）。

故置于中立的 `common/` —— 所有 `model_*` 后端与 `experiment` 都从本模块取 Record，
互不越界依赖：后端之间互不认识，后端也不反向依赖编排层 experiment。

序列化：
    - to_dict / from_dict —— jsonl（保留原始精度，容忍缺键）
    - to_csv_row / from_csv_row —— report.csv（acc 保留 4 位、None → 空串）
"""

from dataclasses import dataclass, fields


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
