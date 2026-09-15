"""领域模型子包：算盘内核的数据层（类型与不变式，零编排逻辑）。

拆分自单一 domain_model.py（2026-09-07），按 DDD 聚合边界分为四个模块：

  enums.py      异常 + 枚举（领域词汇表，最底层）
  structure.py  结构层（AbacusSpec / Beam / Frame / Bead / Rod / Abacus，物理实体）
  state.py      状态层（快照 + 数值→状态构造器 num_to_state）
  delta.py      差分层（监督货币：只存变化）
  action.py     动作层 + 口诀数据类（命令值对象 + 声明式规范）

依赖单向：enums ← structure ← state；enums ← action；state ← delta。

公共 API 统一在此 re-export；推荐 `from evaluate.abacus import ...`，
需要细粒度时 `from evaluate.abacus.domain import ...`。
"""
from evaluate.abacus.domain.enums import (
    AbacusError, InvariantViolation, RhymeNotFound, UnsupportedOperation,
    BeadType, BeadPosition, Finger, OpType, DigitOrder, RhymeCategory,
    ActionType, BeadAddressing,
)
from evaluate.abacus.domain.structure import (
    AbacusSpec, Beam, Frame, Bead, Rod, Abacus,
)
from evaluate.abacus.domain.state import (
    BeadState, RodState, AbacusState,
    num_to_state,
)
from evaluate.abacus.domain.delta import (
    BeadStateDelta, RodStateDelta, StateDelta,
)
from evaluate.abacus.domain.action import (
    AtomicAction, CompositeAction,
    ExpectedRodDelta, Carry, Rhyme,
)

__all__ = [
    "AbacusError", "InvariantViolation", "RhymeNotFound", "UnsupportedOperation",
    "BeadType", "BeadPosition", "Finger", "OpType", "DigitOrder", "RhymeCategory",
    "ActionType", "BeadAddressing",
    "AbacusSpec", "Beam", "Frame", "Bead", "Rod", "Abacus",
    "BeadState", "RodState", "AbacusState",
    "BeadStateDelta", "RodStateDelta", "StateDelta",
    "num_to_state",
    "AtomicAction", "CompositeAction",
    "ExpectedRodDelta", "Carry", "Rhyme",
]
