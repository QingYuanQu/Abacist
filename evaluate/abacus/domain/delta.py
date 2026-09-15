# -*- coding: utf-8 -*-
"""差分层：监督货币（只存变化）。

拆分自原 domain.state（2026-09-07 将状态层与差分层物理分离）。依赖 enums（词汇表）
与 state（AbacusState 快照，用作 diff 的入参类型）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .enums import BeadPosition, BeadType, InvariantViolation
from .state import AbacusState


# ═══════════════════ 4. 差分层（监督货币：只存变化） ═══════════════════

@dataclass(frozen=True)
class BeadStateDelta:
    """单珠变化（与 AtomicAction.before/after 对称；bead_type 使差分自足可算账）"""
    bead_id: int
    bead_type: BeadType
    before: BeadPosition
    after: BeadPosition

    def __post_init__(self):
        if self.before is self.after:
            raise InvariantViolation("差分只记变化珠：before == after")

@dataclass(frozen=True)
class RodStateDelta:
    """单档变化：value_delta 为本档单位（'进一'即 +1，进制无关；绝对值 = ×place_value）"""
    rod_index: int
    place_value: int
    changed_beads: tuple[BeadStateDelta, ...]

    @property
    def value_delta(self) -> int:
        return sum((1 if c.after.is_active else -1) * c.bead_type.value
                   for c in self.changed_beads)

@dataclass(frozen=True)
class StateDelta:
    """整盘变化：裁判的'实际差分'端。

    自带 base（由 diff() 取自 after 状态）：差分应自足可算账——与 BeadStateDelta
    自带 bead_type 同理，不该要求调用方额外传入本该由对象持有的信息。
    """
    step_before: int
    step_after: int
    rod_deltas: tuple[RodStateDelta, ...]
    base: int

    def __post_init__(self):
        if self.step_after != self.step_before + 1:
            raise InvariantViolation("step_after 必须 = step_before + 1")

    @property
    def total_value_delta(self) -> int:
        return sum(rd.value_delta * rd.place_value for rd in self.rod_deltas)

    def matches_semantic(self, value_delta: int, base_rod_index: int) -> bool:
        """内生对账（方案④）：本步实际总盘净值应等于语义净变化 × 基准档位值。
        进位/退位已拆成独立步，单步期望净变化精确等于本档 value_delta，跨档项不再手写。"""
        return self.total_value_delta == value_delta * self._place_value_of(base_rod_index)

    def _place_value_of(self, base_rod_index: int) -> int:
        for rd in self.rod_deltas:
            if rd.rod_index == base_rod_index:
                return rd.place_value
        return self.base ** base_rod_index   # 基准档未变化时才需按进制回退推算

    @classmethod
    def diff(cls, before: AbacusState, after: AbacusState) -> "StateDelta":
        rds = []
        for rb, ra in zip(before.rod_states, after.rod_states):
            changed = tuple(
                BeadStateDelta(bb.bead_id, bb.bead_type, bb.position, ba.position)
                for bb, ba in zip(rb.upper_beads + rb.lower_beads,
                                  ra.upper_beads + ra.lower_beads)
                if bb.position is not ba.position)
            if changed:
                rds.append(RodStateDelta(rb.rod_index, rb.place_value, changed))
        return cls(before.step_index, after.step_index, tuple(rds), after.base)
