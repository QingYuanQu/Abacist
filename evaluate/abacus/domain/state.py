# -*- coding: utf-8 -*-
"""状态层：不可变快照（append-only）。

差分层已拆为独立模块 delta.py（2026-09-07 物理分离）；数值 → 状态的正向构造器
num_to_state 落在状态层（它产出 AbacusState）。

拆分自原 domain_model.py（2026-09-07）。依赖 enums（词汇表）与 structure（物理实体）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .enums import BeadPosition, BeadType, InvariantViolation
from .structure import Abacus, AbacusSpec


# ═══════════════════ 3. 状态层（不可变快照，append-only） ═══════════════════

@dataclass(frozen=True)
class BeadState:
    """单珠语义快照（渲染投影不入语义状态）"""
    bead_id: int
    bead_type: BeadType     # 珠子类型（上/下）
    ordinal: int            # 珠子的「身份/编号」，造出来就固定了。下珠第 0、1、2、3 颗（0 最靠梁）
    position: BeadPosition  # 珠子的「当前状态」，随拨珠动作变化。 随拨珠动作变化。

@dataclass(frozen=True)
class RodState:
    """单档快照"""
    rod_index: int
    place_value: int
    upper_beads: tuple[BeadState, ...]
    lower_beads: tuple[BeadState, ...]

    @property
    def rod_value(self) -> int:
        return sum(b.bead_type.value for b in self.upper_beads + self.lower_beads
                   if b.position.is_active)

    @property
    def is_empty(self) -> bool:
        return self.rod_value == 0

    @property
    def is_full(self) -> bool:
        return self.rod_value == 5 * len(self.upper_beads) + len(self.lower_beads)

    def bead_position(self, bead_id: int) -> BeadPosition:
        for b in self.upper_beads + self.lower_beads:
            if b.bead_id == bead_id:
                return b.position
        raise KeyError(bead_id)

    def with_bead(self, bead_id: int, position: BeadPosition) -> "RodState":
        def swap(bs: BeadState) -> BeadState:
            return replace(bs, position=position) if bs.bead_id == bead_id else bs
        new = RodState(self.rod_index, self.place_value,
                       tuple(swap(b) for b in self.upper_beads),
                       tuple(swap(b) for b in self.lower_beads))
        new.check_physical()
        return new

    def check_physical(self) -> None:
        """守护不变式：下珠激活必须是前缀（0..k-1），否则物理不可能（珠会相撞）"""
        actives = [b.ordinal for b in self.lower_beads if b.position.is_active]
        if actives != list(range(len(actives))):
            raise InvariantViolation(f"档{self.rod_index}下珠激活非前缀: {actives}")

@dataclass(frozen=True)
class AbacusState:
    """整盘快照：step_index 为逻辑步号（物理时间只存于动作 duration_ms）。

    持有 spec（形制 / 进制）：状态必须**知道**自己属于哪种算盘，而不是事后从
    档值序列反推——反推在单档时无解，只能靠哨兵值兜底，那是设计缺陷不是设计。
    """
    step_index: int
    rod_states: tuple[RodState, ...]
    spec: AbacusSpec

    @property
    def rods_left_to_right(self) -> tuple[RodState, ...]:
        """按显示顺序（高位在左 → 个位最右）返回档；统一渲染方向语义，避免各渲染器各自翻转。"""
        return tuple(reversed(self.rod_states))

    @property
    def total_value(self) -> int:
        return sum(rs.rod_value * rs.place_value for rs in self.rod_states)

    @property
    def base(self) -> int:
        """进制：直接取自形制（AbacusSpec.base）。"""
        return self.spec.base

    def rod_state_at(self, rod_index: int) -> RodState:
        if not 0 <= rod_index < len(self.rod_states):
            raise InvariantViolation(f"档越界: {rod_index}")
        return self.rod_states[rod_index]

    def bead_position(self, bead_id: int) -> BeadPosition:
        for rs in self.rod_states:
            for b in rs.upper_beads + rs.lower_beads:
                if b.bead_id == bead_id:
                    return b.position
        raise KeyError(bead_id)

    def with_bead(self, bead_id: int, position: BeadPosition) -> "AbacusState":
        for i, rs in enumerate(self.rod_states):
            if any(b.bead_id == bead_id for b in rs.upper_beads + rs.lower_beads):
                return replace(self, rod_states=self.rod_states[:i]
                               + (rs.with_bead(bead_id, position),) + self.rod_states[i + 1:])
        raise KeyError(bead_id)

    @classmethod
    def empty(cls, abacus: Abacus) -> "AbacusState":
        """清盘态：全部珠位于 default_position。

        原位于 TransitionEngine.initial_state（行为内核），实为"状态层的空状态"，
        2026-09-07 下沉至此——清盘态构造是状态层职责，不该由行为内核承担。
        谁想构造"数值 → 状态"，谁就不再需要依赖 kernel。
        """
        rods = []
        for r in abacus.rods:
            up = tuple(BeadState(b.bead_id, b.bead_type, b.ordinal, b.default_position)
                       for b in r.upper_beads)
            low = tuple(BeadState(b.bead_id, b.bead_type, b.ordinal, b.default_position)
                        for b in r.lower_beads)
            rods.append(RodState(r.index, r.place_value, up, low))
        return cls(0, tuple(rods), abacus.spec)


# ═══════════════════ 数值 → 状态（正向构造器，唯一实现） ═══════════════════

def num_to_state(num: int, abacus: Abacus) -> AbacusState:
    """数值 → 珠态快照：从清盘态起逐档落珠，返回新 state。

    "数值 → 状态"的**唯一**实现（原 bead_codec.num_to_state 迁入，2026-09-07）。
    AbacusEnv._state_of_value 与 RhymeResolver._mock_rod_state 均已删除、改走本函数。

    负数抛 ValueError（AbacusState 无符号位）；
    超出 base ** rod_count 容量抛 ValueError。
    """
    if num < 0:
        raise ValueError(f"AbacusState 无符号位，不支持负数: {num}")
    spec = abacus.spec
    if num >= spec.base ** spec.rod_count:
        raise ValueError(f"{num} 超出 {spec.rod_count} 档 base={spec.base} 容量")
    upper_unit = BeadType.UPPER.value  # 每颗上珠面值（5）
    state = AbacusState.empty(abacus)
    remaining = num
    for rod in abacus.rods:            # rod.index 0 = 个位
        d = remaining % spec.base
        remaining //= spec.base
        if d == 0:
            continue
        upper = min(d // upper_unit, spec.upper_count)
        lower = d - upper_unit * upper
        for k in range(upper):
            state = state.with_bead(rod.bead(BeadType.UPPER, k).bead_id,
                                    BeadPosition.ENGAGED)
        for k in range(lower):
            state = state.with_bead(rod.bead(BeadType.LOWER, k).bead_id,
                                    BeadPosition.ENGAGED)
    return state
