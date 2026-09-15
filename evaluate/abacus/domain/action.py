# -*- coding: utf-8 -*-
"""动作层 + 口诀数据类：命令值对象 与 声明式规范。

拆分自原 domain_model.py（2026-09-07）。依赖 enums（词汇表）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .enums import (
    ActionType, BeadAddressing, BeadPosition, BeadType, Finger, RhymeCategory,
)


# ═══════════════════ 5. 动作层（命令值对象：模板相对寻址） ═══════════════════

@dataclass(frozen=True)
class AtomicAction:
    """原子拨珠动作。模板用相对寻址（rod_offset + 寻址模式），可作用于任意档/任意本体；
    执行时由 TransitionEngine 绑定具体 bead_id（弱引用，可序列化/可token化）"""
    action_type: ActionType
    finger: Finger
    bead_type: BeadType = BeadType.LOWER
    ordinal: int = 0                               # 仅 FIXED 模式使用
    rod_offset: int = 0                            # 0=本档 / +1=进位档 / -1=退位档
    addressing: BeadAddressing = BeadAddressing.FIXED
    duration_ms: int = 120                         # 物理时间唯一存放处
    target_bead_id: int | None = None              # 执行期由引擎回填

    @property
    def before_position(self) -> BeadPosition:
        return self.action_type.before

    @property
    def after_position(self) -> BeadPosition:
        return self.action_type.after

    def bound(self, bead_id: int) -> "AtomicAction":
        return replace(self, target_bead_id=bead_id)

@dataclass(frozen=True)
class CompositeAction:
    """复合动作（联拨）：一组同时发生的原子动作，物理上是一次协调运动。
    模板用 rod_offset 相对寻址、target_bead_id 留空（弱引用），
    执行前由 bound() 按序绑定具体珠子，得到可执行的副本（原模板不变）。

    以 1+9 的"九去一"联拨为例：本档食指拨去下珠 1 颗，同时拇指在进位档推入 1 颗。

    >>> up = AtomicAction(ActionType.RELEASE, Finger.INDEX, BeadType.LOWER,
    ...                   addressing=BeadAddressing.LAST_ACTIVE,
    ...                   rod_offset=0)                       # 本档"去1"（离梁）
    >>> carry = AtomicAction(ActionType.ENGAGE, Finger.THUMB, BeadType.LOWER,
    ...                      ordinal=0, rod_offset=+1,
    ...                      duration_ms=150)                  # 进位档"进1"（靠梁）
    >>> ca = CompositeAction("九去一", "进十", (up, carry))
    >>> ca.duration_ms              # 联拨=两动同时发生，总时长取最慢的那只手
    150
    >>> [a.rod_offset for a in ca.atomic_actions]   # 相对寻址：0=本档 / +1=进位档
    [0, 1]
    >>> [a.target_bead_id for a in ca.atomic_actions]   # 模板本身不绑定珠子
    [None, None]
    >>> bound = ca.bound([41, 42])  # 执行期按序回填绝对 bead_id
    >>> [a.target_bead_id for a in bound.atomic_actions]
    [41, 42]
    >>> [a.target_bead_id for a in ca.atomic_actions]   # 原模板是 frozen，不受影响
    [None, None]
    """
    name: str
    semantic_tag: str                              # 直加/满五/进位/退位/破五…
    atomic_actions: tuple[AtomicAction, ...]

    @property
    def duration_ms(self) -> int:
        return max(a.duration_ms for a in self.atomic_actions)   # 联拨=同时发生

    def bound(self, bead_ids: Sequence[int]) -> "CompositeAction":
        return replace(self, atomic_actions=tuple(
            a.bound(b) for a, b in zip(self.atomic_actions, bead_ids)))


# ═══════════════════ 6. 口诀层（声明式规范：三模态内建对齐） ═══════════════════

@dataclass(frozen=True)
class ExpectedRodDelta:
    """期望差分模板项：rod_offset 相对基准档；value_delta 为该档本位单位（进制无关）"""
    rod_offset: int
    value_delta: int

@dataclass(frozen=True)
class Carry:
    """跨档传播描述：进位(+1) / 退位(−1) 后，对邻档递归同向的 ±digit"""
    rod_offset: int
    digit: int

@dataclass(frozen=True)
class Rhyme:
    """口诀本体：rhyme_text（口诀）+ expected_delta（数值）+ action_mapping（指法）"""
    rhyme_id: int
    category: RhymeCategory
    trigger_digit: int                             # 加减=加/减数；归除=除数（归数）
    rhyme_text: str                                 # "一下五去四"
    modern_text: str                               # 白话释义
    expected_delta: tuple[ExpectedRodDelta, ...]
    action_mapping: tuple[CompositeAction, ...]    # 当前种子均为单一联拨（1:1）
