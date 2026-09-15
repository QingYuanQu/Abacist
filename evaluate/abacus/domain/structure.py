# -*- coding: utf-8 -*-
"""结构层：算盘的物理实体（一次性构建，禁改）。

拆分自原 domain_model.py（2026-09-07）。依赖 enums（词汇表）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .enums import BeadPosition, BeadType, InvariantViolation


# ═══════════════════ 2. 结构层（物理真理：一次性构建，禁改） ═══════════════════

@dataclass(frozen=True)
class AbacusSpec:
    """算盘形态规格：上/下珠数显式化，同一内核可跑二五珠/一四珠"""
    upper_count: int = 1
    lower_count: int = 4
    rod_count: int = 13
    base: int = 10

    def __post_init__(self):
        if min(self.upper_count, self.lower_count, self.rod_count) < 1:
            raise ValueError("上/下珠数与档数必须 ≥ 1")

@dataclass(frozen=True)
class Beam:
    """横梁（全盘唯一；自 Rod 移入，消除 N 档重复持有）"""
    y_position_mm: float = 0.0

@dataclass(frozen=True)
class Frame:
    """外框物理尺寸：持有 Beam"""
    width_mm: float
    height_mm: float
    rod_count: int
    beam: Beam = Beam()

@dataclass(frozen=True)
class Bead:
    """单珠物理定义"""
    bead_id: int
    bead_type: BeadType
    ordinal: int                                   # 同类珠内序号，0 最靠梁
    default_position: BeadPosition = BeadPosition.RESTING

    @property
    def value(self) -> int:
        return self.bead_type.value

@dataclass(frozen=True)
class Rod:
    """档（位值载体）：聚合 Bead；y 坐标实时投影，不持有 Beam"""
    index: int                                     # 0 = 个位
    place_value: int                               # base ** index（构建时物化）
    upper_beads: tuple[Bead, ...]
    lower_beads: tuple[Bead, ...]

    def bead(self, bead_type: BeadType, ordinal: int) -> Bead:
        pool = self.upper_beads if bead_type is BeadType.UPPER else self.lower_beads
        if not 0 <= ordinal < len(pool):
            raise InvariantViolation(f"档{self.index}不存在 {bead_type.name}#{ordinal}")
        return pool[ordinal]

@dataclass(frozen=True)
class Abacus:
    """聚合根：物理结构总集（与运行状态解耦，可被全部 episode 共享）"""
    spec: AbacusSpec
    frame: Frame
    rods: tuple[Rod, ...]
    _bead_index: dict[int, Bead] = field(init=False, repr=False, compare=False,
                                         default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "_bead_index",
                           {b.bead_id: r for rod in self.rods
                            for b in rod.upper_beads + rod.lower_beads
                            for r in [b]})

    def rod_at(self, index: int) -> Rod:
        if not 0 <= index < len(self.rods):
            raise InvariantViolation(f"档越界: {index}")
        return self.rods[index]

    def get_bead_by_id(self, bead_id: int) -> Bead:
        return self._bead_index[bead_id]           # KeyError = 结构错误，直接炸

    @classmethod
    def standard(cls, spec: AbacusSpec | None = None, pitch_mm: float = 10.0) -> "Abacus":
        """标准构建：bead_id = 档×(上+下) + 序（上珠在前）"""
        spec = spec or AbacusSpec()
        frame = Frame(
            width_mm=(spec.rod_count + 2) * pitch_mm,
            height_mm=(spec.upper_count + spec.lower_count + 1) * pitch_mm,
            rod_count=spec.rod_count,
            beam=Beam(y_position_mm=spec.lower_count * pitch_mm),
        )
        per = spec.upper_count + spec.lower_count
        rods = tuple(
            Rod(i, spec.base ** i,
                tuple(Bead(i * per + k, BeadType.UPPER, k) for k in range(spec.upper_count)),
                tuple(Bead(i * per + spec.upper_count + k, BeadType.LOWER, k)
                      for k in range(spec.lower_count)))
            for i in range(spec.rod_count))
        return cls(spec, frame, rods)
