# -*- coding: utf-8 -*-
"""rhymes.py —— 口诀种子数据（领域知识，纯声明式构建器）。

（加法 26 句 / 减法 26 句）与口诀层中文数字词表 cn_number。

约定：
  - 口诀只描述**本档**拨珠；进位 / 退位的跨档 ±1 由 ArithmeticComposer 递归执行，
    不写进口诀（避免口诀与编排器两处记账）。
  - 数值层保持阿拉伯数字，口诀层用中文数字，两套词表并存不合并
    （见 doc/parse-eval对齐方案.md D4）。

本模块只依赖 domain 子包的类型，不含任何编排逻辑。
"""
from __future__ import annotations

from evaluate.abacus.domain import (
    ActionType, AtomicAction, BeadAddressing, BeadType, CompositeAction,
    ExpectedRodDelta, Finger, Rhyme, RhymeCategory,
)



# ═══════════════════ 种子数据：26 句加法口诀  ═══════════════════

CN_DIGITS = "零一二三四五六七八九"   # 公共词表：阿拉伯→中文数字（口诀层；数值层保持阿拉伯）

def cn_number(n: int) -> str:
    """阿拉伯数字(0..9) → 中文数字。口诀词表专用（见 doc/parse-eval对齐方案.md D4：
    数值层阿拉伯 / 口诀层中文，两套词表并存不合并）。

    >>> cn_number(9)
    '九'
    >>> cn_number(0) + cn_number(5)
    '零五'
    """
    if not 0 <= n <= 9:
        raise ValueError(f"口诀数字需 0..9，收到 {n}")
    return CN_DIGITS[n]

def build_addition_rhymes(start_id: int = 1) -> list[Rhyme]:
    UP, LOW = BeadType.UPPER, BeadType.LOWER
    fs: list[Rhyme] = []
    fid = start_id

    def add(category, d, oral, modern, expected, atoms, tag):
        nonlocal fid
        fs.append(Rhyme(fid, category, d, oral, modern, expected,
                          (CompositeAction(oral, tag, tuple(atoms)),)))
        fid += 1

    # —— 直加 9 句：d 上 d ——
    for d in range(1, 10):
        atoms = []
        if d >= 5:
            atoms.append(AtomicAction(ActionType.ENGAGE, Finger.MIDDLE, UP, 0))
        atoms += [AtomicAction(ActionType.ENGAGE, Finger.THUMB, LOW,
                               addressing=BeadAddressing.NEXT_INACTIVE)
                  for _ in range(d % 5)]
        add(RhymeCategory.ADD_DIRECT, d, f"{cn_number(d)}上{cn_number(d)}",
            f"盘面可直加{d}：直接拨入价值{d}的珠",
            (ExpectedRodDelta(0, d),), atoms, "直加")
    # —— 满五加 4 句：d 下五去(5-d) ——
    for d in range(1, 5):
        atoms = [AtomicAction(ActionType.ENGAGE, Finger.MIDDLE, UP, 0)]
        atoms += [AtomicAction(ActionType.RELEASE, Finger.INDEX, LOW,
                               addressing=BeadAddressing.LAST_ACTIVE)
                  for _ in range(5 - d)]
        add(RhymeCategory.ADD_FILL5, d, f"{cn_number(d)}下五去{cn_number(5 - d)}",
            f"盘面≤4 且盘面+{d}≥5：下五(+5)去{5 - d}(−{5 - d})，净+{d}",
            (ExpectedRodDelta(0, d),), atoms, "满五")
    # —— 进十加 9 句：d 去(10-d)（进位 +1 由编排器递归执行，不再写进口诀） ——
    for d in range(1, 10):
        atoms = []
        if 10 - d >= 5:
            atoms.append(AtomicAction(ActionType.RELEASE, Finger.MIDDLE, UP, 0))
        atoms += [AtomicAction(ActionType.RELEASE, Finger.INDEX, LOW,
                               addressing=BeadAddressing.LAST_ACTIVE)
                  for _ in range((10 - d) % 5)]
        add(RhymeCategory.ADD_CARRY10, d, f"{cn_number(d)}去{cn_number(10 - d)}",
            f"盘面+{d}≥10 且本档可直去{10 - d}：去{10 - d}（进位+1 由编排器递归）",
            (ExpectedRodDelta(0, d - 10),), atoms, "进十")
    # —— 破五进十加 4 句：d 上(d-5) 去五（进位 +1 由编排器递归） ——
    for d in range(6, 10):
        atoms = [AtomicAction(ActionType.ENGAGE, Finger.THUMB, LOW,
                              addressing=BeadAddressing.NEXT_INACTIVE)
                 for _ in range(d - 5)]
        atoms.append(AtomicAction(ActionType.RELEASE, Finger.MIDDLE, UP, 0))
        add(RhymeCategory.ADD_BREAK5_CARRY10, d, f"{cn_number(d)}上{cn_number(d - 5)}去五",
            f"盘面≥5、+{d}≥10 且下珠不足以直去{10 - d}：上{d - 5}去五（进位+1 由编排器递归）",
            (ExpectedRodDelta(0, d - 10),), atoms, "破五进十")
    return fs

def build_subtraction_rhymes(start_id: int = 100) -> list[Rhyme]:
    """26 句减法口诀（退位 −1 由编排器递归执行，口诀只描述本档拨珠）。"""
    UP, LOW = BeadType.UPPER, BeadType.LOWER
    fs: list[Rhyme] = []
    fid = start_id

    def add(category, d, oral, modern, expected, atoms, tag):
        nonlocal fid
        fs.append(Rhyme(fid, category, d, oral, modern, expected,
                          (CompositeAction(oral, tag, tuple(atoms)),)))
        fid += 1

    # —— 直减 9 句：d 去 d ——
    for d in range(1, 10):
        atoms = []
        if d >= 5:
            atoms.append(AtomicAction(ActionType.RELEASE, Finger.MIDDLE, UP, 0))
        atoms += [AtomicAction(ActionType.RELEASE, Finger.INDEX, LOW,
                               addressing=BeadAddressing.LAST_ACTIVE)
                  for _ in range(d % 5)]
        add(RhymeCategory.SUB_DIRECT, d, f"{cn_number(d)}去{cn_number(d)}",
            f"盘面可直减{d}：直接拨去价值{d}的珠",
            (ExpectedRodDelta(0, -d),), atoms, "直减")
    # —— 破五减 4 句：d 上(5-d) 去五 ——
    for d in range(1, 5):
        atoms = [AtomicAction(ActionType.ENGAGE, Finger.THUMB, LOW,
                              addressing=BeadAddressing.NEXT_INACTIVE)
                 for _ in range(5 - d)]
        atoms.append(AtomicAction(ActionType.RELEASE, Finger.MIDDLE, UP, 0))
        add(RhymeCategory.SUB_BREAK5, d, f"{cn_number(d)}上{cn_number(5 - d)}去五",
            f"盘面≥5 且下珠不足以直减{d}：上{5 - d}(+{5 - d})去五(−5)，净−{d}",
            (ExpectedRodDelta(0, -d),), atoms, "破五")
    # —— 退十减 9 句：d 退一还(10-d)（退一 −1 由编排器递归） ——
    for d in range(1, 10):
        atoms = []
        if 10 - d >= 5:
            atoms.append(AtomicAction(ActionType.ENGAGE, Finger.MIDDLE, UP, 0))
        atoms += [AtomicAction(ActionType.ENGAGE, Finger.THUMB, LOW,
                               addressing=BeadAddressing.NEXT_INACTIVE)
                  for _ in range((10 - d) % 5)]
        add(RhymeCategory.SUB_BORROW10, d, f"{cn_number(d)}退一还{cn_number(10 - d)}",
            f"盘面<{d}：退一(−10，由编排器递归)+还{10 - d}，本档净+{10 - d}",
            (ExpectedRodDelta(0, 10 - d),), atoms, "退十")
    # —— 退十补五减 4 句：d 退一还五去(d-5)（退一 −1 由编排器递归） ——
    for d in range(6, 10):
        atoms = [AtomicAction(ActionType.ENGAGE, Finger.MIDDLE, UP, 0)]
        atoms += [AtomicAction(ActionType.RELEASE, Finger.INDEX, LOW,
                               addressing=BeadAddressing.LAST_ACTIVE)
                  for _ in range(d - 5)]
        add(RhymeCategory.SUB_BORROW10_FILL5, d, f"{cn_number(d)}退一还五去{cn_number(d - 5)}",
            f"盘面<{d}且还入需满五：退一（递归）+还五(+5)去{d - 5}(−{d - 5})，本档净+{10 - d}",
            (ExpectedRodDelta(0, 10 - d),), atoms, "退十补五")
    return fs
