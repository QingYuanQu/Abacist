"""TextRenderer：零依赖文本渲染器（ASCII 盘面，调试/日志用）。

输出 ASCII 文本盘面（●/·/━），可在无 matplotlib 的环境中独立用于调试/日志。
注：训练数据/机器观测请走 render.registry（make_render），本类仅面向人读。

示例入口：python -m evaluate.abacus.demos text → 输出逐步 txt 到 abacus/output/text/。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluate.abacus.domain import Abacus, AbacusState


class TextRenderer:
    """零依赖文本渲染器（端口默认实现）"""
    def render(self, state: AbacusState, abacus: Abacus, view: str = "front") -> str:
        rods = state.rods_left_to_right                  # 高位在左，个位在最右
        n = len(rods)
        lines = []
        for r in range(abacus.spec.upper_count):
            lines.append("".join("●" if rs.upper_beads[r].position.is_active else "·"
                                 for rs in rods))
        lines.append("━" * n)                            # 梁
        for k in range(abacus.spec.lower_count):         # 下珠：前缀激活
            lines.append("".join("●" if rs.lower_beads[k].position.is_active else "·"
                                 for rs in rods))
        lines.append("".join(str(i % 10) for i in range(n - 1, -1, -1)))
        return "\n".join(lines)

    def describe(self, state: AbacusState) -> dict:
        return {"step": state.step_index,
                "rods": {rs.rod_index: rs.rod_value for rs in state.rod_states},
                "total": state.total_value}


# ═══════════════════ 示例：逐步 ASCII 帧渲染（由薄壳 demo_text.py 调用） ═══════════════════

def demo(output_dir=None) -> list:
    """渲染 47+38 的逐步 ASCII 帧到 output_dir（默认 abacus/output/text/）。

    含空盘初始帧 + 每步终态，每步一个 txt——对齐旧 frames/ 的逐步输出行为。
    返回写出的文件路径列表。
    """
    from pathlib import Path

    from evaluate.abacus import (Abacus, ArithmeticComposer, RhymeResolver, OpType,
                        build_addition_rhymes, build_subtraction_rhymes)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output" / "text"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ab = Abacus.standard()
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    op = composer.compose(OpType.ADD, ab, 47, 38)

    t = TextRenderer()
    states = [op.steps[0].before_state] + [s.after_state for s in op.steps]
    paths = [output_dir / f"add_47_38_step{j:02d}.txt"
             for j in range(len(states))]
    for p, st in zip(paths, states):
        p.write_text(t.render(st, ab), encoding="utf-8")
    return paths
