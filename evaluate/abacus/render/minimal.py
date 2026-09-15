"""极简机器观测渲染：一珠一像素的二值盘面（render_minimal）。

信息论下限的 Vision 表征：一档 = (n_upper + n_lower) 行 × patch_w 列的二值
像素（标准算盘 5×1），一行对应一颗珠，1=靠梁（激活）、0=离梁（静止）。
row 0 为上珠（读出权重 5），row 1..n_lower 为下珠（权重 1，激活必为前缀，
由 RodState.check_physical 守护，故 5bit→10 态自动合法）。

与 render_fixed 的关系：
- render_fixed（80×28）：含梁/杆/珠距/留白，视觉通路需去冗余；
- render_minimal（5×1）：纯珠态位图，无任何结构冗余——模型必须自己
  从行位置学出「上珠×5 + 下珠×1」的读出，这是「极简多模态」实验的
  最小非平凡视觉任务（详见 doc/极简视觉算盘表征.md）。

两者珠位语义同源（都读 BeadState.position.is_active），标签管线
（口诀@档位#）完全复用。patch_w>1 时每档复制该位（预留抖动/鲁棒性实验）。

消费方：VLA 数据管线（作为 render_fixed 的消融对照）。
示例入口：python -m evaluate.abacus.demos minimal → 输出逐步 PNG。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evaluate.abacus.domain import Abacus, AbacusState


def render_minimal(state: AbacusState, abacus: Abacus,
                   patch_w: int = 1) -> np.ndarray:
    """渲染一珠一像素的二值盘面。

    输出 shape: (n_upper + n_lower, n_cols * patch_w)，像素值 {0.0, 1.0}。
    行序：row 0 = 上珠（×5），row 1..n_lower = 下珠（×1，ordinal 升序）；
    列序：与 render_fixed 一致，高位在左、个位在右（rods_left_to_right）。
    """
    if patch_w < 1:
        raise ValueError(f"patch_w 须 ≥1，得到 {patch_w}")

    spec = abacus.spec
    n_upper, n_lower, n_cols = spec.upper_count, spec.lower_count, spec.rod_count
    H, W = n_upper + n_lower, n_cols * patch_w
    img = np.zeros((H, W), dtype=np.float32)

    for col, rod_state in enumerate(state.rods_left_to_right):
        bits = (rod_state.upper_beads + rod_state.lower_beads)
        if len(bits) != H:
            raise ValueError(
                f"档{col}珠数 {len(bits)} 与 spec 行数 {H} 不符")
        x0 = col * patch_w
        for row, bead in enumerate(bits):
            if bead.position.is_active:
                img[row, x0:x0 + patch_w] = 1.0

    return img


# ═══════════════════ 示例：逐步极简盘面渲染（由薄壳 demo_minimal.py 调用） ═══════════════════

def demo(output_dir=None, patch_w: int = 1) -> list:
    """渲染 47+38 的逐步极简盘面到 output_dir（默认 abacus/output/minimal/）。

    与 demo_fixed 的步进完全对齐（空盘初始帧 + 每步终态），
    便于逐帧核对两种分辨率的珠态是否一致。
    patch_w>1 时最近邻放大输出 PNG（矩阵本身仍逐档重复位）。
    返回写出的文件路径列表。
    """
    from pathlib import Path

    from PIL import Image

    from evaluate.abacus import (Abacus, ArithmeticComposer, RhymeResolver, OpType,
                        build_addition_rhymes, build_subtraction_rhymes)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output" / "minimal"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ab = Abacus.standard()
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    op = composer.compose(OpType.ADD, ab, 47, 38)

    states = [op.steps[0].before_state] + [s.after_state for s in op.steps]
    paths = []
    scale = max(1, 20 // patch_w)  # PNG 可视化放大倍数（位图尺寸过小无法肉眼检查）
    for j, st in enumerate(states):
        arr = render_minimal(st, ab, patch_w=patch_w)
        up = np.kron(arr, np.ones((scale, scale), dtype=np.uint8)) * 255
        p = output_dir / f"add_47_38_step{j:02d}.png"
        Image.fromarray(up.astype(np.uint8), mode="L").save(p)
        paths.append(p)
        print(f"step{j:02d} value={st.total_value:>4} bits=\n"
              + "\n".join("  " + "".join(str(v) for v in row)
                          for row in arr.astype(int).T))
    return paths
