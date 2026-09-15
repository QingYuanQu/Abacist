"""机器观测渲染：固定分辨率灰度 patch（render_fixed）。

给模型吃的盘面图（VLA 训练数据 / 闭环推理的观测通道），纯 numpy 实现，
不依赖 matplotlib——VLA 数据管线无需彩色渲染栈即可运行。
消费方：model_vlm/closed_loop、model_vlm/parse_eval_bridge。

珠位几何与 image.py 同源（geometry.bead_centers），保证彩色图与灰度 patch
中的珠子位置完全一致。

示例入口：python -m evaluate.abacus.demos fixed → 输出逐步 patch 到 abacus/output/fixed/。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evaluate.abacus.render.geometry import bead_centers, layout_of

if TYPE_CHECKING:
    from evaluate.abacus.domain import Abacus, AbacusState


def render_fixed(state: AbacusState, abacus: Abacus,
                 patch_h: int = 80, patch_w: int = 28) -> np.ndarray:
    """渲染二值/灰度固定分辨率图像，可直接切分为 n_cols 个视觉 patch。

    复用 geometry.bead_centers() 的真实物理坐标，珠子会随数值上下移动；
    旧版使用代数恒等式计算 active_y/rest_y，导致所有下珠位置相同。

    输出 shape: (patch_h, n_cols * patch_w)，像素值范围 [0, 1]。
    """
    spec = abacus.spec
    n_upper, n_lower, n_cols = spec.upper_count, spec.lower_count, spec.rod_count
    H, W = patch_h, n_cols * patch_w
    img = np.zeros((H, W), dtype=np.float32)

    bead_r = max(2, min(patch_h, patch_w) // 6)
    centers = bead_centers(state, abacus)

    # 与 image.ImageRenderer.render() 一致的外框/梁物理坐标（单一 Layout 来源）
    layout = layout_of(abacus)
    frame_top = layout.frame_top
    frame_bottom = layout.frame_bottom
    beam_y = layout.beam_y

    # y 物理坐标 → 像素坐标（顶部小、底部大）
    y_top_px = bead_r + 1
    y_bottom_px = H - bead_r - 2
    phys_h = frame_top - frame_bottom

    def _phys_to_px(y_phys: float) -> int:
        return int(round(
            y_top_px + (frame_top - y_phys) / phys_h * (y_bottom_px - y_top_px)
        ))

    # 横梁（粗 3 像素）
    beam_row = _phys_to_px(beam_y)
    img[max(0, beam_row - 1):min(H, beam_row + 2), :] = 1.0

    # 竖杆
    for col in range(n_cols):
        x_center = col * patch_w + patch_w // 2
        img[:, x_center] = 1.0

    # 全图坐标网格，用于向量化画圆
    yy, xx = np.indices((H, W))

    for col, rod_state in enumerate(state.rods_left_to_right):
        x_center = col * patch_w + patch_w // 2

        # 上珠
        for b in rod_state.upper_beads:
            _, cy_phys = centers[b.bead_id]
            cy = _phys_to_px(cy_phys)
            value = 1.0 if b.position.is_active else 0.5
            mask = ((xx - x_center) ** 2 + (yy - cy) ** 2) <= bead_r * bead_r
            img[mask] = np.maximum(img[mask], value)

        # 下珠
        for b in rod_state.lower_beads:
            _, cy_phys = centers[b.bead_id]
            cy = _phys_to_px(cy_phys)
            value = 1.0 if b.position.is_active else 0.5
            mask = ((xx - x_center) ** 2 + (yy - cy) ** 2) <= bead_r * bead_r
            img[mask] = np.maximum(img[mask], value)

    return img


# ═══════════════════ 示例：逐步灰度 patch 渲染（由薄壳 demo_fixed.py 调用） ═══════════════════

def demo(output_dir=None) -> list:
    """渲染 47+38 的逐步灰度 patch 到 output_dir（默认 abacus/output/fixed/）。

    含空盘初始帧 + 每步终态，每步一个 PNG——与 text 示例的步进对齐，
    便于人工逐帧核对两种输出形态的珠位是否一致。
    返回写出的文件路径列表。
    """
    from pathlib import Path

    from PIL import Image

    from evaluate.abacus import (Abacus, ArithmeticComposer, RhymeResolver, OpType,
                        build_addition_rhymes, build_subtraction_rhymes)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output" / "fixed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ab = Abacus.standard()
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    op = composer.compose(OpType.ADD, ab, 47, 38)

    states = [op.steps[0].before_state] + [s.after_state for s in op.steps]
    paths = []
    for j, st in enumerate(states):
        arr = render_fixed(st, ab)
        p = output_dir / f"add_47_38_step{j:02d}.png"
        Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(p)
        paths.append(p)
    return paths
