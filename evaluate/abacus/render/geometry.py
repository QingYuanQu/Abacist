"""共享渲染几何：算盘布局常量 + 珠心数据坐标计算。

image.py（matplotlib 彩色渲染）与 fixed.py（numpy 灰度 patch）共用——
保证两种输出形态的珠位完全同源，且几何布局（梁/外框）只有一个权威来源
（此前 frame_top / frame_bottom / beam_y 在 geometry / fixed / image 三处散落，
且 image 里还混着外扩边距的派生值）。

纯域依赖，零第三方依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluate.abacus.domain import Abacus, AbacusState


@dataclass(frozen=True)
class Layout:
    """算盘几何布局（归一化数据坐标，y 轴向上）。

    数值与旧实现逐像素一致：

    - beam_y        横梁中心线 y（= n_lower + 1）
    - frame_bottom  外框底边内沿 y（= 0.4）
    - frame_top     外框顶边内沿 y（= n_upper + n_lower + 1.6）
    - frame_left    外框左缘 x（= 0.3）
    """
    beam_y: float
    frame_bottom: float
    frame_top: float
    frame_left: float


def layout_of(abacus: Abacus) -> Layout:
    """由算盘形制推导几何布局。"""
    n_upper, n_lower = abacus.spec.upper_count, abacus.spec.lower_count
    return Layout(
        beam_y=n_lower + 1,
        frame_bottom=0.4,
        frame_top=n_upper + n_lower + 1.6,
        frame_left=0.3,
    )


def bead_centers(state: AbacusState, abacus: Abacus) -> dict[int, tuple[float, float]]:
    """返回 {bead_id: (x, y)} 数据坐标：靠梁珠贴梁、离梁珠贴框（真实位移）。

    供彩色渲染、灰度 patch 与过渡帧生成共用，保证手与珠位置精确对齐。
    """
    spec = abacus.spec
    n_upper, n_lower, n_cols = spec.upper_count, spec.lower_count, spec.rod_count
    layout = layout_of(abacus)
    beam_y, frame_top, frame_bottom = layout.beam_y, layout.frame_top, layout.frame_bottom
    centers: dict[int, tuple[float, float]] = {}
    rods_ltr = state.rods_left_to_right
    for col, rod_state in enumerate(rods_ltr):
        x = col + 1
        # 上珠（梁上方）：靠梁=向下贴梁，离梁=向上贴顶框
        for k in range(n_upper):
            on = rod_state.upper_beads[k].position.is_active
            cy = beam_y + 0.55 + k * 0.9 if on else frame_top - 0.55 - k * 0.9
            centers[rod_state.upper_beads[k].bead_id] = (x, cy)
        # 下珠（梁下方）：靠梁=向上贴梁，离梁=向下贴底框（k 越大越靠底）
        for k in range(n_lower):
            on = rod_state.lower_beads[k].position.is_active
            cy = beam_y - 0.55 - k * 0.9 if on else frame_bottom + 0.55 + (n_lower - 1 - k) * 0.9
            centers[rod_state.lower_beads[k].bead_id] = (x, cy)
    return centers
