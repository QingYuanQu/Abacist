"""matplotlib 实时窗口动画：复用 image.py 的几何与配色，常驻 ax + FuncAnimation。

与 image.py 的区别：image.py 顶部 `matplotlib.use("Agg")` 适合离线出图（GIF/PNG），
本模块刻意**不**强制 Agg，以保留交互式后端（TkAgg/QtAgg…）用于窗口实时播放。
视觉要素（Style 配色、bead_centers/layout_of 几何、拨珠小手）与 image.py 完全同源。

入口：python -m evaluate.abacus.demos animate
"""
from __future__ import annotations

import matplotlib
# 注意：此处不调用 matplotlib.use("Agg")，以保留交互式后端用于实时播放。
# 若运行环境无 GUI（纯 headless），matplotlib 会回退到 Agg，plt.show() 不弹窗，
# 此时可改用 AbacusAnimator.save() 导出的方式（预留）。
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

from evaluate.abacus.domain import Abacus, AbacusState, StateDelta
from evaluate.abacus.render.style import CJK_FONTS, Style
from evaluate.abacus.render.geometry import bead_centers, layout_of


class AbacusAnimator:
    """常驻轴动画：structure 的框/梁/杆只画一次（静态层），
    每颗珠是一个常驻 Circle artist；动画时只改 Circle 的 center 与 facecolor，
    配合 FuncAnimation 在相邻状态间线性插值拨珠，视觉与 image.py 一致。"""

    def __init__(self, abacus: Abacus, style: Style | None = None, *, scale: float = 1.0):
        self.abacus = abacus
        self.style = style or Style()
        spec = abacus.spec
        self.n_upper, self.n_lower, self.n_cols = (
            spec.upper_count, spec.lower_count, spec.rod_count)
        self.scale = scale
        self._style = self.style

        # 中文字体（候选清单见 style.CJK_FONTS，与 image/vertical 同源）
        plt.rcParams["font.sans-serif"] = list(CJK_FONTS)
        plt.rcParams["axes.unicode_minus"] = False

        layout = layout_of(abacus)
        # 与 image.py 非 tight 模式一致的坐标跨度
        x_left, x_right = -3.2, self.n_cols + 1.6
        y_bottom, y_top = 0, self.n_upper + self.n_lower + 2
        x_span = x_right - x_left
        y_span = y_top - y_bottom
        fig_h = 3.5 * scale
        fig_w = fig_h * (x_span / y_span)
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        self.ax.set_xlim(x_left, x_right)
        self.ax.set_ylim(y_bottom, y_top)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._layout = layout
        self._build_static()
        self._bead_artists: dict[int, Circle] = {}
        self._build_beads()
        self._hand_patches: list = []
        self._build_text()

    # ── 静态结构层（只画一次） ──
    def _build_static(self) -> None:
        s = self._style
        lay = self._layout
        n_cols = self.n_cols
        # 外框
        self.ax.add_patch(Rectangle((lay.frame_left, lay.frame_bottom),
                                    n_cols + 0.4, lay.frame_top - lay.frame_bottom,
                                    facecolor="none", edgecolor=s.frame,
                                    linewidth=2.0, zorder=0))
        # 横梁（厚矩形，zorder 最高压珠）
        self.ax.add_patch(Rectangle((lay.frame_left, lay.beam_y - 0.12),
                                    n_cols + 0.4, 0.24,
                                    facecolor=s.frame, edgecolor="none", zorder=5))
        # 竖杆
        for col in range(n_cols):
            x = col + 1
            self.ax.plot([x, x], [lay.frame_bottom, lay.frame_top],
                         color=s.frame, linewidth=2.5, zorder=1)

    def _build_beads(self) -> None:
        s = self._style
        empty = AbacusState.empty(self.abacus)
        centers = bead_centers(empty, self.abacus)
        for bead_id, (x, y) in centers.items():
            c = Circle((x, y), 0.36, facecolor=s.bead_rest,
                       edgecolor=s.edge, linewidth=1.2, zorder=3, visible=False)
            self.ax.add_patch(c)
            self._bead_artists[bead_id] = c

    def _build_text(self) -> None:
        lay = self._layout
        text_x = -3.0
        # 算式占上 1/3、口诀占下 1/3（与 image.py 同构）
        self._caption_y = lay.frame_bottom + (lay.frame_top - lay.frame_bottom) * (2 / 3)
        self._rhyme_y = lay.frame_bottom + (lay.frame_top - lay.frame_bottom) * (1 / 3)
        self._caption = self.ax.text(text_x, self._caption_y, "",
                                     ha="left", va="center", fontsize=12,
                                     color="#4A2A0A", fontweight="bold", zorder=6)
        self._rhyme = self.ax.text(text_x, self._rhyme_y, "",
                                     ha="left", va="center", fontsize=11,
                                     color="#C0392B", fontweight="bold", zorder=6)

    # ── 动态：把某一快照投影到常驻珠上 ──
    def show_state(self, state: AbacusState, caption: str | None = None,
                   rhyme: str | None = None) -> None:
        s = self._style
        centers = bead_centers(state, self.abacus)
        for bead_id, (x, y) in centers.items():
            c = self._bead_artists[bead_id]
            c.set_center((x, y))
            on = state.bead_position(bead_id).is_active
            c.set_facecolor(s.bead_active if on else s.bead_rest)
            c.set_visible(True)
        if caption is not None:
            self._caption.set_text(caption)
        if rhyme is not None:
            self._rhyme.set_text(rhyme)
        self.fig.canvas.draw_idle()

    # ── 拨珠小手（可选） ──
    def _clear_hand(self) -> None:
        for p in self._hand_patches:
            p.remove()
        self._hand_patches = []

    def _set_hand(self, pos) -> None:
        self._clear_hand()
        if pos is None:
            return
        x, y = pos
        skin, outline = "#F1C27D", "#C07A3A"
        c = Circle((x + 0.55, y), 0.28, facecolor=skin, edgecolor=outline,
                   linewidth=1.0, zorder=7)
        self.ax.add_patch(c)
        self._hand_patches.append(c)
        for dy in (-0.14, 0.0, 0.14):
            ln, = self.ax.plot([x + 0.42, x + 0.18], [y + dy, y + dy],
                               color=skin, linewidth=3.0, solid_capstyle="round", zorder=7)
            self._hand_patches.append(ln)
        ln2, = self.ax.plot([x + 0.72, x + 0.95], [y, y], color=skin,
                            linewidth=4.0, solid_capstyle="round", zorder=7)
        self._hand_patches.append(ln2)

    # ── 帧序列：与 image._step_frames 同构 ──
    def _frames_for_op(self, op, interp: int = 2) -> list:
        """返回帧序列：每帧 = (state, hand_or_None, caption, rhyme)。
        静止 → 手捏起点 → 插值 → 终点 → 松手；多珠同时变化（如"一下五去四"）时手跟随第一颗。"""
        frames = []
        sym = {"ADD": "+", "SUB": "-", "MUL": "×", "DIV": "÷"}[op.op_type.name]
        caption = f"{op.operand_a} {sym} {op.operand_b} = ?"
        prev = op.steps[0].before_state
        frames.append((prev, None, caption, None))
        for step in op.steps:
            after = step.after_state
            rhyme = step.rhyme.rhyme_text
            delta = StateDelta.diff(prev, after)
            moved = [bd for rd in delta.rod_deltas for bd in rd.changed_beads]
            if moved:
                bd = moved[0]
                start = bead_centers(prev, self.abacus)[bd.bead_id]
                end = bead_centers(after, self.abacus)[bd.bead_id]
                frames.append((prev, start, caption, rhyme))
                for t in range(1, interp + 1):
                    f = t / (interp + 1)
                    mx = start[0] + (end[0] - start[0]) * f
                    my = start[1] + (end[1] - start[1]) * f
                    frames.append((prev, (mx, my), caption, rhyme))
                frames.append((after, end, caption, rhyme))
            else:
                frames.append((after, None, caption, rhyme))
            prev = after
        return frames

    def play(self, op, fps: int = 4, interp: int = 2, block: bool = True):
        """播放一次 Operation（四则运算审计链）的拨珠动画。

        fps: 每秒帧数；interp: 每步拨珠的插值中间帧数（越大越顺滑）；
        block=True 时调用 plt.show() 阻塞直到关闭窗口。返回 FuncAnimation 供保存/复用。"""
        frames = self._frames_for_op(op, interp)

        def update(i):
            state, hand, caption, rhyme = frames[i]
            self.show_state(state, caption, rhyme)
            self._set_hand(hand)

        ani = FuncAnimation(self.fig, update, frames=len(frames),
                            interval=1000 / fps, repeat=True)
        if block:
            plt.show()
        return ani


# ═══════════════════ 示例入口（由 demos/__main__.py 的 animate 调用） ═══════════════════

def demo(op_type=None, a=None, b=None, fps: int = 4, interp: int = 2) -> list:
    """交互式动画演示四则运算（默认 47 + 38）。无文件产出，返回空列表。

    命令行：python -m evaluate.abacus.demos animate
    如需指定运算，可在此函数内改 op_type/a/b，或后续扩展为命令行参数。
    """
    from evaluate.abacus import (Abacus, ArithmeticComposer, RhymeResolver, OpType,
                                build_addition_rhymes, build_subtraction_rhymes)

    resolver = RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes())
    composer = ArithmeticComposer(resolver)
    abacus = Abacus.standard()

    if op_type is None:
        op_type, a, b = OpType.ADD, 47, 38
    op = composer.compose(op_type, abacus, a, b)

    animator = AbacusAnimator(abacus)
    animator.play(op, fps=fps, interp=interp, block=True)
    return []
