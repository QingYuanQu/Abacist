"""matplotlib 真实算盘渲染器：单帧 PNG + 逐帧 GIF（给人看的可视化）。

质感对齐原版 datagen/eval/abacus_render.py 的 render_frame：
  - 粗竖杆（2.5px）+ 贯穿全盘
  - 厚横梁（Rectangle）+ zorder 分层压珠
  - 所有珠统一 1.2px 深棕红描边（靠梁=暗红 #C0392B，离梁=暖米 #F5CBA7）
  - 透明背景 + tight_layout（无外框）

机器观测渲染（render_fixed，纯 numpy）已迁至 fixed.py；
共享珠位几何在 geometry.py，两种输出的珠位完全同源。

示例入口：python -m evaluate.abacus.demos image → 输出到 abacus/output/image/。
"""
import matplotlib
matplotlib.use("Agg")  # 离线出图（PNG/GIF）必须；交互式动画见 render/animation.py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from evaluate.abacus.domain import Abacus, AbacusState, StateDelta
from evaluate.abacus.render.geometry import bead_centers, layout_of
from evaluate.abacus.render.style import Style


class ImageRenderer:
    """matplotlib 真实算盘渲染器（彩色 PNG/GIF）；scale=px/档距比例。

    机器观测/训练数据请走 render.registry（make_render），本类用于可视化。"""

    def __init__(self, style: Style | None = None, *, scale: float = 1.0,
                 seed: int | None = None):
        self.style = style or Style()
        self.scale = scale
        self.seed = seed
        # 颜色抖动只在构造时做一次，保证同一渲染器各帧配色恒定（拨珠过程盘面不变色）
        self._style = self._jitter(self.style)

    # ── 数据增强：颜色随机抖动（可选 seed 固定可复现；构造时一次性应用） ──
    def _jitter(self, style: Style) -> Style:
        import colorsys
        import random
        rng = random.Random(self.seed)

        def jitter(hex_color: str) -> str:
            r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            h = (h + rng.uniform(-0.03, 0.03)) % 1.0
            l = max(0.0, min(1.0, l + rng.uniform(-0.05, 0.05)))
            s = max(0.0, min(1.0, s + rng.uniform(-0.06, 0.06)))
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            return f"#{int(r2 * 255):02x}{int(g2 * 255):02x}{int(b2 * 255):02x}"

        return Style(
            frame=jitter(style.frame),
            bead_active=jitter(style.bead_active),
            bead_rest=jitter(style.bead_rest),
            edge=jitter(style.edge),
        )

    def render(self, state: AbacusState, abacus: Abacus, view: str = "front",
               caption: str | None = None, rhyme: str | None = None,
               hand: tuple[float, float] | None = None, tight: bool = False):
        """渲染单帧盘面，返回 PIL.Image（RGB，白底）。
        caption: 盘面上方显示的算式（如 "47 + 38 = ?"）
        rhyme: 盘面左侧显示的当前口诀（如 "七上三去五进一"）
        hand: 可选 (x, y) 数据坐标，表示手捏珠的位置；非 None 时在珠旁画一只小手
        tight: True = 机器观测模式（裁掉文字区与留白，x 取杆对称范围，逐档对齐）"""
        spec = abacus.spec
        n_upper = spec.upper_count
        n_lower = spec.lower_count
        n_cols = spec.rod_count
        layout = layout_of(abacus)

        # 档间距（matplotlib 数据坐标）：1.0，与珠距 0.9 / 珠半径 0.36 同源
        pitch = 1.0
        s = self._style
        # 珠位（靠梁贴梁 / 离梁贴框，真实位移）
        centers = bead_centers(state, abacus)

        # 中文字体（缺省回退，避免乱码/方框）
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                           "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False

        # figsize：按档数自适应（对齐原版 max(3.0, n_cols*1.2) × 3.5）
        # 数据坐标跨度：x = 左侧文字区(-1.8) ~ 外框右缘(0.7+n_cols)；
        # y = 0 ~ n_upper+n_lower+2。figsize 高宽比对齐跨度比，避免 aspect=equal 拉伸留白。
        # 左侧文字区留足空间避免与算盘重叠，右侧外框留白 1.0 保证边框可见。
        if tight:
            # 机器观测模式：x 取杆对称范围 [0.5, n_cols+0.5]，缩放后每根杆中心
            # 精确落在逐档 patch 中心（与 render_fixed 的逐档对齐一致）；
            # y 取外框 [0.4, n_upper+n_lower+1.6] 外扩 0.1 的边，裁掉文字区与留白。
            x_left, x_right = 0.5, n_cols + 0.5
            y_bottom, y_top = layout.frame_bottom - 0.1, layout.frame_top + 0.1
        else:
            x_left, x_right = -3.2, n_cols + 1.6
            y_bottom, y_top = 0, n_upper + n_lower + 2
        # aspect=equal 下，让 figure 高宽比等于数据坐标跨度比，消除左右/上下留白
        x_span = x_right - x_left
        y_span = y_top - y_bottom
        fig_h = 3.5 * self.scale
        fig_w = fig_h * (x_span / y_span)               # 与 y 同比例缩放
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        ax.set_aspect("equal")
        ax.axis("off")

        # ── 外框（矩形描边，最底层） ──
        ax.add_patch(Rectangle((layout.frame_left, layout.frame_bottom),
                               n_cols + 0.4, layout.frame_top - layout.frame_bottom,
                               facecolor="none", edgecolor=s.frame,
                               linewidth=2.0, zorder=0))

        # ── 横梁（厚矩形，zorder 最高压珠） ──
        beam_y = layout.beam_y
        ax.add_patch(Rectangle((layout.frame_left, beam_y - 0.12), n_cols + 0.4, 0.24,
                               facecolor=s.frame, edgecolor="none", zorder=5))

        # 逐档画杆 + 珠
        # 领域模型 rod_index=0 是个位（最右）；渲染时高位在左、个位在最右，
        # 故水平翻转：x 从左到右对应 rod_index 从高到低。
        for col, rod_state in enumerate(state.rods_left_to_right):
            x = col + 1  # 杆中心 x（1~n_cols）

            # 竖杆（粗 2.5，贯穿全盘，最底层）
            ax.plot([x, x], [layout.frame_bottom, layout.frame_top],
                    color=s.frame, linewidth=2.5, zorder=1)

            # 上珠（梁上方，靠梁=向下靠）
            for k in range(n_upper):
                b = rod_state.upper_beads[k]
                x_b, cy = centers[b.bead_id]
                color = s.bead_active if b.position.is_active else s.bead_rest
                ax.add_patch(Circle((x_b, cy), 0.36, facecolor=color,
                                    edgecolor=s.edge, linewidth=1.2, zorder=3))

            # 下珠（梁下方，靠梁=向上靠）
            for k in range(n_lower):
                b = rod_state.lower_beads[k]
                x_b, cy = centers[b.bead_id]
                color = s.bead_active if b.position.is_active else s.bead_rest
                ax.add_patch(Circle((x_b, cy), 0.36, facecolor=color,
                                    edgecolor=s.edge, linewidth=1.2, zorder=3))

        # ── 拨珠小手（可选）：画在指定珠位旁，指示正在拨动的珠 ──
        if hand is not None:
            self._draw_hand(ax, hand[0], hand[1])

        # ── 左侧文字区：算式与口诀在整个盘面高度上垂直均匀分布 ──
        # 文字区 x 取 -2.3，位于外框左缘(0.3)之外，避免与算盘重叠；
        # 垂直方向覆盖外框全高 [0.4, n_upper+n_lower+1.6]，按三等分均布。
        text_x = -3.0
        text_bottom = layout.frame_bottom
        text_top = layout.frame_top
        if caption and rhyme:
            # 两行：算式占上 1/3、口诀占下 1/3，三等分点即垂直均匀分布
            caption_y = text_bottom + (text_top - text_bottom) * (2 / 3)
            rhyme_y = text_bottom + (text_top - text_bottom) * (1 / 3)
        elif caption:
            caption_y = (text_top + text_bottom) / 2
            rhyme_y = None
        else:
            caption_y = None
            rhyme_y = (text_top + text_bottom) / 2

        if caption_y is not None:
            ax.text(text_x, caption_y, caption,
                    ha="left", va="center", fontsize=12, color="#4A2A0A",
                    fontweight="bold", zorder=6)

        if rhyme_y is not None:
            ax.text(text_x, rhyme_y, rhyme,
                    ha="left", va="center", fontsize=11, color="#C0392B",
                    fontweight="bold", zorder=6)

        # axes 填满 figure（figsize 已对齐坐标跨度比，无需 tight_layout 再裁切留白）
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # 转为 PIL.Image（RGB，白底）
        fig.canvas.draw()
        import numpy as np
        buf = np.asarray(fig.canvas.buffer_rgba())
        from PIL import Image
        # RGBA → RGB：透明处用白色垫底（与原版「透明底但保存为不透明图」一致）
        img = Image.fromarray(buf).convert("RGB")
        plt.close(fig)
        return img

    def _draw_hand(self, ax, x: float, y: float) -> None:
        """在珠位 (x, y) 旁画一只简笔小手：掌心圆 + 三根短指，从右侧伸入捏珠。"""
        skin = "#F1C27D"
        outline = "#C07A3A"
        # 掌心（位于珠右侧）
        ax.add_patch(Circle((x + 0.55, y), 0.28, facecolor=skin,
                            edgecolor=outline, linewidth=1.0, zorder=7))
        # 三根手指（朝左捏向珠）
        for dy in (-0.14, 0.0, 0.14):
            ax.plot([x + 0.42, x + 0.18], [y + dy, y + dy],
                    color=skin, linewidth=3.0, solid_capstyle="round", zorder=7)
        # 腕部（向右延伸）
        ax.plot([x + 0.72, x + 0.95], [y, y], color=skin, linewidth=4.0,
                solid_capstyle="round", zorder=7)

    def describe(self, state: AbacusState) -> dict:
        return {"step": state.step_index,
                "rods": {rs.rod_index: rs.rod_value for rs in state.rod_states},
                "total": state.total_value}

    # ── 单步过渡帧：静止 → 手捏珠(起点) → 拨到位(终点) → 松手 ──
    def _step_frames(self, op, prev, step, caption: str,
                     interp: int = 2) -> list:
        """为一个 CalculationStep 生成带手拨珠的过渡帧序列。
        interp: 拨珠过程中额外插值的中间帧数（≥1，越多越顺滑）。"""
        after = step.after_state
        rhyme = step.rhyme.rhyme_text
        delta = StateDelta.diff(prev, after)
        moved = [bd for rd in delta.rod_deltas for bd in rd.changed_beads]

        frames = [self.render(prev, op.abacus, caption=caption, rhyme=rhyme)]
        if moved:
            # 多珠同时变化（如"一下五去四"）时，手跟随第一颗变化珠
            bd = moved[0]
            bead = op.abacus.get_bead_by_id(bd.bead_id)
            start = bead_centers(prev, op.abacus)[bd.bead_id]
            end = bead_centers(after, op.abacus)[bd.bead_id]
            # 手捏珠过渡：起点 → (插值) → 终点
            frames.append(self.render(prev, op.abacus, caption=caption,
                                      rhyme=rhyme, hand=start))
            for t in range(1, interp + 1):
                f = t / (interp + 1)
                mx = start[0] + (end[0] - start[0]) * f
                my = start[1] + (end[1] - start[1]) * f
                frames.append(self.render(prev, op.abacus, caption=caption,
                                          rhyme=rhyme, hand=(mx, my)))
            frames.append(self.render(after, op.abacus, caption=caption, hand=end))
            frames.append(self.render(after, op.abacus, caption=caption))
        else:
            frames.append(self.render(after, op.abacus, caption=caption))
        return frames

    # ── GIF：把一次 Operation 的 before/after 逐帧合成动画 ──
    def render_gif(self, op, path: str, duration_ms: int = 400) -> None:
        import os
        sym = {"ADD": "+", "SUB": "-", "MUL": "×", "DIV": "÷"}[op.op_type.name]
        caption = f"{op.operand_a} {sym} {op.operand_b} = ?"
        frames: list = []
        prev = op.steps[0].before_state
        for step in op.steps:
            frames.extend(self._step_frames(op, prev, step, caption))
            prev = step.after_state
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=duration_ms, loop=0)


# ═══════════════════ 示例：四则运算各渲染 PNG 序列 + GIF（由薄壳 demo_image.py 调用） ═══════════════════

def demo(output_dir=None) -> list:
    """四则运算（÷ + − ×）各渲染 PNG 序列 + GIF 到 output_dir（默认 abacus/output/image/）。

    返回写出的文件路径列表。
    """
    from pathlib import Path

    from evaluate.abacus import (Abacus, ArithmeticComposer, RhymeResolver, OpType,
                        build_addition_rhymes, build_subtraction_rhymes)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output" / "image"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 口诀表：加法 + 减法种子（乘除法内部复用）
    resolver = RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes())
    composer = ArithmeticComposer(resolver)

    abacus = Abacus.standard()
    renderer = ImageRenderer(Style(), scale=1.0)

    written: list = []
    sym = {OpType.ADD: "+", OpType.SUB: "-", OpType.MUL: "×", OpType.DIV: "÷"}
    for op_type, a, b, prefix in [
        (OpType.DIV, 47, 5, "div"),
        (OpType.ADD, 47, 38, "add"),
        (OpType.SUB, 53, 8, "sub"),
        (OpType.MUL, 12, 34, "mul"),
    ]:
        op = composer.compose(op_type, abacus, a, b)
        # 逐步帧：空盘初始帧 + 每步终态（与 text/fixed/minimal 三个示例对齐）
        states = [op.steps[0].before_state] + [s.after_state for s in op.steps]
        for i, st in enumerate(states):
            p = output_dir / f"{prefix}_{i}.png"
            renderer.render(st, abacus).save(p)
            written.append(p)
        gif = output_dir / f"{prefix}_{a}_{b}.gif"
        renderer.render_gif(op, str(gif))
        written.append(gif)
    return written
