"""竖屏短视频渲染器（9:16）：顶部算式 / 中部算盘 / 底部「步骤序号 + 口诀」。

与 image.py **并列，且不共享画布构建**：

- image.py 服务"横条 + 左侧文字"的通用出图，以及 tight 模式的机器观测
  （registry 的 image / image_gray 后端，是 VLA/VLM 的训练数据通道，
  见 geometry 与 registry 的说明）；
- 本模块只服务手机竖屏短视频，**不注册进 render.registry**——那是观测后端表，
  把给人看的版式混进去会污染训练后端的选择面。

共用：geometry（珠位唯一权威）、style（配色/字体）、hand（手势图元）。
不共用：画布与版式——两者诉求不同，硬抽公共层只会互相迁就。

档数不是本模块的旋钮：要几档由调用方构造盘（`AbacusSpec(rod_count=…)`），
`band_ratios` 由几何反解，任意档数自动适配（1080×1920 下：
5 档算盘带约占画面高 63%，7 档约 48%，13 档约 28%）。

两个必须与分辨率解耦的口径，否则画布一大就露馅：

- 字号以"占画布宽度的比例"表达，再按 dpi 换算成磅；
- 描边以**数据坐标**表达，再按 px_per_unit 换算成磅
  （image.py 那套固定磅值在 1080×1920 上会细成头发丝）。

示例入口：python -m evaluate.abacus.demos vertical → 输出到 abacus/output/vertical/。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 离线出图（PNG/GIF/MP4）必须
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

from evaluate.abacus.domain import Abacus, AbacusState
from evaluate.abacus.render.geometry import bead_centers, layout_of
from evaluate.abacus.render.hand import BASE_PX_PER_UNIT, draw_hand
from evaluate.abacus.render.style import CJK_FONTS, Style
from evaluate.abacus.render.timeline import (MovePlan, plan_duration_ms,
                                             plan_hits, plan_operation)


# ── 字号：占**画布宽度**的比例（单行 CJK 约 1 em/字） ──
TEXT_RATIO_CAPTION = 0.085   # 顶部算式
TEXT_RATIO_RHYME = 0.070     # 底部口诀
TEXT_RATIO_STEP = 0.045      # 底部步骤序号（口认上方的小字）
MAX_TEXT_WIDTH_RATIO = 0.90  # 单行最宽不超过画布宽度的 90%，超了自动缩号

# ── 描边宽度：以**数据坐标**表达（不是磅） ──
# 由 image.py 现状在 13 档非 tight 画布（50 px/数据单位）下的磅值反推，
# 故两版视觉粗细一致，且随档数/画布/dpi 自动缩放。
STROKE_FRAME = 0.056   # ← 2.0 pt
STROKE_ROD = 0.069     # ← 2.5 pt
STROKE_BEAD = 0.033    # ← 1.2 pt

# 珠半径 / 梁厚：与 geometry 同源的常量（geometry 只给珠心，半径在绘制层）
BEAD_RADIUS = 0.36
BEAM_HALF_H = 0.12

# 目标档高亮（正在被拨的那一档）：教学片里观众得先知道该看哪一档。
# 色带 + 加粗变色的杆双管齐下——只加粗杆在 1080 宽下不够醒目。
FOCUS_COLOR = "#E67E22"
FOCUS_BAND_ALPHA = 0.18
FOCUS_ROD_LW_SCALE = 1.8


@dataclass(frozen=True)
class VerticalCanvas:
    """竖屏版式（纯数据，可单测）：算盘 axes 跨度 + 三带高度比 + 字号/线宽口径。

    band_ratios = (上文字带, 算盘带, 下文字带)，三者之和为 1；
    算盘带 = 让算盘按自身比例铺满画布宽度所需的高度占比，余量上下均分。
    """
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    canvas_ratio: float                       # 画布 宽/高（9:16 → 0.5625）
    band_ratios: tuple[float, float, float]
    caption_ratio: float
    rhyme_ratio: float
    step_ratio: float
    frame_lw: float                           # 磅（已按 px_per_unit 换算）
    rod_lw: float
    bead_lw: float
    hand_lw_scale: float                      # 手势线宽倍率（1.0 = image.py 基准磅数）
    px_per_unit: float


def vertical_canvas(abacus: Abacus, *, px: tuple[int, int] = (1080, 1920),
                    dpi: int = 100, margin: float = 0.6) -> VerticalCanvas:
    """由算盘形制 + 目标画布像素推导竖屏版式。

    margin: 算盘四周的留白（数据坐标），保证外框描边不被画布裁掉。
    """
    layout = layout_of(abacus)
    n_cols = abacus.spec.rod_count
    px_w, px_h = int(px[0]), int(px[1])

    x_left = layout.frame_left - margin
    x_right = layout.frame_left + (n_cols + 0.4) + margin
    y_bottom = layout.frame_bottom - margin
    y_top = layout.frame_top + margin

    w_data, h_data = x_right - x_left, y_top - y_bottom
    px_per_unit = px_w / w_data
    # 算盘带：算盘按自身比例铺满画布宽度所需的高度占比
    # （>1 表示画面过矮、算盘受高度限制而左右留白，截断为 1）
    band = min(1.0, px_per_unit * h_data / px_h)
    side = (1.0 - band) / 2
    lw = px_per_unit * 72.0 / dpi             # 数据单位 → 磅

    return VerticalCanvas(
        xlim=(x_left, x_right), ylim=(y_bottom, y_top),
        canvas_ratio=px_w / px_h, band_ratios=(side, band, side),
        caption_ratio=TEXT_RATIO_CAPTION, rhyme_ratio=TEXT_RATIO_RHYME,
        step_ratio=TEXT_RATIO_STEP,
        frame_lw=STROKE_FRAME * lw, rod_lw=STROKE_ROD * lw, bead_lw=STROKE_BEAD * lw,
        hand_lw_scale=px_per_unit / BASE_PX_PER_UNIT,
        px_per_unit=px_per_unit,
    )


def _em_width(text: str) -> float:
    """粗略字宽（单位 em）：CJK/全角 = 1，空格 = 0.28，其余（数字/字母/半角）= 0.55。"""
    total = 0.0
    for ch in text:
        if ord(ch) > 0x2E80:
            total += 1.0
        elif ch == " ":
            total += 0.28
        else:
            total += 0.55
    return total


def _fit_pt(text: str, ratio: float, px_w: int, dpi: int) -> float:
    """字号（磅）= ratio × 画布宽，并按字数 clamp 到不超过 MAX_TEXT_WIDTH_RATIO × 画布宽。

    长口诀（如"一下五去四进一"叠加运算式）不会溢出画面。
    """
    px_to_pt = 72.0 / dpi
    pt = ratio * px_w * px_to_pt
    em = _em_width(text)
    if em <= 0:
        return pt
    return min(pt, MAX_TEXT_WIDTH_RATIO * px_w * px_to_pt / em)


class VerticalRenderer:
    """竖屏 9:16 渲染器：单帧 → PIL.Image，帧序列 → GIF / MP4。

    px: 输出像素 (宽, 高)，其比例即画布版式（默认 1080×1920 = 9:16）。

    ⚠️ 1080×1920 × 30 帧的 GIF 会达数十 MB，且 PIL 保存为 256 色；
    快速预览请传 px=(540, 960)，成片走 `save_mp4`（需 imageio-ffmpeg）。
    """

    def __init__(self, style: Style | None = None, *, px: tuple[int, int] = (1080, 1920),
                 dpi: int = 100, margin: float = 0.6):
        self.style = style or Style()
        self.px = (int(px[0]), int(px[1]))
        self.dpi = dpi
        self.margin = margin
        self._canvas: dict[tuple[int, int, int], VerticalCanvas] = {}

    def canvas_of(self, abacus: Abacus) -> VerticalCanvas:
        """取（并缓存）该形制的竖屏版式；同一渲染器内各帧版式恒定。"""
        spec = abacus.spec
        key = (spec.rod_count, spec.upper_count, spec.lower_count)
        if key not in self._canvas:
            self._canvas[key] = vertical_canvas(
                abacus, px=self.px, dpi=self.dpi, margin=self.margin)
        return self._canvas[key]

    # ── 单帧 ──
    def render(self, state: AbacusState, abacus: Abacus, *,
               caption: str | None = None, rhyme: str | None = None,
               step: tuple[int, int] | None = None,
               hand: tuple[float, float] | None = None,
               bead_motion: dict[int, tuple[float, float]] | None = None,
               focus_col: int | None = None) -> Image.Image:
        """渲染单帧盘面，返回 PIL.Image（RGB，白底）。

        caption: 顶部算式（如 "47 + 38 = ?"）
        rhyme:   底部口诀（如 "七上三去五进一"）
        step:    (当前步 1-based, 总步数) → 底部渲染为"第 3 / 5 步"，位于口诀上方
        hand:    可选 (x, y) 数据坐标，在珠旁画拨珠小手
        bead_motion: {bead_id: (x, y)} 覆盖珠心坐标——**珠随手动**的关键接口。
                     盘面状态是离散的（珠要么靠梁要么离梁），拨动中的中间位置
                     只能由调用方（timeline）插值后覆盖，否则珠子只能"瞬移"。
        focus_col: 正在被拨的档（渲染列，0-based）；非 None 时该档铺高亮色带并加粗杆。
        """
        cv = self.canvas_of(abacus)
        s = self.style
        px_w, px_h = self.px
        layout = layout_of(abacus)
        n_cols = abacus.spec.rod_count
        centers = bead_centers(state, abacus)
        if bead_motion:
            centers = {**centers, **bead_motion}

        plt.rcParams["font.sans-serif"] = list(CJK_FONTS)
        plt.rcParams["axes.unicode_minus"] = False

        fig = plt.figure(figsize=(px_w / self.dpi, px_h / self.dpi), dpi=self.dpi)
        gs = fig.add_gridspec(3, 1, height_ratios=list(cv.band_ratios))
        ax_top = fig.add_subplot(gs[0])
        ax_mid = fig.add_subplot(gs[1])
        ax_bot = fig.add_subplot(gs[2])
        for ax in (ax_top, ax_mid, ax_bot):
            ax.axis("off")

        # ── 中部：算盘 ──
        # aspect=equal + adjustable=box + anchor=C → 算盘在带内按自身比例居中、不变形，
        # 带内多余的上下空间自动变成留白（算盘因此精确铺满画布宽度）。
        ax_mid.set_xlim(*cv.xlim)
        ax_mid.set_ylim(*cv.ylim)
        ax_mid.set_aspect("equal", adjustable="box", anchor="C")

        # 外框（矩形描边，最底层）
        ax_mid.add_patch(Rectangle((layout.frame_left, layout.frame_bottom),
                                   n_cols + 0.4, layout.frame_top - layout.frame_bottom,
                                   facecolor="none", edgecolor=s.frame,
                                   linewidth=cv.frame_lw, zorder=0))
        # 横梁（厚矩形，zorder 最高压珠）
        ax_mid.add_patch(Rectangle((layout.frame_left, layout.beam_y - BEAM_HALF_H),
                                   n_cols + 0.4, BEAM_HALF_H * 2,
                                   facecolor=s.frame, edgecolor="none", zorder=5))

        # 目标档高亮色带（铺在杆之下、外框之上）
        if focus_col is not None:
            ax_mid.add_patch(Rectangle((focus_col + 1 - 0.45, layout.frame_bottom), 0.9,
                                       layout.frame_top - layout.frame_bottom,
                                       facecolor=FOCUS_COLOR, edgecolor="none",
                                       alpha=FOCUS_BAND_ALPHA, zorder=0.5))

        # 逐档画杆 + 珠（rods_left_to_right：高位在左、个位在最右）
        for col, rod_state in enumerate(state.rods_left_to_right):
            x = col + 1  # 杆中心 x（1~n_cols）
            focus = (col == focus_col)
            ax_mid.plot([x, x], [layout.frame_bottom, layout.frame_top],
                        color=FOCUS_COLOR if focus else s.frame,
                        linewidth=cv.rod_lw * (FOCUS_ROD_LW_SCALE if focus else 1.0),
                        zorder=1, solid_capstyle="butt")
            for b in rod_state.upper_beads:
                x_b, cy = centers[b.bead_id]
                color = s.bead_active if b.position.is_active else s.bead_rest
                ax_mid.add_patch(Circle((x_b, cy), BEAD_RADIUS, facecolor=color,
                                        edgecolor=s.edge, linewidth=cv.bead_lw, zorder=3))
            for b in rod_state.lower_beads:
                x_b, cy = centers[b.bead_id]
                color = s.bead_active if b.position.is_active else s.bead_rest
                ax_mid.add_patch(Circle((x_b, cy), BEAD_RADIUS, facecolor=color,
                                        edgecolor=s.edge, linewidth=cv.bead_lw, zorder=3))

        if hand is not None:
            draw_hand(ax_mid, hand[0], hand[1], lw_scale=cv.hand_lw_scale)

        # ── 顶部：算式 ──
        if caption:
            ax_top.text(0.5, 0.5, caption, transform=ax_top.transAxes,
                        ha="center", va="center",
                        fontsize=_fit_pt(caption, cv.caption_ratio, px_w, self.dpi),
                        color="#4A2A0A", fontweight="bold")

        # ── 底部：步骤序号（上） + 口诀（下） ──
        step_text = f"第 {step[0]} / {step[1]} 步" if step else None
        if step_text and rhyme:
            step_y, rhyme_y = 0.74, 0.30
        elif step_text:
            step_y, rhyme_y = 0.5, None
        elif rhyme:
            step_y, rhyme_y = None, 0.5
        else:
            step_y = rhyme_y = None

        if step_text and step_y is not None:
            ax_bot.text(0.5, step_y, step_text, transform=ax_bot.transAxes,
                        ha="center", va="center",
                        fontsize=_fit_pt(step_text, cv.step_ratio, px_w, self.dpi),
                        color="#8A6A4A", zorder=6)
        if rhyme and rhyme_y is not None:
            ax_bot.text(0.5, rhyme_y, rhyme, transform=ax_bot.transAxes,
                        ha="center", va="center",
                        fontsize=_fit_pt(rhyme, cv.rhyme_ratio, px_w, self.dpi),
                        color="#C0392B", fontweight="bold", zorder=6)

        fig.canvas.draw()
        import numpy as np
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(buf).convert("RGB")  # RGBA → RGB：透明处垫白
        plt.close(fig)
        return img

    # ── 帧序列：一次 Operation → 按拟人时间轴采样出的逐帧 ──
    def frames(self, op, *, plan: MovePlan | None = None) -> list[Image.Image]:
        """按拟人时间轴渲染一次 Operation 的帧序列。

        plan: 时间预算与节奏（timeline.MovePlan，fps 亦在其中）。
              save_gif / save_mp4 必须使用同一个 plan.fps，否则成片时长与时间轴不符。

        节奏（停顿/缓动/回弹/错峰）与"珠随手动"的中间位置全部由
        `timeline.plan_operation` 算好，本方法只把纯数据 FrameSpec 翻译成图像。
        """
        plan = plan or MovePlan()
        return [self.render(spec.state, op.abacus, caption=spec.caption,
                            rhyme=spec.rhyme, step=spec.step, hand=spec.hand,
                            bead_motion=spec.bead_motion, focus_col=spec.focus_col)
                for spec in plan_operation(op, plan)]

    # ── 封装：帧序列 → GIF / MP4 ──
    def save_gif(self, frames: list, path: str, *, fps: int = 30, loop: int = 0) -> None:
        """保存为 GIF（PIL 会量化为 256 色；大尺寸慢且体积大，预览用小 px）。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=round(1000 / fps), loop=loop)

    def save_mp4(self, frames: list, path: str, *, fps: int = 30) -> None:
        """保存为 MP4（H.264 / yuv420p）。需 imageio + imageio-ffmpeg。

        - pixelformat="yuv420p"：不指定则 QuickTime / 剪映等可能拒读；
        - macro_block_size=1：**必须**。1080 不是 16 的倍数，imageio 默认会把
          1080×1920 静默放大到 1088×1920（横向拉伸 0.74% 且分辨率被改），
          yuv420p 只要求偶数维度（1080/1920 都是偶数），故关闭分块对齐是安全的。
        """
        try:
            import imageio.v2 as iio
        except ImportError as e:      # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "导出 MP4 需要 imageio 与 ffmpeg 插件：pip install imageio-ffmpeg") from e
        import numpy as np

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with iio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))


# ═══════════════════ 示例：47 + 38 竖屏短视频（5 档 / 7 档各一份） ═══════════════════
def demo(output_dir=None) -> list:
    """渲染 47 + 38 的竖屏短视频（7 档 / 1080×1920 / H.264 MP4）→ output/vertical/。

    档数由调用方构造盘时决定（渲染器不管）：7 档算盘带约占画面高 48%，
    珠够大、上下留白适中。节奏由 timeline.MovePlan 控制（默认拟人节奏 ≈ 6~7 秒）。

    返回写出的文件路径列表。
    """
    from evaluate.abacus import (Abacus, AbacusSpec, ArithmeticComposer, OpType,
                                 RhymeResolver, build_addition_rhymes,
                                 build_subtraction_rhymes)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output" / "vertical"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolver = RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes())
    composer = ArithmeticComposer(resolver)

    abacus = Abacus.standard(AbacusSpec(rod_count=7))
    op = composer.compose(OpType.ADD, abacus, 47, 38)

    plan = MovePlan()
    renderer = VerticalRenderer(Style(), px=(1080, 1920))
    print(f"[vertical] {len(op.steps)} 步，时间轴 {plan_duration_ms(plan, len(op.steps)) / 1000:.1f}s"
          f" @ {plan.fps}fps")

    # 音效层（独立于 music.py：这里是珠撞梁的物理撞击音，不是旋律）
    from evaluate.abacus.render.audio import mix_hits, mux_mp4, write_wav
    from evaluate.abacus.render.samples import SampleBank

    frames = renderer.frames(op, plan=plan)
    hits = plan_hits(op, plan)
    print(f"[vertical] 撞击 {len(hits)} 次")

    # 优先真实采样（用户录音），无录音/转码失败则回退合成音色
    bank = None
    try:
        bank = SampleBank.from_recording()
        print(f"[vertical] 音色 = 真实采样（{len(bank)} 层力度）")
    except Exception as e:                        # 文件缺失 / ffmpeg 不可用等
        print(f"[vertical] 音色 = 合成（采样不可用：{e}）")

    written: list = []
    wav = output_dir / "add_47_38_7rod.wav"
    # total_ms 必须给足视频时长：音轨短于视频时 -shortest 会截掉结尾的 lead_out
    write_wav(str(wav), mix_hits(hits, total_ms=len(frames) / plan.fps * 1000.0,
                                 bank=bank))
    written.append(wav)

    # 先出无声 MP4，再与音轨混流（-c:v copy，不二次编码）
    silent = output_dir / "_silent.mp4"
    renderer.save_mp4(frames, str(silent), fps=plan.fps)
    mp4 = output_dir / "add_47_38_7rod.mp4"
    mux_mp4(str(silent), str(wav), str(mp4))
    silent.unlink(missing_ok=True)
    written.append(mp4)

    # 首帧另存 PNG：不开播放器也能核对版式（上算式 / 中算盘 / 下步骤+口诀）
    png = output_dir / "add_47_38_7rod_first.png"
    frames[0].save(png)
    written.append(png)

    # 定位帧另存 PNG：核对目标档高亮（hold 阶段那一帧）
    specs = plan_operation(op, plan)
    idx = next((i for i, sp in enumerate(specs) if sp.focus_col is not None), None)
    if idx is not None:
        focus_png = output_dir / "add_47_38_7rod_focus.png"
        frames[idx].save(focus_png)
        written.append(focus_png)
    return written
