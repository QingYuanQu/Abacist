"""竖屏版式（render/vertical.py）回归测试：版式比例、档数适配、帧序列。

档数不是渲染器的旋钮（由调用方构造 `AbacusSpec(rod_count=…)`），
故这里同时验证"任意档数都能算出合法版式"与"档数只影响饱满度、不影响画布比例"。
"""
import pytest

from evaluate.abacus import (Abacus, AbacusSpec, AbacusState, ArithmeticComposer,
                             OpType, RhymeResolver, build_addition_rhymes,
                             build_subtraction_rhymes)
from evaluate.abacus.render.timeline import MovePlan
from evaluate.abacus.render.vertical import VerticalRenderer, vertical_canvas


def _abacus(rod_count: int) -> Abacus:
    return Abacus.standard(AbacusSpec(rod_count=rod_count))


def test_canvas_ratio_is_9_16():
    cv = vertical_canvas(_abacus(7), px=(1080, 1920))
    assert cv.canvas_ratio == pytest.approx(9 / 16)


def test_bands_are_symmetric_and_geometry_derived():
    """算盘带 = 算盘铺满画布宽度所需的高度占比；余量上下均分。"""
    cv = vertical_canvas(_abacus(7), px=(1080, 1920))
    top, mid, bot = cv.band_ratios
    assert top + mid + bot == pytest.approx(1.0)
    assert top == pytest.approx(bot)
    expect = cv.px_per_unit * (cv.ylim[1] - cv.ylim[0]) / 1920
    assert mid == pytest.approx(expect)
    assert mid == pytest.approx(0.484, abs=0.005)   # 7 档：算盘带约占画面高 48%


def test_rod_count_changes_fill_not_ratio():
    """档数越少算盘越满（5 档 > 7 档），但画布比例恒为 9:16。"""
    five = vertical_canvas(_abacus(5), px=(1080, 1920))
    seven = vertical_canvas(_abacus(7), px=(1080, 1920))
    assert five.canvas_ratio == seven.canvas_ratio == pytest.approx(9 / 16)
    assert five.band_ratios[1] > seven.band_ratios[1]
    assert five.band_ratios[1] == pytest.approx(0.631, abs=0.005)


def test_render_size_matches_px():
    ab = _abacus(5)
    img = VerticalRenderer(px=(540, 960)).render(
        AbacusState.empty(ab), ab, caption="47 + 38 = ?",
        rhyme="七上三去五进一", step=(1, 5))
    assert img.size == (540, 960)


def test_frames_follow_timeline():
    """帧序列按时间轴采样，且全帧尺寸一致（GIF / MP4 的硬要求，漂移会写坏文件）。"""
    ab = _abacus(7)
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    op = composer.compose(OpType.ADD, ab, 47, 38)
    frames = VerticalRenderer(px=(360, 640)).frames(op, plan=MovePlan(fps=10))
    assert len(frames) > len(op.steps)
    assert all(f.size == (360, 640) for f in frames)
