"""image.py 13 档产出的基线锁（新增竖屏后端时立的防线）。

image.py 同时承担 registry 的 image / image_gray 后端——VLA/VLM 训练与推理的
**机器观测**通道。画布尺寸一旦漂移，训练数据分布会静默改变且极难察觉。
竖屏短视频走新增的 `vertical.py`（并列后端，不复用 image.py 的画布构建），
故此处把 image.py 的 non-tight / tight 两种画布比例钉死。
"""
import pytest

from evaluate.abacus import Abacus, AbacusSpec, AbacusState
from evaluate.abacus.render.image import ImageRenderer


def _empty_13():
    ab = Abacus.standard(AbacusSpec(rod_count=13))
    return ab, AbacusState.empty(ab)


def test_nontight_canvas_ratio_locked():
    """非 tight（13 档 + 左侧文字区）：数据跨度 x[-3.2, 14.6] / y[0, 7]。"""
    ab, st = _empty_13()
    img = ImageRenderer().render(st, ab, caption="47 + 38 = ?",
                                 rhyme="七上三去五进一")
    w, h = img.size
    assert h == 350                                   # fig_h = 3.5 in × 100 dpi
    assert w / h == pytest.approx(17.8 / 7, abs=0.01)


def test_tight_canvas_ratio_locked():
    """tight（机器观测，逐档对齐）：数据跨度 x[0.5, 13.5] / y[0.3, 6.7]。"""
    ab, st = _empty_13()
    img = ImageRenderer().render(st, ab, tight=True)
    w, h = img.size
    assert h == 350
    assert w / h == pytest.approx(13 / 6.4, abs=0.01)
