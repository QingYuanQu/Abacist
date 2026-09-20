"""真实采样库（render/samples.py）回归测试：切分、选层、接口对齐。

依赖用户录音 `evaluate/abacus/audio/bead.m4a`（缺失则整文件跳过——
采样是可选增强，合成后端必须始终可用）。
"""
import numpy as np
import pytest

from evaluate.abacus.render.audio import SAMPLE_RATE, mix_hits
from evaluate.abacus.render.samples import RECORDING, SampleBank
from evaluate.abacus.render.timeline import MovePlan, plan_hits

pytestmark = pytest.mark.skipif(not RECORDING.exists(),
                                reason="无用户录音（bead.m4a）")


def _op():
    from evaluate.abacus import (Abacus, AbacusSpec, ArithmeticComposer, OpType,
                                 RhymeResolver, build_addition_rhymes,
                                 build_subtraction_rhymes)
    ab = Abacus.standard(AbacusSpec(rod_count=7))
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    return composer.compose(OpType.ADD, ab, 47, 38)


@pytest.fixture(scope="module")
def bank() -> SampleBank:
    return SampleBank.from_recording()


def test_two_velocity_layers(bank):
    """录音切出两层力度（轻拨 160ms / 重拨 572ms）。"""
    assert len(bank) == 2
    assert sorted(s.velocity for s in bank._samples) == pytest.approx([0.45, 0.90])


def test_heavy_layer_rings_longer(bank):
    """重拨层余响更长（波形探察：572ms 样本带完整衰减尾）。"""
    assert len(bank.click(velocity=1.2)) > len(bank.click(velocity=0.4))


def test_layers_are_actually_different_and_deterministic(bank):
    """不同力度选到不同层；同一参数输出逐样本一致。"""
    light, heavy = bank.click(velocity=0.4), bank.click(velocity=1.2)
    assert not np.array_equal(light, heavy)
    assert np.array_equal(heavy, bank.click(velocity=1.2))


def test_peak_normalized_and_bounded(bank):
    for s in bank._samples:
        assert np.max(np.abs(s.data)) == pytest.approx(1.0, abs=1e-3)
    # click 可叠增益（≤1.25×√k），不得炸出 inf/nan
    w = bank.click(velocity=1.4, count=4)
    assert np.all(np.isfinite(w)) and np.max(np.abs(w)) < 2.0


def test_mix_accepts_bank():
    """mix_hits 走采样后端出完整音轨（与合成后端同一条管线）。"""
    hits = plan_hits(_op(), MovePlan(fps=10))
    track = mix_hits(hits, total_ms=3000.0, bank=SampleBank.from_recording())
    assert len(track) / SAMPLE_RATE >= 3.0
    assert np.count_nonzero(np.abs(track) > 1e-3) > 0
