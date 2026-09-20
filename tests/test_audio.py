"""拨珠撞击音（render/audio.py）回归测试：合成 / 混音 / 写盘。

只测纯函数部分（合成、混音、WAV 读写）；`mux_mp4` 需要真实 ffmpeg，
由 `python -m evaluate.abacus.demos vertical` 实测覆盖（不进自动化测试，避免
把编码耗时与外部二进制依赖带进单测）。
"""
import wave

import numpy as np
import pytest

from evaluate.abacus import (Abacus, AbacusSpec, ArithmeticComposer, OpType,
                             RhymeResolver, build_addition_rhymes,
                             build_subtraction_rhymes)
from evaluate.abacus.render.audio import (SAMPLE_RATE, bead_click, mix_hits,
                                          write_wav)
from evaluate.abacus.render.timeline import MovePlan, plan_hits


def _op(rod_count: int = 7):
    ab = Abacus.standard(AbacusSpec(rod_count=rod_count))
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    return composer.compose(OpType.ADD, ab, 47, 38)


def test_click_is_short_and_deterministic():
    w = bead_click(velocity=1.0, bead_type="lower", seed=0)
    assert len(w) == int(0.055 * SAMPLE_RATE)
    assert np.max(np.abs(w)) == pytest.approx(1.0, abs=0.02)
    assert np.array_equal(w, bead_click(velocity=1.0, bead_type="lower", seed=0))
    assert np.max(np.abs(w[-1000:])) < 0.1        # 短促，没有厅堂余音


def test_click_is_a_snap_not_a_thud():
    """"啪"的三要件：极快起振、低频"咚"几乎为零、高频成分显著。"""
    w = bead_click(velocity=1.0, bead_type="lower", seed=0)
    peak_at = np.argmax(np.abs(w)) / SAMPLE_RATE * 1000
    assert peak_at < 1.5                          # 起振 <1.5ms（>1ms 就是"噗"）
    assert _low_ratio(w) < 0.05                   # <600Hz 占比≈0（没有"咚"）
    assert _high_ratio(w) > 0.25                  # >4kHz 占比显著（"脆"）


def test_upper_bead_is_longer_and_darker_than_lower():
    """上珠更大更重 → 衰减更长、频谱重心更低。"""
    up, low = bead_click(bead_type="upper", seed=0), bead_click(bead_type="lower", seed=0)
    assert len(up) > len(low)
    assert _spectral_centroid(up) < _spectral_centroid(low)


def test_velocity_drives_loudness():
    soft, hard = bead_click(velocity=0.3, seed=0), bead_click(velocity=1.3, seed=0)
    assert np.max(np.abs(soft)) < np.max(np.abs(hard))


def test_mix_covers_every_hit_and_is_deterministic():
    hits = plan_hits(_op(), MovePlan())
    assert hits
    track = mix_hits(hits, seed=0)
    assert len(track) / SAMPLE_RATE > max(h.at_ms for h in hits) / 1000
    assert np.count_nonzero(np.abs(track) > 1e-3) > 0
    assert np.array_equal(track, mix_hits(hits, seed=0))


def test_mix_never_shorter_than_video():
    """音轨必须不短于视频，否则 ffmpeg -shortest 会静默截掉结尾画面。"""
    hits = plan_hits(_op(), MovePlan())
    video_ms = 7100.0
    track = mix_hits(hits, total_ms=video_ms, seed=0)
    assert len(track) / SAMPLE_RATE >= video_ms / 1000 - 1e-6


def test_hits_are_anchored_at_impact():
    """撞击事件：时刻单调、珠型合法、力度在物理区间内。"""
    hits = plan_hits(_op(), MovePlan())
    assert hits == sorted(hits, key=lambda h: h.at_ms)
    assert all(h.bead_type in ("upper", "lower") for h in hits)
    assert all(0.25 <= h.velocity <= 1.4 for h in hits)


def _spectral_centroid(w: np.ndarray) -> float:
    """谱质心（Hz）：比过零率可靠——过零率会被瞬态噪声带偏，
    实测用它判断"钝"会得出与听感相反的结论。"""
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1.0 / SAMPLE_RATE)
    return float((spec * freqs).sum() / spec.sum())


def _band_ratio(w: np.ndarray, lo: float, hi: float = float("inf")) -> float:
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1.0 / SAMPLE_RATE)
    band = spec[(freqs >= lo) & (freqs < hi)].sum()
    return float(band / spec.sum())


def _low_ratio(w: np.ndarray) -> float:
    return _band_ratio(w, 0.0, 600.0)


def _high_ratio(w: np.ndarray) -> float:
    return _band_ratio(w, 4000.0)


def test_cluster_click_is_thicker_and_darker():
    """束音（同档多珠同时撞）= 更长、频谱重心更低的一声，不是 k 声排成一串。"""
    one = bead_click(bead_type="lower", count=1, seed=0)
    four = bead_click(bead_type="lower", count=4, seed=0)
    assert len(four) > len(one)                              # 衰减更长
    assert _spectral_centroid(four) < _spectral_centroid(one)  # 更厚更钝


def test_write_wav_roundtrip(tmp_path):
    w = bead_click(velocity=1.0, seed=0)
    p = tmp_path / "click.wav"
    write_wav(str(p), w)
    with wave.open(str(p)) as f:
        assert (f.getnchannels(), f.getframerate()) == (1, SAMPLE_RATE)
        assert f.getnframes() == len(w)
