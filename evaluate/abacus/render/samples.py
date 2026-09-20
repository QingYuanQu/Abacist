"""真实拨珠采样库：用户录音 → 切分 → 按力度选层 → 播放。

为什么需要它：合成（audio.bead_click）有天花板——真实撞击是非线性接触振动，
正弦+噪声模型只能到"像"。本模块是**采样后端**，`click()` 接口与
`audio.bead_click` 对齐，调用方优先用采样、无录音时回退合成。

样本来源：`evaluate/abacus/audio/bead.m4a`（用户实录，5−3 的两次"去5上2"，
即上珠离梁 + 两颗下珠靠梁的协同动作）。切分点由波形探察确定：
  - 160ms 轻拨（幅度小）——样本在 240ms 的次级碰撞**之前**截断，不带入杂音；
  - 572ms 重拨（干净、余响长）——主力样本。
两段构成力度两层；`click()` 按 velocity 就近选层、层内增益微调。

已知边界：现有样本是**复合动作音**（上珠+下珠混合），暂不区分 upper/lower——
需要分别音色请分别录"只拨上珠"/"只拨下珠"，在 `from_recording` 的 cuts 里加层即可。
"""
from __future__ import annotations

import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluate.abacus.render.audio import SAMPLE_RATE

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
RECORDING = AUDIO_DIR / "bead.m4a"

# 切分点（毫秒）：(start, end, 力度档 0~1)。波形探察确定，改动需重新看波形图。
_CUTS = ((150.0, 235.0, 0.45), (562.0, 850.0, 0.90))


def ensure_wav(src: Path, sr: int = SAMPLE_RATE) -> Path:
    """任意音频 → 同目录同名 44.1kHz 单声道 wav（已存在且不旧于源则跳过）。"""
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".wav")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as e:      # pragma: no cover - 依赖缺失路径
        raise ImportError("采样库需要 imageio-ffmpeg 转码录音") from e
    cmd = [get_ffmpeg_exe(), "-y", "-i", str(src),
           "-ar", str(sr), "-ac", "1", str(dst)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"音频转码失败：{proc.stderr.decode('utf-8', 'ignore')}")
    return dst


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        if w.getnchannels() != 1:
            raise ValueError(f"{path} 不是单声道（转码应产出 mono）")
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return x.astype(np.float64) / 32768.0, sr


def _highpass(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """二阶 Butterworth 高通：去掉麦克风录音的低频隆隆（撞击本身低频很少）。"""
    from scipy.signal import butter, sosfilt
    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    idx = np.linspace(0.0, len(x) - 1, int(len(x) * sr_to / sr_from))
    return np.interp(idx, np.arange(len(x), dtype=np.float64), x)


def _fade(x: np.ndarray, sr: int, ms: float = 3.0) -> np.ndarray:
    """首尾短淡入淡出，避免切分边界咔哒声。"""
    k = min(len(x) // 2, int(ms / 1000 * sr))
    if k:
        ramp = np.linspace(0.0, 1.0, k)
        x = x.copy()
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


@dataclass(frozen=True)
class Sample:
    data: np.ndarray      # float32，[-1, 1]，峰值已归一化，@ SAMPLE_RATE
    velocity: float       # 力度档 0~1


class SampleBank:
    """力度分层的撞击样本库：`click()` 与 `audio.bead_click` 接口对齐。"""

    def __init__(self, samples: list[Sample]):
        self._samples = sorted(samples, key=lambda s: s.velocity)

    def __len__(self) -> int:
        return len(self._samples)

    @classmethod
    def from_recording(cls, recording: Path = RECORDING, cuts=_CUTS,
                       sr: int = SAMPLE_RATE) -> "SampleBank":
        """从录音切出样本（高通去低频隆隆 → 逐段淡入淡出 → 峰值归一化）。"""
        x, file_sr = _read_wav(ensure_wav(recording, sr))
        x = _highpass(_resample(x, file_sr, sr), 180.0, sr)
        samples = []
        for a, b, vel in cuts:
            seg = x[int(a / 1000 * sr):int(b / 1000 * sr)]
            if len(seg) < int(0.02 * sr):
                raise ValueError(f"切分段过短：{a}~{b}ms")
            seg = _fade(seg, sr)
            peak = float(np.max(np.abs(seg))) or 1.0
            samples.append(Sample((seg / peak).astype(np.float32), vel))
        return cls(samples)

    def click(self, *, velocity: float = 1.0, bead_type: str = "lower",
              count: int = 1, seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
        """按力度就近选层播放；接口与 `audio.bead_click` 对齐（可互换）。

        - bead_type 暂不区分（现有样本是"去5上2"复合音，上珠下珠混在一起）；
        - count 仍按 √k 提增益（声能叠加）；
        - 输出可能 >1（增益叠加），由 `mix_hits` 的防削波归一化兜底。
        """
        if not self._samples:
            raise ValueError("空样本库")
        s = min(self._samples, key=lambda t: abs(t.velocity - velocity))
        gain = min(1.25, max(0.55, velocity / s.velocity))
        gain *= min(1.5, math.sqrt(max(1, count)))
        w = s.data if sr == SAMPLE_RATE else _resample(s.data, SAMPLE_RATE, sr)
        return (w * gain).astype(np.float32)
