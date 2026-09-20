"""拨珠撞击音合成与混流：给竖屏短视频配音效（纯 numpy + stdlib wave）。

**与 music.py 无关**：music.py 是"算盘乐器化"（数值即和声、口诀即歌词）的旋律实验，
本模块是**物理撞击音效**——珠撞到梁/框的那一下"嗒"。两者是两回事，不要混。

音色依据（木质珠撞木质梁/框）：

- 瞬态：1~3ms 的宽带噪声脉冲（撞击的宽带成分，决定"脆"）；
- 共振：三条指数衰减正弦——体腔低频（200~400Hz）+ 两条木质峰（1~3kHz）；
- 衰减：T60 量级 40~80ms，**短促**，不能有厅堂感（算盘没有余音绕梁）；
- 上珠（更大更重）频率更低、衰减略长；下珠清脆。

拟人细节（这些才是"像人"而不是"像机器"的地方）：

- **力度**：由撞击速度决定（timeline 用行程/时长算出的峰值速度），
  力度越大越响且越"亮"（高频更多）——真人轻拨与重拨明显不同；
- **不等距**：连拨的间隔有 ±10% 抖动（确定性 seed），等距敲一听就是机器；
- **留白**：只有撞击有声，伸手/收手阶段**没有声音**（给动作配音反而假）。

音节对齐靠 `timeline.HitEvent.at_ms`——撞击瞬间，不是动作起点。
"""
from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100

# 木珠撞木框/木梁的"啪"：**高频非谐波模态 + 极短裂纹瞬态**，低频"咚"很弱。
# 每条模态 (幅度, 频率Hz, 衰减tau s)；频率比刻意非谐波（≈1:2.7:4.6:6.9）——
# 等谐波间隔的正弦听起来像"笛"，木头的模态是非谐的，这是"木感"的来源。
# upper（上珠，更大更重）整体频率更低、衰减更长；lower（下珠）更脆。
_VOICE = {
    "upper": {"dur": 0.075, "modes": ((0.14, 620.0, 0.026), (0.24, 1750.0, 0.018),
                                      (0.26, 2900.0, 0.012), (0.16, 4300.0, 0.008))},
    "lower": {"dur": 0.055, "modes": ((0.10, 780.0, 0.020), (0.22, 2150.0, 0.014),
                                      (0.28, 3600.0, 0.010), (0.18, 5400.0, 0.007))},
}


def bead_click(*, velocity: float = 1.0, bead_type: str = "lower",
               count: int = 1, seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """合成一次撞击音（float32，[-1, 1]，峰值归一化到力度）。

    真实"啪"的三个要件（缺一就退化成"噗/嗒/咚"）：
      1. **极快起振**（0.4ms）——撞击接触刚度极大，起振 >1ms 听感就是"噗"；
      2. **高通裂纹瞬态**——撞击的宽带成分集中在中高频。用一阶差分白噪
         （≈6dB/oct 高通）而不是全谱白噪，后者会"发沙"发闷；
      3. **非谐波模态**（见 _VOICE）——等谐波间隔听着像乐器音笛，不像木头。

    velocity: 0.25~1.4（timeline 的 hit_velocity，已含多珠 √k 的声能叠加）；
    bead_type: "upper"（低沉） / "lower"（清脆）；
    count: 同一档一次推动的珠束数——k>1 是"一记更厚更钝的啪"，不是连响 k 下；
    seed: 确定性（同一 seed 每次出片完全一致）。
    """
    v = _VOICE["upper" if bead_type == "upper" else "lower"]
    k = max(1, int(count))
    thick = 1.0 + 0.22 * (k - 1)          # 束：多珠质量更大 → 衰减更长
    bend = 1.0 - 0.08 * (k - 1)           # 束：音高略降
    tilt = 0.10 * (k - 1)                 # 束：低频模态增强、高频模态减弱（更厚）
    n = int(v["dur"] * (1.0 + 0.12 * (k - 1)) * sr)
    t = np.arange(n, dtype=np.float64) / sr

    # 非谐波模态（木珠+框/梁的共振），高次模态衰减更快
    body = np.zeros(n, dtype=np.float64)
    for i, (amp, freq, tau) in enumerate(v["modes"]):
        if i == 0:
            amp = amp * (1.0 + tilt)
        elif i >= 2:
            amp = amp * max(0.05, 1.0 - 1.5 * tilt)
        body += amp * np.sin(2 * math.pi * freq * bend * t) * np.exp(-t / (tau * thick))

    # 裂纹瞬态：高通差分噪声（差分≈6dB/oct 高通，全谱白噪会发沙），衰减 2.5ms
    rng = np.random.default_rng(seed)
    crack = np.diff(rng.standard_normal(n + 1))[:n] * np.exp(-t / 0.0025)

    # 起振 0.4ms：撞击的"啪"就靠这一下；力度越大瞬态越亮
    env = 1.0 - np.exp(-t / 0.0004)
    sig = env * (body + (0.30 + 0.45 * min(1.4, velocity)) * crack)

    peak = float(np.max(np.abs(sig))) or 1.0
    gain = min(0.98, 0.35 + 0.65 * min(1.4, max(0.25, velocity)))   # ≤0.98 防削波
    return (sig / peak * gain).astype(np.float32)


def mix_hits(events, *, sr: int = SAMPLE_RATE, tail_ms: float = 400.0,
             total_ms: float | None = None, jitter: float = 0.10,
             seed: int = 0, bank=None) -> np.ndarray:
    """把撞击事件混成一条单声道音轨（float32，[-1, 1]）。

    total_ms: 视频总时长（帧数 / fps × 1000）。**必须传**——音轨短于视频时，
              ffmpeg 的 -shortest 会把视频结尾（展示结果的 lead_out）静默截掉，
              实测把 7.1s 成片砍成了 6.47s。

    bank: 真实采样库（samples.SampleBank）。传入则撞击用真实录音播放，
          否则用合成音色（bead_click）。两者接口对齐，可互换。

    jitter: 相邻撞击间隔的抖动比例——真人连拨不可能等距，等距就是机器味；
            抖动量以 min(间隔, 150ms) 为基数，避免长间隔被抖得太多。
    """
    events = list(events)
    if not events:
        return np.zeros(int((total_ms or 0.0) / 1000.0 * sr), dtype=np.float32)

    span = max(e.at_ms for e in events) + tail_ms
    if total_ms is not None:
        span = max(span, total_ms)
    buf = np.zeros(int(span / 1000.0 * sr), dtype=np.float32)

    last = 0.0
    for i, e in enumerate(events):
        gap = max(0.0, e.at_ms - last)
        last = e.at_ms
        rng = np.random.default_rng(seed + i)          # 确定性，不引入全局随机
        shift = rng.uniform(-jitter, jitter) * min(gap, 150.0)
        start = int((e.at_ms + shift) / 1000.0 * sr)
        if start >= len(buf):
            continue
        if bank is not None:
            w = bank.click(velocity=e.velocity, bead_type=e.bead_type,
                           count=getattr(e, "count", 1), seed=seed + i, sr=sr)
        else:
            w = bead_click(velocity=e.velocity, bead_type=e.bead_type,
                           count=getattr(e, "count", 1), seed=seed + i, sr=sr)
        end = min(start + len(w), len(buf))
        buf[start:end] += w[:end - start]

    peak = float(np.max(np.abs(buf)))
    if peak > 0.9:                                     # 防叠加削波
        buf = (buf / peak * 0.9).astype(np.float32)
    return buf


def write_wav(path: str, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """写出 16bit 单声道 WAV（stdlib wave，零第三方依赖）。"""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def mux_mp4(video: str, audio: str, out: str) -> None:
    """无声 MP4 + WAV → 带音轨 MP4（视频流 copy，音频 AAC）。需 imageio-ffmpeg。

    -c:v copy 避免二次编码（画质无损、速度快）；-shortest 以较短者为准收尾。
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as e:      # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "混流需要 imageio-ffmpeg：pip install imageio-ffmpeg") from e

    cmd = [get_ffmpeg_exe(), "-y", "-i", str(video), "-i", str(audio),
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 混流失败：{proc.stderr.decode('utf-8', 'ignore')}")
