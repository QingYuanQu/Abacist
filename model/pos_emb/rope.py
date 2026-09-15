"""RoPE（旋转位置编码）及其外推变体。

子变体（rope_type）走 ROPE_INIT 路由表，结构与 HuggingFace 的
`ROPE_INIT_FUNCTIONS` 一致：字符串 → 频率计算函数。
新增一个 RoPE 变体 = 加一个函数 + 一行注册。

行为约定（自本项目最初实现起保持不变）：
  - cos/sin 采用「前后半拼接」的分块旋转形式；
  - YaRN 仅在 max_len 越过 original_max_position_embeddings 时生效。
"""
import math
from collections.abc import Callable

import torch

from .base import PosEmbedding

# rope_type → 频率计算函数（签名统一，见下）
ROPE_INIT: dict[str, Callable] = {}


def register_rope_type(name: str):
    """注册一个 RoPE 子变体。"""
    def deco(fn):
        ROPE_INIT[name] = fn
        return fn
    return deco


@register_rope_type("default")
def _default_freqs(dim: int, max_len: int, theta: float, scaling: dict | None):
    """原始 RoPE：f(i) = 1 / theta^(2i/dim)。返回 (freqs, attn_factor)。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    return freqs, 1.0


@register_rope_type("yarn")
def _yarn_freqs(dim: int, max_len: int, theta: float, scaling: dict | None):
    """YaRN 外推：f'(i) = f(i) · ((1-γ) + γ/s)，γ 是跨维度的线性斜坡。

    仅在 max_len 越过 original_max_position_embeddings 时改变频率，
    否则退化为 default —— 保持与原实现相同的触发语义。
    """
    freqs, _ = _default_freqs(dim, max_len, theta, None)
    if scaling is None:
        return freqs, 1.0

    orig_max = scaling.get("original_max_position_embeddings", 2048)
    factor = scaling.get("factor", 16)
    beta_fast = scaling.get("beta_fast", 32.0)
    beta_slow = scaling.get("beta_slow", 1.0)
    attn_factor = scaling.get("attention_factor", 1.0)

    if max_len / orig_max <= 1.0:
        return freqs, attn_factor

    def inv_dim(b: float) -> float:
        return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(theta))

    low = max(math.floor(inv_dim(beta_fast)), 0)
    high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
    ramp = torch.clamp(
        (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
        0, 1,
    )
    return freqs * (1 - ramp + ramp / factor), attn_factor


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            unsqueeze_dim: int = 1) -> torch.Tensor:
    """分块旋转：x·cos + rotate_half(x)·sin。"""
    half = x.shape[-1] // 2
    rotate_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return ((x * cos.unsqueeze(unsqueeze_dim))
            + (rotate_half * sin.unsqueeze(unsqueeze_dim))).to(x.dtype)


class RopeEmbedding(PosEmbedding):
    """旋转位置编码 —— 作用在注入点 3（q、k）。"""

    def __init__(self, *, head_dim: int, layer_idx: int, max_len: int,
                 theta: float = 1e6, rope_type: str = "default",
                 scaling: dict | None = None,
                 num_heads: int = 1, hidden_size: int | None = None):
        super().__init__(head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
                         num_heads=num_heads, hidden_size=hidden_size)
        if rope_type not in ROPE_INIT:
            raise ValueError(
                f"未知 rope_type={rope_type!r}，可选: {sorted(ROPE_INIT)}"
            )
        self.theta = theta
        self.rope_type = rope_type
        self.scaling = scaling

        freqs, attn_factor = ROPE_INIT[rope_type](head_dim, max_len, theta, scaling)
        angles = torch.outer(torch.arange(max_len, device=freqs.device), freqs).float()
        cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
        sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor
        # persistent=False：不进 state_dict，换编码实现不影响权重兼容
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def apply_qk(self, q, k, offset: int = 0):
        n = q.shape[1]
        self._check_range(offset, n)
        cos = self.cos[offset:offset + n]
        sin = self.sin[offset:offset + n]
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def build_rope(cfg, *, head_dim: int, layer_idx: int, max_len: int,
               num_heads: int = 1, hidden_size: int | None = None) -> RopeEmbedding:
    """RoPE 工厂 —— 供 registry 分派，从 PosEmbConfig 取参数。"""
    return RopeEmbedding(
        head_dim=head_dim,
        layer_idx=layer_idx,
        max_len=max_len,
        num_heads=num_heads,
        hidden_size=hidden_size,
        theta=cfg.theta,
        rope_type=cfg.rope_type,
        scaling=cfg.rope_scaling,
    )
