"""ALiBi（Attention with Linear Biases）—— 注入点 4：加在注意力分数上。

核心思想：不给 embedding 加任何位置信息，而是给注意力分数加一个与距离
成正比的惩罚：

    bias(h, i, j) = -slope_h · |i - j|

slope 按头几何递减，因此不同头拥有不同的「距离敏感度」——
一部分头关注局部，一部分头看得更远。

与 RoPE 的关键差异：ALiBi 不需要预计算表，也没有「最大长度」概念，
因而不存在位置越界，天然支持任意长度外推。
"""
import math

import torch

from .base import PosEmbedding


def get_slopes(n_heads: int) -> list[float]:
    """ALiBi 官方 slope 生成。

    头数为 2 的幂时是纯几何序列（公比 = 首项）；
    否则用官方的「最近幂 + 隔项插值」方案。
    注意：插值分支不保证整体单调，这是官方实现的行为。
    """
    if n_heads < 1:
        raise ValueError(f"ALiBi 需要至少一个注意力头，得到 {n_heads}")

    def _powers_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return _powers_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return (_powers_of_2(closest)
            + _powers_of_2(2 * closest)[0::2][: n_heads - closest])


class AlibiEmbedding(PosEmbedding):
    """ALiBi —— 只实现注入点 4（bias），其余注入点保持 no-op。"""

    def __init__(self, *, head_dim: int, layer_idx: int, max_len: int,
                 num_heads: int = 1, hidden_size: int | None = None):
        super().__init__(head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
                         num_heads=num_heads, hidden_size=hidden_size)
        slopes = torch.tensor(get_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    def bias(self, n_q: int, n_k: int, offset: int = 0, device=None, dtype=None):
        """返回 [n_heads, n_q, n_k]：每个头一条不同斜率的线性衰减。"""
        q_pos = torch.arange(offset, offset + n_q, device=device)[None, :, None]
        k_pos = torch.arange(n_k, device=device)[None, None, :]
        dist = (q_pos - k_pos).to(dtype).abs()               # [1, n_q, n_k]
        slopes = self.slopes.to(device=device, dtype=dtype)[:, None, None]
        return -slopes * dist                                 # [H, n_q, n_k]


def build_alibi(cfg, *, head_dim: int, layer_idx: int, max_len: int,
                num_heads: int = 1, hidden_size: int | None = None) -> AlibiEmbedding:
    """ALiBi 工厂 —— 供 registry 分派（不消费 PosEmbConfig 的 RoPE 参数）。"""
    return AlibiEmbedding(
        head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
        num_heads=num_heads, hidden_size=hidden_size,
    )
