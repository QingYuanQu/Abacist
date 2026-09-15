"""正弦绝对位置编码（Vaswani 2017）—— 注入点 1：加在 token embedding 上。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

特点：
  - 无参数（固定的三角函数表，注册为 persistent=False 的 buffer，不进 state_dict）；
  - 绝对编码 → 平移位置会改变输出；
  - 与可学习编码的区别只在于「表是固定的还是学出来的」，
    因此它是检验「绝对 vs 相对」这一变量的干净对照组。
"""
import math

import torch

from .base import PosEmbedding, input_only


class SinusoidalEmbedding(PosEmbedding):
    """正弦绝对编码 —— 只实现注入点 1（apply_input）。"""

    def __init__(self, *, head_dim: int, layer_idx: int, max_len: int,
                 num_heads: int = 1, hidden_size: int | None = None,
                 base: float = 10000.0):
        super().__init__(head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
                         num_heads=num_heads, hidden_size=hidden_size)
        self.base = base
        self.register_buffer("pe", self._build_pe(max_len, self.hidden_size, base),
                             persistent=False)

    @staticmethod
    def _build_pe(max_len: int, dim: int, base: float) -> torch.Tensor:
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # 只用 ceil(dim/2) 个频率；dim 为奇数时 cos 列比 sin 列少一个
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        return pe

    def apply_input(self, hidden, offset: int = 0):
        n = hidden.shape[1]
        self._check_range(offset, n)
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"维度不匹配：正弦位置编码宽度 {self.hidden_size}，"
                f"输入隐状态宽度 {hidden.shape[-1]}"
            )
        return hidden + self.pe[offset:offset + n].to(hidden.dtype)


@input_only
def build_sinusoidal(cfg, *, head_dim: int, layer_idx: int, max_len: int,
                     num_heads: int = 1,
                     hidden_size: int | None = None) -> SinusoidalEmbedding:
    """正弦绝对编码工厂 —— 供 registry 分派。"""
    return SinusoidalEmbedding(
        head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
        num_heads=num_heads, hidden_size=hidden_size,
    )
