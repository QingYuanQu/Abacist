"""可学习绝对位置编码（GPT-2 式）—— 注入点 1：加在 token embedding 上。

位置向量作为可训练参数，与 token embedding 同维度相加。
特点：
  - 有参数（max_len × hidden_size），会进 state_dict；
  - 换用本编码后，旧 checkpoint 缺 `input_pos_emb.emb.weight` 键，需重新训练；
  - 绝对编码 → 整体平移位置会改变输出（与 RoPE / ALiBi 的平移不变性相反）。
"""
import torch
from torch import nn

from .base import PosEmbedding, input_only


class LearnedPosEmbedding(PosEmbedding):
    """可学习绝对编码 —— 只实现注入点 1（apply_input）。"""

    def __init__(self, *, head_dim: int, layer_idx: int, max_len: int,
                 num_heads: int = 1, hidden_size: int | None = None,
                 init_std: float = 0.02):
        super().__init__(head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
                         num_heads=num_heads, hidden_size=hidden_size)
        self.emb = nn.Embedding(max_len, self.hidden_size)
        nn.init.normal_(self.emb.weight, std=init_std)

    def apply_input(self, hidden, offset: int = 0):
        n = hidden.shape[1]
        self._check_range(offset, n)
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"维度不匹配：可学习位置编码宽度 {self.hidden_size}，"
                f"输入隐状态宽度 {hidden.shape[-1]}"
            )
        pos = torch.arange(offset, offset + n, device=hidden.device)
        return hidden + self.emb(pos).to(hidden.dtype)


@input_only
def build_learned(cfg, *, head_dim: int, layer_idx: int, max_len: int,
                  num_heads: int = 1, hidden_size: int | None = None) -> LearnedPosEmbedding:
    """可学习绝对编码工厂 —— 供 registry 分派。"""
    return LearnedPosEmbedding(
        head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
        num_heads=num_heads, hidden_size=hidden_size,
    )
