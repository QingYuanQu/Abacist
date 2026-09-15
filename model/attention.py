"""多头注意力 —— 只做投影与调度，计算交给 registry 分派的纯函数（model/attn_fn.py）。

位置编码注入点：
    注入点 3（apply_qk）：作用在 q、k 上，必须在拼接 KV cache 之前
                          （缓存里的 k 已在上一步旋转过，用的是绝对位置）
    注入点 4（bias）    ：作用在注意力分数上，与因果掩码合并
（另两个注入点 apply_input / apply_hidden 分别在 gpt.py / block.py）
"""
import math

import torch
from torch import nn

from model.config import ModelConfig
from model.norm import RMSNorm
from model.registry import get_attention


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    """多头注意力 —— 只做投影与调度，计算交给 registry 分派的纯函数。"""

    def __init__(self, config: ModelConfig, num_heads: int, layer_idx: int):
        super().__init__()
        brain = config.brain
        self.layer_idx = layer_idx
        self.n_local_heads = num_heads
        if brain.num_key_value_heads is not None and brain.num_key_value_heads != brain.num_attention_heads:
            self.n_local_kv_heads = brain.num_key_value_heads
        else:
            self.n_local_kv_heads = self.n_local_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = brain.hidden_size // self.n_local_heads
        self.is_causal = True
        self.q_proj = nn.Linear(brain.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(brain.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(brain.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, brain.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=brain.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=brain.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.attn_fn = get_attention(brain.attn_type)

    def forward(self, x, offset=0, pos_emb=None, past_key_value=None,
                use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 注入点 3（apply_qk）：作用在 q、k 上
        # 必须在拼接缓存之前 —— 缓存里的 k 已在上一步旋转过，用的是绝对位置
        if pos_emb is not None:
            xq, xk = pos_emb.apply_qk(xq, xk, offset)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 注入点 4（bias）：作用在注意力分数上
        # 这里用「q 相对 kv 起点的偏移」而非绝对位置：绝对位置对相对偏置
        # （ALiBi）无意义，且会在全量前向 + pos_offset≠0 时错误地放开因果掩码。
        q_offset = xk.shape[-2] - xq.shape[-2]
        pos_bias = None
        if pos_emb is not None:
            pos_bias = pos_emb.bias(xq.shape[-2], xk.shape[-2], q_offset,
                                    xq.device, xq.dtype)

        output, _ = self.attn_fn(
            self, xq, xk, xv, attention_mask,
            scaling=1.0 / math.sqrt(self.head_dim),
            dropout=self.dropout,
            is_causal=self.is_causal,
            pos_bias=pos_bias,
            offset=q_offset,
        )
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), past_kv
