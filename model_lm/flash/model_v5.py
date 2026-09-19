"""版本 5 —— 演进到 RoPE（旋转位置编码）。

关键跃迁：位置编码不再「加到 embedding 上」（绝对编码），
而是「在注意力内部旋转 Q、K」（相对编码）。这要求注意力是手写的
（版本 2 打好的底）—— nn.MultiheadAttention 没有注入 q,k 旋转的钩子。

特点：零参数；长度外推（预计算长度不是硬上限，可加大）；
天然带相对位置信息。use_pe=False 退回版本 2。
"""
import torch
from torch import nn

from attn import causal_self_attention


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_pe=False, rope_base=10000.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_pe = use_pe
        self.base = rope_base

    def _rope(self, q, k):
        # q,k: [B, nh, T, hd]；把每头最后一维按 (i, i+hd/2) 配对旋转
        hd = self.head_dim
        inv_freq = 1.0 / (self.base ** (torch.arange(0, hd, 2, device=q.device).float() / hd))
        pos = torch.arange(q.size(2), device=q.device).float()
        freqs = torch.outer(pos, inv_freq)        # [T, hd/2]
        # 配对式 RoPE：cos/sin 保持 [T, hd/2]，分别乘前半 q1 与后半 q2，再拼回 hd
        cos = torch.cos(freqs)[None, None, :, :]   # [1, 1, T, hd/2]
        sin = torch.sin(freqs)[None, None, :, :]
        q1, q2 = q[..., :hd // 2], q[..., hd // 2:]
        k1, k2 = k[..., :hd // 2], k[..., hd // 2:]
        q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q, k

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype) * float("-inf"), diagonal=1)
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        nh, hd = self.num_heads, self.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        if self.use_pe:
            q, k = self._rope(q, k)
        out = causal_self_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, T, nh * hd)
        x = x + self.dropout(self.proj(out))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, dropout=0.0, use_pe=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout, use_pe) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
