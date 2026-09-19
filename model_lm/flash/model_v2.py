"""版本 2 —— 手写注意力，无位置编码（注意力支点）。

把版本 1 的 nn.MultiheadAttention 黑盒拆开，逐行看清
Q/K/V 投影 → 缩放点积 → 因果掩码 → softmax → 加权求和。
没有位置编码：靠因果掩码本身提供的「左侧 token 数」弱位置信号，
在「纯加法、可交换」的数据集上通常仍能学会（见 data/data.txt）。
"""
import torch
from torch import nn

from attn import causal_self_attention


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_pe=False):
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

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype) * float("-inf"), diagonal=1)
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        nh, hd = self.num_heads, self.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        out = causal_self_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, T, nh * hd)
        x = x + self.dropout(self.proj(out))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, dropout=0.0, use_pe=False):
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
