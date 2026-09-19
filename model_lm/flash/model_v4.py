"""版本 4 —— 加入「可学习」的绝对位置编码（nn.Embedding，参与训练）。

与版本 3 的唯一区别：位置编码从「固定正弦表」换成「可训练的查找表」。
这是最常见的绝对位置编码写法，也是最初 model.py 用的方案。
use_pe=False 可退回版本 2。
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

    def forward(self, x, pe=None):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype) * float("-inf"), diagonal=1)
        h = self.ln1(x)
        if pe is not None:
            h = h + pe[:T]        # pe 是 [max_len, E]，时间维在 dim0
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
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, dropout=0.0, use_pe=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)   # 可学习
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout, use_pe) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        x = self.token_emb(x)
        pe = self.pos_emb.weight if self.blocks[0].use_pe else None  # [max_len, E]
        for block in self.blocks:
            x = block(x, pe=pe)
        return self.head(self.ln_f(x))
