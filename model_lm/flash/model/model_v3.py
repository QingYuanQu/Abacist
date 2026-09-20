"""版本 3 —— 加入「固定、不可学习」的绝对位置编码（正弦, Vaswani 式）。

与版本 2 仅差一处：token embedding 上叠加一个用 register_buffer 固定的正弦表，
requires_grad=False → 不参与训练。用 use_pe=False 关掉，退回版本 2，
对比「有无固定位置信号」的差异。
注意：这是查表式编码，长度被 max_len 卡死（超长会复用末行，是错误信号）。
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
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout, use_pe) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_len = max_len
        # 固定正弦位置编码：register_buffer + 永不参与梯度
        self.register_buffer("pe", self._build_sinusoidal(max_len, embed_dim), persistent=False)

    @staticmethod
    def _build_sinusoidal(max_len, embed_dim):
        pos = torch.arange(max_len).unsqueeze(1).float()                    # [T, 1]
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / embed_dim))  # [E/2]
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)   # 偶数列 sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数列 cos
        return pe                            # [max_len, embed_dim]

    def forward(self, x):
        x = self.token_emb(x)
        pe = self.pe if self.blocks[0].use_pe else None
        for block in self.blocks:
            x = block(x, pe=pe)
        return self.head(self.ln_f(x))
