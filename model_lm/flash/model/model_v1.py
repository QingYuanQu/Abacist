"""版本 1 —— 极度精简：用 PyTorch 黑盒注意力，无位置编码。

目的：用最少代码跑通一个能学加法的 transformer，粗略体验。
注意力是 nn.MultiheadAttention（不展开内部），没有位置编码。
"""
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_pe=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_pe = use_pe  # 本版本不使用，仅保持统一接口

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, dropout=0.0, use_pe=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout, use_pe) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
