"""
vision.py — 视觉编码器（盘面图 → 视觉 token）

最小可行的多模态入口：把固定尺寸的盘面图按档切成竖条 patch，
每个 patch 用一个线性层投影到 hidden_size，得到「视觉 token」序列，
拼在文本 token 前喂给 GPT。

设计理由（针对 Abacist 的微型 GPT + RTX 2050）：
  - 盘面是逐档独立的结构化低熵信号，无需 CNN/ViT 的强表征能力；
  - 每档一个 patch → 每档一个视觉 token → 序列长度 O(档数)，注意力不爆炸；
  - 线性投影参数极少（784 → hidden_size），与微型 GPT 定位契合。

用法：
  encoder = VisionEncoder(patch_h=28, patch_w=28, hidden_size=64, in_channels=3)
  tokens = encoder(images)   # images: [B, n_cols, C, patch_h, patch_w]（C=1 灰度/C=3 彩色）
  # tokens: [B, n_cols, hidden_size]，拼到 GPT 的 token_emb 之后
"""

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """盘面图 → 逐档 patch → 线性投影 → 视觉 token。

    输入：盘面图，形状 [B, n_cols, C, patch_h, patch_w]（每档一个 patch，
          C=1 灰度 / C=3 彩色 RGB）
    输出：视觉 token，形状 [B, n_cols, hidden_size]
    """

    def __init__(self, patch_h: int = 28, patch_w: int = 28, hidden_size: int = 64,
                 in_channels: int = 1):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_flat = in_channels * patch_h * patch_w
        # 每个 patch 一个线性投影（patch 间共享权重）
        self.proj = nn.Linear(self.patch_flat, hidden_size, bias=False)
        # 位置编码（可学习的档位编码，帮助模型区分高低位）
        self.pos_emb = nn.Parameter(torch.zeros(1, 32, hidden_size))  # 最多 32 档
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, n_cols, C, patch_h, patch_w] → [B, n_cols, hidden_size]"""
        B, n_cols, C, H, W = images.shape
        x = images.reshape(B, n_cols, C * H * W)  # [B, n_cols, patch_flat]
        x = self.proj(x)                          # [B, n_cols, hidden_size]
        x = x + self.pos_emb[:, :n_cols, :]       # 加档位编码
        return x
