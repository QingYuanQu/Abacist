"""model 包 —— 对外收口导出。

包外一律从本文件取公共 API（`from model import GPT` 等），
不写 `from model.gpt import ...`，这样内部模块再拆分重组时外部无需改动。

依赖方向单向无环：
    config / norm / attn_fn / pos_emb ← registry
    attention ← config, norm, registry
    block     ← attention, norm, config, registry
    gpt       ← block, norm, config, registry
"""
from model.attention import Attention, repeat_kv
from model.block import Block, FeedForward
from model.gpt import GPT

__all__ = ["Attention", "Block", "FeedForward", "GPT", "repeat_kv"]
