"""Transformer 块 —— 持有本层专属的注意力与位置编码实例。

位置编码注入点 2（apply_hidden）：加在每层隐状态上，进入注意力之前。
（另三个注入点见 model/attention.py / model/gpt.py）
"""
import torch
from torch import nn

from model.attention import Attention
from model.config import ModelConfig
from model.norm import RMSNorm
from model.registry import build_pos_emb


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate_size = config.brain.eff_intermediate_size
        self.gate_proj = nn.Linear(config.brain.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.brain.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.brain.hidden_size, intermediate_size, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Transformer 块 —— 持有本层专属的位置编码实例。"""

    def __init__(self, layer_id: int, config: ModelConfig, num_attention_heads: int):
        super().__init__()
        self.self_attn = Attention(config, num_heads=num_attention_heads, layer_idx=layer_id)
        self.input_layernorm = RMSNorm(config.brain.hidden_size, eps=config.brain.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.brain.hidden_size, eps=config.brain.rms_norm_eps)
        self.mlp = FeedForward(config)
        # 每层独立实例：attention_heads_pattern 会让各层 head_dim 不同
        self.pos_emb = build_pos_emb(
            config,
            head_dim=self.self_attn.head_dim,
            layer_idx=layer_id,
            num_heads=self.self_attn.n_local_heads,
            hidden_size=config.brain.hidden_size,
        )

    def forward(self, hidden_states, offset=0, past_key_value=None,
                use_cache=False, attention_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.pos_emb.apply_hidden(h, offset)          # 注入点 2
        h, present_key_value = self.self_attn(
            h, offset=offset, pos_emb=self.pos_emb,
            past_key_value=past_key_value, use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + h
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
