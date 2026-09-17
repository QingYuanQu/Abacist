"""GPT 装配层 —— 架构参数全部来自 ModelConfig。

本模块只负责装配：注意力实现（model/attn_fn.py）与位置编码实现
（model/pos_emb 子包）都从 `model.registry` 分派，因此新增一种实现
不需要改动本文件的任何一行 —— 这是本项目的架构验收标准。

位置编码共四个注入点（契约见 model/pos_emb/base.py）：
    注入点 1 apply_input 在 GPT.forward 内，作用于 token embedding，全局只加一次
    注入点 2 apply_hidden 在 Block.forward 内，进入注意力之前
    注入点 3 apply_qk 在 Attention.forward 内，作用于 q、k，KV cache 拼接之前
    注入点 4 bias 在注意力纯函数内，与因果掩码合并
"""
import torch
from torch import nn

from model.block import Block
from model.config import ModelConfig
from model.norm import RMSNorm
from model.registry import build_pos_emb


class GPT(nn.Module):
    """语言模型 —— 架构参数全部来自 ModelConfig（由 brain/pos_emb/词表合成）。

    Args:
        config: 合成后的模型配置
        verbose: 构造时是否打印结构摘要。单模型调试时有用；
                 批量实验（sweep）建议设 False，否则每个变体刷一屏。
                 需要把摘要写进日志时用 `model.summary()` 取字符串。
    """

    def __init__(self, config: ModelConfig, verbose: bool = True):
        super().__init__()
        self.config = config
        embed_dim = config.brain.hidden_size
        num_layers = config.brain.num_hidden_layers
        self.max_seq_len = config.max_seq_len
        self.token_emb = nn.Embedding(config.vocab_size, embed_dim, padding_idx=config.pad_id)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList()
        # 逐层构造：头数（进而 head_dim）随层变化，位置编码也随之逐层独立
        for l in range(num_layers):
            self.blocks.append(Block(l, config, config.brain.heads_for_layer(l)))

        # 输入侧位置编码：绝对编码（learned / sinusoidal）加在 token embedding 上，
        # 全局只加一次；相对编码（rope / alibi）对此是 no-op。
        self.input_pos_emb = build_pos_emb(
            config, head_dim=embed_dim, layer_idx=-1,
            num_heads=1, hidden_size=embed_dim, scope="input",
        )

        self.norm = RMSNorm(embed_dim, eps=config.brain.rms_norm_eps)
        self.head = nn.Linear(embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying
        if verbose:
            print(self.summary())

    def summary(self) -> str:
        """返回模型结构摘要字符串（供打印或写入实验日志）。"""
        config = self.config
        brain = config.brain
        pos = config.pos_emb
        heads = [brain.heads_for_layer(l) for l in range(brain.num_hidden_layers)]
        total = sum(p.numel() for p in self.parameters())
        rope_len = pos.max_position_embeddings or config.max_seq_len

        return "\n".join([
            "=" * 64,
            f"Model GPT | vocab={config.vocab_size}  params={total:,}",
            f"  hidden_size          : {brain.hidden_size}",
            f"  num_hidden_layers    : {brain.num_hidden_layers}",
            f"  attn_type            : {brain.attn_type}",
            f"  pos_type / rope_type : {pos.pos_type} / {pos.rope_type}",
            f"  max_position_embeds  : {rope_len}"
            f"{' (显式)' if pos.max_position_embeddings else ' (取自 max_seq_len)'}",
            f"  rope_theta           : {pos.theta}",
            f"  rope_scaling         : {pos.rope_scaling}",
            f"  heads_per_layer      : {heads}",
            f"  head_dims            : {[brain.hidden_size // h for h in heads]}",
            "=" * 64,
        ])

    def _start_pos(self, past_kvs, pos_offset: int) -> int:
        """缓存已有时从缓存长度续，否则用传入的偏移。"""
        if past_kvs is not None and past_kvs[0] is not None:
            return past_kvs[0][0].shape[1]
        return pos_offset

    def forward_embeddings(self, hidden_states, past_kvs=None, use_cache=False,
                           pos_offset=0, attention_mask=None):
        """从 embedding 直接前向（视觉 token 已投影到 hidden，跳过 token_emb）。

        供多模态拼接使用：视觉 token 与文本 embedding 拼接后传入。
        """
        start_pos = self._start_pos(past_kvs, pos_offset)
        hidden_states = self.input_pos_emb.apply_input(hidden_states, start_pos)

        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            hidden_states, layer_cache = block(
                hidden_states, offset=start_pos,
                past_key_value=past_kvs[i] if past_kvs is not None else None,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            if use_cache:
                new_kvs.append(layer_cache)
        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)
        return logits, new_kvs

    def forward(self, x, past_kvs=None, use_cache=False, pos_offset=0,
                attention_mask=None):
        # token embedding + dropout 后，其余逻辑（输入侧位置编码、逐层前向、
        # 归一化、输出头）与 forward_embeddings 完全一致，直接委托避免重复。
        hidden_states = self.dropout(self.token_emb(x))
        return self.forward_embeddings(
            hidden_states, past_kvs=past_kvs, use_cache=use_cache,
            pos_offset=pos_offset, attention_mask=attention_mask,
        )
