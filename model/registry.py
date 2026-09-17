"""组件注册表 —— 注意力与位置编码的唯一分派点。

扩展方式（这是本项目的架构验收标准）：
    新增一种实现 = 加一个实现文件 + 在本文件的字典里加一行。
    模型装配代码（gpt.py）不因此产生任何改动。

依赖方向单向无环：
    pos_emb/(子包：base ← rope / alibi / learned / sinusoidal)
    attn_fn(纯函数)
    registry ← {pos_emb, attn_fn}
    attention ← registry
    block     ← attention
    gpt       ← block
"""
from collections.abc import Callable

from model.attn_fn import sdpa_attention, eager_attention
from model.config import ModelConfig

from model.pos_emb import (PosEmbedding, build_alibi, build_learned,
                           build_none, build_rope, build_sinusoidal)

# name → 注意力纯函数，签名见 model/attn_fn.py
ATTN_REGISTRY: dict[str, Callable] = {
    "sdpa": sdpa_attention,
    "eager": eager_attention,
}

# name → 位置编码工厂
# 签名统一为 (cfg: PosEmbConfig, *, head_dim, layer_idx, max_len, num_heads, hidden_size)
POS_EMB_REGISTRY: dict[str, Callable] = {
    "rope": build_rope,
    "alibi": build_alibi,
    "learned": build_learned,
    "sinusoidal": build_sinusoidal,
    "none": build_none,
}


def get_attention(name: str):
    """按名字取注意力函数。"""
    if name not in ATTN_REGISTRY:
        raise ValueError(
            f"未知注意力实现 {name!r}，可选: {sorted(ATTN_REGISTRY)}"
        )
    return ATTN_REGISTRY[name]


def build_pos_emb(config: ModelConfig, *,
                  head_dim: int,
                  layer_idx: int,
                  num_heads: int = 1,
                  hidden_size: int | None = None,
                  scope: str = "layer") -> PosEmbedding:
    """构造位置编码模块。

    scope:
        "layer" —— 层内实例，服务于 apply_qk / bias / apply_hidden
        "input" —— 输入侧实例，服务于 apply_input（绝对编码加在 token embedding 上）

    编码的生效位置与 scope 不一致时返回 NoPE 空壳，避免死参数与无谓开销。
    生效位置由工厂自带的 `@input_only` 声明（见 model/pos_emb/base.py），
    注册表之外没有第二份清单。

    逐层调用：trial 级 heads_per_layer 会让每层头数不同 → head_dim 不同，
    因此每层持有独立实例。

    预计算长度取 `PosEmbConfig.max_position_embeddings`，未指定时回退到
    `ModelConfig.max_seq_len`（数据集最长序列）—— 这样训练长度与外推能力解耦。

    num_heads / hidden_size 供按头区分的编码（ALiBi）与绝对编码
    （learned / sinusoidal）使用；只用 q、k 的编码（RoPE）忽略它们。
    """
    cfg = config.pos_emb
    if cfg.pos_type not in POS_EMB_REGISTRY:
        raise ValueError(
            f"未知位置编码 {cfg.pos_type!r}，可选: {sorted(POS_EMB_REGISTRY)}"
        )
    max_len = cfg.max_position_embeddings or config.max_seq_len
    if max_len <= 0:
        raise ValueError(
            "位置编码长度非正：ModelConfig.max_seq_len 与 "
            "PosEmbConfig.max_position_embeddings 均未设置"
        )
    common = dict(head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
                  num_heads=num_heads,
                  hidden_size=hidden_size if hidden_size is not None else head_dim)
    factory = POS_EMB_REGISTRY[cfg.pos_type]
    # 作用域由工厂自带（@input_only 声明），未声明者默认逐层生效
    if getattr(factory, "scope", "layer") != scope:
        return build_none(cfg, **common)
    return factory(cfg, **common)
