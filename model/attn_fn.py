"""注意力纯函数后端 —— 模块外纯函数，签名对齐 HuggingFace AttentionInterface。

本模块只放「计算」本身，不含任何 nn.Module；由 model/attention.py 的
`Attention` 类在 forward 中按 registry 分派调用。

契约：
  - 注意力函数只吃已经算好的 q/k/v，位置编码在外部注入完毕后再传入；
  - 位置偏置统一由 `pos_bias` 传入（形状 [n_q, n_k] 或可广播），
    函数不关心它来自 ALiBi、相对位置还是别的什么；
  - 因果约束由 `is_causal` + `offset` 表达，KV cache 场景下 offset 是
    query 的全局起始位置，保证单步解码也能正确对齐；
  - 返回 (output, weights)，weights 允许为 None。

新增一个后端 = 加一个函数 + 在 model/registry.py 的 ATTN_REGISTRY 加一行，
模型层代码（attention.py / block.py / gpt.py）0 改动。
"""
import torch
import torch.nn.functional as F


def causal_bias(n_q: int, n_k: int, offset: int,
                device=None, dtype=None) -> torch.Tensor:
    """因果掩码的加性形式：可见为 0，不可见为 -inf。

    注意 offset 的语义是 **q 相对 kv 起点的偏移**（= n_k - n_q），
    不是绝对位置。这样全量前向与 KV cache 解码共用一套逻辑：
      - 全量：n_q == n_k → offset = 0 → 标准下三角
      - 单步解码：n_q=1、n_k=past+1 → offset=past → 全部可见（q 在最后）
    """
    q_pos = torch.arange(offset, offset + n_q, device=device)[:, None]
    k_pos = torch.arange(n_k, device=device)[None, :]
    neg_inf = torch.full((), float("-inf"), device=device, dtype=dtype)
    return torch.where(k_pos <= q_pos, torch.zeros((), device=device, dtype=dtype), neg_inf)


def sdpa_attention(module, query, key, value, attention_mask=None, *,
                   scaling=None, dropout=0.0, is_causal=True,
                   pos_bias=None, offset=0, **kwargs):
    """PyTorch SDPA（自动选择 flash / mem-efficient / math 后端）。

    无偏置时走 is_causal 快路径；有偏置时把因果掩码并入 attn_mask
    （SDPA 不允许 is_causal 与 attn_mask 同时给出）。
    """
    mask = pos_bias
    if attention_mask is not None:
        mask = attention_mask if mask is None else mask + attention_mask

    n_q, n_k = query.shape[-2], key.shape[-2]
    # is_causal 快路径只在方阵（q、k 等长）时启用：
    # SDPA 的 is_causal 按「左上对齐」实现，n_q < n_k（KV cache 解码）时
    # 会错误地把最近的 kv 屏蔽掉，因此必须改用显式、按全局位置对齐的掩码。
    if mask is None and is_causal and n_q == n_k:
        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=dropout if module.training else 0.0,
            is_causal=True, scale=scaling,
        )
    else:
        if is_causal:
            cb = causal_bias(n_q, n_k, offset, query.device, query.dtype)
            mask = cb if mask is None else mask + cb
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False, scale=scaling,
        )
    return out, None


def eager_attention(module, query, key, value, attention_mask=None, *,
                    scaling=None, dropout=0.0, is_causal=True,
                    pos_bias=None, offset=0, **kwargs):
    """显式实现 —— 物化完整的注意力矩阵，返回注意力权重。

    用途：调试、需要 attention weights、以及不支持 SDPA 的设备。
    """
    scores = (query @ key.transpose(-2, -1)) * (scaling or 1.0)
    if pos_bias is not None:
        scores = scores + pos_bias
    if attention_mask is not None:
        scores = scores + attention_mask
    if is_causal:
        scores = scores + causal_bias(query.shape[-2], key.shape[-2], offset,
                                      scores.device, scores.dtype)

    probs = F.softmax(scores.float(), dim=-1).type_as(scores)
    if dropout and module.training:
        probs = F.dropout(probs, p=dropout)
    return probs @ value, probs
