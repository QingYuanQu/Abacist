"""手写因果自注意力（教学版）。

把 Q/K/V 的注意力分数计算完全展开，方便对照 nn.MultiheadAttention 的黑盒。
后续 RoPE 版本（model_v5）会复用本函数，只在调用前对 q,k 做旋转。

q,k,v: [B, nh, T, hd]
mask : [T, T]，上三角为 -inf（当前位置看不到未来）
返回   : [B, nh, T, hd]
"""
import torch.nn.functional as F


def causal_self_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)  # [B, nh, T, T]
    scores = scores + mask                                     # 注入因果掩码
    attn = F.softmax(scores, dim=-1)
    return attn @ v
