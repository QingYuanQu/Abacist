"""NoPE（无位置编码）对照组 —— 四个注入点全部 no-op。

为什么需要它：它是消融实验的基线。没有基线就无法区分两件事——

    「RoPE / ALiBi 带来了提升」
    与
    「这个任务根本不在意位置信息」

后者并非不可能：本项目序列长度普遍在 20~160，位置信号可能很弱，
此时任何编码都会打平，NoPE 会直接把这一点暴露出来。

实现说明：NoPE 不需要新类。契约基类 PosEmbedding 的四个注入点默认就是
no-op，直接实例化基类即可，因此本模块只有一个工厂函数。

它还有第二个用途——**作用域空壳**：绝对编码（learned / sinusoidal）只在
输入侧生效，`build_pos_emb(scope="layer")` 会用 NoPE 实例占位，
避免层内出现「有参数却不参与前向」的死参数。
"""
from .base import PosEmbedding


def build_none(cfg, *, head_dim: int, layer_idx: int, max_len: int,
               num_heads: int = 1, hidden_size: int | None = None) -> PosEmbedding:
    """NoPE 工厂 —— 与其它编码签名一致，供 registry 统一分派。"""
    return PosEmbedding(head_dim=head_dim, layer_idx=layer_idx, max_len=max_len,
                        num_heads=num_heads, hidden_size=hidden_size)
