"""位置编码契约 —— 所有位置编码机制的共同接口。

四个注入点，覆盖目前已知的全部位置编码机制：

    apply_input(...)   加在 token embedding 上，全局一次（可学习绝对、正弦绝对）
    apply_hidden(...)  加在每层隐状态上（逐层注入的绝对编码）
    apply_qk(...)      作用在 q、k 上（RoPE 及其全部变体）
    bias(...)          加在注意力分数上（ALiBi、相对位置偏置）

为什么需要 apply_input：绝对编码的语义是「与 token embedding 相加、且只加一次」，
放进 Block 里每层注入会偏离语义（且会被 layernorm 归一化掉幅度）。
因此由 GPT 在 token_emb 之后调用一次，与逐层的注入点分开。

四个方法默认均为 no-op，因此本类可直接实例化，充当 NoPE 对照组；
具体机制只需覆写自己需要的那一个方法。

不使用 ABC：no-op 本身就是合理的默认行为，抽象方法约束在这里只会增加噪音。
"""
from torch import nn


class PosEmbedding(nn.Module):
    """位置编码基类（同时就是 NoPE 实现）。

    Args:
        head_dim:     该层每个注意力头的维度（逐层可能不同，见 heads_per_layer）
        layer_idx:    层序号（供按层定制的编码使用）
        max_len:      预计算/分配的最大位置数
        num_heads:    该层注意力头数（ALiBi 等按头区分的编码需要）
        hidden_size:  隐状态维度（绝对编码需要；缺省时回退到 head_dim）
    """

    def __init__(self, *, head_dim: int, layer_idx: int, max_len: int,
                 num_heads: int = 1, hidden_size: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.max_len = max_len
        self.num_heads = num_heads
        self.hidden_size = hidden_size if hidden_size is not None else head_dim

    def _check_range(self, offset: int, n: int) -> None:
        """越界时明确报错，而不是静默取到越界/截断的编码。"""
        if offset + n > self.max_len:
            raise ValueError(
                f"位置越界：{type(self).__name__}（层 {self.layer_idx}）"
                f"预计算长度 {self.max_len}，但请求区间 [{offset}, {offset + n})。"
                f"请调大 PosEmbConfig.max_position_embeddings。"
            )

    # ---- 注入点 1：加在 token embedding 上（全局一次）----
    def apply_input(self, hidden, offset: int = 0):
        return hidden

    # ---- 注入点 2：加在每层隐状态上 ----
    def apply_hidden(self, hidden, offset: int = 0):
        return hidden

    # ---- 注入点 3：变换 q、k ----
    def apply_qk(self, q, k, offset: int = 0):
        """返回变换后的 (q, k)。默认原样返回。"""
        return q, k

    # ---- 注入点 4：注意力分数偏置 ----
    def bias(self, n_q: int, n_k: int, offset: int = 0, device=None, dtype=None):
        """返回加性偏置，形状 [n_q, n_k] 或 [n_heads, n_q, n_k]（按头区分时）。

        offset 是 **query 相对 kv 起点的偏移**（= n_k - n_q），不是绝对位置。
        用相对偏移才能保证因果掩码与相对偏置在整体平移位置时不变。
        无偏置时返回 None。
        """
        return None


def input_only(build_fn):
    """工厂装饰器：声明该编码只在输入侧（token embedding）生效。

    绝对编码只实现 apply_input，若在层内也实例化会产生
    「有参数却不参与前向」的死参数。`registry.build_pos_emb` 依据此标记，
    在层内改返回 NoPE 空壳。

    作用域信息挂在工厂上，避免与注册表形成两份真相
    （此前用 registry.INPUT_ONLY_POS 集合，加一种编码要改两处）。

    用法：
        @input_only
        def build_learned(cfg, *, head_dim, layer_idx, max_len, ...): ...
    """
    build_fn.scope = "input"
    return build_fn
