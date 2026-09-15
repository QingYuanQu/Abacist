"""模型配置 —— 架构参数与位置编码，模型层的唯一配置来源。

从顶层 `config.py` 迁出（2026-09-04）：这三个类与课程域（Experiment/Trial/Material）
无关，`model_vlm` / `model_vm` 原本为了拿它们而被迫 import 整套课程概念。

依赖方向：`config.py`（课程域）→ `model.config`（模型域）→ 无。
模型子系统（model_vlm / model_vm / model_lm）只依赖本模块，不再感知课程。
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BrainConfig:
    """脑结构 —— 模型架构参数（课程级共享，唯一模型配置来源）。

    字段命名对齐 HF 风格；brain.json 中缺失的字段用默认值兜底。
    """
    # ---- 核心参数（brain.json） ----
    hidden_size: int = 64
    num_attention_heads: int = 8
    num_hidden_layers: int = 3
    # ---- 高级参数（可省略） ----
    num_key_value_heads: int | None = None    # None = 与 num_attention_heads 一致
    intermediate_size: int | None = None      # None = 按 hidden_size·π 对齐 64 取整
    rms_norm_eps: float = 1e-6
    attention_heads_pattern: str = "regular"
    attn_type: str = "sdpa"                   # 注意力实现：sdpa | eager（见 model/registry.py）
    # 显式每层头数（None=回退 attention_heads_pattern 魔法）。供 experiment 型 exp
    # 经 material → head_patterns.json 逐课覆盖；head_dim = hidden_size // heads。
    heads_per_layer: list[int] | None = None

    @classmethod
    def from_json(cls, data: dict) -> "BrainConfig":
        """从 brain.json 字典构造。"""
        return cls(**data)

    @property
    def eff_intermediate_size(self) -> int:
        """FFN 中间维度：未显式指定时按 hidden_size·π 对齐 64 取整。"""
        if self.intermediate_size is not None:
            return self.intermediate_size
        return math.ceil(self.hidden_size * math.pi / 64) * 64

    @staticmethod
    def calculate_attention_heads(
        layer_index: int,  # 当前层索引（0 起）
        n: int,  # 中心层索引，调用方传 num_layers // 2
        pattern: str,  # 头数分布模式: focus_expansion/expansion_focus/balanced_progressive/regular
        num_attention_heads: int,  # 普通模式下指定头数
    ) -> int:
        """计算指定层的注意力头数（可变头注意力模式）

        >>> heads = [BrainConfig.calculate_attention_heads(l, 2, "regular", 8) for l in range(5)]
        >>> print(heads)
        """
        if pattern == "focus_expansion":
            # 聚焦收缩模式: 中间层注意力头多 (1,2,4,8,16,8,4,2,1)
            return 2 ** (n - abs(layer_index - n))
        elif pattern == "expansion_focus":
            # 聚焦发散模式: 两端层注意力头多 (16,8,4,2,1,2,4,8,16)
            return 2 ** (abs(layer_index - n))
        elif pattern == "balanced_progressive":
            # 平衡渐进模式
            progression = [16, 16, 8, 8, 8, 4, 4, 2, 2]
            return progression[layer_index] if layer_index < len(progression) else 2
        elif pattern == "regular":
            return num_attention_heads
        raise ValueError(
            f"未知 attention_heads_pattern={pattern!r}，可选: "
            "focus_expansion / expansion_focus / balanced_progressive / regular"
        )

    def heads_for_layer(self, layer_index: int, n: int) -> int:
        """返回指定层的注意力头数：显式 `heads_per_layer` 优先，否则回退 pattern 魔法。

        这是 GPT 装配/摘要唯一应调用的取头数入口，保证构造与打印一致。
        """
        if self.heads_per_layer is not None:
            return self.heads_per_layer[layer_index]
        return self.calculate_attention_heads(
            layer_index, n, self.attention_heads_pattern, self.num_attention_heads)

    def with_trial_overrides(self, material) -> "BrainConfig":
        """套用单课 material 对头结构的覆盖，返回新的 BrainConfig（唯一覆盖入口）。

        训练 / 评估 / 分析三处共用此入口，避免各自手写覆盖导致 load_state_dict
        时 q_norm/k_norm shape 漂移。material 无 heads_per_layer 覆盖时原样返回 self。
        """
        heads = getattr(material, "heads_per_layer", None)
        if heads is None:
            return self
        return dataclasses.replace(self, heads_per_layer=list(heads))


@dataclass
class PosEmbConfig:
    """位置编码配置（pos_emb.json，与 BrainConfig 并列）。

    两级字段，语义不同，不可合并：
        pos_type  —— 编码「机制」：rope | none（后续可加 alibi / learned / sinusoidal）
        rope_type —— RoPE「机制内部」的子变体：default | yarn | ...
                     仅当 pos_type=rope 时有意义。

    max_position_embeddings 把「预计算长度」与「数据集最长序列」解耦：
    未指定时回退到 ModelConfig.max_seq_len；做长序列外推实验时显式调大。
    """
    theta: float = 1e6                     # 旋转基数 rope_base
    scaling: dict | None = None            # YaRN 推理外推配置（None=不启用）
    pos_type: str = "rope"                 # 编码机制（见 model/registry.py）
    rope_type: str = "default"             # RoPE 子变体（见 model/pos_rope.py）
    max_position_embeddings: int | None = None   # None = 用 ModelConfig.max_seq_len

    @classmethod
    def from_json(cls, data: dict | None) -> "PosEmbConfig":
        """从 pos_emb.json 字典构造；缺失用默认值。"""
        return cls(**data) if data else cls()

    @property
    def rope_scaling(self) -> dict | None:
        """YaRN 推理外推配置（仅 scaling 非空且 enabled 时生效）。

        纯函数，无副作用 —— 打印统一交给 GPT 的结构摘要。
        """
        if not self.scaling or not self.scaling.get("enabled"):
            return None
        s = self.scaling
        return {
            "type": s.get("type", "yarn"),
            "factor": s.get("factor", 8),
            "original_max_position_embeddings": s.get("original_max_position_embeddings", 128),
            "beta_fast": s.get("beta_fast", 4),
            "beta_slow": s.get("beta_slow", 1),
            "attention_factor": s.get("attention_factor", 1.0),
        }


@dataclass
class ModelConfig:
    """合成模型配置 —— 传递给 GPT 的唯一配置对象。

    由 BrainConfig + PosEmbConfig + 词表信息 + dropout 合成而来，
    不在任何 JSON 文件中直接存在。
    """
    brain: BrainConfig
    pos_emb: PosEmbConfig
    vocab_size: int
    pad_id: int = 0
    max_seq_len: int = 0
    dropout: float = 0.0

    @classmethod
    def from_sources(
        cls,
        brain: BrainConfig,
        pos_emb: PosEmbConfig | None,
        vocab_data: Any,
        dropout: float = 0.0,
    ) -> "ModelConfig":
        """合成 ModelConfig：完成 None 兜底与必需字段校验。"""
        if not vocab_data or not vocab_data.max_seq_len:
            raise ValueError("词表缺少 max_seq_len，请重建词表")
        return cls(
            brain=brain,
            pos_emb=pos_emb or PosEmbConfig(),
            vocab_size=vocab_data.vocab_size,
            pad_id=vocab_data.pad_id,
            max_seq_len=vocab_data.max_seq_len,
            dropout=dropout,
        )
