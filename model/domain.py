"""模型配置 —— 架构参数与位置编码，模型层的唯一配置来源。

从 `experiment/domain.py`（原顶层 domain.py）迁出（2026-09-04）：这三个类与实验域（Experiment/Trial/Data）
无关，`model_vlm` / `model_vm` 原本为了拿它们而被迫 import 整套实验概念。

依赖方向：`domain.py`（实验域）→ `model.config`（模型域）→ 无。
模型子系统（model_vlm / model_vm / model_lm）只依赖本模块，不再感知实验。

本模块**不读文件**：配置文件的解析与键名校验统一由 `experiment/config.py` 负责。
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BrainConfig:
    """脑结构 —— 模型架构参数（实验级共享，唯一模型配置来源）。

    字段命名对齐 HF 风格；config.yaml 中缺失的字段用默认值兜底。
    """
    # ---- 核心参数 ----
    hidden_size: int = 64
    num_attention_heads: int = 8
    num_hidden_layers: int = 3
    # ---- 高级参数（可省略） ----
    num_key_value_heads: int | None = None    # None = 与 num_attention_heads 一致
    intermediate_size: int | None = None      # None = 按 hidden_size·π 对齐 64 取整
    rms_norm_eps: float = 1e-6
    attn_type: str = "sdpa"                   # 注意力实现：sdpa | eager（见 model/registry.py）
    # 逐层头数（None = 每层都用 num_attention_heads，即均匀分布）。
    # 非 None 时必须由 trial 级 heads 经 with_trial_overrides 注入，长度 = num_hidden_layers；
    # head_dim = hidden_size // heads[layer]。这是"逐层头结构"的唯一载体。
    heads_per_layer: list[int] | None = None

    @property
    def eff_intermediate_size(self) -> int:
        """FFN 中间维度：未显式指定时按 hidden_size·π 对齐 64 取整。"""
        if self.intermediate_size is not None:
            return self.intermediate_size
        return math.ceil(self.hidden_size * math.pi / 64) * 64

    def heads_for_layer(self, layer_index: int) -> int:
        """返回指定层的注意力头数。

        GPT 装配与结构摘要共用此入口，保证"构造出来的"与"打印出来的"必然一致。
        """
        if self.heads_per_layer is None:
            return self.num_attention_heads
        return self.heads_per_layer[layer_index]

    def with_trial_overrides(self, heads: list[int]) -> "BrainConfig":
        """套用单个 trial 的逐层头数，返回新的 BrainConfig（唯一覆盖入口）。

        训练 / 评估 / 分析三处共用此入口，避免各自手写覆盖导致 load_state_dict
        时 q_norm/k_norm shape 漂移。
        """
        return dataclasses.replace(self, heads_per_layer=list(heads))


@dataclass
class PosEmbConfig:
    """位置编码配置（与 BrainConfig 并列）。

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
    rope_type: str = "default"             # RoPE 子变体（见 model/pos_emb/rope.py）
    max_position_embeddings: int | None = None   # None = 用 ModelConfig.max_seq_len

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
    不在任何配置文件中直接存在。
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
