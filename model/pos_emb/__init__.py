"""位置编码子包 —— 契约 + 全部实现。

目录：
    base.py        契约 PosEmbedding（四个注入点，默认 no-op）
    rope.py        RoPE 及其外推变体（rope_type 路由表）
    alibi.py       ALiBi（按头的线性距离衰减）
    learned.py     可学习绝对编码
    sinusoidal.py  正弦绝对编码
    none.py        NoPE 对照组（消融基线）

导入约定（与 evaluate/abacus/domain 一致，防循环）：
  - 包内模块一律**相对导入**（`from .base import PosEmbedding`）。
  - 包外一律从包根取（`from model.pos_emb import RopeEmbedding`）；
    不写 `from model.pos_emb.rope import ...`，这样内部文件再拆分重组时
    外部代码无需改动。

新增一种编码 = 加一个实现模块 + 在本文件导出 + 在 model/registry.py 注册。
"""
from .alibi import AlibiEmbedding, build_alibi, get_slopes
from .base import PosEmbedding, input_only
from .learned import LearnedPosEmbedding, build_learned
from .none import build_none
from .rope import ROPE_INIT, RopeEmbedding, build_rope, register_rope_type
from .sinusoidal import SinusoidalEmbedding, build_sinusoidal

__all__ = [
    "AlibiEmbedding",
    "LearnedPosEmbedding",
    "PosEmbedding",
    "ROPE_INIT",
    "RopeEmbedding",
    "SinusoidalEmbedding",
    "build_alibi",
    "build_learned",
    "build_none",
    "build_rope",
    "build_sinusoidal",
    "get_slopes",
    "input_only",
    "register_rope_type",
]
