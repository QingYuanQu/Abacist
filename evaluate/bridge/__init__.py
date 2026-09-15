# -*- coding: utf-8 -*-
"""bridge —— 底层中间表示（IR）契约与对齐层。

职责：把 parse 的「纸笔」后序与 eval 的「算盘」两种口径（abacus / digit）
对齐成可落盘的富结构 IR（ExpressionInstance），供所有模型离线投影。

依赖方向：bridge → abacus（内核），零 datagen 依赖。
"""
from .ir import (
    AbacusAction, AbacusTrace, EvalStep, ExpressionInstance,
    to_json, build_instance,
)
from .align import SYM_TO_OP, align, make_default_composer

__all__ = [
    "AbacusAction", "AbacusTrace", "EvalStep", "ExpressionInstance",
    "to_json", "build_instance",
    "SYM_TO_OP", "align", "make_default_composer",
    "emit_ir",
]


def __getattr__(name):
    # emit_ir 惰性导入：避免 `python -m bridge.generate` 时 eager import 造成的
    # runpy 双重导入 RuntimeWarning（generate 的 __main__ 是 CLI 入口）。
    if name == "emit_ir":
        from .generate import emit_ir
        return emit_ir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
