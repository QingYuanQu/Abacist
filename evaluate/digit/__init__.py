# -*- coding: utf-8 -*-
"""digit —— eval 的 digit 口径（逐位竖式展开）。

与 abacus（口诀拨珠）平权的求值口径，当前只支持 + - ×（÷ 未实现）。
"""
from .core import (
    digit_step, make_digit_fn, DigitFn,
    _digit_add, _digit_sub, _digit_mul,
)

__all__ = [
    "digit_step", "make_digit_fn", "DigitFn",
    "_digit_add", "_digit_sub", "_digit_mul",
]
