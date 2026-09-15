# -*- coding: utf-8 -*-
"""digit/core.py —— 逐位竖式展开（eval 的 digit 口径）。

与 abacus 口径（口诀拨珠）平权的第二种求值口径：把 a op b 的二元运算
展开成「逐位 + 进位/借位」的竖式过程，供 model_lm 等生成 DIGIT 思考链。

当前只支持 + - ×（÷ 未实现）；超 max_digits 位退化为普通算式。

本包零依赖（不 import datagen / abacus），数字拆分工具 _int_to_digits 内嵌。
实现由 datagen/eval/digit.py 迁入，展开串格式逐字节兼容。
"""
from __future__ import annotations

from typing import Callable


def _int_to_digits(n: int, base: int = 10) -> list[int]:
    """数字按进制位拆分，高位在前。

    >>> _int_to_digits(56)
    [5, 6]
    >>> _int_to_digits(0)
    [0]
    >>> _int_to_digits(255, 16)
    [15, 15]
    """
    if n == 0:
        return [0]
    d = []
    while n > 0:
        d.append(n % base)
        n //= base
    d.reverse()
    return d


def _digit_add(a: int, b: int, max_digits: int = 4) -> str:
    """加法逐位展开（低位到高位）。

    >>> _digit_add(34, 56)
    '4+6=10进1;3+5+1=9'
    """
    da, db = _int_to_digits(a), _int_to_digits(b)
    if len(da) > max_digits or len(db) > max_digits:
        return f"{a}+{b}={a+b}"

    max_len = max(len(da), len(db))
    da = [0] * (max_len - len(da)) + da
    db = [0] * (max_len - len(db)) + db

    carry = 0
    partials = []
    for i in range(max_len - 1, -1, -1):
        prev_carry = carry
        s = da[i] + db[i] + carry
        if s >= 10:
            carry = 1
            partials.append(f"{da[i]}+{db[i]}{'+'+str(prev_carry) if prev_carry else ''}={s}进1")
        else:
            carry = 0
            partials.append(f"{da[i]}+{db[i]}{'+'+str(prev_carry) if prev_carry else ''}={s}")
    if carry:
        partials.append(str(carry))
    return ';'.join(partials)


def _digit_sub(a: int, b: int, max_digits: int = 4) -> str:
    """减法逐位展开（低位到高位）。a<b 时倒减法末尾加「负」。

    >>> _digit_sub(56, 34)
    '6-4=2;5-3=2'
    """
    if a < 0 or b < 0:
        return f"{a}-{b}={a-b}"

    is_negative = a < b
    if is_negative:
        a, b = b, a  # 倒减法：先算 b-a，末尾加「负」

    da, db = _int_to_digits(a), _int_to_digits(b)
    if len(da) > max_digits or len(db) > max_digits:
        res = a - b
        suffix = "负" if is_negative else ""
        return f"{a}-{b}={res}{suffix}"

    max_len = max(len(da), len(db))
    da = [0] * (max_len - len(da)) + da
    db = [0] * (max_len - len(db)) + db

    borrow = 0
    partials = []
    for i in range(max_len - 1, -1, -1):
        prev_borrow = borrow
        v = da[i] - db[i] - borrow
        if v < 0:
            borrow = 1
            v += 10
            expr = f"{da[i]}-{db[i]}{'-1' if prev_borrow else ''}={v}借1"
        else:
            borrow = 0
            expr = f"{da[i]}-{db[i]}{'-1' if prev_borrow else ''}={v}"

        if da[i] == 0 and db[i] == 0 and prev_borrow == 0:
            continue
        partials.append(expr)

    if is_negative:
        partials.append("负")
    return ';'.join(partials)


def _digit_mul(a: int, b: int, max_digits: int = 4) -> str:
    """乘法逐位展开（竖式乘法，b 按位拆分逐位与 a 相乘后累加）。

    >>> _digit_mul(34, 56)
    '34×6:4×6=24进2;3×6+2=20进2;2|204;34×50:4×5=20进2;3×5+2=17进1;1|1700;204+1700=1904'
    >>> _digit_mul(12, 34)
    '12×4:2×4=8;1×4=4|48;12×30:2×3=6;1×3=3|360;48+360=408'
    """
    if a == 0 or b == 0:
        return f"{a}×{b}=0"
    if a < 0 or b < 0:
        return f"{a}×{b}={a*b}"

    da, db = _int_to_digits(a), _int_to_digits(b)
    if len(da) > max_digits or len(db) > max_digits:
        return f"{a}×{b}={a*b}"

    b_digits = list(reversed(db))  # 低位在前
    partial_products = []

    for pos, bd in enumerate(b_digits):
        if bd == 0:
            continue
        shift = 10 ** pos
        shifted = (a * bd) * shift

        a_digits_rev = list(reversed(da))
        carry = 0
        col_steps = []
        for ad in a_digits_rev:
            prev_carry = carry
            s = ad * bd + carry
            carry = s // 10
            carry_tag = f"进{carry}" if carry else ""
            if prev_carry:
                col_steps.append(f"{ad}×{bd}+{prev_carry}={s}{carry_tag}")
            else:
                col_steps.append(f"{ad}×{bd}={s}{carry_tag}")
        if carry:
            col_steps.append(f"{carry}")

        mul_detail = ';'.join(col_steps)
        label = f"{a}×{bd}" if shift == 1 else f"{a}×{bd}{'0'*pos}"
        partial_products.append(f"{label}:{mul_detail}|{shifted}")

    if len(partial_products) == 1:
        return partial_products[0]

    pp_str = ';'.join(partial_products)
    add_parts = '+'.join(str(int(p.split('|')[-1])) for p in partial_products)
    return f"{pp_str};{add_parts}={a * b}"


def digit_step(a: int, b: int, op: str, max_digits: int = 4) -> str | None:
    """统一入口：逐位展开 a op b。

    op ∈ {'+', '-', '×'} → 对应逐位展开；'÷' 或未知 → None（未实现）。

    >>> digit_step(34, 56, '+')
    '4+6=10进1;3+5+1=9'
    >>> digit_step(34, 56, '÷') is None
    True
    """
    if op == "+":
        return _digit_add(a, b, max_digits)
    if op == "-":
        return _digit_sub(a, b, max_digits)
    if op == "×":
        return _digit_mul(a, b, max_digits)
    return None


# 对齐 bridge.align 的 DigitFn 签名：(a, b, op) -> str | None
DigitFn = Callable[[int, int, str], str | None]


def make_digit_fn(max_digits: int = 4) -> DigitFn:
    """工厂：返回 (a, b, op) -> str | None 的逐位展开函数，供 bridge.align 注入。"""
    def fn(a: int, b: int, op: str) -> str | None:
        return digit_step(a, b, op, max_digits)
    return fn
