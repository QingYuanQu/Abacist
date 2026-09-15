# -*- coding: utf-8 -*-
"""bridge/align.py —— parse 后序 ⇄ eval 注解对齐契约。

这是「纸笔（parse）」与「算盘（eval: abacus / digit）」的唯一结合点：
把 parse 的后序 token 序列逐 token 执行，push 数字入纸笔栈，calc 步调用
eval 的两种口径注解（abacus 口诀链 / digit 逐位展开），并做校验：
  - 每步 abacus 终盘 == 数学真值（独立重算）
  - 端到端 final == expected_ans

任何不一致都是数据自相矛盾（程序错误），直接抛异常。

当前聚焦加减乘（+ - ×）；÷ 尚未实现，遇到即报错。
"""
from __future__ import annotations

from typing import Callable, Sequence

from evaluate.abacus import (
    Abacus, AbacusSpec, ArithmeticComposer, RhymeResolver, OpType,
    build_addition_rhymes, build_subtraction_rhymes,
)

from .ir import AbacusAction, AbacusTrace, EvalStep

# 当前聚焦加减乘；÷ 暂不启用（除法实现未定）
SYM_TO_OP = {"+": OpType.ADD, "-": OpType.SUB, "×": OpType.MUL}

_MATH = {
    OpType.ADD: lambda a, b: a + b,
    OpType.SUB: lambda a, b: a - b,
    OpType.MUL: lambda a, b: a * b,
}

# digit 口径：只 + - ×（超位数字退化）
DigitFn = Callable[[int, int, str], str | None]


def make_default_composer() -> tuple[ArithmeticComposer, Abacus]:
    """默认装配：13 档二五珠算盘 + 加减口诀真值裁判 + 编排器。"""
    abacus = Abacus.standard(AbacusSpec(1, 4, 13, 10))
    resolver = RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes())
    return ArithmeticComposer(resolver), abacus


def align(postfix: Sequence[str],
          composer: ArithmeticComposer,
          abacus: Abacus,
          *,
          digit_fn: DigitFn | None = None,
          expected_ans: int | None = None,
          ) -> tuple[list[EvalStep], int]:
    """把 parse 的后序 token 序列逐 token 执行并注解。

    Args:
        postfix: parse 的 postfix token 序列，如 ['39','9279','+']
        composer: ArithmeticComposer（abacus 口径，空盘起算、单次二元运算）
        abacus: Abacus
        digit_fn: 可选，digit 口径逐位展开函数 (a, b, op) -> str | None
        expected_ans: 可选，端到端校验

    Returns:
        (steps, final_result)：steps 逐步注解，final_result 纸笔栈最终值。
    """
    stack: list[int] = []
    steps: list[EvalStep] = []

    for tok in postfix:
        if tok.isdigit():
            # push：数字入纸笔栈
            stack.append(int(tok))
            steps.append(EvalStep(kind="push", token=tok, stack=list(stack)))
            continue

        if tok not in SYM_TO_OP:
            raise ValueError(f"未知 token（÷ 暂不支持，聚焦加减乘）: {tok!r}")

        # calc：弹出两数，算盘执行 a op b
        op_type = SYM_TO_OP[tok]
        if len(stack) < 2:
            raise ValueError(f"后序序列非法：运算符 {tok} 前栈深不足")
        b, a = stack.pop(), stack.pop()
        stack_before = list(stack)          # 弹出 a、b 后的栈（结果未入栈，无泄漏）
        truth = _MATH[op_type](a, b)

        # abacus 口径：空盘起算、单次二元运算
        op = composer.compose(op_type, abacus, a, b)
        if op.expected_result != truth:
            raise AssertionError(
                f"算盘结果 {op.expected_result} ≠ 数学真值 {truth}（{a} {tok} {b}）")
        trace = AbacusTrace(actions=[
            AbacusAction(oral=s.rhyme.rhyme_text,
                         rod=s.base_rod_index,
                         value=s.after_state.total_value)
            for s in op.steps
        ])

        # digit 口径：逐位竖式展开（可选，注入点）
        digit_str = digit_fn(a, b, tok) if digit_fn is not None else None

        stack.append(op.expected_result)
        steps.append(EvalStep(kind="calc", token=tok, stack=list(stack),
                              stack_before=stack_before,
                              a=a, b=b, abacus=trace, digit=digit_str))

    if not stack:
        raise ValueError("空后序序列")
    final_result = stack[0]

    if expected_ans is not None and final_result != expected_ans:
        raise AssertionError(
            f"端到端不一致：{final_result} ≠ parse ANS {expected_ans}")

    return steps, final_result
