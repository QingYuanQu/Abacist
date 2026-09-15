"""交互演示：输入算式 → 算盘拨珠动画（GIF）+ 口诀逐步展示。

用法：
    python tools/demo.py "3+5×2"                 # 一步命令，GIF 输出到 output/demo/
    python tools/demo.py                         # 无参数 → 交互模式（q 退出）
    python tools/demo.py "12÷4" --model data/full_h128_60e_bal.pth
                                                  # VLM 模型看盘说口诀，控制台对比真值

说明：
    - 每个运算生成一个 GIF：左侧显示算式与当前口诀，珠子带手拨动画。
    - 仅支持整数运算；÷ 要求整除（与训练数据分布一致）。
"""
import os
import re
import sys
from pathlib import Path

# anaconda 下 PIL/imageio 与 torch 各带一份 OpenMP 运行时，先声明放行
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluate.abacus import (                        # noqa: E402
    Abacus, ArithmeticComposer, RhymeResolver, OpType,
    build_addition_rhymes, build_subtraction_rhymes,
)
from evaluate.abacus.render.image import ImageRenderer  # noqa: E402
from evaluate.abacus.render.style import Style  # noqa: E402

SYM_TO_OPTYPE = {'+': OpType.ADD, '-': OpType.SUB,
                 '×': OpType.MUL, '÷': OpType.DIV}
_PRECEDENCE = {'+': 1, '-': 1, '×': 2, '÷': 2}
_TYPE_SYM = {v: k for k, v in SYM_TO_OPTYPE.items()}


def _normalize(expr: str) -> str:
    """全角/别名符号归一。"""
    return (expr.replace('（', '(').replace('）', ')')
                .replace('*', '×').replace('/', '÷')
                .replace('x', '×').replace('X', '×')
                .replace('−', '-').replace('－', '-')
                .replace('＋', '+').replace('＝', '').replace('=', ''))


def infix_to_postfix(expr: str) -> list[str]:
    """中缀算式 → 后缀 token 序列（shunting-yard）。

    数字可多位；抛 ValueError 表示语法错误。
    """
    expr = _normalize(expr).replace(' ', '')
    if not expr:
        raise ValueError('空算式')
    tokens = re.findall(r'\d+|[+\-×÷()]', expr)
    if ''.join(tokens) != expr:
        raise ValueError(f'无法识别的字符: {expr!r}')

    out: list[str] = []
    ops: list[str] = []
    prev = None                                    # 上一个 token（判断负号/语法）
    for t in tokens:
        if t.isdigit():
            out.append(t)
        elif t == '(':
            ops.append(t)
        elif t == ')':
            while ops and ops[-1] != '(':
                out.append(ops.pop())
            if not ops:
                raise ValueError('括号不匹配')
            ops.pop()
        else:                                      # 运算符
            if prev is None or prev in '(+-×÷':
                raise ValueError(f'运算符 {t!r} 位置错误')
            while ops and ops[-1] != '(' and \
                    _PRECEDENCE[ops[-1]] >= _PRECEDENCE[t]:
                out.append(ops.pop())
            ops.append(t)
        prev = t
    while ops:
        if ops[-1] == '(':
            raise ValueError('括号不匹配')
        out.append(ops.pop())
    return out


def _check_expr(tokens: list[str]) -> None:
    """校验后缀序列合法（数字比运算符多 1，且逐步栈不空）。"""
    if not any(t in SYM_TO_OPTYPE for t in tokens):
        raise ValueError('算式里没有运算符')
    stack = 0
    for t in tokens:
        if t in SYM_TO_OPTYPE:
            if stack < 2:
                raise ValueError('运算符缺少操作数（表达式不完整？）')
            stack -= 1
        else:
            stack += 1
    if stack != 1:
        raise ValueError('数字与运算符数量不匹配')


def _quick_eval(tokens: list[str]) -> int:
    """纯数字栈机求值（供模型对照用，不渲染）。"""
    stack: list[int] = []
    for t in tokens:
        if t not in SYM_TO_OPTYPE:
            stack.append(int(t))
        else:
            b, a = stack.pop(), stack.pop()
            if t == '+':
                stack.append(a + b)
            elif t == '-':
                stack.append(a - b)
            elif t == '×':
                stack.append(a * b)
            else:
                if a % b:
                    raise ValueError(f'{a} ÷ {b} 不是整除')
                stack.append(a // b)
    return stack[0]


def solve_and_render(expr: str, out_dir: Path, renderer: ImageRenderer,
                     composer: ArithmeticComposer, abacus: Abacus,
                     model_loop=None) -> int:
    """执行算式：逐步算盘运算 + GIF 渲染 + 口诀打印。返回最终结果。"""
    tokens = infix_to_postfix(expr)
    _check_expr(tokens)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n算式: {expr}')
    print(f'后序: {" ".join(tokens)}')

    # 可选：VLM 模型先做一遍真闭环（PARSE→EVAL→READ）
    model_result = None
    if model_loop is not None:
        res = model_loop.solve(expr, ANS=_quick_eval(tokens), model_driven=True)
        model_result = res['final_result']
        ok = res['correct']
        status = '✓' if ok else '✗' if ok is False else '(无对照)'
        print(f'\n[VLM 模型] POST={res["post"] or "(生成失败)"}'
              f'  最终答案={model_result}'
              f'  端到端{status}\n')

    stack: list[int] = []
    op_id = 0
    for tok in tokens:
        if tok not in SYM_TO_OPTYPE:
            stack.append(int(tok))
            continue

        b, a = stack.pop(), stack.pop()
        op_type = SYM_TO_OPTYPE[tok]
        if op_type == OpType.DIV and a % b != 0:
            raise ValueError(f'{a} ÷ {b} 不是整除（演示仅支持整除，与训练分布一致）')

        op = composer.compose(op_type, abacus, a, b, op_id)
        rhyme = '；'.join(s.rhyme.rhyme_text for s in op.steps)
        sym = _TYPE_SYM[op_type]

        print(f'[第{op_id + 1}步] {a} {sym} {b} = {op.expected_result}')
        print(f'  口诀链: {rhyme}')

        gif = out_dir / f'{op_id + 1:02d}_{a}{sym}{b}.gif'
        renderer.render_gif(op, str(gif))
        print(f'  动画  : {gif}')

        stack.append(op.expected_result)
        op_id += 1

    result = stack[0]
    print(f'\n最终结果: {result}')
    if model_loop is not None and model_result is not None:
        agree = '与模型一致 ✓' if model_result == result else '与模型不一致 ✗'
        print(f'模型结果: {model_result}（{agree}）')
    print(f'动画输出目录: {out_dir}')
    return result


def make_demo_components():
    rhymes = build_addition_rhymes() + build_subtraction_rhymes()
    resolver = RhymeResolver(rhymes)
    composer = ArithmeticComposer(resolver)
    abacus = Abacus.standard()
    renderer = ImageRenderer(Style(), scale=1.0)
    return composer, abacus, renderer


def main():
    import argparse
    ap = argparse.ArgumentParser(description='算盘交互演示：算式 → 拨珠动画 + 口诀')
    ap.add_argument('expr', nargs='?', help='算式，如 "3+5×2"；缺省进入交互模式')
    ap.add_argument('--model', default=None,
                    help='VLM checkpoint（三任务共用），开启模型看盘说口诀模式')
    ap.add_argument('--out', default='output/demo', help='GIF 输出目录')
    args = ap.parse_args()

    composer, abacus, renderer = make_demo_components()

    model_loop = None
    if args.model:
        from model_vlm.closed_loop import VLMClosedLoop
        model_loop = VLMClosedLoop(args.model, args.model, args.model)

    out_root = _PROJECT_ROOT / args.out

    if args.expr:
        solve_and_render(args.expr, out_root, renderer, composer,
                         abacus, model_loop)
        return

    # ── 交互模式 ──
    print('算盘交互演示（输入 q 退出）')
    print('示例: 3+5×2 / (4×7-1)×7 / 12÷4')
    while True:
        try:
            expr = input('\n算式> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not expr or expr.lower() in ('q', 'quit', 'exit'):
            break
        try:
            solve_and_render(expr, out_root, renderer, composer,
                             abacus, model_loop)
        except ValueError as e:
            print(f'[错误] {e}')


if __name__ == '__main__':
    main()
