"""demo.py —— 内核端到端冒烟演示（人工验收用，非自动化测试）。

从原 abacus_domain.py 的 `_demo()` 拆出（2026-09-07）。此前库模块同时承担
"跑 doctest"与"跑演示"两职；自动化测试已迁至 tests/，本文件只保留人读演示。

运行：python -m evaluate.abacus.demo
"""
from evaluate.abacus import (
    Abacus, AbacusSpec, ArithmeticComposer, DigitOrder, RhymeResolver, OpType,
    build_addition_rhymes, build_subtraction_rhymes,
)



# ═══════════════════ Demo：数据生成主链端到端 ═══════════════════

def _demo() -> None:
    from evaluate.abacus.render.text import TextRenderer   # 渲染已迁出内核，演示期按需导入
    abacus = Abacus.standard(AbacusSpec(1, 4, 13, 10))
    resolver = RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes())
    resolver.assert_total(OpType.ADD)                  # 真值裁判验收：90 格全命中
    resolver.assert_total(OpType.SUB)                  # 减法 90 格全命中
    composer = ArithmeticComposer(resolver)
    renderer = TextRenderer()

    for op_type, a, b in [(OpType.ADD, 4, 1), (OpType.ADD, 9, 1),
                          (OpType.ADD, 47, 38), (OpType.ADD, 99, 1),
                          (OpType.ADD, 999, 1), (OpType.ADD, 49, 1),
                          (OpType.SUB, 5, 2), (OpType.SUB, 7, 3),
                          (OpType.SUB, 12, 5), (OpType.SUB, 50, 1),
                          (OpType.SUB, 100, 1), (OpType.SUB, 53, 8),
                          (OpType.MUL, 3, 24), (OpType.MUL, 12, 34),
                          (OpType.DIV, 48, 6), (OpType.DIV, 72, 8),
                          (OpType.DIV, 144, 12), (OpType.DIV, 100, 4),
                          (OpType.DIV, 100, 7), (OpType.DIV, 7, 3),
                          (OpType.DIV, 47, 5)]:
        op = composer.compose(op_type, abacus, a, b)
        sym = {OpType.ADD: "+", OpType.SUB: "-", OpType.MUL: "×", OpType.DIV: "÷"}[op_type]
        label = f"{op.expected_result}" + \
            (f" 余 {op.remainder}" if op_type is OpType.DIV and op.remainder else "")
        print(f"\n══ {a} {sym} {b} = {label} ══")
        for s in op.steps:
            print(f"  step{s.step_index:>2} 档{s.base_rod_index} [{s.rhyme.rhyme_text}]"
                  f"  {s.before_state.total_value:>4} → {s.after_state.total_value:<4}"
                  f"  valid={s.is_valid}")
        if (a, b) == (47, 38):
            print(renderer.render(op.final_state, abacus))
        op.replay()                                    # 确定性重放自检

    # 文本渲染出口（渲染统一走 abacus.render，内核不持有渲染抽象）
    op = composer.compose(OpType.ADD, abacus, 47, 38)
    print("\nCoT   :", "；".join(f"{s.step_index}:{s.rhyme.rhyme_text}" for s in op.steps))
    print("终盘  :")
    print(renderer.render(op.final_state, abacus))

    # —— 双模式等价性自检：两种 digit_order 终盘一致、审计链均通过 ——
    composer_lsd = ArithmeticComposer(resolver, digit_order=DigitOrder.LSD)
    for op_type, a, b in [(OpType.ADD, 47, 38), (OpType.ADD, 999, 1),
                          (OpType.SUB, 100, 1), (OpType.SUB, 53, 8),
                          (OpType.MUL, 12, 34), (OpType.DIV, 100, 7)]:
        op_msd = composer.compose(op_type, abacus, a, b)
        op_lsd = composer_lsd.compose(op_type, abacus, a, b)
        assert op_msd.expected_result == op_lsd.expected_result
        assert op_msd.final_state.total_value == op_lsd.final_state.total_value == op_msd.expected_result \
            or op_type is OpType.DIV                             # 除法终盘=商+余数并排，只比终盘相等
        assert op_msd.final_state.total_value == op_lsd.final_state.total_value
        op_msd.replay(); op_lsd.replay()
    print("\n双模式等价自检：MSD / LSD 终盘一致，审计链全部通过 [OK]")

if __name__ == "__main__":
    _demo()
