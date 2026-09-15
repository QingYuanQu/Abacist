# -*- coding: utf-8 -*-
"""bridge/selfcheck.py —— IR 契约自检（独立入口，避免 `-m bridge.align` 的双重导入警告）。

用法：python -m bridge.selfcheck
"""
from .align import make_default_composer, align
from .ir import build_instance, to_json
from evaluate.digit import make_digit_fn


def main() -> None:
    composer, abacus = make_default_composer()
    digit_fn = make_digit_fn()  # digit 口径注入点（阶段 1 落地）

    cases = [
        {"n": 2, "Q": "39+9279", "pre": "+ 39 9279", "post": "39 9279 +",
         "ANS": 9318, "Ic": 0, "bk": 0, "sp": "train"},
        {"n": 2, "Q": "12×34", "pre": "× 12 34", "post": "12 34 ×",
         "ANS": 408, "Ic": 0, "bk": 0, "sp": "train"},
        {"n": 3, "Q": "1+2+3", "pre": "+ 1 + 2 3", "post": "1 2 + 3 +",
         "ANS": 6, "Ic": 0, "bk": 0, "sp": "train"},
    ]

    for rec in cases:
        steps, result = align(rec["post"].split(), composer, abacus,
                              digit_fn=digit_fn, expected_ans=rec["ANS"])
        inst = build_instance(rec, steps)
        oral = "；".join(a.oral for s in inst.steps if s.abacus
                         for a in s.abacus.actions)
        digit_chain = "；".join(s.digit for s in inst.steps if s.digit)
        assert result == rec["ANS"], f"{rec['Q']}: {result} != {rec['ANS']}"
        print(f"Q={rec['Q']}  result={result}")
        print(f"  口诀链 = {oral}")
        print(f"  digit链 = {digit_chain}")

    print("\n序列化冒烟（首条 IR JSON）:")
    print(to_json(build_instance(cases[0], align(
        cases[0]["post"].split(), composer, abacus,
        digit_fn=digit_fn, expected_ans=cases[0]["ANS"])[0])))
    print("\n[OK] bridge/selfcheck 自检通过")


if __name__ == "__main__":
    main()
