# -*- coding: utf-8 -*-
"""bridge/generate.py —— 读 parse 的 dataset_D → align → 落盘 IR jsonl。

这是底层 IR 的唯一生产入口：消费 parse 的纸笔记录，调用 eval 的两种口径
（abacus 口诀 / digit 逐位）对齐注解，落盘富结构 IR，供各模型离线投影。
"""
from __future__ import annotations

import json
from pathlib import Path

from .align import make_default_composer, align
from .ir import build_instance, to_json
from evaluate.digit import make_digit_fn


def emit_ir(dataset_path, out_path, *, n_samples: int = 0) -> int:
    """读 dataset_D.jsonl，逐条 align，落盘 IR jsonl。

    Args:
        dataset_path: parse 的 dataset_D.jsonl 路径（含 Q/pre/post/ANS/ops/tree/gid）
        out_path: IR 输出 jsonl 路径
        n_samples: >0 只处理前 N 条（抽检）；<=0 全量

    Returns:
        生成的 IR 条数。
    """
    composer, abacus = make_default_composer()
    digit_fn = make_digit_fn()

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"[skip] 数据集不存在: {dataset_path}")
        return 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines) if n_samples <= 0 else min(n_samples, len(lines))

    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for line in lines[:total]:
            rec = json.loads(line)
            steps, _ = align(rec["post"].split(), composer, abacus,
                             digit_fn=digit_fn, expected_ans=rec["ANS"])
            inst = build_instance(rec, steps)
            out.write(to_json(inst) + "\n")
            count += 1

    print(f"[IR] 已生成 {count} 条 → {out_path}")
    return count


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="生成底层 IR jsonl")
    ap.add_argument("dataset", nargs="?", default=None,
                    help="dataset_D 路径（默认 parse/dataset/dataset_D.jsonl）")
    ap.add_argument("--out", default=None,
                    help="IR 输出路径（默认 <dataset>_ir.jsonl 同目录）")
    ap.add_argument("-n", type=int, default=0, help="只处理前 N 条（0=全量）")
    args = ap.parse_args()

    _root = Path(__file__).resolve().parent.parent
    dataset = Path(args.dataset) if args.dataset else _root / "parse" / "dataset" / "dataset_D.jsonl"
    out = Path(args.out) if args.out else dataset.with_name(dataset.stem + "_ir.jsonl")
    emit_ir(dataset, out, n_samples=args.n)
