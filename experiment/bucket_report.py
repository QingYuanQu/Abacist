# -*- coding: utf-8 -*-
"""experiment/bucket_report.py —— 分桶评测报告（分析工具，不参与训练主链路）。

职责：加载已训练的单课模型，在测试集上批量推理，按**结构等价**判定正确性，
并按 n × bk / prec_switch / ans_digits 分桶出报告。

为什么需要独立脚本：
  1. 结构等价（生成 ∈ 该 Q 的合法解集合）不在现有 acc / acc_ans / acc_think 三口径内。
     dataset_C 存在一题多解（同一 Q 有多棵等价树），严格串等会系统性低估——
     例如全加的 n=5 样本有多达 14 个合法 POST，命中率上限仅 1/14；
  2. 分桶报告是**分析产物**，不是实验编排的一部分，不应侵入训练与 report.csv 回写链路。

判定口径：
  - `结构正确率`：生成串去掉停止符后 ∈ 该样本的 `alt` 集合（一题多解的正确处理方式）；
  - `严格串等`：生成串与期望 A 完全相同（对照项，会被多解压低，不代表真实能力）。

用法：
    python -m experiment.bucket_report --experiments PILOT_h64
    python -m experiment.bucket_report --experiments P_regular,P_focus,P_expansion,P_balanced
    python -m experiment.bucket_report --experiments P_regular --csv bucket.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from model import GPT
from model.config import ModelConfig
from model.vocab import load_vocab
from model_lm.eval import generate_batch
from experiment.loader import load_experiment

# 分桶档位（闭区间；99 表示「及以上」）
N_BANDS = [(2, 5), (6, 8), (9, 12), (13, 20)]
BK_BANDS = [(0, 2), (3, 5), (6, 7), (8, 9)]
DIGIT_BANDS = [(1, 1), (2, 2), (3, 99)]
PS_BANDS = [(0, 0), (1, 1), (2, 99)]
MIN_CELL = 30          # 低于此样本数的格子不解释，输出 "—"
PS_BAND_N = (4, 6)     # prec_switch 分析只在 n ∈ [4,6] 内做（大 n 几乎必然混合运算）

_COL_W = 15
_ROW_W = 9


def _strip_stop(s: str) -> str:
    """去掉停止符 '#'（生成到 max_len 停止时没有）。"""
    s = s.rstrip()
    return s[:-1] if s.endswith('#') else s


def _band(v, bands) -> str | None:
    for lo, hi in bands:
        if lo <= v <= hi:
            return f"{lo}-{hi}" if hi < 99 else f"{lo}+"
    return None


def _load_rows(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _eval_trial(exp, trial, device, batch_size: int) -> list[dict] | None:
    """批量推理单 trial 测试集，返回逐样本判定结果（None = 无 ckpt 跳过）。"""
    ckpt = trial.paths.best_model
    if not os.path.isfile(ckpt):
        return None

    vocab = load_vocab(exp.vocab_path)

    # 套用该 trial 的逐层头数（唯一入口，训练/评测/分析共用）
    brain_config = exp.brain.with_trial_overrides(trial.heads)

    model = GPT(ModelConfig.from_sources(brain_config, exp.pos_emb, vocab))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    rows = _load_rows(trial.paths.test_data)
    prompts = [r['Q'] for r in rows]
    stop_ids = {vocab.stoi[vocab.stop_token]}

    preds = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        preds.extend(generate_batch(
            model, device, vocab.stoi, vocab.itos, vocab.tokenizer,
            chunk, vocab.max_seq_len, stop_ids, vocab.pad_id,
        ))

    out = []
    for r, pred in zip(rows, preds):
        # alt 缺失时退化为「期望输出即唯一解」（等价于严格串等）
        alts = set((r.get('alt') or _strip_stop(r['A'])).split('|'))
        out.append({
            "n": r.get("n"), "bk": r.get("bk"),
            "prec_switch": r.get("prec_switch"), "ans_digits": r.get("ans_digits"),
            "struct_ok": _strip_stop(pred) in alts,
            "exact_ok": pred == r["A"],
        })
    return out


def _cell(correct: int, total: int) -> str:
    if total == 0:
        return "—"
    if total < MIN_CELL:
        return f"— ({total})"
    return f"{correct / total * 100:.1f}% ({total})"


def _matrix(title: str, row_labels, col_labels, counter: dict) -> None:
    """打印二维表；counter 的键为 (row_label, col_label) -> [correct, total]。"""
    print(f"\n{title}")
    print(" " * _ROW_W + "".join(c.ljust(_COL_W) for c in col_labels))
    for rl in row_labels:
        line = rl.ljust(_ROW_W)
        for cl in col_labels:
            correct, total = counter.get((rl, cl), (0, 0))
            line += _cell(correct, total).ljust(_COL_W)
        print(line)


def _buckets(res: list[dict], row_key: str, col_key: str):
    counter: dict[tuple, list] = {}
    rows_seen, cols_seen = [], []
    for r in res:
        rl, cl = row_key(r), col_key(r)
        if rl is None or cl is None:
            continue
        if rl not in rows_seen:
            rows_seen.append(rl)
        if cl not in cols_seen:
            cols_seen.append(cl)
        slot = counter.setdefault((rl, cl), [0, 0])
        slot[1] += 1
        slot[0] += int(r["struct_ok"])
    return sorted(rows_seen), sorted(cols_seen), counter


def _one_dim(res: list[dict], key, label: str) -> None:
    counter: dict[str, list] = {}
    for r in res:
        k = key(r)
        if k is None:
            continue
        slot = counter.setdefault(k, [0, 0])
        slot[1] += 1
        slot[0] += int(r["struct_ok"])
    print(f"\n{label}")
    for k in sorted(counter):
        correct, total = counter[k]
        print(f"  {k.ljust(8)}{_cell(correct, total)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="分桶评测报告（结构等价 + n×bk）")
    ap.add_argument("--experiments", required=True, help="逗号分隔的 exp 名")
    ap.add_argument("--trial", type=int, default=None, help="只评第 N 课（默认全部）")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--csv", default=None, help="额外输出长表 CSV")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary = []
    csv_rows = []

    for name in [s.strip() for s in args.experiments.split(',') if s.strip()]:
        exp = load_experiment(name)
        trials = exp.trials if args.trial is None else [exp.trials[args.trial]]
        for trial in trials:
            res = _eval_trial(exp, trial, device, args.batch_size)
            if res is None:
                print(f"[skip] {name}/L{trial.id} {trial.name}: 无 ckpt，跳过")
                continue

            total = len(res)
            struct = sum(r["struct_ok"] for r in res) / total
            exact = sum(r["exact_ok"] for r in res) / total
            print("=" * 70)
            print(f"{name} / L{trial.id} {trial.name}")
            print(f"  结构正确率 {struct * 100:.2f}%   严格串等 {exact * 100:.2f}%   (total={total})")

            # 表 A：n × bk
            rows, cols, counter = _buckets(
                res, lambda r: _band(r["n"], N_BANDS) if r["n"] is not None else None,
                lambda r: _band(r["bk"], BK_BANDS) if r["bk"] is not None else None)
            if rows and cols:
                _matrix(f"表A  n × bk（结构正确率，格子样本数 <{MIN_CELL} 不解释）", rows, cols, counter)
                for rl in rows:
                    for cl in cols:
                        c, t = counter.get((rl, cl), (0, 0))
                        csv_rows.append({"exp": name, "trial": f"L{trial.id}_{trial.name}",
                                         "table": "n×bk", "row": rl, "col": cl,
                                         "correct": c, "total": t})

            # 表 B：n ∈ [4,6] 内按 prec_switch
            sub = [r for r in res
                   if r["n"] is not None and PS_BAND_N[0] <= r["n"] <= PS_BAND_N[1]]
            if sub:
                _one_dim(sub, lambda r: _band(r["prec_switch"], PS_BANDS)
                         if r["prec_switch"] is not None else None,
                         f"表B  n∈[{PS_BAND_N[0]},{PS_BAND_N[1]}] 内按 prec_switch（优先级切换次数）")

            # 表 C：n 档 × ans_digits（混淆控制）
            rows2, cols2, counter2 = _buckets(
                res, lambda r: _band(r["n"], N_BANDS) if r["n"] is not None else None,
                lambda r: _band(r["ans_digits"], DIGIT_BANDS) if r["ans_digits"] is not None else None)
            if rows2 and cols2:
                _matrix("表C  n × ans_digits（混淆控制：答案位数）", rows2, cols2, counter2)
                for rl in rows2:
                    for cl in cols2:
                        c, t = counter2.get((rl, cl), (0, 0))
                        csv_rows.append({"exp": name, "trial": f"L{trial.id}_{trial.name}",
                                         "table": "n×ans_digits", "row": rl, "col": cl,
                                         "correct": c, "total": t})

            summary.append((name, trial.name, struct, exact, total))

    # 表 D：跨 exp × trial 总表
    if summary:
        print("\n" + "=" * 70)
        print("表D  总表（结构正确率）")
        trials_seen = []
        for _, lname, *_ in summary:
            if lname not in trials_seen:
                trials_seen.append(lname)
        experiments_seen = []
        for ename, *_ in summary:
            if ename not in experiments_seen:
                experiments_seen.append(ename)
        print(" " * _ROW_W + "".join(l.ljust(_COL_W) for l in trials_seen))
        for sname in experiments_seen:
            line = sname.ljust(_ROW_W)
            for lname in trials_seen:
                hit = next((x for x in summary if x[0] == sname and x[1] == lname), None)
                line += (f"{hit[2] * 100:.1f}%".ljust(_COL_W) if hit else "".ljust(_COL_W))
            print(line)

    if args.csv and csv_rows:
        with open(args.csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[CSV] 已写入 {args.csv}")


if __name__ == "__main__":
    main()
