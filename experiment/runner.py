"""runner.py — 实验编排：逐 trial 独立训练并写回结果。

experiment 语义：每个 trial 从零训练，无复习、无权重传递，失败不阻塞。
词表按数据源去重后共享（trial_vocabs 即 ensure_experiment_vocab 返回的 {trial_id: vocab}）。
"""

import os
import torch

from experiment.domain import Experiment, ExecutionContext
from experiment.data_adapter import generate_trial
from experiment.tools.store import ReportTable
from experiment.trial.train_and_eval import train_and_eval_trial
from model.vocab import ensure_experiment_vocab


def prepare_all(exp: Experiment) -> dict:
    """公共前置：生成全部 trial 的数据 + 构建共享词表。

    Returns:
        {trial_id: vocab_data}（experiment 语义下所有 trial 共享同一份词表）
    """
    print(f"[实验] 步骤1: 确保全部 trial 数据就绪...")
    os.makedirs(exp.data_dir, exist_ok=True)
    for trial in exp.trials:
        paths = trial.paths
        if not os.path.isfile(paths.train_data) or \
                (paths.test_data and not os.path.isfile(paths.test_data)):
            generate_trial(trial, exp.data_seed)

    # 词表按去重后的数据文件集合构建一份（experiment 语义：数据/词表共享，训练独立）
    print(f"[实验] 步骤2: 构建共享词表...")
    return ensure_experiment_vocab(exp, exp.trials)


def _print_progress(exp: Experiment) -> None:
    """打印实验整体进度（已通过/未通过/待评估）。"""
    passed = failed = pending = 0
    acc_mode = exp.eval.acc_mode if exp.eval else "acc"

    print(f"\n[实验] 进度总览（判定口径: {acc_mode}）:")
    for trial in exp.trials:
        r = trial.record
        if r is None or getattr(r, acc_mode) is None:
            status, pending = "待评估", pending + 1
        else:
            acc = getattr(r, acc_mode)
            if trial.passed:
                status, passed = f"通过 ({acc*100:.2f}%)", passed + 1
            else:
                status, failed = f"未通过 ({acc*100:.2f}%)", failed + 1
        print(f"  #{trial.id}: {trial.name} — {status}")

    print(f"[实验] 汇总: 共 {len(exp.trials)} 个 trial | 通过 {passed} | "
          f"未通过 {failed} | 待评估 {pending}")


def run_experiment(exp: Experiment, start_trial=None, end_trial=None) -> None:
    """跑实验：从 start_trial 到 end_trial（含），逐 trial 独立训练并写回结果。"""
    print("=" * 60)
    print(f"[实验] Abacist 实验编排器")
    print(f"[实验] Experiment: {exp.name}")
    print("=" * 60)

    _print_progress(exp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[实验] 设备: {device}")

    print(f"\n[实验] 公共前置：确保全部 trial 数据就绪 + 构建词表...")
    trial_vocabs = prepare_all(exp)

    total = len(exp.trials)
    last_id = total - 1
    start_trial = 0 if start_trial is None else max(0, start_trial)
    end_trial = last_id if end_trial is None else min(last_id, end_trial)
    if start_trial > end_trial:
        print(f"[实验] [WARN] start={start_trial} > end={end_trial}，无 trial 需要运行。")
        return

    for trial_id in range(start_trial, end_trial + 1):
        trial = exp.trials[trial_id]

        vocab_data = trial_vocabs.get(trial_id)
        if vocab_data is None:
            print(f"[实验] [WARN] trial {trial_id} 词表缺失，跳过。")
            continue

        # 已通过且记录完整 → 跳过（passed 与 acc 同源，不会出现"无 acc 却 passed"）
        if trial.passed and trial.record:
            print(f"\n[实验] trial {trial_id} ({trial.name}) 已通过 "
                  f"(acc={trial.record.acc*100:.2f}%)，跳过。")
            continue

        ctx = ExecutionContext(
            vocab_data=vocab_data,
            device=device,
            seed=exp.train.train_seed,
        )
        record, passed = train_and_eval_trial(trial_id=trial_id, exp=exp, ctx=ctx)

        # 写回结果（无论通过与否）；acc_mode/pass_threshold 一并落表，让每行自带判定口径
        ReportTable(exp.report_path).save_result(
            trial_id, trial.name, record, passed,
            exp.eval.acc_mode, exp.eval.pass_threshold)

        if not passed:
            print(f"\n[实验] [WARN] trial {trial_id} ({trial.name}) 未通过，继续下一个。")

    print(f"\n{'=' * 60}")
    print(f"[实验] [DONE] 对比实验完成！ L{start_trial} → L{end_trial}")
    print(f"{'=' * 60}")
