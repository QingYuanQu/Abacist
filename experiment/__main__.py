"""experiment 的 CLI 入口 —— 只做参数解析与派发，编排逻辑在 runner.py。

用法：
    python -m experiment --experiment PATTERN_in2post              # 跑全部未通过 trial
    python -m experiment --experiment PATTERN_in2post --start 3 --end 7
    python -m experiment --experiment PATTERN_in2post --dry-run    # 只读检查
    python -m experiment --experiment PATTERN_in2post --status
    python -m experiment --experiment NEW --init                   # 生成 config.yaml 模板
    python -m experiment --experiment NEW --init --clone OLD
"""

import argparse
import os
import sys

from experiment.domain import Experiment
from experiment.tools.cleanup import reset_data, reset_eval, reset_from, reset_full
from experiment.tools.loader import clone_experiment, init_experiment, load_experiment
from experiment.runner import run_experiment
from experiment.config import ConfigError, UnsupportedFeature
from experiment.tools.store import ReportMismatch

TAG = "[实验]"


# ==================== 只读报告 ====================

def _dry_run_report(exp: Experiment):
    """只读检查：不生成数据、不训练，只报告配置与产物就绪情况。"""
    print(f"\n{TAG} 只读检查（--dry-run：不生成数据、不训练）")
    print(f"  配置    : {exp.config_path}")
    print(f"  结果表  : {exp.report_path}")
    print(f"  词表    : {'就绪' if os.path.isfile(exp.vocab_path) else '待构建'}")
    print(f"  trials  : {len(exp.trials)}")
    print(f"  {'id':>3}  {'name':<18} {'heads':<26} {'数据':<6} {'模型':<6} 结果")
    for t in exp.trials:
        data_ok = "就绪" if os.path.isfile(t.paths.train_data) else "缺失"
        model_ok = "就绪" if os.path.isfile(t.paths.best_model) else "缺失"
        result = "—"
        if t.record is not None:
            acc_mode = exp.eval.acc_mode if exp.eval else "acc"
            acc = getattr(t.record, acc_mode)
            result = f"acc={acc*100:.2f}% passed={t.passed}" if acc is not None else "—"
        print(f"  {t.id:>3}  {t.name:<18} {str(t.heads):<26} {data_ok:<6} {model_ok:<6} {result}")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Abacist 实验编排器")
    parser.add_argument("--experiment", type=str, required=True,
                        help="实验名（对应 experiment/studies/<name>/ 目录，须与 config.yaml 的 name 一致）")
    parser.add_argument("--start", type=int, default=None,
                        help="起始 trial 号（默认 0）")
    parser.add_argument("--end", type=int, default=None,
                        help="终止 trial 号（默认最后一个）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读检查配置与产物就绪情况，不生成数据、不训练")
    parser.add_argument("--reset", action="store_true",
                        help="清除 report.csv 结果 + memory + data + vocab + logs，回到初始状态")
    parser.add_argument("--reset-eval", action="store_true",
                        help="清除 memory 和 report.csv 结果（保留数据和词表）")
    parser.add_argument("--reset-data", action="store_true",
                        help="清除数据和词表（保留 memory 和评测记录）")
    parser.add_argument("--reset-full", action="store_true",
                        help="清除 memory + data + vocab + logs（保留评测记录）")
    parser.add_argument("--reset-from", type=int, metavar="N",
                        help="清除第 N 个 trial 及其后所有 trial 的模型、结果与日志")
    parser.add_argument("--status", action="store_true",
                        help="查看当前实验状态（来自 report.csv）")
    parser.add_argument("--init", action="store_true",
                        help="初始化实验目录结构（生成 config.yaml 模板 + 空 report.csv）")
    parser.add_argument("--clone", type=str, default=None, metavar="SRC",
                        help="配合 --init：从源实验克隆配置（不复制数据/词表/模型/日志）")

    args = parser.parse_args()

    # ---- --init 模式：创建实验脚手架 ----
    if args.init:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            if args.clone:
                clone_experiment(args.experiment, project_root, args.clone)
            else:
                init_experiment(args.experiment, project_root)
        except (FileExistsError, FileNotFoundError, ConfigError) as e:
            print(f"{TAG} 错误: {e}")
            sys.exit(1)
        return

    # ---- 加载配置（唯一输入 config.yaml） ----
    try:
        exp = load_experiment(args.experiment)
    except (FileNotFoundError, ConfigError, UnsupportedFeature, ReportMismatch) as e:
        print(f"{TAG} 错误: {e}")
        sys.exit(1)

    if args.dry_run:
        _dry_run_report(exp)
        return

    if args.reset:
        reset_eval(exp)
        reset_data(exp)
        print(f"{TAG} 已重置到初始状态，下次运行将重新生成全部数据并训练。")
        return

    if args.reset_eval:
        reset_eval(exp)
        print(f"{TAG} 已清除评测产物（memory + report.csv 结果列 + logs）")
        return

    if args.reset_data:
        reset_data(exp)
        print(f"{TAG} 已清除数据产物（data + vocab）")
        return

    if args.reset_full:
        reset_full(exp)
        print(f"{TAG} 已清除模型和数据（memory + data + vocab + logs）")
        return

    if args.reset_from is not None:
        reset_from(exp, args.reset_from)
        print(f"{TAG} 已清除 #{args.reset_from} 及之后的模型、结果与日志")
        return

    if args.status:
        print("=" * 40)
        print(f"{TAG} 当前状态 (来自 {exp.name}/report.csv)")
        print("=" * 40)
        acc_mode = exp.eval.acc_mode if exp.eval else "acc"
        for trial in exp.trials:
            r = trial.record
            acc = None if r is None else getattr(r, acc_mode)
            if acc is None:
                print(f"  #{trial.id}: {trial.name} — 未评估")
            else:
                print(f"  #{trial.id}: {trial.name} — {acc_mode}={acc*100:.2f}%, "
                      f"correct={getattr(r, 'correct', None)}, total={r.total}, "
                      f"passed={trial.passed}")
        return

    run_experiment(exp=exp, start_trial=args.start, end_trial=args.end)


if __name__ == "__main__":
    main()
