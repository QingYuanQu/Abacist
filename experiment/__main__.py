import argparse
import os
import sys

import torch

from experiment.runner import _run_experiment
from experiment.material_adapter import generate_trial
from model.vocab import ensure_experiment_vocab
from experiment.cleanup import reset_eval, reset_data, reset_from, reset_full
from experiment.loader import load_experiment, init_experiment, clone_experiment
from config import Experiment

# ==================== 公共前置函数 ====================

def prepare_all(exp: Experiment):
    """公共前置：生成全部课程数据 + 构建词表。"""
    # 1. 生成全部课程数据
    print("[prepare] 步骤1: 确保全部课程数据就绪...")
    os.makedirs(exp.material_dir, exist_ok=True)
    for cfg in exp.trials:
        m = cfg.material
        if not os.path.isfile(m.train_data_path) or (m.test_data_path and not os.path.isfile(m.test_data_path)):
            print(f"[prepare] 生成第{cfg.id}课数据: {cfg.name}")
            generate_trial(cfg, exp.data_seed)

    # 2. 构建词表（experiment 语义：按数据源去重后共享一份词表）
    print("[prepare] 步骤2: 构建词表...")
    vocab_data = ensure_experiment_vocab(exp, exp.trials)
    return vocab_data

def _print_study_progress(exp: Experiment):
    """打印学习单元整体进度（已通过/未通过/待评估）。"""
    tag = "[实验]"
    total = len(exp.trials)
    passed = failed = pending = 0

    print(f"\n{tag} 学习进度总览:")
    for trial in exp.trials:
        r = trial.record
        if r is None or r.acc is None:
            status = "待评估"
            pending += 1
        elif trial.passed:
            status = f"通过 ({r.acc*100:.2f}%)"
            passed += 1
        else:
            status = f"未通过 ({r.acc*100:.2f}%)"
            failed += 1
        print(f"  L{trial.id}: {trial.name} — {status}")

    print(f"{tag} 汇总: 共 {total} 课 | 通过 {passed} | 未通过 {failed} | 待评估 {pending}")


def run_study(exp: Experiment = None, start_trial=None, end_trial=None):

    if exp is None:
        exp = load_experiment("PATTERN_in2post")

    tag = "[实验]"

    print("=" * 60)
    print(f"{tag} Abacist 学习编排器")
    print(f"{tag} Experiment: {exp.name}")
    print("=" * 60)

    # 从 exp.csv 加载结果
    _print_study_progress(exp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{tag} 设备: {device}")

    # 确保 checkpoint 目录存在
    os.makedirs(exp.ckpt_dir, exist_ok=True)

    print(f"\n{tag} 公共前置：确保全部课程数据就绪 + 构建词表...")
    vocab_data = prepare_all(exp)

    # 3. 分派（统一 experiment 语义：各课独立训练）
    total = len(exp.trials)
    last_id = total - 1
    if end_trial is None:
        end_trial = last_id

    start_trial = 0
    _run_experiment(exp, start_trial, end_trial, vocab_data, device)

# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Abacist 课程编排器")
    parser.add_argument("--experiment", type=str, default="default",
                        help="学习单元名称（对应 experiment/studies/<name>/ 目录），默认=default")
    parser.add_argument("--start", type=int, default=None,
                        help="起始课号（默认自动从首个未通过课开始）")
    parser.add_argument("--end", type=int, default=None,
                        help="终止课号（默认跑完全部）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查数据和状态，不实际训练")
    parser.add_argument("--reset", action="store_true",
                        help="清除 report.csv 结果 + brain + material + vocab，回到初始状态")
    parser.add_argument("--reset-eval", action="store_true",
                        help="清除 brain 和 report.csv 结果（保留数据和词表）")
    parser.add_argument("--reset-data", action="store_true",
                        help="清除数据和词表（保留 brain 和评测记录）")
    parser.add_argument("--reset-full", action="store_true",
                        help="清除模型和数据（brain + material + vocab，保留评测记录）")
    parser.add_argument("--reset-from", type=int, metavar="N",
                        help="清除第 N 课及其后所有课程的模型和评测结果")
    parser.add_argument("--status", action="store_true",
                        help="查看当前学习状态（来自 exp.csv）")
    parser.add_argument("--init", action="store_true",
                        help="初始化学习单元目录结构（生成 exp.csv/brain.json/policy.json）")
    parser.add_argument("--pattern", type=str, default="regular",
                        help="配合 --init 使用：注意力头分布模式 (focus_expansion/expansion_focus/balanced_progressive/regular)")
    parser.add_argument("--clone", type=str, default=None, metavar="SRC",
                        help="配合 --init 使用：从指定 exp 克隆结构配置（不复制数据/词表/模型/日志）")

    args = parser.parse_args()

    # --init 模式：创建学习单元脚手架
    if args.init:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        experiment_name = args.experiment or "default"
        if args.clone:
            clone_experiment(experiment_name, project_root, args.clone)
            return
        init_experiment(experiment_name, project_root, brain_pattern=args.pattern)
        return

    # 加载 exp
    experiment_name = args.experiment or "PATTERN_in2post"
    try:
        exp = load_experiment(experiment_name)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[实验] 错误: {e}")
        sys.exit(1)

    # 根据 kind 选择用户可见标签
    tag = "[实验]"

    if args.reset:
        reset_eval(exp)
        reset_data(exp)
        print(f"{tag} 已重置到初始状态，下次运行将从 L0 重新生成全部数据并训练。")
        return

    if args.reset_eval:
        reset_eval(exp)
        print(f"{tag} 已清除评测产物（brain + exp.csv 结果列）")
        return

    if args.reset_data:
        reset_data(exp)
        print(f"{tag} 已清除数据产物（material + vocab）")
        return

    if args.reset_full:
        reset_full(exp)
        print(f"{tag} 已清除模型和数据（brain + material + vocab）")
        return

    if args.reset_from is not None:
        reset_from(exp, args.reset_from)
        print(f"{tag} 已清除 L{args.reset_from} 及之后的模型和评测结果")
        return

    if args.status:
        print("=" * 40)
        print(f"{tag} 当前学习状态 (来自 {exp.name}/report.csv)")
        print("=" * 40)
        for trial in exp.trials:
            r = trial.record
            if r is None or r.acc is None:
                print(f"  L{trial.id}: 未评估")
            else:
                acc_str = f"{r.acc*100:.2f}%"
                print(f"  L{trial.id}: acc={acc_str}, correct={r.correct}, total={r.total}, "
                      f"passed={trial.passed}")
        if not exp.trials:
            print("  (无评估记录)")
        return

    run_study(
        exp=exp,
        start_trial=args.start,
        end_trial=args.end,
    )


if __name__ == "__main__":
    main()
