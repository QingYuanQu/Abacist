"""cleanup.py — 实验产物清理模块。

按"产物类别"提供原子清理函数，每个函数支持可选的范围过滤；
experiment/__main__.py 的 --reset 系列命令是这些原子函数的组合。

产物类别（路径一律取自 config.TrialPaths，此处不重复拼字符串）：
  purge_models          memory/trial_{N}.pth + trial_{N}_ckpt.pt（按 trial 号索引）
  purge_data            data/*.jsonl（按 trial 号索引）
  purge_vocab           vocab/*.json（全局，无范围）
  purge_logs            logs/trial_{N}_*.jsonl（按 trial 号索引）
  purge_trial_results   report.csv 的结果列

范围参数：
  from_trial=None  → 全部
  from_trial=N     → N 及之后所有 trial
"""

import glob
import os

from experiment.domain import Experiment
from experiment.tools.store import ReportTable


def _resolve_trial_ids(trials, from_trial=None):
    """统一解析 trial 范围 → trial 号集合。None 表示全部。"""
    if from_trial is not None:
        return set(range(from_trial, len(trials)))
    return None  # None = 全部


def _scope_label(from_trial) -> str:
    return f"#{from_trial}~" if from_trial is not None else "全部"


# ==================== 原子清理函数 ====================

def purge_models(exp: Experiment, from_trial=None):
    """删除模型权重与断点（trial_{N}.pth / trial_{N}_ckpt.pt）。"""
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        targets += glob.glob(os.path.join(exp.ckpt_dir, "trial_*.pth"))
        targets += glob.glob(os.path.join(exp.ckpt_dir, "trial_*_ckpt.pt"))
    else:
        for tid in trial_ids:
            paths = exp.trials[tid].paths
            for fp in (paths.best_model, paths.checkpoint):
                if os.path.isfile(fp):
                    targets.append(fp)

    for fp in targets:
        os.remove(fp)

    print(f"[清理] 已删除 {len(targets)} 个模型文件 ({_scope_label(from_trial)})")


def purge_data(exp: Experiment, from_trial=None):
    """删除训练/测试/bead 数据文件（*.jsonl）。

    全清时 glob 匹配；按范围清时遍历 exp.trials 取路径（dataset 类型的
    train/test 可能被多个 trial 共享，故需去重）。
    """
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        targets = glob.glob(os.path.join(exp.data_dir, "*.jsonl"))
    else:
        for tid in trial_ids:
            paths = exp.trials[tid].paths
            for fp in (paths.train_data, paths.test_data, paths.bead_data):
                if fp and os.path.isfile(fp):
                    targets.append(fp)
        targets = list(set(targets))

    for fp in targets:
        os.remove(fp)

    print(f"[清理] 已删除 {len(targets)} 个数据文件 ({_scope_label(from_trial)})")


def purge_vocab(exp: Experiment):
    """删除实验全部词表文件（vocab/*.json，全局无范围）。"""
    targets = glob.glob(os.path.join(exp.vocab_dir, "*.json"))

    for fp in targets:
        os.remove(fp)

    print(f"[清理] 已删除 {len(targets)} 个词表文件")


def purge_logs(exp: Experiment, from_trial=None):
    """删除训练/评估日志（logs/trial_{N}_*.jsonl）。"""
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        targets = glob.glob(os.path.join(exp.log_dir, "trial*_*.jsonl"))
    else:
        for tid in trial_ids:
            paths = exp.trials[tid].paths
            for fp in (paths.train_log, paths.test_log):
                if os.path.isfile(fp):
                    targets.append(fp)

    for fp in targets:
        os.remove(fp)

    print(f"[清理] 已删除 {len(targets)} 个日志文件 ({_scope_label(from_trial)})")


def purge_trial_results(exp: Experiment, from_trial=None):
    """清除 report.csv 中指定 trial 的结果列（Record 字段 + 口径 + passed）。

    不清除 id/name 这类身份列。
    """
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)
    cleared = ReportTable(exp.report_path).clear_results(trial_ids)
    print(f"[清理] 已清除 {cleared} 个 trial 的评测结果 ({_scope_label(from_trial)})")


# ==================== 便捷组合函数（供 experiment/__main__.py CLI 调用） ====================

def reset_eval(exp: Experiment):
    """清除评测产物：模型权重 + report.csv 结果 + 日志。"""
    purge_models(exp)
    purge_trial_results(exp)
    purge_logs(exp)


def reset_data(exp: Experiment):
    """清除数据产物：训练数据 + 词表。"""
    purge_data(exp)
    purge_vocab(exp)


def reset_full(exp: Experiment):
    """清除模型和数据产物（memory + data + vocab + logs），保留评测记录。"""
    purge_models(exp)
    purge_data(exp)
    purge_vocab(exp)
    purge_logs(exp)


def reset_from(exp: Experiment, trial_id):
    """清除第 trial_id 个 trial 及其后所有 trial 的模型、评测结果与日志。

    保留 L0~trial_id-1 的模型和评测记录。
    """
    purge_models(exp, from_trial=trial_id)
    purge_trial_results(exp, from_trial=trial_id)
    purge_logs(exp, from_trial=trial_id)
