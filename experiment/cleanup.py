"""
cleanup.py — 学习单元产物清理模块

按"产物类别"提供原子清理函数，每个函数支持可选的范围过滤。
experiment/__main__.py 的 --reset 系列命令是原子函数的组合（原细粒度 CLI clean.py 已删除）。

产物类别：
  purge_models          exp 的 brain/*.pt + *.pth（按课号可索引）
  purge_data            exp 的 material/*.jsonl（通过 trial_cfgs 映射到课号）
  purge_vocab           exp 的 vocab/*.json（全局，无范围）
  purge_review          exp 的 material/*_review_*.jsonl（按课号索引）
  purge_logs            exp 的 logs/L{N}_*.jsonl（按课号索引）
  purge_trial_results  exp.csv 的结果列
  purge_standalone      项目根 saved_models/*.pt + *.pth + eval_results.csv（单课模式）

范围参数：
  from_trial=None  → 全部
  from_trial=N     → N 及之后所有课
"""

import glob
import os

from config import Experiment
from experiment.store import ReportTable


def _get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_trial_ids(trial_cfgs, from_trial=None):
    """统一解析课程范围 → 课号集合。None 表示全部。"""
    if from_trial is not None:
        return set(range(from_trial, len(trial_cfgs)))
    return None  # None = 全部


# ==================== 原子清理函数 ====================

def purge_models(exp: Experiment, from_trial=None):
    """删除学习单元模型权重和 checkpoint。

    - study_L{N}.pth（最终通过的权重）
    - checkpoint_L{N}.pt（训练中断点）
    """
    ckpt_dir = exp.ckpt_dir
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        # 全部：glob 通配
        targets += glob.glob(os.path.join(ckpt_dir, "study_L*.pth"))
        targets += glob.glob(os.path.join(ckpt_dir, "checkpoint_L*.pt"))
    else:
        for lid in trial_ids:
            art = exp.trials[lid].artifacts
            for fp in (art.best_model_path, art.checkpoint_path):
                if os.path.isfile(fp):
                    targets.append(fp)

    for fp in targets:
        os.remove(fp)

    scope = f"L{from_trial}~" if from_trial is not None else "全部"
    print(f"[清理] 已删除 {len(targets)} 个模型文件 ({scope})")


def purge_data(exp: Experiment, from_trial=None):
    """删除训练/测试/bead 数据文件（*.jsonl）。

    全清时 glob 匹配；按范围清时遍历 exp.trials 取路径。
    """
    material_dir = exp.material_dir
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        targets = glob.glob(os.path.join(material_dir, "*.jsonl"))
    else:
        for lid in trial_ids:
            cfg = exp.trials[lid]
            m = cfg.material
            for fp in (m.train_data_path, m.test_data_path, m.bead_data_path):
                if fp and os.path.isfile(fp):
                    targets.append(fp)
        # 去重（test_data_path 可能与 train_data_path 相同）
        targets = list(set(targets))

    for fp in targets:
        os.remove(fp)

    scope = f"L{from_trial}~" if from_trial is not None else "全部"
    print(f"[清理] 已删除 {len(targets)} 个数据文件 ({scope})")


def purge_vocab(exp: Experiment):
    """删除学习单元所有词表文件（vocab/*.json）。"""
    targets = glob.glob(os.path.join(exp.vocab_dir, "*.json"))

    for fp in targets:
        os.remove(fp)

    print(f"[清理] 已删除 {len(targets)} 个词表文件")


def purge_review(exp: Experiment, from_trial=None):
    """删除复习临时文件（*_review_*.jsonl）。"""
    material_dir = exp.material_dir
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        targets = glob.glob(os.path.join(material_dir, "*_review_*.jsonl"))
    else:
        for lid in trial_ids:
            cfg = exp.trials[lid]
            train_path = cfg.material.train_data_path
            if train_path:
                pattern = train_path.replace('.jsonl', '_review_*.jsonl')
                targets += glob.glob(pattern)

    for fp in targets:
        if os.path.isfile(fp):
            os.remove(fp)

    scope = f"L{from_trial}~" if from_trial is not None else "全部"
    print(f"[清理] 已删除 {len(targets)} 个复习临时文件 ({scope})")


def purge_logs(exp: Experiment, from_trial=None):
    """删除学习单元日志文件（logs/L{N}_*.jsonl）。

    Args:
        exp: Experiment 对象
        from_trial: None=全部, N=N及之后所有课
    """
    log_dir = exp.log_dir
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)

    targets = []
    if trial_ids is None:
        targets = glob.glob(os.path.join(log_dir, "L*_*.jsonl"))
    else:
        for lid in trial_ids:
            art = exp.trials[lid].artifacts
            for fp in (art.train_log_path, art.test_log_path):
                if os.path.isfile(fp):
                    targets.append(fp)

    for fp in targets:
        os.remove(fp)

    scope = f"L{from_trial}~" if from_trial is not None else "全部"
    print(f"[清理] 已删除 {len(targets)} 个日志文件 ({scope})")


def purge_trial_results(exp: Experiment, from_trial=None):
    """清除 report.csv 中指定课的评估结果列（Record 字段 + passed）。

    不清除 material/method 等静态配置列。
    """
    trial_ids = _resolve_trial_ids(exp.trials, from_trial)
    table = ReportTable(exp.report_csv_path)
    cleared = table.clear_results(trial_ids)
    scope = f"L{from_trial}~" if from_trial is not None else "全部"
    print(f"[清理] 已清除 {cleared} 门课程的评测结果 ({scope})")


def purge_standalone():
    """删除单课模式产物：saved_models/*.pt + *.pth + eval_results.csv。

    这些文件无法按课号索引，只能全删。
    """
    root = _get_project_root()
    targets = []

    targets += glob.glob(os.path.join(root, "saved_models", "*.pt"))
    targets += glob.glob(os.path.join(root, "saved_models", "*.pth"))

    eval_csv = os.path.join(root, "saved_models", "eval_results.csv")
    if os.path.isfile(eval_csv):
        targets.append(eval_csv)

    for fp in targets:
        if os.path.isfile(fp):
            os.remove(fp)

    print(f"[清理] 已删除 {len(targets)} 个单课模式产物")


# ==================== 便捷组合函数（供 experiment/__main__.py CLI 调用） ====================

def reset_eval(exp: Experiment):
    """清除评测产物：模型权重 + exp.csv 结果 + 日志 + 单课模式产物。"""
    purge_models(exp)
    purge_trial_results(exp)
    purge_logs(exp)
    purge_standalone()


def reset_data(exp: Experiment):
    """清除数据产物：训练数据 + 词表 + 复习临时文件。"""
    purge_data(exp)
    purge_vocab(exp)
    purge_review(exp)


def reset_full(exp: Experiment):
    """清除模型和数据产物：brain + material + vocab + logs，保留评测记录。"""
    purge_models(exp)
    purge_data(exp)
    purge_vocab(exp)
    purge_review(exp)
    purge_logs(exp)
    purge_standalone()


def reset_from(exp: Experiment, trial_id):
    """清除第 trial_id 课及其后所有课的模型和评测结果。

    保留 L0~trial_id-1 的模型和评测记录。
    同时清理这些课的复习临时文件和日志。
    """
    purge_models(exp, from_trial=trial_id)
    purge_trial_results(exp, from_trial=trial_id)
    purge_review(exp, from_trial=trial_id)
    purge_logs(exp, from_trial=trial_id)
