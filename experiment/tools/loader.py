"""loader.py — Experiment 的装配层：config.yaml + 磁盘产物 → 聚合根。

职责边界（刻意收窄）：
    experiment/schema.py  认识文件格式（键名、校验）—— 唯一读 config.yaml 的地方
    experiment/loader.py  认识磁盘布局（产物放哪）并把 spec + 结果装配成 Experiment

因此本模块只有三个函数：load / init / clone。
"""

import os

from experiment.config import Experiment, Trial, TrialPaths
from experiment.schema import clone_text, load_spec, skeleton
from experiment.tools.store import ReportTable

# 实验目录下的产物子目录（均为可再生，不参与版本控制）
ARTIFACT_DIRS = ("material", "vocab", "memory", "logs")


def studies_root(project_root: str) -> str:
    """全部实验的父目录: <project_root>/experiment/studies/"""
    return os.path.join(project_root, "experiment", "studies")


def experiment_dir(project_root: str, name: str) -> str:
    """单个实验的目录: <project_root>/experiment/studies/<name>/"""
    return os.path.join(studies_root(project_root), name)


def load_experiment(name: str, project_root: str | None = None) -> Experiment:
    """加载实验：校验 config.yaml，装配 trials，并回填 report.csv 中的结果。

    Args:
        name: 实验名（目录名；config.yaml 里的 name 必须与之一致）
        project_root: 项目根目录，默认自动检测
    Returns:
        Experiment 聚合根
    """
    if project_root is None:
        # 由 experiment 包位置反推仓库根，避免依赖本文件所在层级（曾因挪入 tools/ 而错位）
        import experiment
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(experiment.__file__)))
    root_dir = experiment_dir(project_root, name)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"实验目录不存在: {root_dir}")

    spec = load_spec(os.path.join(root_dir, "config.yaml"))

    exp = Experiment(
        name=name,
        project_root=project_root,
        brain=spec.brain,
        pos_emb=spec.pos_emb,
        train=spec.train,
        eval=spec.eval,
        data_seed=spec.data_seed,
    )
    for sub in ARTIFACT_DIRS:
        os.makedirs(os.path.join(root_dir, sub), exist_ok=True)

    # 结果按 id 回填；name 指纹不符会抛 ReportMismatch（旧结果不可复用）
    results = ReportTable(exp.report_path).load([t.name for t in spec.trials])
    exp.trials = [
        Trial(
            id=ts.id,
            name=ts.name,
            material=ts.material,
            method=ts.method,
            heads=ts.heads,
            paths=TrialPaths.for_trial(exp, ts.id, ts.name, ts.material),
            record=results.get(ts.id, (None, None))[0],
            passed=results.get(ts.id, (None, None))[1],
        )
        for ts in spec.trials
    ]
    return exp


def init_experiment(name: str, project_root: str) -> str:
    """脚手架：创建 experiment/studies/<name>/ 与一份 config.yaml 模板。

    只生成"配置 + 空结果表 + 空产物目录"，不生成任何真实数据。

    Returns:
        新建的 config.yaml 路径
    """
    root_dir = experiment_dir(project_root, name)
    if os.path.isdir(root_dir) and os.listdir(root_dir):
        raise FileExistsError(f"实验目录已存在且非空: {root_dir}")

    for sub in ARTIFACT_DIRS:
        os.makedirs(os.path.join(root_dir, sub), exist_ok=True)
    config_path = os.path.join(root_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(skeleton(name))
    ReportTable(os.path.join(root_dir, "report.csv")).write_header()
    print(f"[init] 已创建 {config_path}（请编辑它填写 trial 定义）")
    return config_path


def clone_experiment(name: str, project_root: str, source_name: str) -> str:
    """克隆：从已有实验复制配置，不复制数据/词表/权重/日志/结果。

    新实验的 report.csv 重建为空表（回到待评估状态）。

    Returns:
        新建的 config.yaml 路径
    """
    src_dir = experiment_dir(project_root, source_name)
    src_config = os.path.join(src_dir, "config.yaml")
    if not os.path.isfile(src_config):
        raise FileNotFoundError(f"源实验配置不存在: {src_config}")

    dst_dir = experiment_dir(project_root, name)
    if os.path.isdir(dst_dir) and os.listdir(dst_dir):
        raise FileExistsError(f"实验目录已存在且非空: {dst_dir}")

    for sub in ARTIFACT_DIRS:
        os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)
    config_path = os.path.join(dst_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(clone_text(src_config, name))
    ReportTable(os.path.join(dst_dir, "report.csv")).write_header()
    print(f"[clone] 已从 {source_name} 克隆出 {config_path}（不含数据/词表/权重/日志）")
    return config_path
