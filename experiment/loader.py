# common/loader.py —— Experiment 的加载 / 脚手架 / CSV IO（common → config，无环）
import json
import os
import shutil
import sys

from config import (
    EvalConfig, Trial, TrialArtifacts,
    Method, Experiment, TrainConfig,
)
from model.config import BrainConfig, PosEmbConfig
from experiment.store import (
    MATERIAL_COLS, REPORT_COLS,
    MaterialTable, MethodTable, ReportTable,
)

# ==================== 公开 API ====================

def load_trials(material_csv: str, method_csv: str, report_csv: str,
                 material_dir: str, default_method: "Method | None" = None) -> list[Trial]:
    """从三表加载全部课程配置（按 id 关联，report 可缺；name 以 material 表为准）。

    default_method: 当 method.csv 缺失或某课无对应行时兜底（experiment 型统一 method 用）。
    """
    methods = {r["id"].strip(): r for r in MethodTable(method_csv).all()}
    reports = {r["id"].strip(): r for r in ReportTable(report_csv).all()}
    trials = []
    for mrow in MaterialTable(material_csv).all():
        lid = mrow["id"].strip()
        row = dict(mrow)
        row.update(methods.get(lid, {}))
        row.update(reports.get(lid, {}))
        row["name"] = mrow["name"]  # name 冗余存于三表，以 material 为权威
        trials.append(Trial.from_row(row, material_dir, default_method))
    return trials


def _dump_json(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[init] 创建 {path}")


def init_experiment(name: str, project_root: str, brain_pattern: str = "regular"):
    """脚手架：创建 experiment/studies/<name>/ 目录结构及默认配置文件。

    生成产物：
        experiment/studies/<name>/
            material.csv     -- 学什么表（仅表头），由用户填写
            method.csv       -- 怎么学表（仅表头），由用户填写
            report.csv       -- 学得怎么样表（仅表头），评估时自动写入
            brain.json       -- 默认架构参数
            policy.json      -- {data_seed}
            eval.json        -- {eval_batch_size, print_interval, acc_mode, pass_threshold}
            train.json       -- {enable_validation, early_stop_patience, use_cosine_schedule, lr_decay_per_trial, train_seed}
            material/ vocab/ memory/ logs/  -- 空子目录

    Args:
        name: 学习单元名称（目录名）
        project_root: 项目根目录
        brain_pattern: 注意力头分布模式（透传给 brain.json 的 attention_heads_pattern）
    """
    root_dir = os.path.join(project_root, "experiment", "studies", name)

    # 检查学习单元是否已存在
    if os.path.isdir(root_dir):
        print(f"[init] 错误: 学习单元 '{name}' 已存在，无法重复创建。")
        print(f"  目录: {root_dir}")
        sys.exit(1)

    # 创建目录结构（config 子目录承载结构化 JSON 配置）
    for sub in ("config", "material", "vocab", "memory", "logs"):
        os.makedirs(os.path.join(root_dir, sub), exist_ok=True)

    # 生成 CSV（仅表头）：experiment 型各课 method 统一由 train.json 合成，无需 method.csv
    csv_files = [("material.csv", MATERIAL_COLS), ("report.csv", REPORT_COLS)]
    for fname, cols in csv_files:
        path = os.path.join(root_dir, fname)
        with open(path, 'w', encoding='utf-8', newline="") as f:
            f.write(",".join(cols) + "\n")
        print(f"[init] 创建 {path}")

    # 表驱动生成各 JSON 配置文件（路径 -> 内容）
    configs = {
        os.path.join(root_dir, "config", "brain.json"): {
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "attention_heads_pattern": brain_pattern,
        },
        os.path.join(root_dir, "config", "pos_emb.json"): {
            "theta": 1e6,
            "scaling": {
                "enabled": False,
                "type": "yarn",
                "factor": 8,
                "original_max_position_embeddings": 128,
                "beta_fast": 4,
                "beta_slow": 1,
                "attention_factor": 1.0,
            },
        },
        os.path.join(root_dir, "config", "policy.json"): {
            "data_seed": 42,
        },
        os.path.join(root_dir, "config", "eval.json"): {
            "eval_batch_size": 1024,
            "print_interval": 1000,
            "acc_mode": "acc",
            "pass_threshold": 0.95,
        },
        os.path.join(root_dir, "config", "train.json"): {
            "enable_validation": True,
            "early_stop_patience": 5,
            "use_cosine_schedule": False,
            "lr_decay_per_trial": 1.0,
            "dropout": 0.0,
            "train_seed": 42,
            # experiment 型共享 method（course 型忽略，以 method.csv 每课为准）
            "epochs": 50,
            "repeat_factor": 1,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "shuffle": True,
        },
    }
    for path, data in configs.items():
        _dump_json(path, data)

    print(f"[init] 学习单元 '{name}' 初始化完成。请编辑 exp.csv 填写 trial 定义。")


def clone_experiment(name: str, project_root: str, source_name: str):
    """克隆脚手架：从已存在的 exp 复制结构配置，不复制数据/词表/模型/日志。

    复制产物（结构 + 配置）：
        brain.json / pos_emb.json / policy.json / eval.json / train.json
        material.csv / method.csv（课定义原样复制）
    其中 report.csv 不复制，重建为空表（回到待评估状态）。

    不复制：
        material/（数据）、vocab/（词表）、memory/（模型）、logs/（日志）

    Args:
        name: 新学习单元名称（目标目录名）
        project_root: 项目根目录
        source_name: 源学习单元名称（已存在的 exp）
    """
    src_root = os.path.join(project_root, "experiment", "studies", source_name)
    if not os.path.isdir(src_root):
        print(f"[clone] 错误: 源学习单元 '{source_name}' 不存在。")
        print(f"  目录: {src_root}")
        sys.exit(1)

    dst_root = os.path.join(project_root, "experiment", "studies", name)
    if os.path.isdir(dst_root):
        print(f"[clone] 错误: 学习单元 '{name}' 已存在，无法重复创建。")
        print(f"  目录: {dst_root}")
        sys.exit(1)

    # 创建目录结构（只建空子目录，不复制内容）
    for sub in ("config", "material", "vocab", "memory", "logs"):
        os.makedirs(os.path.join(dst_root, sub), exist_ok=True)

    # 直接复制的配置文件（5 个 JSON，位于 config/ 子目录）
    for fname in ("brain.json", "pos_emb.json", "policy.json", "eval.json", "train.json"):
        src = os.path.join(src_root, "config", fname)
        dst = os.path.join(dst_root, "config", fname)
        if not os.path.isfile(src):
            print(f"[clone] 警告: 源文件缺失，跳过 {fname}")
            continue
        shutil.copy2(src, dst)
        print(f"[clone] 复制 config/{fname}")

    # material/method 原样复制；report 重建空表
    for fname in ("material.csv", "method.csv"):
        src = os.path.join(src_root, fname)
        if not os.path.isfile(src):
            print(f"[clone] 警告: 源文件缺失，跳过 {fname}")
            continue
        shutil.copy2(src, os.path.join(dst_root, fname))
        print(f"[clone] 复制 {fname}")
    ReportTable(os.path.join(dst_root, "report.csv")).write_header(REPORT_COLS)
    print(f"[clone] 重建空 report.csv")

    print(f"[clone] 学习单元 '{name}' 已从 '{source_name}' 克隆完成（不含数据/词表/模型/日志）。")


def _load_json(path: str) -> dict | None:
    """安全加载 JSON 文件，不存在则返回 None。"""
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _apply_head_patterns(trials, head_patterns: dict, num_layers: int) -> None:
    """按 name(族名)+variant(变体) 查 head_patterns，填每课 heads_per_layer 并校验。

    head_patterns 为空（无 head_patterns.json）时跳过，各课 heads_per_layer 保持 None，
    建模型时经 with_trial_overrides 回退 exp.brain 的 attention_heads_pattern。
    """
    if not head_patterns:
        return
    for trial in trials:
        m = trial.material
        family = head_patterns.get(trial.name)
        if family is None:
            raise ValueError(
                f"head_patterns.json 缺少族 {trial.name!r}（第 {trial.id} 课）")
        heads = family.get(m.variant)
        if heads is None:
            raise ValueError(
                f"head_patterns.json 的族 {trial.name!r} 缺少变体 {m.variant!r}（第 {trial.id} 课）")
        if len(heads) != num_layers:
            raise ValueError(
                f"第 {trial.id} 课 {trial.name}-{m.variant} 的 heads_per_layer 长度 "
                f"{len(heads)} != num_hidden_layers {num_layers}")
        m.heads_per_layer = [int(h) for h in heads]


def load_experiment(name: str, project_root: str | None = None) -> Experiment:
    """从 experiment/studies/<name>/ 目录加载 Experiment。

    目录结构:
        experiment/studies/<name>/
            material.csv     -- 学什么（必选）
            method.csv       -- 怎么学（experiment 可选，缺省由 train.json 合成）
            report.csv       -- 学得怎么样（可选，缺失视为全部未评估）
            brain.json       -- BrainConfig（可选，缺失用默认值）
            policy.json      -- data_seed（必选，缺失报错）
            eval.json        -- EvalConfig（必选，缺失报错）
            train.json       -- TrainConfig + train_seed（必选，缺失报错）

    Args:
        name: 学习单元名称（目录名）
        project_root: 项目根目录，默认自动检测
    Returns:
        Experiment 聚合根对象
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(project_root, "experiment", "studies", name)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"学习单元目录不存在: {root_dir}")

    material_csv = os.path.join(root_dir, "material.csv")
    method_csv = os.path.join(root_dir, "method.csv")
    report_csv = os.path.join(root_dir, "report.csv")
    if not os.path.isfile(material_csv):
        raise FileNotFoundError(f"material.csv 不存在: {material_csv}")
    config_dir = os.path.join(root_dir, "config")
    policy_json_path = os.path.join(config_dir, "policy.json")

    if not os.path.isfile(method_csv):
        # experiment 型 method.csv 可选（各课 method 统一，缺省用 train 合成默认）
        method_csv = ""

    # 加载 brain.json（可选，缺失字段用默认值）
    brain_json = _load_json(os.path.join(config_dir, "brain.json"))
    brain = BrainConfig.from_json(brain_json) if brain_json else BrainConfig()

    # 加载 pos_emb.json（可选，缺失用默认值）
    pos_emb = PosEmbConfig.from_json(_load_json(os.path.join(config_dir, "pos_emb.json")))

    # 加载 policy.json（data_seed 等，缺失报错）
    policy_json = _load_json(policy_json_path)
    if not policy_json:
        raise FileNotFoundError(
            f"policy.json 缺失，请重建学习单元: {policy_json_path}"
        )

    data_seed = policy_json.get("data_seed", 42)

    # 严格加载 eval.json（必须存在）
    eval_json = _load_json(os.path.join(config_dir, "eval.json"))
    if not eval_json:
        raise FileNotFoundError(f"eval.json 缺失，请重建学习单元: {os.path.join(config_dir, 'eval.json')}")
    eval_cfg = EvalConfig.from_json(eval_json)


    # 严格加载 train.json（必须存在）
    train_json = _load_json(os.path.join(config_dir, "train.json"))
    if not train_json:
        raise FileNotFoundError(f"train.json 缺失，请重建学习单元: {os.path.join(config_dir, 'train.json')}")
    train_cfg = TrainConfig.from_json(train_json)

    # 构造 Experiment
    exp = Experiment(
        name=name,
        project_root=project_root,
        brain=brain,
        pos_emb=pos_emb,
        eval=eval_cfg,
        train=train_cfg,
        data_seed=data_seed,
    )
    # 确保子目录存在
    os.makedirs(exp.material_dir, exist_ok=True)
    os.makedirs(exp.vocab_dir, exist_ok=True)
    os.makedirs(exp.ckpt_dir, exist_ok=True)
    os.makedirs(exp.log_dir, exist_ok=True)
    # 加载 trials（三表按 id 关联）；experiment 型缺 method.csv 时用 train 合成默认 Method
    default_method = Method.default_from_train(train_cfg) if not method_csv else None
    exp.trials = load_trials(material_csv, method_csv, report_csv, exp.material_dir, default_method)
    # 加载 head_patterns.json（可选）：按 name(族名)+variant 填每课 heads_per_layer，并校验
    head_patterns = _load_json(os.path.join(config_dir, "head_patterns.json")) or {}
    _apply_head_patterns(exp.trials, head_patterns, brain.num_hidden_layers)
    # 填充每课产物路径（best/checkpoint/train_log/test_log）
    for trial in exp.trials:
        trial.artifacts = TrialArtifacts.for_trial(exp, trial.id)
    return exp
