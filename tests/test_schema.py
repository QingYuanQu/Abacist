"""config.yaml 的唯一加载器（experiment/schema.py）—— 键白名单、作用域与三轴校验。

这些用例是配置格式的回归网：旧格式（material.csv + config/*.json）之所以能
静默跑出错误结论，就是因为"拼错的键永远不报错"。凡白名单/校验被放宽，
这里的用例必须先红。
"""
import copy
import os

import pytest

from experiment.schema import (ConfigError, UnsupportedFeature, load_spec,
                               parse_spec)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "experiment", "studies")


def raw_experiment() -> dict:
    """一份最小可用配置（两个 expr trial，头结构各写一种写法）。"""
    return {
        "name": "EXP",
        "data_seed": 42,
        "brain": {"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2},
        "pos_emb": {"theta": 1.0e6, "pos_type": "rope", "rope_type": "default"},
        "train": {"enable_validation": True, "early_stop_patience": 5, "dropout": 0.0},
        "eval": {"eval_batch_size": 8, "print_interval": 10},
        "trial_defaults": {
            "method": {"epochs": 1, "repeat_factor": 1, "batch_size": 4,
                       "learning_rate": 1.0e-3, "shuffle": True},
            "material": {"type": "expr", "ops": "+-", "repeat": 2, "start": 0, "end": 9},
        },
        "trials": [
            {"name": "a", "heads": [4, 4]},
            {"name": "b", "heads": [2, 2]},
        ],
    }


def errors_of(raw, where="EXP") -> list[str]:
    """断言解析失败并返回错误列表。"""
    with pytest.raises(ConfigError) as ei:
        parse_spec(copy.deepcopy(raw), where)
    return ei.value.errors


# ==================== 正常路径 ====================

def test_minimal_config_ok():
    spec = parse_spec(copy.deepcopy(raw_experiment()), "EXP")
    assert spec.name == "EXP"
    assert spec.data_seed == 42
    assert [t.id for t in spec.trials] == [0, 1]
    assert [t.name for t in spec.trials] == ["a", "b"]
    assert spec.trials[0].heads == [4, 4]
    assert spec.trials[0].method.epochs == 1
    # trial_defaults.material 被继承，trial 里没重复写
    assert spec.trials[1].material.ops == "+-"
    assert spec.trials[1].material.type == "expr"


def test_heads_default_is_uniform():
    """heads 缺席 → [num_attention_heads] * num_hidden_layers（等价旧 regular）。"""
    raw = raw_experiment()
    for t in raw["trials"]:
        t.pop("heads")
    spec = parse_spec(raw, "EXP")
    assert spec.trials[0].heads == [4, 4]


def test_trial_material_merges_over_default():
    """trial 的 material 是 defaults 的差量：缺的字段继承，同名键覆盖。"""
    raw = raw_experiment()
    raw["trials"][1]["material"] = {"repeat": 5}        # 只写差量，其余继承
    spec = parse_spec(raw, "EXP")
    assert spec.trials[1].material.repeat == 5          # 差量覆盖
    assert spec.trials[1].material.ops == "+-"          # 其余字段继承 defaults
    assert spec.trials[1].material.type == "expr"
    # 同名键覆盖
    raw = raw_experiment()
    raw["trials"][1]["material"] = {"ops": "*/"}
    assert parse_spec(raw, "EXP").trials[1].material.ops == "*/"


def test_acc_mode_from_registry():
    raw = raw_experiment()
    raw["eval"]["acc_mode"] = "acc_ans"
    assert parse_spec(raw, "EXP").eval.acc_mode == "acc_ans"


# ==================== 键白名单与作用域 ====================

def test_unknown_top_key_rejected():
    raw = raw_experiment()
    raw["material.csv"] = "legacy"
    assert any("material.csv" in e for e in errors_of(raw))


def test_experiment_level_key_in_trial_rejected():
    """trial 里偷写 brain/train/eval → 报错（防止某课偷改 lr 导致不可比）。"""
    raw = raw_experiment()
    raw["trials"][0]["lr"] = 0.1
    errs = errors_of(raw)
    assert any("lr" in e and "trials[0]" in e for e in errs)


def test_heads_per_layer_not_a_brain_key():
    """唯一的头分布入口是 trial 级 heads；brain 里写 heads_per_layer 要报错。"""
    raw = raw_experiment()
    raw["brain"]["heads_per_layer"] = [4, 4]
    errs = errors_of(raw)
    assert any("heads_per_layer" in e for e in errs)


def test_unknown_method_field_rejected():
    raw = raw_experiment()
    raw["trial_defaults"]["method"]["lr_decay_per_trial"] = 0.9
    assert any("lr_decay_per_trial" in e for e in errors_of(raw))


def test_method_requires_all_fields():
    raw = raw_experiment()
    raw["trial_defaults"]["method"].pop("batch_size")
    assert any("batch_size" in e for e in errors_of(raw))


# ==================== 身份 ====================

def test_name_must_match_directory():
    assert any("目录名" in e for e in errors_of(raw_experiment(), where="OTHER"))


def test_duplicate_trial_name_rejected():
    raw = raw_experiment()
    raw["trials"][1]["name"] = "a"
    errs = errors_of(raw)
    assert any("重复" in e for e in errs)


def test_trial_name_rejects_path_separator():
    raw = raw_experiment()
    raw["trials"][0]["name"] = "a/b"
    assert any("路径分隔符" in e for e in errors_of(raw))


# ==================== 头结构三轴 ====================

def test_heads_length_must_equal_num_layers():
    raw = raw_experiment()
    raw["trials"][0]["heads"] = [4, 4, 4]
    assert any("num_hidden_layers" in e for e in errors_of(raw))


def test_heads_must_divide_hidden_size():
    raw = raw_experiment()
    raw["trials"][0]["heads"] = [3, 3]
    assert any("整除" in e for e in errors_of(raw))


def test_heads_and_head_dims_are_mutually_exclusive():
    raw = raw_experiment()
    raw["trials"][0]["head_dims"] = [[32, 32], [32, 32]]
    assert any("互斥" in e for e in errors_of(raw))


def test_head_dims_sum_must_equal_hidden_size():
    raw = raw_experiment()
    raw["trials"][0].pop("heads")
    raw["trials"][0]["head_dims"] = [[32, 16], [32, 32]]
    assert any("hidden_size" in e for e in errors_of(raw))


def test_head_dims_declared_but_unsupported():
    """异质头宽是"声明未实现"：配置合法但拒绝加载，绝不静默忽略。"""
    raw = raw_experiment()
    raw["trials"][0].pop("heads")
    raw["trials"][0]["head_dims"] = [[32, 32], [32, 32]]
    with pytest.raises(UnsupportedFeature) as ei:
        parse_spec(raw, "EXP")
    assert "head_dims" in str(ei.value)


def test_theta_per_head_declared_but_unsupported():
    raw = raw_experiment()
    raw["trials"][0]["theta_per_head"] = [1e7, 1e6, 1e3, 1e2]
    with pytest.raises(UnsupportedFeature) as ei:
        parse_spec(raw, "EXP")
    assert "theta_per_head" in str(ei.value)


def test_theta_per_head_length_is_per_head_not_per_layer():
    """theta_per_head 逐头对齐 → 长度 = 每层头数（不是层数）。"""
    raw = raw_experiment()
    raw["trials"][0]["theta_per_head"] = [1e7, 1e6]      # 层数=2，但每层 4 头
    errs = errors_of(raw)
    assert any("每层头数" in e for e in errs)


def test_theta_per_head_requires_uniform_head_count():
    raw = raw_experiment()
    raw["trials"][0]["heads"] = [2, 4]                   # 各层头数不一致 → 头序无法对齐
    raw["trials"][0]["theta_per_head"] = [1e7, 1e6]
    errs = errors_of(raw)
    assert any("各层头数一致" in e for e in errs)


# ==================== material / 枚举 ====================

def test_dataset_requires_source():
    raw = raw_experiment()
    raw["trials"][0]["material"] = {"type": "dataset"}
    assert any("source" in e for e in errors_of(raw))


def test_dataset_parse_mode_restricted():
    """dataset 走纯结构投影，只认 pre/post；其余会被静默当成 post。"""
    raw = raw_experiment()
    raw["trials"][0]["material"] = {"type": "dataset", "source": "d.jsonl", "parse": "fixed"}
    assert any("只支持" in e for e in errors_of(raw))


def test_expr_requires_enumeration_range():
    raw = raw_experiment()
    raw.pop("trial_defaults")        # 去掉基值，单独验证 expr 必填枚举范围（合并后仍缺字段才报错）
    raw["trials"][0]["material"] = {"type": "expr", "ops": "+-"}
    errs = errors_of(raw)
    assert any("repeat" in e for e in errs)


def test_unknown_attn_type_rejected():
    raw = raw_experiment()
    raw["brain"]["attn_type"] = "turbo"
    assert any("attn_type" in e for e in errors_of(raw))


def test_unknown_pos_emb_type_rejected():
    raw = raw_experiment()
    raw["pos_emb"]["pos_type"] = "quantum"
    assert any("pos_type" in e for e in errors_of(raw))


def test_unknown_acc_mode_rejected():
    raw = raw_experiment()
    raw["eval"]["acc_mode"] = "vibes"
    assert any("acc_mode" in e for e in errors_of(raw))


# ==================== 标量类型归一（YAML 1.1 陷阱） ====================

def test_yaml_11_exponent_string_coerced_to_float():
    """`theta: 1.0e6` 在 YAML 1.1 里是字符串；配置入口必须转正，不能漏到 RoPE 里。"""
    raw = raw_experiment()
    raw["pos_emb"]["theta"] = "1.0e6"
    assert parse_spec(raw, "EXP").pos_emb.theta == 1000000.0


def test_unparsable_number_rejected_with_yaml_tip():
    raw = raw_experiment()
    raw["pos_emb"]["theta"] = "1e6秒"
    errs = errors_of(raw)
    assert any("theta" in e and "YAML 1.1" in e for e in errs)


def test_numeric_string_for_int_field_coerced():
    raw = raw_experiment()
    raw["eval"]["eval_batch_size"] = "8"
    assert parse_spec(raw, "EXP").eval.eval_batch_size == 8


def test_fractional_value_for_int_field_rejected():
    raw = raw_experiment()
    raw["eval"]["print_interval"] = 1.5
    assert any("print_interval" in e for e in errors_of(raw))


def test_bool_string_coerced_but_bool_field_rejects_other_strings():
    raw = raw_experiment()
    raw["train"]["enable_validation"] = "false"
    assert parse_spec(raw, "EXP").train.enable_validation is False

    raw = raw_experiment()
    raw["train"]["enable_validation"] = "half"
    assert any("enable_validation" in e for e in errors_of(raw))


def test_int_field_rejects_bool():
    """Python 里 bool 是 int 的子类，`enable_validation: 1` 不该被当成等价写法。"""
    raw = raw_experiment()
    raw["eval"]["print_interval"] = True
    assert any("print_interval" in e for e in errors_of(raw))


def test_string_field_rejects_number():
    raw = raw_experiment()
    raw["brain"]["attn_type"] = 1
    assert any("attn_type" in e for e in errors_of(raw))


# ==================== 错误聚合 ====================

def test_all_errors_reported_at_once():
    """一次收集全部错误，避免"改一处、跑一次"。"""
    raw = raw_experiment()
    raw["trials"][0]["heads"] = [3, 3]
    raw["eval"]["acc_mode"] = "vibes"
    raw["brain"]["attn_type"] = "turbo"
    errs = errors_of(raw)
    assert len(errs) >= 3
    assert any("整除" in e for e in errs)
    assert any("acc_mode" in e for e in errs)
    assert any("attn_type" in e for e in errs)


# ==================== 仓库内真实配置 ====================

def test_repo_configs_load_or_declare_unsupported():
    """9 个真实实验：要么能加载，要么明确以 UnsupportedFeature 拒绝（PATTERN_A/B）。"""
    names = sorted(d for d in os.listdir(STUDIES)
                   if not d.startswith("_")          # 下划线开头 = 临时/冒烟实验
                   and os.path.isfile(os.path.join(STUDIES, d, "config.yaml")))
    assert len(names) == 9
    for name in names:
        path = os.path.join(STUDIES, name, "config.yaml")
        try:
            spec = load_spec(path)
        except UnsupportedFeature:
            assert name in ("PATTERN_A", "PATTERN_B"), f"{name} 不应被拒绝加载"
            continue
        assert spec.name == name
        assert spec.trials, f"{name} 无 trial"
        assert len({t.name for t in spec.trials}) == len(spec.trials)
