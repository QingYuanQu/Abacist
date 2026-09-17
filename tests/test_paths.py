"""TrialPaths —— 单个 trial 全部磁盘位置的唯一计算入口。

旧代码把同一个 trial 的位置拆在两处（数据路径挂 Material、产物路径挂
TrialArtifacts），任何一处改名都会错位；本用例把命名约定钉死。
"""
import os

import pytest

from config import Experiment, Material, TrialPaths


@pytest.fixture
def exp(tmp_path):
    return Experiment(name="E", project_root=str(tmp_path))


def test_dataset_paths_shared_by_projection_key(exp):
    """dataset 按 (source, input_format, parse, eval) 四元组共享 train/test。"""
    m = Material(type="dataset", source="parse/dataset/dataset_C.jsonl",
                 input_format="infix", parse="post", eval="none", split=0.2)
    a = TrialPaths.for_trial(exp, 0, "regular-1", m)
    b = TrialPaths.for_trial(exp, 6, "regular-64", m)   # id/name 不同，投影参数相同

    assert a.train_data == b.train_data
    assert a.test_data == b.test_data
    assert os.path.basename(a.train_data) == "dataset_C_infix_post_none_train.jsonl"
    assert os.path.basename(a.test_data) == "dataset_C_infix_post_none_test.jsonl"
    assert a.bead_data == ""


def test_dataset_projection_key_depends_on_eval_and_parse(exp):
    base = dict(type="dataset", source="parse/dataset/dataset_C.jsonl",
                input_format="infix", parse="post", eval="none")
    other_eval = dict(base, eval="digit")
    other_fmt = dict(base, input_format="postfix")
    p0 = TrialPaths.for_trial(exp, 0, "t", Material(**base))
    assert TrialPaths.for_trial(exp, 0, "t", Material(**other_eval)).train_data != p0.train_data
    assert TrialPaths.for_trial(exp, 0, "t", Material(**other_fmt)).train_data != p0.train_data


def test_expr_path_encodes_all_knobs(exp):
    m = Material(type="expr", ops="+-", repeat=2, start=0, end=9,
                 input_format="infix", parse="post", eval="none", split=0.2)
    p = TrialPaths.for_trial(exp, 3, "foo", m)

    assert os.path.basename(p.train_data) == \
        "trial3_foo_expr_+-_post_none_infix_2_0-9.jsonl"
    assert os.path.basename(p.test_data) == \
        "trial3_foo_expr_+-_post_none_infix_2_0-9_test.jsonl"
    assert os.path.basename(p.bead_data) == "trial3_foo_bead.jsonl"


def test_expr_ops_tag_avoids_filesystem_unsafe_chars(exp):
    m = Material(type="expr", ops="&|*/", repeat=2, start=0, end=9, parse="post")
    p = TrialPaths.for_trial(exp, 0, "t", m)
    assert "andormuldiv" in os.path.basename(p.train_data)
    assert not set("&|*/") & set(os.path.basename(p.train_data))


def test_split_non_positive_means_same_file(exp):
    m = Material(type="expr", ops="+-", repeat=2, start=0, end=9, split=0.0)
    p = TrialPaths.for_trial(exp, 0, "t", m)
    assert p.test_data == p.train_data


def test_artifact_names_are_id_based(exp):
    """产物按 trial **id** 命名（name 只做数据文件标签，不参与产物，避免改名就孤立旧权重）。"""
    m = Material(type="expr", ops="+-", repeat=2, start=0, end=9, split=0.2)
    p = TrialPaths.for_trial(exp, 4, "whatever", m)
    assert os.path.basename(p.best_model) == "trial_4.pth"
    assert os.path.basename(p.checkpoint) == "trial_4_ckpt.pt"
    assert os.path.basename(p.train_log) == "trial_4_train.jsonl"
    assert os.path.basename(p.test_log) == "trial_4_test.jsonl"


def test_paths_live_under_experiment_dir(exp):
    m = Material(type="bead", start=0, end=4)
    p = TrialPaths.for_trial(exp, 1, "t", m)
    for path in (p.train_data, p.bead_data, p.best_model, p.checkpoint,
                 p.train_log, p.test_log):
        assert path.startswith(os.path.join(exp.root_dir, ""))
    assert os.path.dirname(p.best_model) == exp.ckpt_dir
    assert os.path.dirname(p.train_log) == exp.log_dir
