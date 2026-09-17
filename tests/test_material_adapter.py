"""dataset 投影契约测试。

关键点：dataset 源（dataset_C.jsonl 等）自带的难度结构字段
（n / bk / prec_switch / ans_digits / alt）必须透传进 material.jsonl，
否则 bucket_report 的 n×bk 分桶表拿不到维度会静默失效。
这条透传在 2026-09-17 之前是丢掉的（投影只写 {category,Q,A}）。
"""
import json
import os

import pytest

from config import Experiment, Material, TrialPaths
from experiment.material_adapter import _generate_dataset

_STRUCT_FIELDS = ("n", "bk", "prec_switch", "ans_digits", "alt")


@pytest.fixture
def exp(tmp_path):
    return Experiment(name="E", project_root=str(tmp_path))


def _write_source(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_dataset_projection_passthrough_struct_fields(exp, tmp_path):
    src = tmp_path / "src.jsonl"
    _write_source(src, [
        {"n": 2, "bk": 0, "ops": "+", "prec_switch": 0, "ans_digits": 1,
         "alt": "1 5 +|5 1 +",
         "Q": "1+5", "pre": "+ 1 5", "post": "1 5 +", "ANS": 6, "sp": "train"},
        {"n": 5, "bk": 3, "ops": "+-", "prec_switch": 1, "ans_digits": 2,
         "alt": "7 3 - 2 +",
         "Q": "7-3+2", "pre": "+ - 7 3 2", "post": "7 3 - 2 +", "ANS": 6, "sp": "test"},
    ])
    m = Material(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none", split=0.2)
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    # 真实管线里 material/ 由 --init 预建；此处模拟该前置条件
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)

    train = [json.loads(l) for l in open(paths.train_data, encoding="utf-8") if l.strip()]
    test = [json.loads(l) for l in open(paths.test_data, encoding="utf-8") if l.strip()]
    assert len(train) == 1 and len(test) == 1

    for row in (train[0], test[0]):
        for k in _STRUCT_FIELDS:
            assert k in row, f"结构字段 {k} 未透传，bucket 分桶将失效"
        assert row["Q"] and row["A"]              # 投影本身仍正常


def test_dataset_projection_keeps_only_known_struct_fields(exp, tmp_path):
    """未知字段不应被无脑塞进 material.jsonl（投影契约是 category/Q/A + 已知结构字段）。"""
    src = tmp_path / "src.jsonl"
    _write_source(src, [
        {"n": 2, "bk": 0, "ops": "+", "stray": "x",
         "Q": "1+5", "pre": "+ 1 5", "post": "1 5 +", "ANS": 6, "sp": "train"},
    ])
    m = Material(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none")
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)
    row = json.loads(open(paths.train_data, encoding="utf-8").readline())
    assert "stray" not in row
    assert "n" in row and "bk" in row


def test_dataset_projection_routes_by_sp_field(exp, tmp_path):
    src = tmp_path / "src.jsonl"
    _write_source(src, [
        {"n": 2, "ops": "+", "Q": "1+5", "pre": "+ 1 5", "post": "1 5 +", "ANS": 6, "sp": "train"},
        {"n": 3, "ops": "-", "Q": "7-3", "pre": "- 7 3", "post": "7 3 -", "ANS": 4, "sp": "test"},
    ])
    m = Material(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none")
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)
    train = [json.loads(l) for l in open(paths.train_data, encoding="utf-8") if l.strip()]
    test = [json.loads(l) for l in open(paths.test_data, encoding="utf-8") if l.strip()]
    assert len(train) == 1 and train[0]["n"] == 2
    assert len(test) == 1 and test[0]["n"] == 3
