"""dataset 投影契约测试。

关键点：dataset 源（dataset_C.jsonl 等）自带的难度结构字段
（n / bk / prec_switch / ans_digits / alt）必须透传进 data.jsonl，
否则 bucket_report 的 n×bk 分桶表拿不到维度会静默失效。
这条透传在 2026-09-17 之前是丢掉的（投影只写 {category,Q,A}）。
"""
import json
import os

import pytest

from experiment.domain import Experiment, Data, TrialPaths
from experiment.data_adapter import _generate_dataset

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
         "infix": "1+5", "prefix": "+ 1 5", "postfix": "1 5 +", "answer": 6, "sp": "train"},
        {"n": 5, "bk": 3, "ops": "+-", "prec_switch": 1, "ans_digits": 2,
         "alt": "7 3 - 2 +",
         "infix": "7-3+2", "prefix": "+ - 7 3 2", "postfix": "7 3 - 2 +", "answer": 6, "sp": "test"},
    ])
    m = Data(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none", split=0.2)
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    # 真实管线里 data/ 由 --init 预建；此处模拟该前置条件
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)

    all_rows = [json.loads(l) for p in (paths.train_data, paths.test_data)
                for l in open(p, encoding="utf-8") if l.strip()]
    assert len(all_rows) == 2                      # 两条都被分流到某一侧

    for row in all_rows:
        for k in _STRUCT_FIELDS:
            assert k in row, f"结构字段 {k} 未透传，bucket 分桶将失效"
        assert row["Q"] and row["A"]              # 投影本身仍正常


def test_dataset_projection_keeps_only_known_struct_fields(exp, tmp_path):
    """未知字段不应被无脑塞进 data.jsonl（投影契约是 category/Q/A + 已知结构字段）。"""
    src = tmp_path / "src.jsonl"
    _write_source(src, [
        {"n": 2, "bk": 0, "ops": "+", "stray": "x",
         "infix": "1+5", "prefix": "+ 1 5", "postfix": "1 5 +", "answer": 6, "sp": "train"},
    ])
    m = Data(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none")
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)
    all_rows = [json.loads(l) for p in (paths.train_data, paths.test_data)
                for l in open(p, encoding="utf-8") if l.strip()]
    assert all_rows, "记录应被分流到 train 或 test 一侧"
    row = all_rows[0]
    assert "stray" not in row
    assert "n" in row and "bk" in row


def test_dataset_projection_splits_by_config_and_groups_by_q(exp, tmp_path):
    """P0b：dataset 分流由 config 的 split/seed 驱动，且同一 Q（含多解）整体进同侧防泄漏。"""
    src = tmp_path / "src.jsonl"
    # 多解组：同 Q='1+5' 两棵姊妹树；单解 Q='7-3'。源的 sp 故意相反，验证不被 sp 影响。
    _write_source(src, [
        {"n": 2, "ops": "+", "gid": 1, "tree": "(1+5)",
         "infix": "1+5", "prefix": "+ 1 5", "postfix": "1 5 +", "answer": 6, "Ic": 0, "bk": 0, "sp": "train"},
        {"n": 2, "ops": "+", "gid": 1, "tree": "(5+1)",
         "infix": "1+5", "prefix": "+ 5 1", "postfix": "5 1 +", "answer": 6, "Ic": 0, "bk": 0, "sp": "test"},
        {"n": 3, "ops": "-", "gid": 2, "tree": "(7-3)",
         "infix": "7-3", "prefix": "- 7 3", "postfix": "7 3 -", "answer": 4, "Ic": 0, "bk": 0, "sp": "test"},
    ])
    m = Data(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none", split=0.2)
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)   # seed 默认 0

    train = [json.loads(l) for l in open(paths.train_data, encoding="utf-8") if l.strip()]
    test = [json.loads(l) for l in open(paths.test_data, encoding="utf-8") if l.strip()]
    assert len(train) + len(test) == 3

    # 多解组（同 Q='1+5'）整体进同侧：要么两条都在 train，要么都在 test（防泄漏）
    n_train = sum(1 for r in train if r["Q"].rstrip("=") == "1+5")
    n_test = sum(1 for r in test if r["Q"].rstrip("=") == "1+5")
    assert (n_train == 2 and n_test == 0) or (n_train == 0 and n_test == 2)

    # 确定性：相同 seed 重跑，划分一致（与源 sp 无关）
    _generate_dataset(m, paths)
    train2 = [json.loads(l) for l in open(paths.train_data, encoding="utf-8") if l.strip()]
    test2 = [json.loads(l) for l in open(paths.test_data, encoding="utf-8") if l.strip()]
    assert {r["Q"] for r in train} == {r["Q"] for r in train2}
    assert {r["Q"] for r in test} == {r["Q"] for r in test2}


def _read_rows(*paths):
    rows = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                rows[r["Q"].rstrip("=")] = r    # 投影 Q 带 '='，逻辑 Q 去尾
    return rows


def test_dataset_projection_aggregates_alt_from_real_format(exp, tmp_path):
    """真实 dataset_D 格式：源**不含** alt/prec_switch/ans_digits，只带 gid + 多解姊妹树。

    投影必须按 Q 聚合多解 alt，否则 bucket_report 的结构正确率会退化为严格串等
    （命中率上限 1/14），这正是 dataset_D 引入多解设计要避免的系统性低估。
    此测试用真实格式源，修复前会因透传不存在的键而 KeyError（红），修复后绿。
    """
    src = tmp_path / "src.jsonl"
    # 一个多解组（gid=7，Q='2+3+5' 两棵姊妹树）+ 一条单解（gid=8）
    _write_source(src, [
        {"n": 3, "ops": "++", "gid": 7, "tree": "((2+3)+5)",
         "infix": "2+3+5", "prefix": "+ + 2 3 5", "postfix": "2 3 + 5 +",
         "stack": "2→[2] 3→[2,3] +→[5] 5→[5,5] +→[10]", "answer": 10, "Ic": 0, "bk": 0, "sp": "train"},
        {"n": 3, "ops": "++", "gid": 7, "tree": "(2+(3+5))",
         "infix": "2+3+5", "prefix": "+ 2 + 3 5", "postfix": "2 3 5 + +",
         "stack": "2→[2] 3→[2,3] 5→[2,3,5] +→[2,8] +→[10]", "answer": 10, "Ic": 0, "bk": 0, "sp": "train"},
        {"n": 2, "ops": "+", "gid": 8, "tree": "(1+5)",
         "infix": "1+5", "prefix": "+ 1 5", "postfix": "1 5 +",
         "stack": "1→[1] 5→[1,5] +→[6]", "answer": 6, "Ic": 0, "bk": 0, "sp": "test"},
    ])
    m = Data(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none", split=0.2)
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)

    rows = _read_rows(paths.train_data, paths.test_data)
    # 多解组：每条记录的 alt 必须包含该 Q 的全部合法 post 解
    multi_alt = set(rows["2+3+5"]["alt"].split("|"))
    assert multi_alt == {"2 3 + 5 +", "2 3 5 + +"}, multi_alt
    # 单解：alt 即自身那条 post
    assert rows["1+5"]["alt"] == "1 5 +"


def test_dataset_projection_computes_prec_switch_and_ans_digits(exp, tmp_path):
    """真实 dataset_D 格式：源**不含** prec_switch/ans_digits，必须现算（而非透传）。

    修复前输出无这两个键，断言 KeyError（红）；修复后 bucket_report 的表 B/表 C
    才能拿到维度出表，而非静默空白。
    """
    src = tmp_path / "src.jsonl"
    _write_source(src, [
        {"n": 5, "ops": "+-", "gid": 1, "tree": "((7-3)+2)",
         "infix": "7-3+2", "prefix": "+ - 7 3 2", "postfix": "7 3 - 2 +",
         "stack": "7→[7] 3→[7,3] -→[4] 2→[4,2] +→[6]", "answer": 6, "Ic": 0, "bk": 3, "sp": "test"},
        {"n": 3, "ops": "+×", "gid": 2, "tree": "(4+2×3)",
         "infix": "4+2×3", "prefix": "+ 4 × 2 3", "postfix": "4 2 3 × +",
         "stack": "4→[4] 2→[4,2] 3→[4,2,3] ×→[4,6] +→[10]", "answer": 10, "Ic": 0, "bk": 0, "sp": "train"},
    ])
    m = Data(type="dataset", source=str(src),
                 input_format="infix", parse="post", eval="none")
    paths = TrialPaths.for_trial(exp, 0, "t", m)
    os.makedirs(os.path.dirname(paths.train_data), exist_ok=True)
    _generate_dataset(m, paths)

    rows = _read_rows(paths.train_data, paths.test_data)
    # 7-3+2：运算符 +,- 优先级相同 → 0 次切换
    assert rows["7-3+2"]["prec_switch"] == 0
    # 4+2×3：运算符 +,× 优先级不同 → 1 次切换
    assert rows["4+2×3"]["prec_switch"] == 1
    # ans_digits：answer 十进制位数
    assert rows["7-3+2"]["ans_digits"] == 1    # 6
    assert rows["4+2×3"]["ans_digits"] == 2    # 10
