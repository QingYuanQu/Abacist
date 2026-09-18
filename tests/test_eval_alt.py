"""alt 判定评测（一题多解结构等价）单元测试。

锁定：load_test_dataset 解析 alt；compute_accuracy 的 acc 在样本带 alt 时
改为「去停止符后的 pred ∈ alt 集合」即正确，无 alt 退回严格串等；
acc_ans / acc_think 不受 alt 影响（始终严格）。
"""
import json
from pathlib import Path
from unittest import mock

from model_lm.eval import _strip_stop, compute_accuracy, load_test_dataset


class _FakeVocab:
    stoi = {"#": 0, "x": 1}
    itos = {0: "#", 1: "x"}
    tokenizer = None
    max_seq_len = 32
    stop_token = "#"
    pad_id = 0
    vocab_size = 2


class _FakeModel:
    def eval(self):
        pass


def test_strip_stop():
    assert _strip_stop("POST 1 2 +#") == "POST 1 2 +"
    assert _strip_stop("POST 1 2 +") == "POST 1 2 +"
    assert _strip_stop("a b #  ") == "a b "


def test_load_test_dataset_parses_alt(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps({"Q": "1+2", "A": "POST 1 2 +#",
                    "alt": "POST 1 2 +|POST 2 1 +"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prompts, expected, alts, total = load_test_dataset(str(p))
    assert total == 1
    assert prompts == ["1+2"]
    assert expected == ["POST 1 2 +#"]
    assert alts[0] == {"POST 1 2 +", "POST 2 1 +"}


def test_load_test_dataset_no_alt_is_none(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps({"Q": "3+4", "A": "POST 3 4 +#"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _, _, alts, _ = load_test_dataset(str(p))
    assert alts == [None]


def test_compute_accuracy_alt_accepts_valid_alternative(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [
        # 多解：模型输出另一个合法解 POST 2 1 + 应判对
        {"Q": "1+2", "A": "POST 1 2 +#", "alt": "POST 1 2 +|POST 2 1 +"},
        # 单解（无 alt）：输出错误应判错
        {"Q": "3+4", "A": "POST 3 4 +#"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")

    fake_preds = ["POST 2 1 +#", "WRONG#"]

    with mock.patch("model_lm.eval.generate_batch", return_value=fake_preds):
        acc, acc_ans, acc_think, correct, *_ = compute_accuracy(
            _FakeModel(), "cpu", _FakeVocab(), 8, 0, test_data_path=str(p))

    assert correct == 1          # 第 1 条命中 alt；第 2 条严格不匹配
    assert acc == 0.5


def test_compute_accuracy_no_alt_strict_only(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"Q": "3+4", "A": "POST 3 4 +#"}]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")

    # 模型输出一个「合法但未被记录为 alt」的等价串——无 alt 时仍判错（退回严格）
    fake_preds = ["POST 4 3 +#"]

    with mock.patch("model_lm.eval.generate_batch", return_value=fake_preds):
        acc, _, _, correct, *_ = compute_accuracy(
            _FakeModel(), "cpu", _FakeVocab(), 8, 0, test_data_path=str(p))

    assert correct == 0
    assert acc == 0.0
