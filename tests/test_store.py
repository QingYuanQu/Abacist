"""report.csv 读写 —— 身份指纹与非空合并语义。

两条不变量都在修真实 bug：
  1) 身份指纹：trials 被重排/改名后旧结果不可复用，必须报错而不是错位对上；
  2) 非空合并：eval-only 重跑（Record 无 train_loss/lr/耗时）不得抹掉训练元信息。
"""
import csv
import os

import pytest

from config import Record
from experiment.store import REPORT_COLS, ReportMismatch, ReportTable


def full_record() -> Record:
    return Record(epoch=3, timestamp="2026-09-17 10:00:00", train_loss=0.5, lr=1e-3,
                  acc=0.25, acc_ans=0.0, acc_think=0.0, correct=5, correct_ans=0,
                  correct_think=0, total=20, epoch_time_s=1.5, elapsed_s=4.5)


def partial_record() -> Record:
    """只有评估结果、没有训练元信息（模拟 eval-only 重跑）。"""
    return Record(epoch=3, timestamp="2026-09-17 11:00:00", acc=0.30,
                  correct=6, total=20)


def test_write_header_then_load_empty(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    with open(table.path, encoding="utf-8") as f:
        assert next(csv.reader(f)) == REPORT_COLS
    assert table.load(["a"]) == {}


def test_save_and_load_roundtrip(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    table.save_result(0, "a", full_record(), True, "acc", 0.95)

    out = table.load(["a"])
    record, passed = out[0]
    assert passed is True
    assert record.epoch == 3
    assert record.train_loss == pytest.approx(0.5)
    assert record.acc == pytest.approx(0.25)
    assert record.total == 20


def test_partial_row_without_acc_is_not_a_result(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    table.save_result(0, "a", Record(epoch=1, train_loss=1.0), False, "acc", 0.95)
    # acc 为空 = 未评估 → 不产生 Record（避免"有行但没成绩"被当成已评估）
    record, _ = table.load(["a"])[0]
    assert record is None


def test_non_empty_merge_keeps_training_meta(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    table.save_result(0, "a", full_record(), False, "acc", 0.95)
    table.save_result(0, "a", partial_record(), True, "acc", 0.95)

    record, passed = table.load(["a"])[0]
    assert passed is True
    assert record.acc == pytest.approx(0.30)          # 新值覆盖
    assert record.train_loss == pytest.approx(0.5)    # 旧训练元信息不被空值抹掉
    assert record.lr == pytest.approx(1e-3)
    assert record.epoch_time_s == pytest.approx(1.5)
    assert len(table.all()) == 1                      # 追加而非新行


def test_fingerprint_name_mismatch(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    table.save_result(0, "a", full_record(), True, "acc", 0.95)
    with pytest.raises(ReportMismatch):
        table.load(["renamed"])


def test_fingerprint_id_out_of_range(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    table.save_result(7, "x", full_record(), True, "acc", 0.95)
    with pytest.raises(ReportMismatch):
        table.load(["a"])


def test_non_integer_id_rejected(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    with open(table.path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLS)
        w.writeheader()
        w.writerow({c: "" for c in REPORT_COLS} | {"id": "L0", "name": "a"})
    with pytest.raises(ReportMismatch):
        table.load(["a"])


def test_clear_results_keeps_identity(tmp_path):
    table = ReportTable(str(tmp_path / "report.csv"))
    table.write_header()
    table.save_result(0, "a", full_record(), True, "acc", 0.95)
    table.save_result(1, "b", full_record(), True, "acc", 0.95)

    assert table.clear_results([0]) == 1
    record, passed = table.load(["a", "b"])[0]
    assert record is None and passed is None          # 已清空
    assert table.load(["a", "b"])[1][1] is True       # 未指定的行保持原样
    rows = table.all()
    assert [r["name"] for r in rows] == ["a", "b"]    # id/name 身份列保留
