"""loader —— config.yaml + 磁盘产物 → Experiment 聚合根。

loader 刻意只有 load / init / clone 三个函数：认识"文件格式"的是 schema，
认识"磁盘布局"的才是 loader。这里守住这条边界。
"""
import os

import pytest

from experiment.config import Experiment
from experiment.tools.loader import (ARTIFACT_DIRS, clone_experiment, experiment_dir,
                               init_experiment, load_experiment, studies_root)

NAME = "TMP_EXP"


def test_init_creates_scaffold(tmp_path):
    config_path = init_experiment(NAME, str(tmp_path))
    assert os.path.isfile(config_path)
    assert os.path.isfile(os.path.join(experiment_dir(str(tmp_path), NAME), "report.csv"))
    for sub in ARTIFACT_DIRS:
        assert os.path.isdir(os.path.join(experiment_dir(str(tmp_path), NAME), sub))


def test_init_then_load(tmp_path):
    init_experiment(NAME, str(tmp_path))
    exp = load_experiment(NAME, str(tmp_path))

    assert isinstance(exp, Experiment)
    assert exp.root_dir == os.path.join(studies_root(str(tmp_path)), NAME)
    assert len(exp.trials) == 1
    trial = exp.trials[0]
    # heads 缺席 → 均匀默认（skeleton 里 brain 是 4 头 × 5 层）
    assert trial.heads == [exp.brain.num_attention_heads] * exp.brain.num_hidden_layers
    assert trial.record is None and trial.passed is None
    assert trial.paths.best_model.endswith("trial_0.pth")


def test_init_refuses_non_empty_dir(tmp_path):
    init_experiment(NAME, str(tmp_path))
    with pytest.raises(FileExistsError):
        init_experiment(NAME, str(tmp_path))


def test_load_missing_experiment(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment("NOPE", str(tmp_path))


def test_clone_copies_config_but_not_results(tmp_path):
    init_experiment(NAME, str(tmp_path))
    clone_experiment("TMP_CLONE", str(tmp_path), NAME)

    cloned = load_experiment("TMP_CLONE", str(tmp_path))
    source = load_experiment(NAME, str(tmp_path))
    assert cloned.name == "TMP_CLONE"
    assert len(cloned.trials) == len(source.trials)
    assert [t.name for t in cloned.trials] == [t.name for t in source.trials]
    assert cloned.trials[0].record is None
    # 克隆不复制数据/词表/权重/日志
    assert not os.listdir(cloned.material_dir) if os.path.isdir(cloned.material_dir) else True


def test_load_backfills_results_from_report_csv(tmp_path):
    from model.record import Record
    from experiment.tools.store import ReportTable

    init_experiment(NAME, str(tmp_path))
    exp = load_experiment(NAME, str(tmp_path))
    record = Record(epoch=2, timestamp="t", train_loss=1.0, lr=1e-3, acc=0.5, total=10)
    ReportTable(exp.report_path).save_result(0, exp.trials[0].name, record, False, "acc", 0.95)

    reloaded = load_experiment(NAME, str(tmp_path))
    assert reloaded.trials[0].record.acc == pytest.approx(0.5)
    assert reloaded.trials[0].passed is False
