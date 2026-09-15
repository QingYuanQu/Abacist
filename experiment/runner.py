from experiment.store import ReportTable
from config import Experiment, ExecutionContext
from experiment.trial.train_and_eval import train_and_eval_trial


def _run_experiment(exp: Experiment, start_trial, end_trial, trial_vocabs, device):
    """experiment 语义：每课从零训练，无复习，失败不阻塞。

    各 trial 独立：prev_ckpt=None, review_files=[], 失败仅记录继续下一个。
    词表按数据源去重后共享（trial_vocabs 即 ensure_experiment_vocab 返回的字典）。
    """
    for trial_id in range(start_trial, end_trial + 1):
        trial_cfg = exp.trials[trial_id]

        vocab_data = trial_vocabs.get(trial_id)
        if vocab_data is None:
            print(f"[实验] [WARN] 第{trial_id}课词表缺失，跳过。")
            continue

        # 检查该课是否已通过
        if trial_cfg.passed and trial_cfg.record:
            print(f"\n[实验] 第{trial_id}课已通过 (acc={trial_cfg.record.acc*100:.2f}%)，跳过。")
            continue

        ctx = ExecutionContext(
            vocab_data=vocab_data,
            device=device,
            epochs=exp.train.epochs,
            seed=exp.train.train_seed,
        )
        record, passed = train_and_eval_trial(trial_id=trial_id, exp=exp, ctx=ctx)

        # 写回结果到 report.csv（无论通过与否）
        ReportTable(exp.report_csv_path).save_result(trial_id, trial_cfg.display_name, record, passed)

        if passed:
            pass
        else:
            print(f"\n[实验] [WARN] 第{trial_id}课 {trial_cfg.name} 未通过，继续下一课。")

    print(f"\n{'=' * 60}")
    print(f"[实验] [DONE] 对比实验完成！ L{start_trial} → L{end_trial}")
    print(f"{'=' * 60}")
