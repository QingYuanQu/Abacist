from experiment.store import ReportTable
from config import Experiment, ExecutionContext
from experiment.trial.train_and_eval import train_and_eval_trial


def _run_experiment(exp: Experiment, start_trial, end_trial, trial_vocabs, device):
    """experiment 语义：每个 trial 从零训练，无复习、无权重传递，失败不阻塞。

    各 trial 独立；词表按数据源去重后共享（trial_vocabs 即
    ensure_experiment_vocab 返回的 {trial_id: vocab}）。
    """
    for trial_id in range(start_trial, end_trial + 1):
        trial = exp.trials[trial_id]

        vocab_data = trial_vocabs.get(trial_id)
        if vocab_data is None:
            print(f"[实验] [WARN] trial {trial_id} 词表缺失，跳过。")
            continue

        # 已通过且记录完整 → 跳过（passed 与 acc 同源，不会出现"无 acc 却 passed"）
        if trial.passed and trial.record:
            print(f"\n[实验] trial {trial_id} ({trial.name}) 已通过 "
                  f"(acc={trial.record.acc*100:.2f}%)，跳过。")
            continue

        ctx = ExecutionContext(
            vocab_data=vocab_data,
            device=device,
            seed=exp.train.train_seed,
        )
        record, passed = train_and_eval_trial(trial_id=trial_id, exp=exp, ctx=ctx)

        # 写回结果（无论通过与否）；acc_mode/pass_threshold 一并落表，让每行自带判定口径
        ReportTable(exp.report_path).save_result(
            trial_id, trial.name, record, passed,
            exp.eval.acc_mode, exp.eval.pass_threshold)

        if not passed:
            print(f"\n[实验] [WARN] trial {trial_id} ({trial.name}) 未通过，继续下一个。")

    print(f"\n{'=' * 60}")
    print(f"[实验] [DONE] 对比实验完成！ L{start_trial} → L{end_trial}")
    print(f"{'=' * 60}")
