import os
from datetime import datetime

import torch
from experiment.trial.eval import eval_one_trial
from config import Experiment, ExecutionContext, Record
from experiment.trial.trial_train import train_one_trial

# ==================== 单 trial 训练+评估循环 ====================
def train_and_eval_trial(trial_id: int, exp: Experiment, ctx: ExecutionContext) -> tuple[Record, bool]:
    """对单个 trial 执行训练和评估。

    验证集 = 测试集，训练中每 epoch 已做完整评估，
    最优 Record 的成绩即最终成绩，无需训练后重复评估。

    Args:
        trial_id: trial 序号（experiment.trials 下标）
        exp: 实验聚合根（含 trials/train/eval/brain）
        ctx: 运行时上下文（vocab_data/device/prev_model_path/review_files/seed）
    Returns:
        (best_record, passed): 最优 epoch 的 Record 与通过判定
    """
    trial = exp.trials[trial_id]
    trial_name = trial.name
    method = trial.method
    pass_threshold = exp.eval.pass_threshold if exp.eval else 0.95
    model_path = trial.paths.best_model

    # 根据 kind 选择用户可见标签
    tag = "[实验]"

    print(f"\n{'=' * 60}")
    print(f"{tag} 开始训练第{trial_id}个 trial: {trial_name}")
    print(f"{tag}   训练参数: epochs={method.epochs}, lr={method.learning_rate}, "
          f"batch={method.batch_size}, repeat_factor={method.repeat_factor}")
    print(f"{tag}   通过阈值: {pass_threshold*100:.0f}%")
    print(f"{'=' * 60}")

    # ---- 零样本迁移评估：用上一 trial 权重直接解当前 trial 测试题 ----
    if ctx.prev_model_path and os.path.isfile(ctx.prev_model_path):
        torch.cuda.empty_cache()
        zs_acc, _, _, _, _, _, _ = eval_one_trial(trial_id,
                                                      exp,
                                                      ctx,
                                                      model_path=ctx.prev_model_path,
                                                      verbose=False)
        print(f"{tag} [零样本迁移] L{trial_id - 1} 权重 → L{trial_id} 测试集: {zs_acc * 100:.2f}%")

    # 训练（每 epoch 产出 Record 写日志，返回最优 Record）
    best_record = train_one_trial(trial_id, exp, ctx)
    torch.cuda.empty_cache()

    # 兜底：无验证 或 训练被跳过（断点续训已完成）时，完整评估一次构造 Record
    if best_record is None:
        print(f"\n{tag} 评估第{trial_id}个 trial...")
        acc, acc_ans, acc_think, correct, correct_ans, correct_think, total = eval_one_trial(
            trial_id, exp, ctx, model_path=model_path, verbose=False)
        best_record = Record(
            epoch=method.epochs,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            acc=acc, acc_ans=acc_ans, acc_think=acc_think,
            correct=correct, correct_ans=correct_ans, correct_think=correct_think,
            total=total,
        )

    # 按 acc_mode 选主指标判定 passed
    acc_mode = exp.eval.acc_mode if exp.eval else "acc"
    primary_acc = {"acc": best_record.acc, "acc_ans": best_record.acc_ans,
                   "acc_think": best_record.acc_think}.get(acc_mode, best_record.acc)

    passed = primary_acc >= pass_threshold
    if passed:
        print(f"{tag} [PASS] 第{trial_id}个 trial {trial_name} 通过！(主指标 {acc_mode}={primary_acc*100:.2f}%"
              f"@epoch {best_record.epoch})")
    else:
        print(f"{tag} [FAIL] 第{trial_id}个 trial {trial_name} 未达标（{primary_acc*100:.2f}% < {pass_threshold*100:.0f}%"
              f"@epoch {best_record.epoch}）")

    return best_record, passed
