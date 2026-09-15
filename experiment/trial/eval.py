"""单课评估 + 单课 CLI（课程编排层，2026-09-04 从 model_lm/eval.py 迁出）。

迁出原因：eval_one_trial 依赖 experiment/trial 聚合对象（exp.trials[].artifacts/material、
exp.brain/pos_emb/eval、ReportTable 报告回写），属于课程编排而非通用 LM 能力；
底层的批量推理、准确率统计留在 model_lm/eval.py，此处只做课程侧装配。

用法：
    python -m experiment.trial.eval --experiment default --trial 0 --batch
    python -m experiment.trial.eval --experiment default --trial 0 --chat
"""
import argparse
import os
from datetime import datetime

import torch

from experiment.store import ReportTable
from experiment.loader import load_experiment
from config import ExecutionContext, Record
from model.config import ModelConfig
from model import GPT
from model.vocab import load_vocab
from model_lm.eval import (          # 通用 LM 能力（含内部自检 _verify_generate_batch）
    _verify_generate_batch,
    chat,
    compute_accuracy,
    load_test_dataset,
)


def eval_one_trial(trial_id: int, exp, ctx, model_path=None, verbose=True):
    """评估单课测试集，返回三种口径准确率与计数。

    Args:
        trial_id: 单课配置（material/artifacts）
        exp: 学习单元聚合根（brain/eval）
        ctx: 运行时上下文（vocab_data/device）
        model_path: 模型权重路径（默认 trial.artifacts.best_model_path）
        verbose: 是否打印详细信息
    Returns:
        (acc, acc_ans, acc_think, correct, correct_ans, correct_think, total)
    """
    trial = exp.trials[trial_id]

    brain_config = exp.brain.with_trial_overrides(trial.material)
    eval_config = exp.eval
    vocab_data = ctx.vocab_data
    device = ctx.device

    if model_path is None:
        model_path = trial.artifacts.best_model_path

    # stop_token 统一从词表读取，消除调用方散传导致的不一致
    stop_token = vocab_data.stop_token


    stoi = vocab_data.stoi
    itos = vocab_data.itos
    vocab_size = vocab_data.vocab_size
    tokenizer = vocab_data.tokenizer
    max_seq_len = vocab_data.max_seq_len

    model_config = ModelConfig.from_sources(brain_config, exp.pos_emb, vocab_data)
    model = GPT(model_config)
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 测试数据路径
    test_data_path = trial.material.test_data_path
    print(f"[评估] 测试数据: {test_data_path}")

    if not test_data_path or not os.path.isfile(test_data_path):
        print(f"[评估] 测试数据不存在: {test_data_path}")
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    prompts, expected, total = load_test_dataset(test_data_path)
    if total == 0:
        print(f"[评估] 测试集为空: {test_data_path}")
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    # 小样本阈值：低于此数打印全部结果（含每条详情），否则按间隔采样
    small_sample_threshold = 50
    is_small = total <= small_sample_threshold

    if verbose or is_small:
        print(f"\n开始测试 {total} 个样本...")
        print("=" * 60)

    pad_id = vocab_data.pad_id
    if eval_config is not None:
        eval_batch_size = eval_config.eval_batch_size
    else:
        eval_batch_size = 1024

    # ---- 等价性自检（仅小样本时触发，确保批量生成正确） ----
    if is_small:
        stop_ids = {stoi[stop_token]}
        _verify_generate_batch(model, device, stoi, itos, tokenizer,
                               max_seq_len, stop_ids, pad_id,
                               prompts)

    # ---- 分块批量推理（共用 compute_accuracy，返回完整 predictions） ----
    accuracy, acc_ans, acc_think, correct, correct_ans, correct_think, total, predictions = compute_accuracy(
        model, device, vocab_data, eval_batch_size, pad_id,
        prompts=prompts, expected=expected,
        max_seq_len=max_seq_len, stop_token=stop_token
    )
    print_interval = eval_config.print_interval

    # ---- 逐条打印 ----
    if is_small:
        for i, (pred, exp) in enumerate(zip(predictions, expected)):
            idx = i + 1
            is_correct = pred == exp
            status = "✅" if is_correct else "❌"
            print(f"[{idx}/{total}] {status}  \tQ: {prompts[i]}", end="\t")
            print(f"期望: {exp}\t预测: {pred}\t累计准确率: {correct/idx*100:.2f}%")
    else:
        for i, (pred, exp) in enumerate(zip(predictions, expected)):
            idx = i + 1
            is_correct = pred == exp
            if verbose and (idx % print_interval == 0 or idx == total):
                status = "✅" if is_correct else "❌"
                pred_flat = pred.replace("\n", " ")
                exp_flat = exp.replace("\n", " ")
                print(f"[{idx}/{total}] {status}  \tQ: {prompts[i]}", end="\t")
                print(f"期望: {exp_flat}\t预测: {pred_flat}\t累计准确率: {correct/idx*100:.2f}%")

    if verbose or is_small:
        print("=" * 60)
        print(f"\n测试结果汇总:")
        print(f"  总样本数: {total}")
        print(f"  整串正确: {correct}")
        print(f"  答案正确(ANS): {correct_ans}")
        print(f"  思考正确(think): {correct_think}")
        print(f"  整串准确率(acc): {accuracy*100:.2f}%")
        print(f"  答案准确率(acc_ans): {acc_ans*100:.2f}%")
        print(f"  思考准确率(acc_think): {acc_think*100:.2f}%")

    return accuracy, acc_ans, acc_think, correct, correct_ans, correct_think, total


# ---------- 主程序（单课评估 CLI） ----------
def main():
    parser = argparse.ArgumentParser(description="Abacist 评估")
    parser.add_argument("--experiment", type=str, default="default",
                        help="学习单元名称（对应 experiment/studies/<name>/ 目录）")
    parser.add_argument("--trial", type=int, default=0,
                        help="指定第 N 课的模型")
    parser.add_argument("--model", type=str, default=None,
                        help="单课模式：指定模型权重路径（覆盖 SAVE['model_path']）")
    parser.add_argument("--batch", action="store_true",
                        help="批量评估模式：对指定课测试集做批量推理并统计准确率")
    parser.add_argument("--chat", action="store_true",
                        help="聊天模式：加载指定课模型后交互式生成")

    args = parser.parse_args()

    exp = load_experiment(args.experiment)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 聊天模式 =====
    if args.chat:
        trial_id = args.trial
        if trial_id < 0 or trial_id >= len(exp.trials):
            print(f"[聊天] 错误：trial={trial_id} 超出范围（0~{len(exp.trials)-1}）")
            return
        trial = exp.trials[trial_id]
        model_path = trial.artifacts.best_model_path
        if not os.path.isfile(model_path):
            print(f"[聊天] 错误：课程模型不存在: {model_path}")
            print(f"[聊天] 请先运行: python -m experiment --start {trial_id} --end {trial_id}")
            return

        vocab_path = exp.vocab_path
        if not os.path.isfile(vocab_path):
            print(f"[聊天] 错误：词表不存在: {vocab_path}")
            return
        vocab_data = load_vocab(vocab_path)

        model = GPT(ModelConfig.from_sources(
            exp.brain.with_trial_overrides(trial.material), exp.pos_emb, vocab_data))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        stop_ids = {vocab_data.stoi[vocab_data.stop_token]}
        chat(model, device, vocab_data.stoi, vocab_data.itos, vocab_data.tokenizer,
             vocab_data.max_seq_len, stop_ids)
        return

    # ===== 批量评估模式：复用 eval_one_trial =====
    if args.batch:
        trial_id = args.trial
        if trial_id < 0 or trial_id >= len(exp.trials):
            print(f"[评估] 错误：trial={trial_id} 超出范围（0~{len(exp.trials)-1}）")
            return
        trial = exp.trials[trial_id]
        model_path = trial.artifacts.best_model_path
        if not os.path.isfile(model_path):
            print(f"[评估] 错误：课程模型不存在: {model_path}")
            print(f"[评估] 请先运行: python -m experiment --start {trial_id} --end {trial_id}")
            return

        # 统一共享词表
        vocab_path = exp.vocab_path
        if not os.path.isfile(vocab_path):
            print(f"[评估] 错误：词表不存在: {vocab_path}")
            return
        vocab_data = load_vocab(vocab_path)
        ctx = ExecutionContext(vocab_data=vocab_data, device=device)
        accuracy, acc_ans, acc_think, correct, correct_ans, correct_think, total = eval_one_trial(
            trial_id, exp=exp, ctx=ctx, model_path=model_path, verbose=True)
        # 写回结果到 report.csv（独立评估无训练过程信息，仅记录评估字段）
        acc_mode = exp.eval.acc_mode if exp.eval else "acc"
        primary_acc = {"acc": accuracy, "acc_ans": acc_ans, "acc_think": acc_think}.get(acc_mode, accuracy)
        pass_threshold = exp.eval.pass_threshold if exp.eval else 0.95
        record = Record(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            acc=accuracy, acc_ans=acc_ans, acc_think=acc_think,
            correct=correct, correct_ans=correct_ans, correct_think=correct_think,
            total=total,
        )
        ReportTable(exp.report_csv_path).save_result(trial_id, trial.display_name, record,
                                                       primary_acc >= pass_threshold)
        return

    # 两个入口都未指定时给出提示
    parser.print_help()


if __name__ == "__main__":
    main()
