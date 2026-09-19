import os

import torch
from model_lm.train import train
from model.train_utils import set_seed, load_checkpoint

from model import GPT
from model_lm.dataset import ArithmeticReasoningDataset
from functools import partial
from torch.utils.data import DataLoader
from torch import optim

from experiment.domain import Experiment, Trial, ExecutionContext
from common.record import Record
from model.domain import ModelConfig

def collate_fn(pad_id, batch):
    """将批次中的序列填充到相同长度"""
    max_len = max(len(seq) for seq in batch)
    padded_batch = []
    for seq in batch:
        pad_len = max_len - len(seq)
        padded = torch.cat([seq, torch.full((pad_len,), pad_id, dtype=seq.dtype)])  # 0 是 padding idx
        padded_batch.append(padded)
    return torch.stack(padded_batch)

# -------------------- 单 trial 训练函数（供 train_and_eval_trial 调用） --------------------
def train_one_trial(trial_id: int, exp: Experiment, ctx: ExecutionContext) -> "Record | None":
    """单 trial 训练。

    Args:
        trial_id: trial 配置（data/method/artifacts）
        exp: 实验聚合根（brain/train/eval）
        ctx: 运行时上下文（vocab_data/device/prev_model_path/review_files/epochs/seed）
    Returns:
        Record | None: 最优 epoch 的 Record；无验证或未训练任何 epoch 时为 None
    """
    trial = exp.trials[trial_id]
    method = trial.method
    paths = trial.paths

    # ---- 从 exp 读取实验级配置 ----
    train_config = exp.train
    brain_config = exp.brain.with_trial_overrides(trial.heads)
    eval_batch_size = exp.eval.eval_batch_size
    enable_val = train_config.enable_validation
    early_stop = train_config.early_stop_patience
    pass_threshold = exp.eval.pass_threshold if exp.eval else 0.95

    # ---- 从 ctx 取运行时状态 ----
    vocab_data = ctx.vocab_data
    pad_id = vocab_data.pad_id
    vocab_size = vocab_data.vocab_size
    max_seq_len = vocab_data.max_seq_len

    device = ctx.device
    prev_model_path = ctx.prev_model_path
    seed = ctx.seed

    # ---- 从 trial 取 trial 级配置 ----
    epochs = method.epochs
    lr = method.learning_rate
    train_data_path = paths.train_data
    model_path = paths.best_model
    ckpt_path = paths.checkpoint
    train_log_path = paths.train_log

    set_seed(seed)

    ##################################### 模型初始化 #####################################
    train_dataset = ArithmeticReasoningDataset(
        file_path=train_data_path,
        vocab_data=vocab_data,
        repeat_factor=method.repeat_factor,
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=method.batch_size,
                              shuffle=method.shuffle,
                              collate_fn=partial(collate_fn, pad_id),
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False,
                              )
    model_config = ModelConfig.from_sources(
        brain_config, exp.pos_emb, vocab_data, train_config.dropout)
    model = GPT(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    ##################################### 判断开始位置 #####################################
    start_epoch = 0
    best_val_acc, no_improve = -1.0, 0

    # 确保 checkpoint 目录存在
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    # 1. 本 trial 断点续训
    if ckpt_path and os.path.isfile(ckpt_path):
        start_epoch, best_val_acc, no_improve = load_checkpoint(model, optimizer, ckpt_path, device)
    # 2. 上一 trial 续训
    elif prev_model_path and os.path.isfile(prev_model_path):
        print(f"[训练] 从上一 trial 权重初始化: {prev_model_path}")
        prev_state = torch.load(prev_model_path, map_location=device)
        prev_vocab_size = prev_state.get('token_emb.weight', prev_state.get('head.weight')).shape[0]
        if prev_vocab_size != vocab_size:
            raise RuntimeError(
                f"词表大小不匹配！\n"
                f"  权重文件 {prev_model_path} 的 vocab_size = {prev_vocab_size}\n"
                f"  当前词表 vocab_size = {vocab_size}\n"
                f"  原因：权重是用不同的词表训练的。\n"
                f"  解决：运行 python -m experiment --experiment <name> --reset 重新从 L0 开始训练。"
            )
        model.load_state_dict(prev_state)
    # 3. 从头开始训练
    else:
        print("[训练] 从头开始训练")

    ##################################### 检查空转 #####################################
    # start_epoch >= epochs 时直接跳过
    if start_epoch >= epochs:
        print(f"[训练] [WARN] checkpoint 已在 epoch {start_epoch}，目标 epochs={epochs}，无需训练。")
        # 断点续训已完成：当前 model 已是训练后权重，刷新 best_model_path，避免评估读到残留的未训练快照
        if model_path:
            torch.save(model.state_dict(), model_path)
        return None

    ##################################### 训练 #####################################
    set_seed(seed)
    return train(device,
                 model,
                 train_loader,
                 optimizer,
                 epochs,
                 pad_id,
                 start_epoch,
                 ckpt_path,
                 test_data_path=paths.test_data,
                 vocab_data=vocab_data,
                 eval_batch_size=eval_batch_size,
                 enable_val=enable_val,
                 early_stop=early_stop,
                 model_path=model_path,
                 train_log_path=train_log_path,
                 pass_threshold=pass_threshold,
                 best_val_acc=best_val_acc,
                 count=no_improve,
                 acc_mode=exp.eval.acc_mode)
