# -*- coding: utf-8 -*-
"""model/train_utils.py —— 训练基础设施（原 common/utils.py 迁入）。

被 model_lm 与 exp 的训练共同消费：随机种子设置、checkpoint 保存/恢复。
"""
import os
import random

import numpy as np
import torch


def set_seed(seed=99):
    """设置随机种子以保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, path, best_val_acc=-1.0, no_improve=0):
    """保存完整训练状态，支持断点续训。"""
    checkpoint = {
        'epoch': epoch,           # 当前完成的 epoch 编号（0-based）
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,   # 历史最佳验证准确率（-1 表示无验证）
        'no_improve': no_improve,       # 连续无提升计数（早停用）
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """从 checkpoint 恢复训练状态。返回 (起始 epoch, best_val_acc, no_improve)。"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 继续
    best_val_acc = checkpoint.get('best_val_acc', -1.0)
    no_improve = checkpoint.get('no_improve', 0)
    print(f"[Checkpoint] 已恢复 epoch {start_epoch} (best={best_val_acc:.4f}, no_improve={no_improve}) ← {os.path.basename(path)}")
    return start_epoch, best_val_acc, no_improve
