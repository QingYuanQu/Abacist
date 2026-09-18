import json
import os
import time
from datetime import datetime

import torch
from torch import nn
from model.train_utils import save_checkpoint
from model.record import Record
from model_lm.eval import compute_accuracy


def _append_record(log_path, record: Record):
    """将单 epoch Record 追加写入 jsonl（一行一条）。"""
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')


# -------------------- 5. 训练（每 epoch 产出 Record，返回最优 Record）------------------------
def train(device,
          model,
          dataloader,
          optimizer,
          epochs,
          pad_id,
          start_epoch,
          ckpt_path,
          test_data_path,
          vocab_data,
          eval_batch_size,
          enable_val,
          early_stop,
          model_path,
          train_log_path,
          pass_threshold,
          best_val_acc=-1.0,
          count=0,
          acc_mode="acc") -> Record | None:
    """训练单 trial，每 epoch 构造完整 Record 写入 jsonl 日志。

    验证集 = 测试集，每 epoch 做完整评估；按 acc_mode 主指标判优，
    最优时保存权重并留存 best_record。

    Returns:
        best_record: 主指标最优 epoch 的 Record；无验证或未训练任何 epoch 时为 None。
    """
    ################################## 0. 加载验证集 ##################################
    has_val = False
    pass_threshold_flag = False
    if enable_val:
        if test_data_path and os.path.isfile(test_data_path):
            print(f"[训练] 验证集: {os.path.basename(test_data_path)}")
            has_val = True
        else:
            print(f"[训练] [WARN] 测试数据不存在: {test_data_path}，跳过验证早停。")

    ################################## 1. 模型训练 ##################################
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 初始化日志文件（如果提供了路径）
    if train_log_path:
        os.makedirs(os.path.dirname(train_log_path), exist_ok=True)

    best_record: Record | None = None
    best_model_state = None  # 内存中保存最优 epoch 的权重，收尾时兜底写 best_model_path
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # 本 epoch 的 Record（无验证时评估字段为 None）
        record = Record(
            epoch=epoch + 1,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            train_loss=avg_loss,
            lr=current_lr,
            epoch_time_s=round(epoch_time, 2),
            elapsed_s=round(elapsed, 2),
        )

        ################################## 1.1验证评估 + 早停 ##################################
        status = ""
        early_stop_now = False
        if has_val:
            model.eval()
            (acc, acc_ans, acc_think, correct, correct_ans,
             correct_think, total, _) = compute_accuracy(
                model, device, vocab_data, eval_batch_size, pad_id,
                test_data_path=test_data_path
            )
            model.train()
            record.acc, record.acc_ans, record.acc_think = acc, acc_ans, acc_think
            (record.correct, record.correct_ans, record.correct_think,
             record.total) = correct, correct_ans, correct_think, total

            # 按 acc_mode 选主指标做早停判断（"acc" | "acc_ans" | "acc_think"）
            val_acc = {"acc": acc, "acc_ans": acc_ans, "acc_think": acc_think}.get(acc_mode, acc)

            if pass_threshold_flag:
                count += 1

            # 到达阈值后：5次未提升则提前停止
            if val_acc >= pass_threshold:
                pass_threshold_flag = True
                # 有提升
                if val_acc > best_val_acc:
                    status = f" | val_acc={val_acc:.4f} ≥ {best_val_acc:.4f}  提升 ({count}/{early_stop})"
                    if model_path:
                        torch.save(model.state_dict(), model_path)  # 有提升就存，不依赖阈值
                    best_val_acc = val_acc
                    best_record = record
                    best_model_state = model.state_dict()
                # 未达阈值：没提升，计数
                else:
                    status = f" | val_acc={val_acc:.4f} < {best_val_acc:.4f} 未提升({count}/{early_stop})"

                # 连续 patience 次都没超过历史最佳
                if count >= early_stop:
                    status = f" | val_acc={val_acc:.4f} ≥ {pass_threshold:.2f} 提前停止"
                    early_stop_now = True
            else:
                if val_acc > best_val_acc:
                    if model_path:
                        torch.save(model.state_dict(), model_path)  # 有提升就存，不依赖阈值
                    status = f" | val_acc={val_acc:.4f} ≥ {best_val_acc:.4f} 更新权重"
                    best_val_acc = val_acc
                    best_record = record
                    best_model_state = model.state_dict()
                else:
                    status = f" | val_acc={val_acc:.4f} < {pass_threshold:.2f} 未达阈值"

        ################################## 1.2 日志记录（每 epoch 一条 Record） ##################################
        _append_record(train_log_path, record)

        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | lr={current_lr:.2e}{status} | {epoch_time:.1f}s | 累计 {elapsed:.1f}s")

        if early_stop_now:
            break

        if ckpt_path:
            save_checkpoint(model, optimizer, epoch, ckpt_path, best_val_acc, count)

    # 收尾兜底：确保 best_model_path 始终对应最优/最新训练权重。
    # 仅依赖循环内「有提升才存」会在断点续训（best_val_acc 已很高）时不再触发，
    # 导致 best_model_path 残留早期甚至未训练的快照（评估读到 0 的根因）。
    if model_path:
        torch.save(best_model_state if best_model_state is not None else model.state_dict(), model_path)
    return best_record
