"""
train.py — 视觉→珠态训练（阶段 3.3 端到端验证）

训练「看图读珠态」：输入盘面图，输出文本珠态 [L3|L4]#。

架构：
  [盘面图 n_cols×H×W] → VisionEncoder → 视觉 token [n_cols, hidden]
  [视觉 token | 文本 token(bead#)] → GPT → 自回归预测 bead#
  仅对文本部分计算 loss（视觉 token 是输入，不预测）。

用法（无参数 = 训练 model_vm/data/ 下默认数据）：
  python -m model_vm.train_vision
  python -m model_vm.train_vision --img-dir <图像目录> --jsonl <标签.jsonl> \
      --vocab <词表.json> --out <模型.pth> --epochs 50
"""

import os
import sys
# 让 model_vm 下的独立模块可直接运行 / 被引用（项目根 + 本目录加入 path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.domain import BrainConfig, ModelConfig, PosEmbConfig
from model import GPT
from model.vision import VisionEncoder
from model_vm.dataset import VisionBeadDataset, vision_collate
from model_vm.vocab import build_vision_vocab
from evaluate.abacus.render.registry import DEFAULT_RENDER, resolve_patch


def build_model(vocab_data, hidden_size=64, n_cols=5, patch_h=None, patch_w=None,
                in_channels=1):
    """构建 VisionEncoder + GPT 组合模型。

    patch 尺寸缺省时取默认后端（DEFAULT_RENDER）的尺寸；建议由 train() 从实际
    数据形状传入——尺寸必须与数据生成所用后端一致。in_channels 区分灰度(1)/彩色(3)。
    """
    patch_h, patch_w = resolve_patch(DEFAULT_RENDER, patch_h, patch_w)
    brain = BrainConfig(hidden_size=hidden_size)
    pos_emb = PosEmbConfig()
    model_cfg = ModelConfig.from_sources(brain, pos_emb, vocab_data)
    gpt = GPT(model_cfg)
    encoder = VisionEncoder(patch_h=patch_h, patch_w=patch_w,
                            hidden_size=hidden_size, in_channels=in_channels)
    return encoder, gpt


def train(img_dir, jsonl_path, vocab_path, out_path, epochs=50, batch_size=128,
          lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if vocab_path and os.path.isfile(vocab_path):
        from model.vocab import load_vocab
        vocab = load_vocab(vocab_path)
    else:
        vocab = build_vision_vocab()

    ds = VisionBeadDataset(img_dir, jsonl_path, vocab)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=vision_collate)

    # patch 尺寸与通道数从数据形状推导（数据是事实来源，不再硬编码 28×28）
    n_cols_actual, C, H, W = ds.image_shape
    encoder, gpt = build_model(vocab, hidden_size=64, n_cols=n_cols_actual,
                               patch_h=H, patch_w=W, in_channels=C)
    encoder.to(device)
    gpt.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gpt.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, bead_ids, _nums in dl:
            images = images.to(device)      # [B, n_cols, H, W]
            bead_ids = bead_ids.to(device)  # [B, L]

            n_cols = images.shape[1]
            # 视觉 token
            vis_tokens = encoder(images)  # [B, n_cols, hidden]

            # 文本 token embedding（去掉最后的 #，作为输入，预测后续）
            text_input = bead_ids[:, :-1]  # [B, L-1]
            text_emb = gpt.token_emb(text_input)  # [B, L-1, hidden]

            # 拼接：[视觉 token | 文本 token]
            seq_emb = torch.cat([vis_tokens, text_emb], dim=1)  # [B, n_cols+L-1, hidden]

            # 前向
            logits, _ = gpt.forward_embeddings(seq_emb)  # [B, n_cols+L-1, vocab]
            # 只对文本部分算 loss：第 1 个 bead token 由最后一个视觉 token 位置预测，
            # 后续 bead token 由前一个文本 token 位置预测。
            # 因此 text_logits 取 [n_cols-1 : n_cols-1+L]，预测完整 bead_ids。
            start = n_cols - 1
            text_logits = logits[:, start:start + bead_ids.size(1), :]  # [B, L, vocab]
            loss = criterion(text_logits.reshape(-1, text_logits.size(-1)),
                             bead_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(gpt.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dl)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | loss={avg:.4f}")

    # 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "gpt": gpt.state_dict()}, out_path)
    print(f"[视觉训练] 模型已保存: {out_path}")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_img_dir = os.path.join(_here, "data", "imgs")
    _default_jsonl = os.path.join(_here, "data", "labels.jsonl")
    _default_vocab = os.path.join(_here, "data", "vocab.json")
    _default_out = os.path.join(_here, "data", "model.pth")

    parser = argparse.ArgumentParser(
        description="视觉→珠态训练（无参数时训练 model_vm/data/ 下默认数据）")
    parser.add_argument("--img-dir", default=_default_img_dir,
                        help=f"图像目录，默认 {_default_img_dir}")
    parser.add_argument("--jsonl", default=_default_jsonl,
                        help=f"标签 jsonl，默认 {_default_jsonl}")
    parser.add_argument("--vocab", default=_default_vocab,
                        help=f"词表 JSON，默认 {_default_vocab}（不存在时用内置词表）")
    parser.add_argument("--out", default=_default_out,
                        help=f"模型输出，默认 {_default_out}")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.img_dir, args.jsonl, args.vocab, args.out,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
