"""
train.py — VLM 训练（阶段 3.4.2）

训练「看图 + 指令 → 口诀动作」：
  输入盘面图 + 指令"加 7 ="，输出口诀"7去3进1#"。

架构（与 model_vm/train.py 同构，文本部分从「珠态标签」换成「指令 + 口诀」）：
  [盘面图] → VisionEncoder → 视觉 token [n_cols, hidden]
  [视觉 token | prompt token | rhyme 预测] → GPT
  仅对 rhyme 部分计算 loss。

Loss 对齐（关键）：
  序列长度 = n_cols + L_prompt + L_rhyme
  第 1 个 rhyme token 由「prompt 最后一个 token」位置预测，
  即 rhyme logits 取 [n_cols + L_prompt - 1 : n_cols + L_prompt - 1 + L_rhyme]。

用法：
  python -m model_vlm.train_vlm --npy <图像.npy> --jsonl <标签.jsonl> \
      --out <模型.pth> --epochs 100
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path


import torch
from torch import nn
from numpy import integer as np_integer
from torch.utils.data import ConcatDataset, DataLoader

from model.domain import BrainConfig, ModelConfig, PosEmbConfig
from model import GPT
from model.vision import VisionEncoder
from model_vlm.dataset import VLMDataset, vlm_collate
from model.vocab import PAD_TOKEN, STOP_TOKEN, Vocab, build_vocab_from_files  # noqa: E402
from evaluate.abacus.render.registry import (
    DEFAULT_RENDER,
    backend_names,
    resolve_patch,
    default_patch,
)

# 项目根加入 sys.path（支持从任意目录运行 / 直接 python model_vlm/train.py）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_VLM_TOKENS = [
    r'\[PARSE\]', r'\[EVAL\]', r'\[READ\]',
    'PRE', 'POST', 'STACK', 'ANS', 'INFIX',
    ';', '#', '=', r'\+', r'\-', r'\×', r'\÷', r'\(', r'\)', r'\d', r'\D',
]
VLM_TOKEN_PATTERN = '|'.join(_VLM_TOKENS)


VLM_MAX_SEQ_LEN = 320  # 多位数数据集（PRE/POST 空格分隔）n=20 最长约 271；
                        # RoPE 频率表按此预计算，须 ≥ 实际最大序列长度


def build_vocab(*jsonl_paths: str) -> Vocab:
    """从训练 jsonl 自动构建 Vocab，覆盖 prompt / A 中全部字符。"""
    paths = [p for p in jsonl_paths if p is not None]
    vocab_list, _ = build_vocab_from_files(paths, VLM_TOKEN_PATTERN)
    if PAD_TOKEN not in vocab_list:
        vocab_list.append(PAD_TOKEN)
    return Vocab(vocab=vocab_list, pad_id=vocab_list.index(PAD_TOKEN),
                 max_seq_len=VLM_MAX_SEQ_LEN, stop_token=STOP_TOKEN)


def build_model(vocab_data, hidden_size=64, n_cols=13,
                patch_h=None, patch_w=None, in_channels=1,
                render=DEFAULT_RENDER):
    """构建 VisionEncoder + GPT 组合模型。

    patch 尺寸显式给出时优先；缺省按 `render` 后端解析——尺寸是**后端的属性**，
    必须与数据生成所用后端一致（此前取全局默认 → 5×1，用 fixed 数据会不匹配）。

    in_channels：图像通道数（C=1 灰度 / C=3 彩色），须与数据生成
    后端一致（fixed/minimal/image_gray → 1，image → 3）。
    """
    patch_h, patch_w = resolve_patch(render, patch_h, patch_w)
    brain = BrainConfig(hidden_size=hidden_size)
    pos_emb = PosEmbConfig()
    model_cfg = ModelConfig.from_sources(brain, pos_emb, vocab_data)
    gpt = GPT(model_cfg)
    encoder = VisionEncoder(patch_h=patch_h, patch_w=patch_w,
                            hidden_size=hidden_size, in_channels=in_channels)
    return encoder, gpt


def validate_npy_images(images, labels, *, n_cols, patch_h, patch_w,
                        npy_path="", name="train"):
    """训练前自动校验 npy 图像堆叠与 patch 参数的一致性。

    兼容两种存储形态：
      - 新版 [N, n_cols, C, H, W]（parse_eval_bridge 生成，uint8）
      - 旧版 [N, n_cols, H, W]（灰度，加载时补 C 维）

    校验项：
      1. 维度/形状：n_cols、patch_h、patch_w 必须与命令行参数一致，
         否则 VisionEncoder 投影会炸出难定位的 matmul 维度错误；
      2. image 下标越界：jsonl 的 `image` 字段（去重数组下标）必须 < N。
      3. 多数据集（train + parse）间通道数一致。

    形状不匹配时给出后端修正提示（自动比对已知后端默认尺寸）。
    返回通道数 C（PNG 模式无 npy，返回 None）。
    """
    if images is None:
        return None
    where = f"{name}（{npy_path}）"

    if images.ndim == 5:
        n, nc, c, h, w = images.shape
    elif images.ndim == 4:
        n, nc, h, w = images.shape
        c = 1
    else:
        raise ValueError(
            f"[校验失败] {where}: npy 维度 {images.ndim} 不合法，"
            "应为 [N, n_cols, C, H, W]（新版）或 [N, n_cols, H, W]（旧版灰度），"
            f"实际 shape={images.shape}")

    problems = []
    if nc != n_cols:
        problems.append(f"档位数不匹配: npy={nc} vs --n-cols {n_cols}")
    if (h, w) != (patch_h, patch_w):
        # 自动比对已知后端默认尺寸，给出最可能的修正命令
        hints = [f"{b}: --patch-h {bh} --patch-w {bw} --render {b}"
                 for b in ("fixed", "minimal", "image", "image_gray")
                 for bh, bw in [default_patch(b)] if (bh, bw) == (h, w)]
        hint = f"（该尺寸与已知后端匹配: {'; '.join(hints)}）" if hints else ""
        problems.append(
            f"patch 尺寸不匹配: npy={h}x{w} vs --patch-h {patch_h} --patch-w {patch_w}"
            f" {hint}")
    if problems:
        raise ValueError(
            f"[校验失败] {where}: npy shape={images.shape}\n  - "
            + "\n  - ".join(problems))

    max_idx = max((int(rec["image"]) for rec in labels
                   if isinstance(rec.get("image"), (int, np_integer))),
                  default=-1)
    if max_idx >= n:
        raise ValueError(
            f"[校验失败] {where}: jsonl 中 image 下标最大值 {max_idx} "
            f"越界（npy 只有 {n} 张图），数据与图像堆叠不配套")

    return c


def train(jsonl_path, out_path, *, image_dir=None, npy_path=None,
          parse_jsonl=None, epochs=100, batch_size=128, lr=1e-3, n_cols=13,
          hidden_size=64, patch_h=None, patch_w=None,
          render=DEFAULT_RENDER, device=None):
    # patch 尺寸缺省跟随 render 后端（而非全局默认），保证与数据生成后端一致
    patch_h, patch_w = resolve_patch(render, patch_h, patch_w)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = build_vocab(jsonl_path, parse_jsonl)

    ds = VLMDataset(jsonl_path, vocab, image_dir=image_dir,
                    npy_path=npy_path, n_cols=n_cols,
                    patch_h=patch_h, patch_w=patch_w, render=render)
    datasets = [ds]
    if parse_jsonl is not None:
        parse_ds = VLMDataset(parse_jsonl, vocab, image_dir=image_dir,
                              npy_path=npy_path, n_cols=n_cols,
                              patch_h=patch_h, patch_w=patch_w, render=render)
        datasets.append(parse_ds)

    ds_all = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # 训练前自动校验：npy 形状 vs patch 参数、image 下标越界、通道一致
    tags = ["train", "parse"] if parse_jsonl is not None else ["train"]
    chan = None
    for d, tag in zip(datasets, tags):
        c = validate_npy_images(d.images, d.labels, n_cols=n_cols,
                                patch_h=patch_h, patch_w=patch_w,
                                npy_path=npy_path or "", name=tag)
        if c is not None:
            if chan is not None and chan != c:
                raise ValueError(
                    f"[校验失败] 多数据集通道数不一致: train C={chan} vs {tag} C={c}")
            chan = c
    print(f"[校验通过] npy shape 与 patch 参数一致 (n_cols={n_cols}, "
          f"patch={patch_h}x{patch_w}{f', C={chan}' if chan is not None else ''})")

    collate = partial(vlm_collate, pad_id=vocab.pad_id)
    dl = DataLoader(ds_all, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # 通道数从数据集推导（image 后端 → 3，其余 → 1），勿手填
    in_channels = getattr(ds, 'in_channels', 1)
    encoder, gpt = build_model(vocab, hidden_size=hidden_size, n_cols=n_cols,
                               patch_h=patch_h, patch_w=patch_w,
                               in_channels=in_channels, render=render)
    encoder.to(device)
    gpt.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gpt.parameters()), lr=lr)
    # pad 位置不进 loss，由 targets 的 -100 屏蔽（尾部 pad 在 causal 下无害）
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, seq_ids, rhyme_ids, prompt_lens, _nums in dl:
            images = images.to(device)        # [B, n_cols, H, W]
            seq_ids = seq_ids.to(device)      # [B, L]
            rhyme_ids = rhyme_ids.to(device)  # [B, Lm]
            prompt_lens = prompt_lens.to(device)

            B, n_cols, C, H, W = images.shape
            L = seq_ids.size(1)

            # 布局：[vis(13) | prompt | rhyme_in] 逐样本连续，pad 只在尾部
            vis_tokens = encoder(images)          # [B, n_cols, hidden]
            text_emb = gpt.token_emb(seq_ids)     # [B, L, hidden]
            seq_emb = torch.cat([vis_tokens, text_emb], dim=1)
            logits, _ = gpt.forward_embeddings(seq_emb)  # [B, n_cols+L, V]

            # 逐样本构造 target：位置 (n_cols+Lp-1+k) 预测 rhyme[k]
            # 该位置的输入序列 = prompt末token, rhyme[0..k-1]，因果链与推理一致
            targets = torch.full((B, n_cols + L), -100,
                                 dtype=torch.long, device=device)
            for b in range(B):
                s = n_cols + int(prompt_lens[b].item()) - 1
                m = rhyme_ids[b]
                m_len = int((m != vocab.pad_id).sum().item())  # pad 只在尾部
                targets[b, s:s + m_len] = m[:m_len]

            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(gpt.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dl)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | loss={avg:.4f}")

    # 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "gpt": gpt.state_dict(),
                "vocab": vocab.vocab, "pad_id": vocab.pad_id,
                "n_cols": n_cols, "patch_h": patch_h, "patch_w": patch_w,
                "in_channels": in_channels, "render": render,
                "max_seq_len": VLM_MAX_SEQ_LEN}, out_path)
    print(f"[VLM训练] 模型已保存: {out_path}")
    return encoder, gpt, vocab


if __name__ == "__main__":
    # 默认参数指向当前数据集 model_vlm/dataset/（npy 为 minimal 5×1 后端），
    # 全部可被命令行覆盖；路径相对项目根，任意 cwd 运行均有效
    _DS_DIR = _ROOT / "model_vlm" / "data"

    parser = argparse.ArgumentParser(
        description="VLM 训练（默认数据集 model_vlm/dataset/，直接运行即可训练）")
    parser.add_argument("--jsonl", default=str(_DS_DIR / "train.jsonl"),
                        help="训练标签 jsonl（默认 %(default)s）")
    parser.add_argument("--out", default=str(_ROOT / "model_vlm" / "ckpt" / "vlm.pth"),
                        help="输出模型 .pth（默认 %(default)s）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image-dir", dest="image_dir",
                       help="图片目录（与 jsonl 中的 image 相对路径组合）")
    group.add_argument("--npy", default=str(_DS_DIR / "images.npy"),
                       help="图像堆叠 npy（默认 %(default)s；已去重，"
                            "jsonl 的 image 字段为数组下标）")
    parser.add_argument("--parse-jsonl", dest="parse_jsonl",
                        default=str(_DS_DIR / "parse.jsonl"),
                        help="parse 纯文本样本 jsonl（默认 %(default)s，"
                             "三任务合训；传空串 '' 禁用）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-cols", type=int, default=13)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--patch-h", type=int, default=None,
                        help="视觉 patch 高度（默认跟随 --render 后端）")
    parser.add_argument("--patch-w", type=int, default=None,
                        help="视觉 patch 宽度（默认跟随 --render 后端）")
    parser.add_argument("--render", default=DEFAULT_RENDER,
                        choices=backend_names(),
                        help=f"渲染后端名（默认 {DEFAULT_RENDER}，写入 checkpoint "
                             "供推理/闭环自恢复；--patch-h/--patch-w 缺省时自动取"
                             "该后端默认尺寸；image = 彩色 RGB → C=3，"
                             "其余为灰度 → C=1。可选值随 registry 注册自动扩展）")
    args = parser.parse_args()
    train(args.jsonl, args.out,
          image_dir=args.image_dir, npy_path=args.npy,
          parse_jsonl=args.parse_jsonl or None,  # 空串 → 不合训 parse
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          n_cols=args.n_cols, hidden_size=args.hidden_size,
          patch_h=args.patch_h, patch_w=args.patch_w, render=args.render)
