"""
infer_vlm.py — VLM 三任务推理脚本

加载 `model_vlm.train_vlm` 训练保存的 .pth，对 parse/eval/read 做 greedy decode，
并统计每类任务准确率。

用法：
  python -m model_vlm.infer_vlm \
      --checkpoint model_vlm/dataset/train.jsonl.pth \
      --jsonl model_vlm/dataset/train.jsonl \
      --image-dir model_vlm/dataset \
      --max-len 64 --n-samples 200
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 项目根加入 sys.path（支持从任意目录运行 / 直接 python model_vlm/infer_vlm.py）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import GPT
from model.vision import VisionEncoder
from model_vlm.train import VLM_TOKEN_PATTERN, build_model
from model.vocab import Vocab
from evaluate.abacus.render.registry import DEFAULT_RENDER, resolve_patch


STOP_TOKEN_IDS = set()  # 遇到 # 即停，不在词表里单独设 stop token


def load_checkpoint(checkpoint_path: str, device: str):
    """加载训练保存的 checkpoint。

    返回 (encoder, gpt, vocab, n_cols, in_channels, patch_h, patch_w, render)。
    render / patch 尺寸 / 通道数均自恢复；老 checkpoint（无 render 与
    patch_h 字段，那时只有 fixed 80×28 灰度一种数据）回退 fixed / C=1。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab_list = ckpt["vocab"]
    pad_id = ckpt.get("pad_id", vocab_list.index("<PAD>") if "<PAD>" in vocab_list else 0)
    n_cols = ckpt.get("n_cols", 13)
    # 渲染后端 / patch 尺寸 / 通道数自恢复：优先用 ckpt 记录值；
    # 老 ckpt 两个字段都没有（那时只有 fixed 80×28 一种）→ 回退 fixed。
    has_meta = ("render" in ckpt) or ("patch_h" in ckpt)
    render = ckpt.get("render", DEFAULT_RENDER if has_meta else "fixed")
    patch_h = ckpt.get("patch_h")
    patch_w = ckpt.get("patch_w")
    patch_h, patch_w = resolve_patch(render, patch_h, patch_w)
    in_channels = ckpt.get("in_channels", 1)
    max_seq_len = ckpt.get("max_seq_len", 192)

    vocab = Vocab(vocab=vocab_list, pad_id=pad_id,
                  max_seq_len=max_seq_len, stop_token="#")
    # 强制使用训练时同样的 tokenizer pattern，避免 build_tokenizer_regex 与 pattern 分词差异
    vocab.tokenizer = re.compile(VLM_TOKEN_PATTERN, flags=re.DOTALL)

    hidden_size = ckpt["gpt"]["token_emb.weight"].shape[1]
    encoder, gpt = build_model(vocab, hidden_size=hidden_size, n_cols=n_cols,
                               patch_h=patch_h, patch_w=patch_w,
                               in_channels=in_channels)
    encoder.load_state_dict(ckpt["encoder"])
    gpt.load_state_dict(ckpt["gpt"])
    encoder.to(device)
    gpt.to(device)
    encoder.eval()
    gpt.eval()
    return encoder, gpt, vocab, n_cols, in_channels, patch_h, patch_w, render


def _infer_image_shape(image_dir: Path):
    """从 image_dir 里任意一张真实图推断 (H, W, C)。"""
    for p in image_dir.iterdir():
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            arr = np.array(Image.open(p), dtype=np.float32)
            c = arr.shape[2] if arr.ndim == 3 else 1
            return arr.shape[0], arr.shape[1], c
    raise FileNotFoundError(f"{image_dir} 下没有可用图片")


def load_image(record: dict, image_dir: Path, n_cols: int, zero_image: np.ndarray | None):
    """加载单张图并整理为 [1, n_cols, C, H, W]；parse 样本用全零图。

    zero_image 已是 [n_cols, C, H, W]（由调用方按真实图通道数构造）；
    真实图按 PIL 原生通道加载（RGB → C=3），与训练侧 vlm_dataset 的
    to_rod_patches 布局逐元素一致。
    """
    if "image" not in record or not record["image"]:
        image = zero_image.copy()
    else:
        img_path = image_dir / record["image"]
        img = Image.open(img_path)                     # 保留原通道
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:                              # 灰度 → [n_cols, 1, H, W]
            H, W = arr.shape
            image = arr.reshape(H, n_cols, W // n_cols) \
                       .transpose(1, 0, 2)[:, None]
        else:                                          # 彩色 → [n_cols, C, H, W]
            H, W, C = arr.shape
            image = arr.reshape(H, n_cols, W // n_cols, C) \
                       .transpose(1, 3, 0, 2)
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


def tokenize(text: str, vocab: Vocab):
    tokens = vocab.tokenizer.findall(text)
    return [vocab.stoi[t] for t in tokens]


def decode(ids: list[int], vocab: Vocab) -> str:
    return "".join(vocab.vocab[i] for i in ids)


@torch.no_grad()
def generate(encoder, gpt, image: torch.Tensor, prompt_ids: list[int],
             vocab: Vocab, n_cols: int, max_len: int = 64) -> str:
    """greedy decode：与训练时同样的拼接方式生成 rhyme。"""
    device = next(gpt.parameters()).device
    image = image.to(device)
    prompt_ids_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    vis_tokens = encoder(image)  # [1, n_cols, hidden]
    prompt_emb = gpt.token_emb(prompt_ids_t)  # [1, Lp, hidden]
    prefix_emb = torch.cat([vis_tokens, prompt_emb], dim=1)

    logits, past_kvs = gpt.forward_embeddings(prefix_emb, use_cache=True)
    start = n_cols + len(prompt_ids) - 1
    next_logits = logits[:, start, :]

    generated = []
    for _ in range(max_len):
        next_id = int(next_logits.argmax(dim=-1).item())
        generated.append(next_id)
        tok = vocab.vocab[next_id]
        if tok == "#":
            break

        # 自回归单步（past_kvs 非空时 start_pos 自动取 KV 缓存长度，无需 pos_offset）
        token_emb = gpt.token_emb(torch.tensor([[next_id]], device=device))
        logits, past_kvs = gpt.forward_embeddings(
            token_emb, past_kvs=past_kvs, use_cache=True)
        next_logits = logits[:, -1, :]

    return decode(generated, vocab)


def run(checkpoint_path: str, jsonl_path: str, image_dir: str | None,
        max_len: int = 64, n_samples: int | None = None, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder, gpt, vocab, n_cols, in_channels, patch_h, patch_w, render = \
        load_checkpoint(checkpoint_path, device)
    print(f"[VLM推理] render={render} patch={patch_h}×{patch_w} "
          f"C={in_channels} n_cols={n_cols}")

    image_dir = Path(image_dir) if image_dir else None
    if image_dir is not None and image_dir.exists():
        H, W, C = _infer_image_shape(image_dir)
        zero_image = np.zeros((n_cols, C, H, W // n_cols), dtype=np.float32)
    else:
        # 无图目录时 fallback 到 checkpoint 保存的 patch 尺寸
        zero_image = np.zeros((n_cols, in_channels, patch_h, patch_w),
                              dtype=np.float32)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    if n_samples is not None:
        records = records[:n_samples]

    stats = {"parse": {"total": 0, "right": 0}, "eval": {"total": 0, "right": 0},
             "read": {"total": 0, "right": 0}}

    for i, rec in enumerate(records):
        task = rec["task"]
        prompt = rec["prompt"]
        ref = str(rec["A"])          # 真值（参考答案，仅用于比对评分）
        image = load_image(rec, image_dir, n_cols, zero_image)
        pred = generate(encoder, gpt, image, tokenize(prompt, vocab),
                        vocab, n_cols, max_len=max_len)

        stats[task]["total"] += 1
        if pred == ref:
            stats[task]["right"] += 1

        if i < 5:
            print(f"[{task}] prompt={prompt!r} ref={ref!r} pred={pred!r}")

    print("\n[推理统计]")
    for task, s in stats.items():
        acc = s["right"] / s["total"] if s["total"] else 0.0
        print(f"  {task:5}: {s['right']}/{s['total']} = {acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="训练保存的 .pth")
    parser.add_argument("--jsonl", required=True, help="测试 jsonl")
    parser.add_argument("--image-dir", default=None, help="图片目录")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="只测前 N 条（默认全量）")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.checkpoint, args.jsonl, args.image_dir,
        max_len=args.max_len, n_samples=args.n_samples, device=args.device)
