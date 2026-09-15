"""
eval.py — 视觉→珠态识数（看图读珠态推理入口）

加载训练好的模型（VisionEncoder + GPT），对盘面图直接输出识数结果
（珠态文本）。不评估准确率，重在用户体验：无参数时对三张默认示例图
推理并打印结果；也可用 --img 指定任意盘面 PNG。

用法：
  python -m model_vm.eval                              # 三张默认示例图
  python -m model_vm.eval --img x.png                  # 指定一张
  python -m model_vm.eval --img a.png b.png c.png      # 指定多张
  python -m model_vm.eval --model m.pth --vocab v.json --n-cols 5
"""

import os
import sys
# 让 model_vm 下的独立模块可直接运行 / 被引用（项目根 + 本目录加入 path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from model_vm.vocab import build_vision_vocab
from model_vm.train import build_model

# 三张默认示例图（位于 model_vm/data/imgs/，文件名 = 数字真值补零）
_DEFAULT_IMGS = ["000007.png", "000034.png", "000099.png"]


def load_image_patches(img_path, n_cols):
    """整盘面 PNG → [n_cols, C, H, patch_w] float32 0~1（C=1 灰度/C=3 彩色）。"""
    img = Image.open(img_path)
    if img.mode == "L":
        arr = np.asarray(img, dtype=np.float32) / 255.0      # [H, W]
        arr = arr[..., None]                                 # [H, W, 1]
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
    H, W, C = arr.shape
    if W % n_cols != 0:
        raise ValueError(f"图片宽 {W} 无法按 {n_cols} 档切分（{img_path}）")
    patch_w = W // n_cols
    return arr.reshape(H, n_cols, patch_w, C).transpose(1, 3, 0, 2)


def load_model(vocab_path, model_path, n_cols, patch_h, patch_w, in_channels, device):
    """加载词表 + VisionEncoder + GPT 权重。"""
    if vocab_path and os.path.isfile(vocab_path):
        from model.vocab import load_vocab
        vocab = load_vocab(vocab_path)
    else:
        vocab = build_vision_vocab()

    encoder, gpt = build_model(vocab, hidden_size=64, n_cols=n_cols,
                               patch_h=patch_h, patch_w=patch_w,
                               in_channels=in_channels)
    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    gpt.load_state_dict(ckpt["gpt"])
    encoder.to(device).eval()
    gpt.to(device).eval()
    return encoder, gpt, vocab


@torch.no_grad()
def infer(encoder, gpt, vocab, image_patches, device, max_len=40):
    """盘面 patch [n_cols,C,H,W] → 珠态文本（到 # 或 max_len 截断）。"""
    stop_id = vocab.stoi['#']
    x = torch.tensor(image_patches, dtype=torch.float32).unsqueeze(0).to(device)
    emb = encoder(x)  # [1, n_cols, hidden]
    generated = []
    for _ in range(max_len):
        logits, _ = gpt.forward_embeddings(emb)
        nxt = logits[0, -1].argmax().item()
        if nxt == stop_id:
            break
        generated.append(nxt)
        nxt_emb = gpt.token_emb(torch.tensor([[nxt]], device=device))
        emb = torch.cat([emb, nxt_emb], dim=1)
    return ''.join(vocab.itos[t] for t in generated)


def _guess_num(path):
    """从文件名 <数字>.png 提取数字真值（失败返回 None）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return None


if __name__ == "__main__":
    import argparse

    _here = os.path.dirname(os.path.abspath(__file__))
    _default_img_dir = os.path.join(_here, "data", "imgs")
    _default_model = os.path.join(_here, "data", "model.pth")
    _default_vocab = os.path.join(_here, "data", "vocab.json")

    parser = argparse.ArgumentParser(
        description="视觉→珠态识数推理（无参数时对三张默认示例图输出识数结果）")
    parser.add_argument("--img", nargs="*", default=None,
                        help="盘面 PNG 路径（可多个）；缺省用 data/imgs/ 下三张示例图")
    parser.add_argument("--model", default=_default_model,
                        help=f"模型权重，默认 {_default_model}")
    parser.add_argument("--vocab", default=_default_vocab,
                        help=f"词表 JSON，默认 {_default_vocab}（不存在时用内置词表）")
    parser.add_argument("--n-cols", type=int, default=5, help="档位数，默认 5")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 确定图片列表：--img 指定，否则三张默认示例图
    if args.img:
        img_paths = args.img
    else:
        img_paths = [os.path.join(_default_img_dir, n) for n in _DEFAULT_IMGS]

    # 检查文件
    missing = [p for p in img_paths if not os.path.isfile(p)]
    if missing:
        print("[识数] 找不到图片：")
        for p in missing:
            print(f"   - {p}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[识数] 未找到模型权重 {args.model}")
        print("       请先训练：python -m model_vm.train_vision")
        sys.exit(1)

    # 从第一张图推导 patch 尺寸与通道数（数据是事实来源）
    first = load_image_patches(img_paths[0], args.n_cols)
    _, in_channels, patch_h, patch_w = first.shape
    print(f"[识数] 设备={device} 档位={args.n_cols} "
          f"patch={patch_h}×{patch_w}×{in_channels} 模型={os.path.basename(args.model)}")

    encoder, gpt, vocab = load_model(args.vocab, args.model, args.n_cols,
                                     patch_h, patch_w, in_channels, device)

    print("-" * 46)
    for p in img_paths:
        patches = load_image_patches(p, args.n_cols)
        bead = infer(encoder, gpt, vocab, patches, device)
        num = _guess_num(p)
        suffix = f"（真值 {num}）" if num is not None else ""
        print(f"  {os.path.basename(p)}  →  {bead}{suffix}")
    print("-" * 46)
