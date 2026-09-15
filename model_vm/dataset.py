"""
dataset.py — 视觉数据集（阶段 3.3）

加载视觉→珠态数据（PNG 图像目录 + jsonl 标签），返回 (images, text_tokens)。
供训练「看图读珠态」任务使用。

样本构造（对齐文本训练格式）：
  Q: 图像（n_cols 个 patch）
  A: <bead>#   （珠态文本 + 停止符 #）

训练时：
  - 图像 → VisionEncoder → 视觉 token [n_cols, hidden]
  - 文本 "盘面=？" + bead + "#" → 文本 token
  - 拼接后自回归训练（只对 bead 部分计算 loss）
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisionBeadDataset(Dataset):
    """视觉→珠态数据集。

    每个样本返回 (image, bead_ids, num)：
      image:  [n_cols, C, patch_h, patch_w] float32（0~1，C=1 灰度/C=3 彩色）
      bead_ids: 珠态文本的 token id 列表（含 # 停止符）
      num: 数字真值
    """

    def __init__(self, img_dir, jsonl_path, vocab_data):
        self.img_dir = img_dir
        self.stoi = vocab_data.stoi
        self.tokenizer = vocab_data.tokenizer

        self.labels = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.labels.append(json.loads(line.strip()))
        assert self.labels, f"标签文件为空: {jsonl_path}"

        n_png = sum(1 for f in os.listdir(img_dir)
                    if f.lower().endswith(".png"))
        assert n_png == len(self.labels), \
            f"图像数 {n_png} != 标签数 {len(self.labels)}"

        n_cols_set = {lab["n_cols"] for lab in self.labels}
        assert len(n_cols_set) == 1, f"n_cols 不一致: {sorted(n_cols_set)}"
        self.n_cols = n_cols_set.pop()

        # 从首个样本推导整图形状（全数据集同 n_cols，形状恒定）
        n_cols_actual, C, patch_h, patch_w = self._load_image(self.labels[0]["img"]).shape
        self.image_shape = (self.n_cols, C, patch_h, patch_w)

        print(f"[视觉数据集] 加载 {len(self.labels)} 条 "
              f"(每样本 {self.n_cols}×{C}×{patch_h}×{patch_w}, 图像目录 {img_dir})")

    def _load_image(self, filename):
        """整图 PNG → 每档 patch [n_cols, C, H, patch_w]（float32 0~1，C=1 灰度/C=3 彩色）。"""
        img = Image.open(os.path.join(self.img_dir, filename))
        if img.mode == "L":
            arr = np.asarray(img, dtype=np.float32) / 255.0      # [H, W]
            arr = arr[..., None]                                 # [H, W, 1]
        else:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
        H, W, C = arr.shape
        assert W % self.n_cols == 0, f"整图宽 {W} 无法按 {self.n_cols} 档切分"
        patch_w = W // self.n_cols
        return arr.reshape(H, self.n_cols, patch_w, C).transpose(1, 3, 0, 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lab = self.labels[idx]
        image = self._load_image(lab["img"])  # [n_cols, C, H, W]
        # 文本目标：bead + "#"
        text = f'{lab["bead"]}#'
        bead_ids = [self.stoi[t] for t in self.tokenizer.findall(text)]
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(bead_ids, dtype=torch.long),
            lab["num"],
        )


def vision_collate(batch):
    """整理 batch：图像堆叠，文本 pad。"""
    images = torch.stack([b[0] for b in batch])          # [B, n_cols, C, H, W]
    bead_seqs = [b[1] for b in batch]
    nums = [b[2] for b in batch]

    # pad 文本到最大长度
    max_len = max(len(s) for s in bead_seqs)
    padded = torch.zeros(len(bead_seqs), max_len, dtype=torch.long)
    for i, s in enumerate(bead_seqs):
        padded[i, :len(s)] = s
    return images, padded, nums


if __name__ == "__main__":
    import argparse
    import sys
    # 让本模块可直接运行 / 被引用（项目根 + 本目录加入 path）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    _here = os.path.dirname(os.path.abspath(__file__))
    _default_img_dir = os.path.join(_here, "data", "imgs")
    _default_jsonl = os.path.join(_here, "data", "labels.jsonl")

    parser = argparse.ArgumentParser(
        description="视觉→珠态数据集自检（无参数时读取 model_vm/data/ 下默认数据）")
    parser.add_argument("--img-dir", default=_default_img_dir,
                        help=f"PNG 图像目录，默认 {_default_img_dir}")
    parser.add_argument("--jsonl", default=_default_jsonl,
                        help=f"标签 jsonl，默认 {_default_jsonl}")
    parser.add_argument("--n", type=int, default=5, help="抽样打印的样本数")
    args = parser.parse_args()

    from model_vm.vocab import build_vision_vocab

    vocab = build_vision_vocab()
    ds = VisionBeadDataset(args.img_dir, args.jsonl, vocab)
    print(f"image_shape={ds.image_shape}")

    for i in range(min(args.n, len(ds))):
        image, bead_ids, num = ds[i]
        bead = ''.join(vocab.itos[t] for t in bead_ids.tolist())
        print(f"  [{i}] num={num} image={tuple(image.shape)} "
              f"bead_ids={bead_ids.tolist()} bead={bead!r}")

    # collate 冒烟
    batch = [ds[i] for i in range(min(4, len(ds)))]
    images, padded, nums = vision_collate(batch)
    print(f"collate → images={tuple(images.shape)} text={tuple(padded.shape)} nums={nums}")
