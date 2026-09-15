"""
dataset.py — VLM 数据集（阶段 3.4.2）

加载 VLM 数据（npy 图像 + jsonl 标签），返回 (image, prompt_ids, rhyme_ids)。
供训练「看图 + 指令 → 口诀动作」任务使用。

样本构造（对齐 3.3 视觉训练格式，但文本分两段）：
  输入:  [视觉 token | prompt token("加 7 =")]
  输出:  rhyme token("7去3进1#")

训练时 loss 对齐（关键）：
  第 1 个 rhyme token 由「prompt 最后一个 token」位置预测，
  后续 rhyme token 由「前一个 rhyme token」位置预测。
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from evaluate.abacus.render.registry import DEFAULT_RENDER, resolve_patch


def to_rod_patches(arr: np.ndarray, n_cols: int) -> np.ndarray:
    """整幅盘面 → 逐档 patch，统一输出 [n_cols, C, H, W]。

    输入两种形态（registry 各后端的返回）：
      - [H, n_cols*W]      灰度（minimal/fixed/image_gray）→ C=1
      - [H, n_cols*W, C]   彩色（image，C=3）→ 原通道数
    输出与 VisionEncoder 的输入契约 [B, n_cols, C, H, W] 逐元素对齐
    （去掉 B 维）。PNG 路径与 npy 路径都必须经此函数，保证两条
    加载路径产出完全一致的布局。
    """
    if arr.ndim == 2:                                   # 灰度
        h, w = arr.shape
        return arr.reshape(h, n_cols, w // n_cols) \
                  .transpose(1, 0, 2)[:, None]          # [n_cols, 1, H, W]
    if arr.ndim == 3:                                   # 彩色（末维通道）
        h, w, c = arr.shape
        return arr.reshape(h, n_cols, w // n_cols, c) \
                  .transpose(1, 3, 0, 2)                # [n_cols, C, H, W]
    raise ValueError(f"无法切分的图像维度: {arr.shape}")


class VLMDataset(Dataset):
    """VLM 数据集。

    每个样本返回 (image, prompt_ids, rhyme_ids, num)：
      image:      [n_cols, C, patch_h, patch_w] float32（C=1 灰度 / C=3 彩色）
      prompt_ids: 指令文本 "加 <digit> =" 的 token id 列表
      rhyme_ids: 口诀文本 "<rhyme>#" 的 token id 列表
      num:        当前盘面值（调试）
    """

    def __init__(self, jsonl_path, vocab_data, *, npy_path=None,
                 image_dir=None, n_cols=13,
                 patch_h=None, patch_w=None, render=DEFAULT_RENDER):
        """支持两种图像来源：
          - npy_path: 图像堆叠 npy（新版 [N, n_cols, C, H, W]；旧版
            [N, n_cols, H, W] 灰度自动补 C 维）
          - image_dir: 独立 PNG，jsonl 中 `image` 字段存相对路径
            （L 灰度 → C=1；RGB 彩色 → C=3，按 PIL mode 推导）

        patch_h/patch_w 显式给出时优先；缺省按 `render` 后端解析——尺寸是
        **后端的属性**，不是全局常量（此前取全局默认 → 5×1，与 fixed 数据不匹配）。
        零占位图 fallback 在无真实图可推断时按同一规则解析。
        """
        self.stoi = vocab_data.stoi
        self.tokenizer = vocab_data.tokenizer
        self.pad_id = vocab_data.pad_id
        self.n_cols = n_cols
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.render = render
        self.image_dir = Path(image_dir) if image_dir else None

        if npy_path is not None:
            self.images = np.load(npy_path)
        else:
            self.images = None
        self.labels = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.labels.append(json.loads(line.strip()))

        # parse 样本没有 image 字段，需要全零占位图；通道数由真实图
        # 推导（npys.shape / PNG mode），供 train_vlm 构建编码器用。
        self._zero_image, self.in_channels = self._build_zero_image()

        if self.images is not None:
            # 新版（image 字段=int 下标）图像已去重，数量与标签数无关；
            # 仅旧版「堆叠图与行一一对应」形态才要求行数相等；
            # 纯文本样本（如 parse）无 image 字段，用零占位图，也不要求行数相等
            indexed = any(isinstance(rec.get("image"), (int, np.integer))
                          for rec in self.labels)
            has_image = any("image" in rec for rec in self.labels)
            if has_image and not indexed:
                assert len(self.images) == len(self.labels), \
                    f"图像数 {len(self.images)} != 标签数 {len(self.labels)}"
            print(f"[VLM数据集] 加载 {len(self.labels)} 条 (npy 图像 {self.images.shape})")
        else:
            print(f"[VLM数据集] 加载 {len(self.labels)} 条 (图片目录 {self.image_dir})")

    def _build_zero_image(self):
        """构造 parse 样本使用的全零占位图，返回 ([n_cols, C, H, W], C)。"""
        if self.images is not None:
            z = np.zeros_like(self.images[0])
            if z.ndim == 3:            # 旧版 [n_cols, H, W] 灰度 → 补 C 维
                z = z[None]
            return z, z.shape[1]
        if self.image_dir is not None:
            for label in self.labels:
                img_name = label.get("image")
                if not img_name:
                    continue
                img_path = self.image_dir / img_name
                if img_path.exists():
                    img = Image.open(img_path)
                    c = 3 if img.mode == "RGB" else 1
                    # 尺寸从首图推导（patch_h=H, patch_w=W//n_cols）
                    w, h = img.size
                    return (np.zeros((self.n_cols, c, h, w // self.n_cols),
                                     dtype=np.float32), c)
        # 无图可推断：按 render 后端解析尺寸（C=1 灰度）
        h, w = resolve_patch(self.render, self.patch_h, self.patch_w)
        return (np.zeros((self.n_cols, 1, h, w), dtype=np.float32), 1)

    def __len__(self):
        return len(self.labels)

    def _load_image(self, idx):
        """从 npy 或 PNG 路径加载并整理为 [n_cols, C, H, W]。

        npy 两种形态：
          - 旧版：堆叠图与 jsonl 行一一对应（images[idx]）；
          - 新版（parse_eval_bridge 全量生成）：jsonl 的 `image` 字段为
            去重数组下标（int），存储 dtype=uint8（0..255）。
        parse 样本没有 `image` 字段，返回全零占位图。
        """
        if self.images is not None:
            rec = self.labels[idx]
            if "image" not in rec:            # 纯文本样本（parse）→ 零占位图
                return self._zero_image.copy()
            img = (self.images[rec["image"]]
                   if isinstance(rec.get("image"), (int, np.integer))
                   else self.images[idx])
            if img.ndim == 3:          # 旧版灰度 npy [n_cols, H, W] → 补 C 维
                img = img[None]
            if img.dtype == np.uint8:      # 新版 uint8 存储 → 归一化到 0..1
                return img.astype(np.float32) / 255.0
            return img

        rec = self.labels[idx]
        if "image" not in rec:
            return self._zero_image.copy()

        img_path = self.image_dir / rec["image"]
        # 保留原通道：L 灰度 → C=1，RGB 彩色 → C=3（勿 convert("L") 抹掉颜色）
        img = Image.open(img_path)
        arr = np.array(img, dtype=np.float32) / 255.0
        return to_rod_patches(arr, self.n_cols)

    def __getitem__(self, idx):
        image = self._load_image(idx)  # [n_cols, C, H, W]
        prompt = self.labels[idx]["prompt"]
        rhyme = self.labels[idx]["A"]
        num = self.labels[idx].get("num", self.labels[idx].get("value", 0))

        prompt_ids = [self.stoi[t] for t in self.tokenizer.findall(prompt)]
        rhyme_ids = [self.stoi[t] for t in self.tokenizer.findall(rhyme)]
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(prompt_ids, dtype=torch.long),
            torch.tensor(rhyme_ids, dtype=torch.long),
            num,
        )


def vlm_collate(batch, pad_id=0):
    """整理 batch：图像堆叠；文本逐样本拼接 [prompt | rhyme前缀]，pad 只放尾部。

    关键约束（与推理 generate 布局一致）：
      - rhyme 前缀（A[:-1]）紧跟 prompt 真实 token 之后，中间不允许夹 pad，
        否则"rhyme[k] 由 rhyme[k-1] 位置预测"的自回归因果链断裂
        （短 prompt 样本会被 batch 内最大 prompt 长度撑出 PAD 夹层）；
      - pad 只出现在序列尾部，causal attention 下对有效位置无害。

    返回 (images, seq_ids, rhyme_ids, prompt_lens, nums)：
      images:     [B, n_cols, C, H, W]
      seq_ids:    [B, L] 拼接文本 token ids（L = max(Lp_b + Lm_b - 1)），尾 pad
      rhyme_ids: [B, Lm] 完整 rhyme 标签（loss 目标），尾 pad
      prompt_lens:[B] 每个样本 prompt 真实长度
    """
    images = torch.stack([b[0] for b in batch])
    prompts = [b[1] for b in batch]
    rhymes = [b[2] for b in batch]
    nums = [b[3] for b in batch]

    B = len(prompts)
    seqs = []
    prompt_lens = torch.zeros(B, dtype=torch.long)
    for i, (p, m) in enumerate(zip(prompts, rhymes)):
        m_in = m[:-1] if len(m) > 1 else torch.empty(0, dtype=torch.long)
        seqs.append(torch.cat([p, m_in]))
        prompt_lens[i] = len(p)

    L = max(len(s) for s in seqs)
    seq_ids = torch.full((B, L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq_ids[i, :len(s)] = s

    m_len = max(len(m) for m in rhymes)
    rhyme_ids = torch.full((B, m_len), pad_id, dtype=torch.long)
    for i, m in enumerate(rhymes):
        rhyme_ids[i, :len(m)] = m

    return images, seq_ids, rhyme_ids, prompt_lens, nums
