"""
gen.py — 视觉→珠态数据生成器（阶段 3.3）

目标：训练模型「看图读珠态」——输入盘面图，输出文本珠态 [L3|L4]。

2026-09-04 阶段 1 切换到新内核 abacus/（原 datagen/eval 旧内核）：
  - 珠态构造：abacus.bead_codec.num_to_state（替代旧 Abacus(num, base)）
  - 渲染：abacus.render.registry.make_render（替代旧 render_frame_fixed）
  - 标签：state_to_bead_text（与旧 to_bead() 逐字节一致，已交叉验证）
  - 渲染后端默认 = 本模块顶部 DEFAULT_RENDER（minimal 5×1）；
    传 render='fixed'（灰度 patch）/ 'image'（真实算盘图）可切。
    换后端 = 换输入分布，需重新生成数据并重训。

2026-09-04 数据格式改版：npy 数组 → PNG 图像目录（可直接肉眼观察数据）。

数据格式（双输出）：
  图像目录 <img_dir>/  每样本一张灰度 PNG，文件名 <num:06d>.png（数字真值补零）
  标签文件 <prefix>.jsonl 每行 {"bead": "[L3|L4]", "num": 34, "img": "000034.png", "n_cols": 5}
  读回：像素 uint8 0~255 → float32 /255（minimal 二值后端无损；fixed 灰度含 1/255 量化）

用法：
  generate_vision(start, end, img_dir, jsonl_path, n_cols, ...)
"""

import json
import os
import sys
# 让 model_vm 下的独立模块可直接运行 / 被引用（项目根加入 path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from evaluate.abacus import Abacus, AbacusSpec
from evaluate.abacus import num_to_state, state_to_bead_text
from evaluate.abacus.render.registry import make_render

# model_vm 默认渲染类型：minimal / fixed / image（改这里一键切换生成哪种盘面）
DEFAULT_RENDER = "image"


def generate_vision(start: int, end: int, img_dir: str, jsonl_path: str,
                    n_cols: int = 5, render: str | None = None,
                    patch_h: int | None = None, patch_w: int | None = None,
                    base: int = 10, seed: int = 42):
    """生成视觉→珠态数据集。

    对 [start, end) 范围内每个数字：
      1. num_to_state 在 n_cols 档算盘上落珠（超出 base**n_cols 容量报错）
      2. registry 渲染盘面图（固定 n_cols 档，前导空档保留，个位在右）
      3. 盘面图存 PNG（灰度 uint8，文件名 <num:06d>.png），珠态文本存 jsonl

    Args:
        start, end: 数字范围 [start, end)
        img_dir: 图像输出目录（每样本一张 <num:06d>.png）
        jsonl_path: 标签文本输出路径（每行含 img 文件名与 n_cols）
        n_cols: 固定档位数
        render: 渲染后端名（None = 全局默认 DEFAULT_RENDER）
        patch_h, patch_w: 每档 patch 尺寸（None = 后端默认，如 minimal 5×1）
        base: 珠态进制（10=上1下4，16=上2下5）
        seed: 随机种子（当前未用，预留）
    """
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)

    spec = AbacusSpec(
        rod_count=n_cols, base=base,
        upper_count=2 if base == 16 else 1,
        lower_count=5 if base == 16 else 4,
    )
    abacus = Abacus.standard(spec)
    renderer = make_render(render, patch_h=patch_h, patch_w=patch_w)

    numbers = list(range(start, end))
    labels = []

    for num in numbers:
        state = num_to_state(num, abacus)
        img = renderer(state, abacus)          # (H, W) 灰度 或 (H, W, 3) 彩色
        name = f"{num:06d}.png"
        arr = (img * 255).round().astype(np.uint8)
        mode = "L" if arr.ndim == 2 else "RGB"
        Image.fromarray(arr, mode=mode).save(os.path.join(img_dir, name))
        labels.append({"bead": state_to_bead_text(state), "num": num, "img": name, "n_cols": n_cols})

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + '\n')

    shape_str = f"{img.shape[0]}×{img.shape[1]}" if numbers else "空"
    if numbers and img.ndim == 3:
        shape_str += f"×{img.shape[2]}"
    print(f"[视觉数据] 已生成 {len(numbers)} 条 → 图像目录 {img_dir} ({shape_str}), "
          f"标签 {jsonl_path}")
    return len(numbers)


if __name__ == "__main__":
    import argparse

    # 默认输出到本模块旁的 data/（model_vm/data），便于直接观察测试数据
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_img_dir = os.path.join(_here, "data", "imgs")
    _default_jsonl = os.path.join(_here, "data", "labels.jsonl")

    parser = argparse.ArgumentParser(description="视觉→珠态数据生成器（数据格式见模块 docstring；"
                    "无参数时生成 0~19 测试数据到 model_vm/data/）")
    parser.add_argument("--start", type=int, default=0, help="起始数字（含），默认 0")
    parser.add_argument("--end", type=int, default=100, help="结束数字（不含），默认 20")
    parser.add_argument("--img-dir", default=_default_img_dir,
                        help=f"PNG 图像输出目录，默认 {_default_img_dir}")
    parser.add_argument("--jsonl", default=_default_jsonl,
                        help=f"标签 jsonl 输出路径，默认 {_default_jsonl}")
    parser.add_argument("--n-cols", type=int, default=5, help="档位数")
    parser.add_argument("--render", default=DEFAULT_RENDER,
                        help=f"渲染后端名（minimal / fixed / image / image_gray），默认 {DEFAULT_RENDER}")
    parser.add_argument("--base", type=int, default=10, help="珠态进制")
    parser.add_argument("--doctest", action="store_true",
                        help="只跑 docstring 自检（临时目录，数据即用即弃）")
    args = parser.parse_args()

    if args.doctest:
        import doctest
        result = doctest.testmod(verbose=False)
        print(f"doctest: {result.attempted} 项, {result.failed} 失败")
        sys.exit(1 if result.failed else 0)

    generate_vision(args.start, args.end, args.img_dir, args.jsonl,
                    n_cols=args.n_cols, render=args.render, base=args.base)
