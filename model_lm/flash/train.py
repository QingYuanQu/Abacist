"""训练入口（多版本选择器）。

用法：
    python train.py                 # 默认训练 original（即原 model.py），存 model.pth
    python train.py v1              # 极简版：黑盒注意力、无 PE
    python train.py v2              # 手写注意力、无 PE（对照 v1 看注意力内部）
    python train.py v3              # 固定不可学习正弦 PE
    python train.py v3 --no-pe      # 同结构关掉 PE → 退回 v2，直观看出 PE 贡献
    python train.py v4              # 可学习 PE（= 原 model.py 的 PE 思路）
    python train.py v5              # RoPE
    python train.py v5 --epochs 40
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import ArithmeticDataset
from utils import train, set_seed

from model import GPT as GPT_original
from model_v1 import GPT as GPT_v1
from model_v2 import GPT as GPT_v2
from model_v3 import GPT as GPT_v3
from model_v4 import GPT as GPT_v4
from model_v5 import GPT as GPT_v5

MODELS = {
    "original": GPT_original,
    "v1": GPT_v1, "v2": GPT_v2, "v3": GPT_v3, "v4": GPT_v4, "v5": GPT_v5,
}
VOCAB = ["加", "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "等于"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", default="original", choices=list(MODELS))
    ap.add_argument("--no-pe", action="store_true", help="关闭位置编码（退回无 PE 基线）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--embed-dim", type=int, default=8)
    ap.add_argument("--num-heads", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ArithmeticDataset("data/data.txt", VOCAB, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=87, shuffle=True, num_workers=0, drop_last=False)

    GPT = MODELS[args.version]
    if args.version == "original":
        # 原版签名无 use_pe，保持原样以兼容 eval.py / eval_user.py
        model = GPT(vocab_size=dataset.vocab_size, embed_dim=args.embed_dim,
                    num_heads=args.num_heads, num_layers=args.num_layers,
                    max_len=args.max_len).to(device)
        save_path = "model.pth"
    else:
        model = GPT(vocab_size=dataset.vocab_size, embed_dim=args.embed_dim,
                    num_heads=args.num_heads, num_layers=args.num_layers,
                    max_len=args.max_len, use_pe=not args.no_pe).to(device)
        # 关掉 PE 的 checkpoint 单独命名，避免与带 PE 的互相覆盖（教学对比时会混）
        save_path = f"model_{args.version}{'_nope' if args.no_pe else ''}.pth"

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.version}] Total parameters: {total_params:,}  (use_pe={not args.no_pe})")
    train(model, dataloader, optimizer, device, args.epochs)
    torch.save(model.state_dict(), save_path)
    print(f"saved -> {save_path}")


if __name__ == "__main__":
    main()
