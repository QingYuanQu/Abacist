import time
import torch
from datasets import ArithmeticDataset
from model import GPT
from utils import generate

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = ["加", "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "等于"]
    dataset = ArithmeticDataset("data/data.txt", vocab, max_len=6)

    model = GPT(
        vocab_size=dataset.vocab_size,
        embed_dim=8,
        num_heads=1,
        num_layers=1,
        max_len=6
    ).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    print("════════════════════════════════════════")
    print("          UltraTinyLLM                ")
    print("  🧠 732 个参数，学会加法 — 世界上最小的 GPT")
    print("  📚 以极小窥宏大，是您入门与探秘大模型的最佳起点")
    print("  🔗 github.com/QingYuanQu/UltraTinyLLM")
    print("════════════════════════════════════════")
    print("  🧮 中文加法推理 (输入 quit 退出)")

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
            break

        if not prompt.endswith("等于"):
            prompt += "等于"

        try:
            chars = []
            for ch in generate(model, device, dataset, prompt, max_new=2):
                chars.append(ch)
                print(ch, end='', flush=True)
                time.sleep(0.3)
            print()
        except ValueError as e:
            print(f"无法识别: {e}")