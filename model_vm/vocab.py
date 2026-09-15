"""
vocab.py — 视觉任务的词表构建

视觉→珠态任务的文本输出是珠态字符串（[L3|L4]#），token 集合小且确定。
直接手动构建词表，避免依赖 Q/A 格式的 build_vocab_from_file。

token 集合：
  数字 0-9、'['、']'、'|'、'L'、'U'、'空'、'+'、'#'、'<PAD>'
"""

import os
import sys
# 让 model_vm 下的独立模块可直接运行 / 被引用（项目根加入 path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vocab import PAD_TOKEN, STOP_TOKEN, Vocab


def build_vision_vocab() -> Vocab:
    """构建视觉任务词表。
    """
    tokens = [str(i) for i in range(10)]  # 0-9 固定在前（与主词表一致）
    extra = ['[', ']', '|', 'L', 'U', '空', '+', '#']
    tokens += sorted(t for t in extra if t not in tokens)
    tokens.append(PAD_TOKEN)

    vocab = Vocab(
        vocab=tokens,
        pad_id=len(tokens) - 1,
        max_seq_len=32,  # 珠态字符串最长为 5 档 × "L4+U1" 级，32 足够
        stop_token=STOP_TOKEN,
    )
    return vocab


if __name__ == "__main__":
    import argparse

    _here = os.path.dirname(os.path.abspath(__file__))
    _default_save = os.path.join(_here, "data", "vocab.json")

    parser = argparse.ArgumentParser(
        description="构建视觉→珠态任务词表（token 集合见模块 docstring）")
    parser.add_argument("--save", default=_default_save,
                        help=f"保存词表为 JSON，默认 {_default_save}")
    args = parser.parse_args()

    vocab = build_vision_vocab()
    print(f"词表大小: {vocab.vocab_size}")
    print(f"pad_id={vocab.pad_id}, stop_token={vocab.stop_token!r}")
    for i, tok in enumerate(vocab.vocab):
        print(f"  {i:>3}: {tok!r}")

    # tokenizer 冒烟：切分一个典型珠态文本
    sample = "L3|L4#"
    toks = vocab.tokenizer.findall(sample)
    ids = [vocab.stoi[t] for t in toks]
    print(f"示例 {sample!r} → tokens={toks} ids={ids}")

    if args.save:
        from model.vocab import save_vocab_multi
        save_vocab_multi(vocab.vocab, args.save,
                         source_files=[],
                         max_seq_len=vocab.max_seq_len,
                         config_hash="vision")
        print(f"词表已保存: {args.save}")
