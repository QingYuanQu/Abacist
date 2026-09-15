"""
vocab.py — VLM 任务词表构建（主线，阶段 3.4.2）

视觉语言动作任务的文本输出是「指令 + 口诀」，token 集合在视觉珠态
词表基础上加入指令与口诀动作 token。原始 vocab.py 中一并定义的
build_vision_vocab（看图读珠态）已随独立 vision 子系统迁至 model_vm/，
本文件只保留主线 VLM 词表构造。
"""

from model.vocab import PAD_TOKEN, STOP_TOKEN, Vocab


def build_vlm_vocab() -> Vocab:
    """构建 VLM 任务词表（视觉语言动作）。

    在视觉词表基础上，加入：
      - 指令 token：'加'、'='（空格由 tokenizer 处理）
      - 口诀动作 token：'上'、'下'、'去'、'进'、'退'、'还'
    """
    tokens = [str(i) for i in range(10)]  # 0-9 固定在前
    extra = ['[', ']', '|', 'L', 'U', '空', '+', '#',
             '加', '=', '上', '下', '去', '进', '退', '还']
    tokens += sorted(t for t in extra if t not in tokens)
    tokens.append(PAD_TOKEN)

    vocab = Vocab(
        vocab=tokens,
        pad_id=len(tokens) - 1,
        max_seq_len=64,  # 指令 + 口诀最长约 16 字符，64 足够
        stop_token=STOP_TOKEN,
    )
    return vocab
