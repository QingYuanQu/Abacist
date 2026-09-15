import json
import os
import torch
from torch.utils.data import Dataset

class ArithmeticReasoningDataset(Dataset):
    def __init__(self, file_path, vocab_data, repeat_factor=100, is_train=True,
                 extra_files=None):
        """加载算术推理数据集。

        Args:
            file_path: 主数据文件路径（算式数据）
            vocab_data: 词表数据（含 stoi 映射）
            repeat_factor: 数据重复因子
            is_train: 是否训练模式（影响 repeat_factor）
            extra_files: 额外的数据文件列表（如复习数据），仅在训练时加载
        """
        self.data = []
        self.repeat_factor = repeat_factor if is_train else 1
        self.stoi = vocab_data.stoi
        self.tokenizer = vocab_data.tokenizer

        # 加载主数据文件
        files_to_load = [file_path]
        # 训练模式下加载复习数据
        if is_train and extra_files:
            files_to_load.extend(extra_files)

        for fp in files_to_load:
            if not os.path.isfile(fp):
                print(f"[数据集] 跳过不存在的文件: {fp}")
                continue
            count = 0
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line.strip())
                    seq = f"{item['Q']}{item['A']}"
                    tokens = self.tokenizer.findall(seq)
                    self.data.append(tokens)
                    count += 1
            print(f"[数据集] 加载 {os.path.basename(fp)}: {count} 条")

        print(f"[数据集] 总计 {len(self.data)} 条样本, repeat_factor={self.repeat_factor}")

    def __len__(self):
        return len(self.data) * self.repeat_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        tokens = self.data[real_idx]
        token_ids = [self.stoi[tok] for tok in tokens]
        return torch.tensor(token_ids, dtype=torch.long)


# vocab = load_vocab("vocab/vocab1.json")
# tokenizer = build_tokenizer_regex(vocab)
# print(tokenizer.findall("<think>\nPRE + 1 1\nPOST 1 1 +\n</think>\nANS 2#"))