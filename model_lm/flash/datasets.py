import torch
from torch.utils.data import Dataset

class ArithmeticDataset(Dataset):
    def __init__(self, file_path, chars, max_len, repeat_factor=100):
        self.chars = chars
        self.data = []
        self.repeat_factor = repeat_factor
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip() for line in f if line.strip() and len(line.strip()) <= max_len * 2]  # 字符数可能 > token数

    def __len__(self):
        return len(self.data) * self.repeat_factor

    def _tokenize(self, text):
        """最长匹配分词，支持多字符 token（如'等于'）"""
        tokens = []
        i = 0
        sorted_chars = sorted(self.chars, key=len, reverse=True)
        while i < len(text):
            matched = False
            for ch in sorted_chars:
                if text[i:i+len(ch)] == ch:
                    tokens.append(self.stoi[ch])
                    i += len(ch)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"无法识别的字符 '{text[i]}' 在位置 {i}，文本: {text}")
        return tokens

    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        seq = self.data[real_idx]
        tokens = self._tokenize(seq)
        return torch.tensor(tokens, dtype=torch.long)