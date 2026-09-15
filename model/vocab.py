import hashlib
import os
import re
import json
import argparse
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

PAD_TOKEN = "<PAD>"   # 全局常量，方便其他模块引用
STOP_TOKEN = "#"        # 停止 token，所有词表统一使用

def build_vocab_from_file(file_path, tokenizer_pattern=None):
    if tokenizer_pattern is None:
        tokenizer_pattern = r'<think>|</think>|#|INFIX|PRE|POST|STACK|ANS|.'

    tokenizer = re.compile(tokenizer_pattern, flags=re.DOTALL)
    dataset_vocab = set()
    max_seq_len = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                # 兼容新格式 prompt / 旧格式 Q
                seq = f"{item.get('prompt', item.get('Q', ''))}{item['A']}"
                tokens = tokenizer.findall(seq)
                dataset_vocab.update(tokens)
                max_seq_len = max(max_seq_len, len(tokens))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: 跳过无效行 - {e}")
                continue

    return sorted(list(dataset_vocab)), max_seq_len


def build_vocab_from_files(file_paths, tokenizer_pattern=None):
    """从多个数据文件构建统一词表。

    Args:
        file_paths: 数据文件路径列表
        tokenizer_pattern: 分词正则（可选）
    Returns:
        (vocab_list, max_seq_len)
    """
    all_vocab = set()
    max_seq_len = 0
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"[词表] 跳过不存在的文件: {path}")
            continue
        vocab, max_len = build_vocab_from_file(path, tokenizer_pattern)
        all_vocab.update(vocab)
        max_seq_len = max(max_seq_len, max_len)
        print(f"[词表]   {os.path.basename(path)}: vocab={len(vocab)}, max_len={max_len}")
    return sorted(list(all_vocab)), max_seq_len


def save_vocab(vocab, output_path, source_file, max_seq_len, config_hash):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 固定数字 0-9 的索引为 0-9，其他 token 按顺序排列
    digits = [str(i) for i in range(10)]
    other_tokens = sorted([tok for tok in vocab if tok not in digits])
    # <PAD> 强制追加到最后，保证 id 与所有数据 token 不冲突
    if PAD_TOKEN not in other_tokens:
        other_tokens.append(PAD_TOKEN)
    sorted_vocab = digits + other_tokens
    stoi = {tok: i for i, tok in enumerate(sorted_vocab)}
    itos = {str(i): tok for i, tok in enumerate(sorted_vocab)}  # JSON key 必须是字符串

    # ← 新增：元数据
    source_mtime = None
    if source_file and os.path.isfile(source_file):
        source_mtime = os.path.getmtime(source_file)

    vocab_data = {
        "vocab": sorted_vocab,
        "vocab_size": len(sorted_vocab),
        "stoi": stoi,
        "itos": itos,
        "pad_id": stoi[PAD_TOKEN],
        "metadata": {
            "source_file": source_file,
            "source_mtime": source_mtime,
            "config_hash": config_hash,
            "max_seq_len": max_seq_len,
            "stop_token": STOP_TOKEN,
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"词表已保存到: {output_path}")
    print(f"词表大小: {len(sorted_vocab)}")
    if max_seq_len:
        print(f"最大序列长度: {max_seq_len}")

def load_vocab(vocab_path):
    """加载词表为 Vocab 对象（替代裸 dict）。"""
    return Vocab.load(vocab_path)


def build_tokenizer_regex(vocab):
    sorted_tokens = sorted(vocab, key=len, reverse=True)
    escaped = [re.escape(tok) for tok in sorted_tokens]
    return re.compile('|'.join(escaped), flags=re.DOTALL)


@dataclass
class Vocab:
    """词表领域对象 —— 替代裸 dict 承载词表数据。

    单一数据源是 `vocab` 列表；stoi/itos/vocab_size/tokenizer 均为派生属性，
    消除存读两套的不一致。原 metadata 兜底 dict 被拆平为顶层字段。
    """
    vocab: list[str]                     # 排序 token 列表（含 <PAD>）
    pad_id: int                          # 填充 token 的 id
    max_seq_len: int = 0                 # 从 metadata 提上来
    stop_token: str = STOP_TOKEN         # 从 metadata 提上来
    source_files: list[str] = field(default_factory=list)
    config_hash: str = ""

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @cached_property
    def stoi(self) -> dict[str, int]:
        return {tok: i for i, tok in enumerate(self.vocab)}

    @cached_property
    def itos(self) -> dict[int, str]:
        return {i: tok for i, tok in enumerate(self.vocab)}

    @cached_property
    def tokenizer(self):
        return build_tokenizer_regex(self.vocab)

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        """从现有词表 JSON（dict）构造，兼容旧格式（忽略存里的 stoi/itos）。"""
        meta = d.get("metadata") or {}
        source_files = meta.get("source_files") or [meta.get("source_file", "")]
        source_files = [f for f in source_files if f]
        pad_id = d.get("pad_id")
        if pad_id is None:
            stoi = d.get("stoi") or {tok: i for i, tok in enumerate(d["vocab"])}
            pad_id = stoi.get(PAD_TOKEN, 0)
        return cls(
            vocab=list(d["vocab"]),
            pad_id=int(pad_id),
            max_seq_len=int(meta.get("max_seq_len", 0)),
            stop_token=meta.get("stop_token", STOP_TOKEN),
            source_files=source_files,
            config_hash=meta.get("config_hash", ""),
        )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """从词表文件加载 Vocab（替代 load_vocab 的对象化入口）。"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


def data_config_hash(data_config):
    """从 DATA 关键参数生成指纹，用于判断词表是否需要重建。"""
    key_fields = ['ops', 'mode', 'input_fmt', 'repeat', 'start', 'end',
                  'allow_negative']
    key = ','.join(f"{k}={data_config.get(k)}" for k in key_fields)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def ensure_vocab(vocab_config, data_config, model_config=None, extra_files=None):
    """确保词表与当前配置匹配：自动重建。

    自动检测珠态数据（bead_data.jsonl），若存在则合并入词表。

    Args:
        vocab_config: VOCAB 配置字典
        data_config: DATA 配置字典
        model_config: (保留兼容，不再使用)
        extra_files: 额外的数据文件列表（可选）
    Returns: vocab_data (dict)
    """
    vocab_path = vocab_config['vocab_path']
    data_path = data_config['train_data_path']
    current_hash = data_config_hash(data_config)

    # 自动检测珠态数据
    bead_path = data_config.get('bead_data_path', '')
    data_files = [data_path]
    if extra_files:
        data_files.extend(extra_files)
    if os.path.isfile(bead_path):
        data_files.append(bead_path)

    need_rebuild = False
    reason = ""

    if not os.path.isfile(vocab_path):
        need_rebuild = True
        reason = "词表文件不存在"
    else:
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            meta = existing.get('metadata', {})
            stored_hash = meta.get('config_hash')
            stored_files = meta.get('source_files', [meta.get('source_file', '')])

            if stored_hash != current_hash:
                need_rebuild = True
                reason = f"配置指纹不匹配 (旧={stored_hash}, 新={current_hash})"
            elif set(stored_files) != set(data_files):
                need_rebuild = True
                reason = f"数据文件列表变更"
            else:
                for fp in data_files:
                    if os.path.isfile(fp) and fp not in stored_files:
                        need_rebuild = True
                        reason = f"新增数据文件: {fp}"
                        break
        except (json.JSONDecodeError, KeyError):
            need_rebuild = True
            reason = "词表文件格式异常"

    if need_rebuild:
        print(f"[词表] 需要重建: {reason}")
        print(f"[词表] 正在从 {len(data_files)} 个文件构建词表...")
        vocab_list, max_seq_len = build_vocab_from_files(data_files)
        save_vocab_multi(vocab_list, vocab_path,
                         source_files=data_files,
                         max_seq_len=max_seq_len,
                         config_hash=current_hash)

    vocab_data = load_vocab(vocab_path)

    print(f"[词表] 就绪 | vocab_size={vocab_data.vocab_size} "
          f"| max_seq_len={vocab_data.max_seq_len}")
    return vocab_data


def save_vocab_multi(vocab, output_path, source_files, max_seq_len, config_hash):
    """保存词表（多文件版本），与 save_vocab 兼容。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 固定数字 0-9 的索引为 0-9，其他 token 按顺序排列
    digits = [str(i) for i in range(10)]
    other_tokens = sorted([tok for tok in vocab if tok not in digits])
    if PAD_TOKEN not in other_tokens:
        other_tokens.append(PAD_TOKEN)
    sorted_vocab = digits + other_tokens
    stoi = {tok: i for i, tok in enumerate(sorted_vocab)}
    itos = {str(i): tok for i, tok in enumerate(sorted_vocab)}

    # 记录每个源文件的 mtime，用于缓存检查
    source_mtimes = {}
    for fp in source_files:
        if os.path.isfile(fp):
            source_mtimes[fp] = os.path.getmtime(fp)

    vocab_data = {
        "vocab": sorted_vocab,
        "vocab_size": len(sorted_vocab),
        "stoi": stoi,
        "itos": itos,
        "pad_id": stoi[PAD_TOKEN],
        "metadata": {
            "source_files": source_files,
            "source_mtimes": source_mtimes,
            "config_hash": config_hash,
            "max_seq_len": max_seq_len,
            "stop_token": STOP_TOKEN,
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"词表已保存到: {output_path}")
    print(f"词表大小: {len(sorted_vocab)}")
    if max_seq_len:
        print(f"最大序列长度: {max_seq_len}")

def _build_vocab_with_cache(vocab_path, data_files):
    """带缓存检查的词表构建，共享逻辑。"""
    # 缓存检查
    if os.path.isfile(vocab_path):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            meta = existing.get('metadata', {})
            stored_files = meta.get('source_files', [])
            stored_mtimes = meta.get('source_mtimes', {})
            if set(stored_files) == set(data_files):
                mtimes_match = True
                for fp in data_files:
                    current_mtime = os.path.getmtime(fp) if os.path.isfile(fp) else None
                    stored_mtime = stored_mtimes.get(fp)
                    if current_mtime is None or stored_mtime is None or abs(current_mtime - stored_mtime) > 1e-6:
                        mtimes_match = False
                        break
                if mtimes_match:
                    print(f"[课程词表] 缓存命中（{len(data_files)} 个源文件未变），跳过重建")
                    vocab_data = load_vocab(vocab_path)
                    print(f"[课程词表] 就绪 | vocab_size={vocab_data.vocab_size} "
                          f"| max_seq_len={vocab_data.max_seq_len}")
                    return vocab_data
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    print(f"[课程词表] 正在从 {len(data_files)} 个文件构建词表...")
    vocab_list, max_seq_len = build_vocab_from_files(data_files)
    save_vocab_multi(vocab_list, vocab_path,
                     source_files=data_files,
                     max_seq_len=max_seq_len,
                     config_hash="curriculum")

    vocab_data = load_vocab(vocab_path)
    print(f"[课程词表] 就绪 | vocab_size={vocab_data.vocab_size} "
          f"| max_seq_len={vocab_data.max_seq_len}")
    return vocab_data


def ensure_study_vocab(trial_configs, curriculum_config):
    """扫描全部课程数据文件，构建统一词表（带缓存）—— shared 策略。

    确保词表覆盖所有课程，这样模型权重在课程间
    传递时 token id 不会错位。

    缓存机制：若词表文件已存在，检查 source_files 列表一致
    且每个文件 mtime 未变 → 直接复用，跳过重建。

    Args:
        trial_configs: Trial 列表
        curriculum_config: CURRICULUM 配置字典
    Returns:
        vocab_data (dict)
    """
    vocab_path = curriculum_config.get('study_vocab_path',
                                   curriculum_config.get('vocab_path', 'vocab/study_vocab.json'))

    data_files = []
    seen = set()
    for cfg in trial_configs:
        m = cfg.material
        for fp in (m.train_data_path,
                   m.test_data_path,
                   m.bead_data_path):
            if fp and os.path.isfile(fp) and fp not in seen:
                data_files.append(fp)
                seen.add(fp)

    return _build_vocab_with_cache(vocab_path, data_files)


def ensure_experiment_vocab(exp, trial_configs):
    """experiment 类型：按去重后的数据文件集合建一份共享词表。

    experiment 的"独立"只指模型权重/训练每课独立，而非数据/词表独立——
    数据共享时（同 source + 同投影参数 → 同 train/test 路径），词表也共享。
    因此对全部 trial 的数据文件去重后，建一份共享词表到 exp.vocab_path，
    返回 {trial_id: 同一 vocab_data}，保持调用方以 trial_id 取用的接口不变。

    Returns:
        dict[int, vocab_data] — 所有 trial_id 指向同一份共享词表（无数据则空 dict）
    """
    os.makedirs(exp.vocab_dir, exist_ok=True)
    data_files = []
    seen = set()
    for cfg in trial_configs:
        m = cfg.material
        for fp in (m.train_data_path, m.test_data_path, m.bead_data_path):
            if fp and os.path.isfile(fp) and fp not in seen:
                data_files.append(fp)
                seen.add(fp)

    if not data_files:
        return {}

    vocab_data = _build_vocab_with_cache(exp.vocab_path, data_files)
    return {cfg.id: vocab_data for cfg in trial_configs}


def main():
    parser = argparse.ArgumentParser(description="从数据文件构建词表")
    parser.add_argument("--input", nargs='+', default=None,
                        help="输入数据文件路径列表 (.jsonl)")
    parser.add_argument("--output", default="vocab/default_vocab.json", help="输出词表文件路径 (.json)")
    parser.add_argument("--pattern", default=None, help="自定义分词正则表达式")

    args = parser.parse_args()

    if args.input is None:
        print("[词表] 错误：请通过 --input 指定输入数据文件")
        return

    data_files = args.input

    print(f"正在从 {len(data_files)} 个文件构建词表:")
    for fp in data_files:
        print(f"  - {fp}")
    vocab_list, max_seq_len = build_vocab_from_files(data_files, args.pattern)
    save_vocab_multi(vocab_list, args.output,
                     source_files=data_files,
                     max_seq_len=max_seq_len,
                     config_hash="cli")


if __name__ == "__main__":
    main()
