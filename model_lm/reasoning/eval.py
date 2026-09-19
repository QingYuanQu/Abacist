"""通用 LM 评估能力（与实验无关）。

实验相关的单 trial 评估（依赖 experiment/trial 聚合对象）见 experiment/trial/eval.py。
"""
import json

import torch

@torch.no_grad()
def chat(model, device, stoi, itos, tokenizer, model_max_len, stop_id):
    """交互式聊天：读取用户输入的算式，逐条生成答案。"""
    print("=" * 60)
    print("Abacist 交互模式（输入 quit/exit 退出）")
    print("示例输入: 10+13  或  123+456")
    print("=" * 60)
    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if user.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user:
            continue

        # 与训练数据对齐：Q 形如 "(10+13)="
        prompt = f"{user}="
        result = generate(model, device, stoi, itos, tokenizer, prompt,
                          model_max_len=model_max_len, stop_id=stop_id)
        print(f"模型: {result}")

@torch.no_grad()
def generate(model, device, stoi, itos, tokenizer, prompt, model_max_len=10, stop_id=None):
    """
    使用 KV Cache 的高效生成函数。
    第一步：完整 prompt 前向，缓存所有 KV。
    后续步：每次只输入上一个 token，复用 KV Cache，只做 O(T) 计算。
    """
    model.eval()
    ids = [stoi[t] for t in tokenizer.findall(prompt)]
    prompt_tensor = torch.tensor([ids], device=device)
    generated = ids[:]

    # 第一步：处理完整 prompt，缓存 KV
    logits, past_kvs = model(prompt_tensor, use_cache=True, pos_offset=0)

    # 后续步：逐步生成
    for step in range(model_max_len - len(prompt)):
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)

        # 停止条件
        if stop_id and next_id in stop_id:
            break

        # 只输入新生成的 token，复用 KV Cache
        current_pos = len(generated) - 1  # 新 token 的位置
        next_input = torch.tensor([[next_id]], device=device)
        logits, past_kvs = model(next_input, past_kvs=past_kvs, use_cache=True, pos_offset=current_pos)

    output_ids = generated[len(ids):]
    result = ''.join([itos[i] for i in output_ids])
    return result


@torch.no_grad()
def generate_batch(model, device, stoi, itos, tokenizer, prompts,
                   model_max_len, stop_ids, pad_id):
    """按 prompt 长度分桶，同长度批量贪心解码。

    设计原理：
    - 同长度 prompt 凑一批 → 无需 padding → 不需要 attention mask
    - pos_offset 标量对整批一致 → 和单条 generate() 数值逐 token 等价
    - 变长答案通过 done 掩码处理：已停样本喂占位 token，输出忽略

    Args:
        model: GPT 模型
        device: torch.device
        stoi: token → id 映射
        itos: id → token 映射
        tokenizer: 编译好的正则 tokenizer
        prompts: 字符串列表
        model_max_len: 模型最大序列长度
        stop_ids: set of stop token ids
        pad_id: 填充 token id（用于已停样本的占位输入）

    Returns:
        list[str]: 每个 prompt 的生成结果（不含 prompt，与 generate() 语义一致）
    """
    model.eval()
    B_total = len(prompts)

    # ---- 1. 分词 + 按长度分桶 ----
    buckets = {}  # {plen: [(orig_idx, ids), ...]}
    for i, p in enumerate(prompts):
        ids = [stoi[t] for t in tokenizer.findall(p)]
        plen = len(ids)
        buckets.setdefault(plen, []).append((i, ids))

    results = [None] * B_total
    # 把 stop_ids 转成 Python set（支持 set/list）
    stop_set = set(stop_ids)

    # ---- 2. 逐桶批量生成 ----
    for plen, items in buckets.items():
        B = len(items)
        orig_idx = [x[0] for x in items]
        # [B, plen]，所有样本同长，零 padding
        batch = torch.tensor([x[1] for x in items], device=device)
        max_new = model_max_len - plen

        # ---- prefill：一次前向处理整批 prompt ----
        logits, kvs = model(batch, use_cache=True, pos_offset=0)
        next_ids = logits[:, -1].argmax(-1)  # [B]

        done = torch.zeros(B, dtype=torch.bool, device=device)
        gen_ids = [[] for _ in range(B)]

        # ---- decode：同步推进，done 掩码处理变长答案 ----
        for s in range(max_new):
            # 记录本轮生成的 token（已停样本跳过记录）
            for b in range(B):
                if not done[b]:
                    tid = next_ids[b].item()
                    gen_ids[b].append(tid)
                    if tid in stop_set:
                        done[b] = True

            if done.all():
                break

            # 构造下一步输入：已停样本喂 pad_id，输出会被忽略
            inp = next_ids.unsqueeze(1).clone()  # [B, 1]
            inp[done] = pad_id

            logits, kvs = model(inp, past_kvs=kvs, use_cache=True,
                                pos_offset=plen + s)
            next_ids = logits[:, -1].argmax(-1)
            # 已停样本冻结输出为 pad_id（确保不影响后续逻辑）
            next_ids[done] = pad_id

        # ---- 拼串结果 ----
        for b, oi in enumerate(orig_idx):
            results[oi] = ''.join(itos[t] for t in gen_ids[b])

    return results


def _verify_generate_batch(model, device, stoi, itos, tokenizer,
                           model_max_len, stop_ids, pad_id,
                           sample_prompts):
    """等价性自检：逐条 generate() vs generate_batch() 对比。

    选取若干条样本，先逐条生成，再用 generate_batch 批量生成，
    逐字符比对。全部一致才通过，否则抛出 AssertionError。

    Args:
        sample_prompts: 用于自检的 prompt 列表（建议 20 条以上）
    """
    print("[自检] 逐条 vs 批量等价性验证...")
    # 逐条
    single_results = []
    for p in sample_prompts:
        single_results.append(
            generate(model, device, stoi, itos, tokenizer, p,
                     model_max_len=model_max_len, stop_id=stop_ids)
        )

    # 批量（同批次）
    batch_results = generate_batch(
        model, device, stoi, itos, tokenizer, sample_prompts,
        model_max_len=model_max_len, stop_ids=stop_ids, pad_id=pad_id
    )

    mismatches = []
    for i in range(len(sample_prompts)):
        if single_results[i] != batch_results[i]:
            mismatches.append(i)

    if mismatches:
        for i in mismatches:
            print(f"  [FAIL] 第 {i} 条不一致:")
            print(f"    逐条: {single_results[i][:200]!r}")
            print(f"    批量: {batch_results[i][:200]!r}")
        raise AssertionError(
            f"generate_batch 等价性自检失败: {len(mismatches)}/{len(sample_prompts)} 条不一致"
        )

    print(f"  [PASS] {len(sample_prompts)} 条全部一致 [OK]")

def _strip_stop(s: str) -> str:
    """去掉停止符 '#'（生成到 max_len 停止时没有）；与 bucket_report 口径一致。"""
    s = s.rstrip()
    return s[:-1] if s.endswith('#') else s


def load_test_dataset(test_data_path):
    """读取 JSONL 测试集，返回 (prompts, expected, alts, total)。

    每行要求含 'Q'（prompt）与 'A'（期望答案）字段；可选含 'alt'
    （该 Q 的全部合法后序解，用 '|' 分隔）——用于一题多解的结构等价判定。
    alts[i] 为 set[str]；样本缺 'alt' 时为 None（退回严格串等）。

    Returns:
        prompts (list[str]): 各样本 prompt
        expected (list[str]): 各样本期望答案
        alts (list[set[str] | None]): 各样本合法解集合（去停止符的规范形）
        total (int): 样本总数
    """
    raw_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line.strip()))
    prompts = [item['Q'] for item in raw_data]
    expected = [item['A'] for item in raw_data]
    alts = []
    for item in raw_data:
        alt_raw = item.get('alt')
        if alt_raw:
            alts.append({_strip_stop(a.strip()) for a in alt_raw.split('|') if a.strip()})
        else:
            alts.append(None)
    return prompts, expected, alts, len(raw_data)


def _extract_ans(text: str) -> str | None:
    """提取 ANS 后的答案本体，忽略思考过程；无 ANS 返回 None。

    A 形如 "PRE + 0 + 0 0\\nANS 0#"，只取 ANS 与 #（或换行/串尾）之间的内容。
    """
    idx = text.find('ANS')
    if idx < 0:
        return None
    body = text[idx + len('ANS'):].lstrip()
    for term in ('#', '\n'):
        p = body.find(term)
        if p >= 0:
            body = body[:p]
    return body.strip()


def _extract_think(text: str) -> str | None:
    """提取思考过程部分（ANS 之前的内容）；无 ANS 返回 None。

    用于 acc_think：只比思考过程（忽略答案本体）。
    """
    idx = text.find('ANS')
    if idx < 0:
        return None
    return text[:idx].strip()


@torch.no_grad()
def compute_accuracy(model, device, vocab_data, eval_batch_size, pad_id,
                     prompts=None, expected=None, test_data_path=None,
                     alts=None, max_seq_len=None, stop_token=None):
    """分块批量推理，统计准确率并返回完整预测列表。

    tokenizer 构建、stop_token 解析（统一走词表 metadata，缺失报错）、
    分块 generate_batch、计数都在此完成，训练验证与 trial 末评测共用。

    同时计算三种口径的准确率：
      - acc（整串对比；若样本带 alt，改为结构等价：去停止符后的 pred ∈ alt 集合，否则 pred == exp）
      - acc_ans（只比 ANS 答案本体，始终严格）
      - acc_think（只比思考过程，ANS 之前部分，始终严格）

    Args:
        model: GPT 模型
        device: torch.device
        vocab_data: 词表数据（需含 metadata.stop_token）
        eval_batch_size: 批量推理 batch size
        pad_id: 填充 token id
        prompts: 输入文本列表（与 test_data_path 二选一）
        expected: 期望输出列表（与 test_data_path 二选一）
        test_data_path: 测试数据文件路径，提供时自动加载 prompts/expected
        data_max_len: 模型最大序列长度（None=从 vocab_data.max_seq_len 读取）
        stop_token: 停止 token（None=从词表 metadata 严格读取，缺失报错）

    Returns:
        (accuracy, acc_ans, acc_think, correct, correct_ans, correct_think, total, predictions)
        predictions: 与 prompts 等长的完整预测字符串列表
    """
    if test_data_path is not None:
        prompts, expected, alts, _ = load_test_dataset(test_data_path)
    if max_seq_len is None:
        max_seq_len = vocab_data.max_seq_len
    if stop_token is None:
        stop_token = vocab_data.stop_token

    model.eval()
    stoi = vocab_data.stoi
    itos = vocab_data.itos
    tokenizer = vocab_data.tokenizer

    stop_ids = {stoi[stop_token]}

    total = len(prompts)
    predictions = []
    correct = 0
    correct_ans = 0
    correct_think = 0
    for chunk_start in range(0, total, eval_batch_size):
        chunk_end = min(chunk_start + eval_batch_size, total)
        chunk_prompts = prompts[chunk_start:chunk_end]
        chunk_expected = expected[chunk_start:chunk_end]
        chunk_alts = alts[chunk_start:chunk_end] if alts is not None else [None] * len(chunk_expected)
        preds = generate_batch(
            model, device, stoi, itos, tokenizer,
            chunk_prompts, max_seq_len, stop_ids, pad_id
        )
        predictions.extend(preds)
        for pred, exp, alt in zip(preds, chunk_expected, chunk_alts):
            # acc：结构等价优先（alt 命中即正确），无 alt 退回严格串等
            if alt is not None:
                if _strip_stop(pred) in alt:
                    correct += 1
            elif pred == exp:
                correct += 1
            pa, ea = _extract_ans(pred), _extract_ans(exp)
            if pa is not None and pa == ea:
                correct_ans += 1
            pt, et = _extract_think(pred), _extract_think(exp)
            if pt is not None and pt == et:
                correct_think += 1

    accuracy = correct / total if total > 0 else 0.0
    acc_ans = correct_ans / total if total > 0 else 0.0
    acc_think = correct_think / total if total > 0 else 0.0
    return accuracy, acc_ans, acc_think, correct, correct_ans, correct_think, total, predictions


