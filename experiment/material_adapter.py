# -*- coding: utf-8 -*-
"""experiment/material_adapter.py —— 实验材料适配器（阶段 3+4）。

职责：读 Trial.material 配置 → 调底层 parse（表达式生成）+ eval（abacus/digit，经 bridge）
→ 投影成 model_lm 的 material.jsonl（{category, Q, A}）。

这是 datagen/ 的「新内核替代」（阶段 4 起 datagen/ 已删除）：
  - 树枚举 / 全括号中缀 / 前后序  → parse.dataset_generator（替代 datagen/parse/traverse）
  - 求值（口诀 / 逐位）            → bridge.align（替代 datagen/eval/evaluate）
  - A 行式投影                     → project_a（替代 datagen/parse/format.format_a）
  - 珠态数据（数字↔珠态）          → abacus.bead_codec（原 datagen/gen_bead.py 迁入）

以新的为准：运算符 × ÷、空盘起算口诀、digit 逐位（已清进0）。
env（单步交互）类型已删除：未被任何 exp 使用，逐步交互语义由 VLA closed_loop 覆盖。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random

from evaluate.abacus.bead_codec import num_to_bead_text
from evaluate.bridge.align import align, make_default_composer
from evaluate.bridge.ir import build_instance, ExpressionInstance
from evaluate.digit import make_digit_fn
from parse.dataset_generator import (
    enumerate_trees, assign_leaves, assign_ops, replace_leaves,
    infix_full, prefix, postfix, evaluate_num, LETTERS,
)


# ==================== 投影层：IR → Q / A 行式文本 ====================

def _op_step_content(s, eval_, abacus_bead: bool = False) -> str:
    """单个 calc 步的操作步内容：后缀切片 / 口诀 / 逐位。"""
    if eval_ in ('none', None):
        return f"{s.a},{s.b},{s.token}"
    if eval_ == 'abacus':
        oral = ';'.join(a.oral for a in s.abacus.actions)
        if abacus_bead and s.abacus.actions:
            init = num_to_bead_text(0)
            final = num_to_bead_text(s.abacus.actions[-1].value)
            return f"{init}~{oral}~{final}"
        return oral
    if eval_ == 'digit':
        return s.digit or ''
    return ''


def _project_stack(inst: ExpressionInstance, eval_) -> str:
    """STACK 行（prepost_stack）：栈快照 + 操作步交织，'|' 分隔。"""
    parts = []
    for s in inst.steps:
        if s.kind == 'push':
            parts.append(','.join(str(x) for x in s.stack))
        else:  # calc
            parts.append(_op_step_content(s, eval_))
            parts.append(','.join(str(x) for x in s.stack))
    return '|'.join(parts)


def _project_eval_steps(inst: ExpressionInstance, eval_, abacus_bead: bool = False) -> str:
    """ABACUS/DIGIT 行（none + eval）：独立操作步，'|' 分隔。"""
    return '|'.join(_op_step_content(s, eval_, abacus_bead)
                    for s in inst.steps if s.kind == 'calc')


def project_q(inst: ExpressionInstance, input_fmt: str = 'infix') -> str:
    """投影 Q（中缀/前缀/后缀 + '='）。"""
    if input_fmt == 'prefix':
        return inst.prefix + '='
    if input_fmt == 'postfix':
        return inst.postfix + '='
    return inst.infix + '='  # infix（默认）


def project_a(inst: ExpressionInstance, parse: str, eval_: str,
              input_fmt: str = 'infix', abacus_bead: bool = False) -> str:
    """从 IR 投影 A（model_lm 的行式思考链）。

    对照 datagen/parse/format.format_a 的投影规则，逐字节兼容。
    """
    ans_str = str(inst.answer)
    if parse == 'fixed':
        n = inst.answer
        return f"0{n}" if 0 <= n < 10 else str(n)
    if parse == 'direct':
        return ans_str + '#'

    reps = []
    if input_fmt != 'infix':
        reps.append(('INFIX', inst.infix))
    if parse in ('pre', 'prepost', 'prepost_stack'):
        if input_fmt != 'prefix':
            reps.append(('PRE', inst.prefix))
    if parse in ('post', 'prepost', 'prepost_stack'):
        if input_fmt != 'postfix':
            reps.append(('POST', inst.postfix))
    if parse == 'prepost_stack':
        reps.append(('STACK', _project_stack(inst, eval_)))
    elif eval_ in ('abacus', 'digit'):
        label = 'ABACUS' if eval_ == 'abacus' else 'DIGIT'
        content = _project_eval_steps(inst, eval_, abacus_bead)
        if content:
            reps.append((label, content))

    think = '\n'.join(f'{label} {content}' for label, content in reps)
    if eval_ in ('none', 'digit', 'abacus', None):
        return f'{think}\nANS {ans_str}#'
    if eval_ == 'no_ans':
        return f'{think}#'
    return think


def project_structure_a(inst, parse: str) -> str:
    """纯结构转换的 A 投影：目标记法序列 + '#'（无思考链标记、无 ANS 求值）。

    用于 experiment 型 dataset 实验（前中后缀记法转换对比）——任务只学记法结构，
    不学算术求值，故 A 就是目标记法序列本身：
      parse='pre'  → 前序序列 '#'（如 "+ + + 6 1 3 9#"）
      parse='post' → 后序序列 '#'（如 "6 1 3 9 + + +#"）

    区别于 project_a：后者是 course 型 expr 用的富结构思考链（POST/PRE 标记 + ANS）。
    """
    seq = inst.prefix if parse == 'pre' else inst.postfix
    return f'{seq}#'


# ==================== 生成层：material 配置 → 表达式 → IR → material.jsonl ====================

def _iter_exprs(start, end, repeat, op_list, sample, rng):
    """生成 (tree_idx, op_combo, operands) 流。sample>0 随机采样（去重），否则全枚举。"""
    trees = list(enumerate_trees(repeat))
    op_combos = list(itertools.product(op_list, repeat=repeat - 1))
    n_trees = len(trees)
    n_ops = len(op_combos)
    n_operand_combos = (end - start) ** repeat
    total = n_trees * n_ops * n_operand_combos

    if sample is not None and sample > 0:
        seen = set()
        yielded = 0
        attempts = 0
        max_attempts = max(sample * 3, total)
        while yielded < sample and attempts < max_attempts and len(seen) < total:
            tree_idx = rng.randrange(n_trees)
            op_combo = op_combos[rng.randrange(n_ops)]
            operands = tuple(rng.randint(start, end - 1) for _ in range(repeat))
            key = (tree_idx, op_combo, operands)
            attempts += 1
            if key in seen:
                continue
            seen.add(key)
            yield tree_idx, op_combo, operands
            yielded += 1
    else:
        for tree_idx in range(n_trees):
            for op_combo in op_combos:
                for operands in itertools.product(range(start, end), repeat=repeat):
                    yield tree_idx, op_combo, operands


def _generate_expr(m, paths, seed) -> tuple[int, int]:
    """生成 expr 类型 trial 数据（流式写 train/test 双文件）。"""
    composer, abacus = make_default_composer()
    digit_fn = make_digit_fn()

    trees = list(enumerate_trees(m.repeat))
    op_list = list(m.ops)
    rng = random.Random(seed)

    train_f = open(paths.train_data, 'w', encoding='utf-8')
    test_f = open(paths.test_data, 'w', encoding='utf-8') if \
        (paths.test_data and paths.test_data != paths.train_data) else None

    train_count = test_count = filtered = 0
    try:
        for tree_idx, op_combo, operands in _iter_exprs(
                m.start, m.end, m.repeat, op_list, m.sample, rng):
            # 1. 组装数值树
            t = assign_ops(assign_leaves(trees[tree_idx]), list(op_combo))
            values = {LETTERS[i]: v for i, v in enumerate(operands)}
            nt = replace_leaves(t, values)

            # 2. 求值（数学真值）+ 允许负结果过滤
            ans = evaluate_num(nt)
            if not m.allow_negative and ans < 0:
                filtered += 1
                continue

            # 3. 组装 record + align（abacus/digit 注解）
            record = {
                'n': m.repeat, 'ops': ''.join(op_combo),
                'Q': infix_full(nt), 'pre': prefix(nt), 'post': postfix(nt),
                'ANS': ans,
            }
            try:
                steps, _ = align(record['post'].split(), composer, abacus,
                                 digit_fn=digit_fn, expected_ans=ans)
            except Exception:
                # abacus 中间态负数 / 档位越界（新内核无倒减法），整条过滤
                filtered += 1
                continue

            # 4. 投影 Q/A
            inst = build_instance(record, steps)
            q = project_q(inst, m.input_format)
            a = project_a(inst, m.parse, m.eval, m.input_format, m.abacus_bead)
            line = json.dumps({"category": ''.join(op_combo), "Q": q, "A": a},
                              ensure_ascii=False) + '\n'

            # 5. 确定性分流（MD5，跨进程一致）
            key = f"{operands},{op_combo},{tree_idx},{seed}"
            h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            if test_f is not None and (h / 0xFFFFFFFF) < m.split:
                test_f.write(line)
                test_count += 1
            else:
                train_f.write(line)
                train_count += 1
    finally:
        train_f.close()
        if test_f:
            test_f.close()

    print(f"[material_adapter] expr 生成完成：训练 {train_count} / 测试 {test_count} / 过滤 {filtered}")
    return train_count, test_count


def _sample_bead_numbers(bead_start, bead_end, samples_per_digit: int = 2, seed: int = 42):
    """基于算盘珠态数学结构的精简采样（原 datagen/gen_bead.py 迁入）。

    三层采样：位值核心（每位 0-9 至少一次）/ 组合多样性 / 边界覆盖 + 负数覆盖。
    """
    rng = random.Random(seed)
    if bead_end <= bead_start:
        return []

    max_abs = max(abs(bead_start), abs(bead_end - 1))
    n_digits = len(str(max_abs))
    has_negative = bead_start < 0

    result = set()

    # Layer 1：位值核心 — 每个十进制位上 0-9 至少一次
    for pos in range(n_digits):
        power = 10 ** pos
        for digit_val in range(10):
            for _ in range(samples_per_digit):
                fill = 0
                if pos < n_digits - 1:
                    max_fill = (bead_end - 1) // (power * 10)
                    if max_fill > 0:
                        fill = rng.randint(0, min(max_fill, 99)) * (power * 10)
                n = fill + digit_val * power
                if bead_start <= n < bead_end:
                    result.add(n)

    # Layer 2：组合多样性 — 相邻位随机组合
    for _ in range(max(50, n_digits * 30)):
        result.add(rng.randint(bead_start, bead_end - 1))

    # Layer 3：边界覆盖
    if bead_end - bead_start <= 100:
        for n in range(bead_start, bead_end):
            result.add(n)
    if bead_start <= 0 < bead_end:
        result.add(0)
    for k in range(n_digits):
        base = 10 ** k
        for offset in [-1, 0, 1]:
            n = base + offset
            if bead_start <= n < bead_end:
                result.add(n)
    if bead_start < bead_end:
        result.add(bead_start)
        result.add(bead_end - 1)
        for offset in range(-3, 0):
            n = bead_end - 1 + offset
            if bead_start <= n < bead_end:
                result.add(n)

    # 负数覆盖
    if has_negative:
        for k in range(n_digits):
            base = 10 ** k
            for offset in [-1, 0, 1]:
                n = -(base + offset)
                if bead_start <= n < bead_end:
                    result.add(n)
        for pos in range(n_digits):
            power = 10 ** pos
            for digit_val in range(1, 10):
                n = -(digit_val * power + rng.randint(0, power - 1))
                if bead_start <= n < bead_end:
                    result.add(n)
        for _ in range(30):
            result.add(rng.randint(bead_start, -1))

    return sorted(result)


def _generate_bead(start, end, out_path, repeat, ops, allow_negative,
                   max_samples, test_out_path, split, seed, base: int = 10) -> None:
    """生成珠态训练数据（数字↔珠态，用新内核 abacus.bead_codec）。"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if test_out_path:
        os.makedirs(os.path.dirname(test_out_path) or '.', exist_ok=True)

    bead_start, bead_end = start, end
    n_digits = len(str(max(abs(bead_start), abs(bead_end - 1))))
    print(f"[珠态数据] 操作数范围: [{start}, {end}), repeat={repeat}, ops='{ops}', 位数={n_digits}")

    numbers = _sample_bead_numbers(bead_start, bead_end, seed=seed)
    if max_samples is not None and len(numbers) > max_samples:
        step = max(1, len(numbers) // max_samples)
        numbers = numbers[::step][:max_samples]

    def _lines(n):
        bead = num_to_bead_text(n, base)
        # 与 expr 型 trial 的 project_a 对齐：答案统一用 "ANS <ans>#" 包裹，
        # 使 acc_mode=acc_ans 对所有 trial 一致生效（认算盘无思考链，ANS 前为空）。
        return (json.dumps({"category": "盘面", "Q": str(n), "A": f"ANS {bead}#"},
                           ensure_ascii=False) + '\n',
                json.dumps({"category": "珠态", "Q": bead, "A": f"ANS {n}#"},
                           ensure_ascii=False) + '\n')

    if test_out_path:
        train_count = test_count = 0
        with open(out_path, 'w', encoding='utf-8') as train_f, \
             open(test_out_path, 'w', encoding='utf-8') as test_f:
            for n in numbers:
                pos_line, neg_line = _lines(n)
                key = f"{n},{seed}"
                h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
                if (h / 0xFFFFFFFF) < split:
                    test_f.write(pos_line); test_f.write(neg_line); test_count += 2
                else:
                    train_f.write(pos_line); train_f.write(neg_line); train_count += 2
        print(f"[珠态数据] 已生成 {train_count + test_count} 条 → train={out_path}, test={test_out_path}")
    else:
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            for n in numbers:
                pos_line, neg_line = _lines(n)
                f.write(pos_line); f.write(neg_line); count += 2
        print(f"[珠态数据] 已生成 {count} 条 → {out_path}")


# dataset 投影用的运算符优先级（与 dataset_generator.PREC 同源，但本模块不 import 生成器）
_PREC = {'+': 1, '-': 1, '×': 2, '÷': 2}


def _count_prec_switch(q: str) -> int:
    """Q 中相邻运算符优先级不同的次数（去括号后中缀的难度度量）。

    提取 Q 的运算符序列（跳过数字/空格/括号），统计相邻优先级不同的相邻对。
    例：'2+3×5'（+,×）→ 1；'2+3+5'（+,+）→ 0；'2+3×5-1'（+,×,-）→ 2。
    与 bucket_report 的 PS_BANDS(0/1/2+) 对齐。
    """
    ops = [c for c in q if c in _PREC]
    return sum(1 for a, b in zip(ops, ops[1:]) if _PREC[a] != _PREC[b])


def _ans_digits(ans: int) -> int:
    """答案十进制位数（ANS 恒非负，abs 仅为防御）。"""
    return len(str(abs(ans)))


def _generate_dataset(m, paths, seed: int = 0) -> None:
    """从 source 投影 dataset 类型 trial 数据（按 config 划分，按 Q 分组防泄漏）。

    dataset 源（如 dataset_D.jsonl）是原始 IR（infix/prefix/postfix/answer/gid…，已固化 prec_switch/
    ans_digits 两个单记录难度元数据；alt 不在此固化，因它跨记录且按记法派生）。它不是直接
    可训练的 {Q, A}。这里 Q 用 project_q（输入记法 + '='），A 用 project_structure_a（目标
    记法 + '#'，纯结构转换、不求值）；并补齐 bucket_report 需要的字段。
    （数字集 dataset_D.jsonl 带 answer；字母集 dataset_C.jsonl 不带 answer，评测走结构性接受。）

    分流（P0b 修复）：不再信任源文件写死的 sp 字段（那由上游生成器按固定 0.8 写死、绕过
    experiment 配置），改由本实验的 split（Material.split，默认 0.2 = 20% 测试，与历史一致）
    + seed（experiment.data_seed，经 generate_trial 传入）按 Q 分组确定性划分——同一 Q 的
    所有记录（含多解姊妹树）整体进 train 或 test，杜绝多解泄漏；且与 expr/bead 型用同一 seed
    做确定性分流，复现性语义统一。改 dataset 的 split / data_seed 立即生效、无需重生成上游
    dataset_D。

      - alt         ：同 infix 的多解集合（'|' 分隔的目标记法串）。按 infix 聚合源里同一 infix 的
                      所有合法解（dataset_D 靠 gid 标识多解组，但按 infix 聚合更通用：同 infix
                      必对应同一解集合）。缺失时退化为空（bucket_report 退化为严格串等）。
      - prec_switch ：从 infix 现算（源若自带则优先，兼容旧/手工格式）。
      - ans_digits  ：从 answer 现算（源若自带则优先；字母集无 answer，置 0）。

    注意：alt 必须在此按 infix 聚合派生（跨记录 + 分记法，上游固化会冗余且耦合记法）；
    prec_switch/ans_digits 已由 dataset_generator 在生成阶段固化，下游优先透传
    （record.get(..., 现算)），仅 fallback 兼容旧格式/手工源。
    """
    proj_parse = m.parse if m.parse in ('pre', 'post') else 'post'
    # 第一遍：全读源，按 Q 聚合多解 alt（目标记法）+ 收集唯一 Q 顺序
    src_rows = []                              # (record, inst)
    alt_by_q: dict[str, set[str]] = {}
    q_keys: list[str] = []                      # 唯一 Q 首次出现顺序
    with open(m.source, 'r', encoding='utf-8') as src:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            inst = build_instance(record, [])
            src_rows.append((record, inst))
            q = inst.infix
            sol = project_structure_a(inst, proj_parse).rstrip('#')
            alt_by_q.setdefault(q, set()).add(sol)
            if q not in q_keys:
                q_keys.append(q)

    # 分流：每个 Q 组整体 train/test（防多解泄漏），由 split + seed 确定性决定
    rng = random.Random(seed)
    q_dest: dict[str, str] = {}
    for q in sorted(q_keys):                    # sorted 保证与源顺序无关、只看 Q 集合
        q_dest[q] = 'test' if rng.random() < m.split else 'train'

    train_f = open(paths.train_data, 'w', encoding='utf-8')
    test_f = open(paths.test_data, 'w', encoding='utf-8') if \
        (paths.test_data and paths.test_data != paths.train_data) else None

    train_count = test_count = 0
    try:
        for record, inst in src_rows:
            q = project_q(inst, m.input_format)
            a = project_structure_a(inst, proj_parse)

            # alt：优先用源自带；否则按 Q 聚合（真实 dataset_C 走此路）
            alt = record.get('alt') or '|'.join(sorted(alt_by_q.get(inst.infix, set())))
            struct = {
                'n': record.get('n'),
                'bk': record.get('bk', 0),
                'alt': alt,
                'prec_switch': record.get('prec_switch', _count_prec_switch(inst.infix)),
                # ans_digits 仅在源带 answer 时有意义（数字集）；字母集无 answer -> 置 0
                'ans_digits': record.get('ans_digits',
                                        _ans_digits(inst.answer) if inst.answer else 0),
            }
            out_line = json.dumps(
                {"category": record.get('ops', ''), "Q": q, "A": a, **struct},
                ensure_ascii=False) + '\n'

            dest = q_dest[inst.infix]
            if test_f is not None and dest == 'test':
                test_f.write(out_line)
                test_count += 1
            else:
                train_f.write(out_line)
                train_count += 1
    finally:
        train_f.close()
        if test_f:
            test_f.close()
    print(f"[material_adapter] dataset 投影完成：训练 {train_count} / 测试 {test_count}")


def generate_trial(cfg, seed) -> None:
    """根据单个 trial 的 material 与 paths 生成数据。cfg 为 Trial 对象。"""
    m = cfg.material
    paths = cfg.paths
    print("=" * 60)
    print(f"[数据生成] trial {cfg.id}: {cfg.name} (type={m.type})")
    print("=" * 60)

    if m.type == 'expr':
        _generate_expr(m, paths, seed)
    elif m.type == 'bead':
        _generate_bead(m.start, m.end, paths.train_data, m.repeat, m.ops,
                       m.allow_negative, m.sample, paths.test_data, m.split,
                       seed, base=m.abacus_base)
    elif m.type == 'dataset':
        _generate_dataset(m, paths, seed)
    else:
        raise ValueError(f"未知 material.type: {m.type!r}（env 类型已删除）")
