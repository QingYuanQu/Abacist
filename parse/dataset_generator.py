#!/usr/bin/env python3
"""
LLM Attention Evaluation Dataset Generator
==========================================
Generates datasets A, B, C, D for arithmetic expression parsing and evaluation.

Stage 1 (A): Letters, fully parenthesized infix. Test parse ability (Q->pre,post,stack).
Stage 2 (B): Letters, redundant parens removed (rules #1-#3). Still test parse.
Stage 3 (C): Real numbers, more parens removed (rules #1-#5). Group by Q, test ANS.
Stage 4 (D): Same as C with explicit Zipf-1 length distribution (T(n) = T(6)*6/n for n=6..20).

Tree representation:
  Leaf:     ['L', char]
  Internal: ['N', left, op, right]
"""

import json
import random
import itertools
import os
from collections import defaultdict
from math import comb, gcd

# ============================ Constants ============================
# ÷ 暂时禁用（2026-08-30）：叶值 1..9 下 ÷ 密集树不可行率高，且值生成在嵌套 ÷
# 上最坏仍需 24^深度 次重试。div_ok/count_div/评估器里的 ÷ 分支全部保留，
# 重新启用只需把 USE_DIV 改回 True（并先修复嵌套 ÷ 的快速不可行判定）。
# 叶子操作数上限：默认 9（一位数）。改成 99 可生成两位数操作数数据集。
LEAF_MAX = 9

# × 节点值上限（防结果超出算盘档数）。默认 9999；多位数数据集可提到 999999
# （6 位，仍远小于 13 档）。注意：LEAF_MAX=99 时若保持 9999，嵌套 × 大量
# 不可行 → make_dataset_c 约束求解重试爆炸（实测 5h+ 不出结果）。
MUL_CAP = 9999

# ÷ 除数子树值上限。默认 10^9（原始行为）；两位数数据集建议 99——
# 上限过大时 mm=m·rv 取模约束在叶子几乎无解，值生成重试爆炸（实测卡组数分钟）。
DIV_RV_CAP = 10 ** 9

USE_DIV = False
OPS = ['+', '-', '×', '÷'] if USE_DIV else ['+', '-', '×']
PREC = {'+': 1, '-': 1, '×': 2, '÷': 2}   # ÷ 保留在优先级表：解析/去括号/评估代码兼容
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
# 输出目录：相对项目根（__file__ 在 parse/ 下，上一级即项目根），避免硬编码旧路径 d:\TinyAbacus
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'parse', 'dataset')
SEED = 42

# ============================ Tree Utilities ============================

def is_leaf(t):
    return t[0] == 'L'

def num_leaves(t):
    if is_leaf(t):
        return 1
    return num_leaves(t[1]) + num_leaves(t[3])

def tree_key(t):
    """Hashable representation for deduplication."""
    if is_leaf(t):
        return ('L', t[1])
    return ('N', tree_key(t[1]), t[2], tree_key(t[3]))


def ops_of_tree(t):
    """树的中序运算符序列（如 '+-×'），叶子为 ''。

    >>> ops_of_tree(['N', ['L', 'a'], '+', ['N', ['L', 'b'], '-', ['L', 'c']]])
    '+-'
    """
    if is_leaf(t):
        return ''
    return ops_of_tree(t[1]) + t[2] + ops_of_tree(t[3])

# ============================ Catalan & Tree Generation ============================

def num_trees(n):
    """Number of full binary trees with n leaves = C(n-1)."""
    if n <= 1:
        return 1
    return comb(2 * (n - 1), n - 1) // n

def enumerate_trees(n):
    """Generate all full binary tree structures with n leaves."""
    if n == 1:
        yield ['L', None]
        return
    for i in range(1, n):
        for left in enumerate_trees(i):
            for right in enumerate_trees(n - i):
                yield ['N', left, None, right]

def sample_tree(n, rng):
    """Sample a random binary tree with n leaves, uniformly from all C(n-1) structures."""
    if n == 1:
        return ['L', None]
    splits = list(range(1, n))
    probs = [num_trees(i) * num_trees(n - i) for i in splits]
    total = sum(probs)
    r = rng.uniform(0, total)
    cum = 0
    for i, p in zip(splits, probs):
        cum += p
        if r <= cum:
            return ['N', sample_tree(i, rng), None, sample_tree(n - i, rng)]
    i = splits[-1]
    return ['N', sample_tree(i, rng), None, sample_tree(n - i, rng)]

def make_balanced_tree(n):
    """Create a fully balanced binary tree with n leaves."""
    if n == 1:
        return ['L', None]
    h = n // 2
    return ['N', make_balanced_tree(h), None, make_balanced_tree(n - h)]

# ============================ Assignment ============================

def assign_leaves(tree):
    """Assign letters a, b, c, ... to leaves in left-to-right order."""
    ctr = [0]
    def _go(t):
        if is_leaf(t):
            c = LETTERS[ctr[0]]
            ctr[0] += 1
            return ['L', c]
        return ['N', _go(t[1]), t[2], _go(t[3])]
    return _go(tree)

def assign_ops(tree, ops):
    """Assign operators to internal nodes (pre-order)."""
    ctr = [0]
    def _go(t):
        if is_leaf(t):
            return t
        left = _go(t[1])
        op = ops[ctr[0]]
        ctr[0] += 1
        right = _go(t[3])
        return ['N', left, op, right]
    return _go(tree)

def replace_leaves(tree, values):
    """Replace letter leaves with numeric string leaves."""
    if is_leaf(tree):
        return ['L', str(values[tree[1]])]
    return ['N', replace_leaves(tree[1], values), tree[2],
            replace_leaves(tree[3], values)]

# ============================ Colless Index ============================

def colless(t):
    """Compute Colless imbalance index."""
    if is_leaf(t):
        return 0
    return (abs(num_leaves(t[1]) - num_leaves(t[3]))
            + colless(t[1]) + colless(t[3]))

def max_colless(n):
    """Maximum Colless index for n leaves (caterpillar tree)."""
    if n <= 2:
        return 0
    return (n - 1) * (n - 2) // 2

def colless_bucket(ic, n):
    """Bucket 0-9 based on normalized Colless index."""
    mic = max_colless(n)
    if mic == 0:
        return 0
    return min(int(ic / mic * 10), 9)

# ============================ Expression Generation ============================

def infix_full(t):
    """Fully parenthesized infix."""
    if is_leaf(t):
        return t[1]
    return f"({infix_full(t[1])}{t[2]}{infix_full(t[3])})"

def infix_no_outer(t):
    """Infix without the outermost parentheses.
    去掉最外层的括号。"""
    s = infix_full(t)
    if len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        return s[1:-1]
    return s

def prefix(t):
    if is_leaf(t):
        return t[1]
    return f"{t[2]} {prefix(t[1])} {prefix(t[3])}"

def postfix(t):
    if is_leaf(t):
        return t[1]
    return f"{postfix(t[1])} {postfix(t[3])} {t[2]}"

def postfix_tokens(t):
    if is_leaf(t):
        return [t[1]]
    return postfix_tokens(t[1]) + postfix_tokens(t[3]) + [t[2]]

def stack_eval_sym(t):
    """Stack evaluation with symbolic (subexpression) results.
    具有符号（子表达式）结果的堆栈评估。"""
    tokens = postfix_tokens(t)
    stack = []
    steps = []
    for tok in tokens:
        if tok not in PREC:
            stack.append(tok)
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(f"({a}{tok}{b})")
        steps.append(f"{tok}→[{','.join(stack)}]")
    return ' '.join(steps)

def stack_eval_num(t):
    """Stack evaluation with numeric results."""
    tokens = postfix_tokens(t)
    stack = []
    steps = []
    for tok in tokens:
        if tok not in PREC:
            stack.append(tok)
        else:
            b = int(stack.pop())
            a = int(stack.pop())
            if tok == '+':
                r = a + b
            elif tok == '-':
                r = a - b
            elif tok == '×':
                r = a * b
            elif tok == '÷':
                r = a // b
            stack.append(str(r))
        steps.append(f"{tok}→[{','.join(stack)}]")
    return ' '.join(steps)

# ============================ Parenthesis Removal ============================
#
# Rules (S=child root op, P=parent op, L/R=child is left/right of parent):
#   #1  L, prec(S) > prec(P)              -> remove (safe)
#   #2  R, prec(S) > prec(P)              -> remove (safe)
#   #3  L, prec(S)==prec(P), S in {-,÷}   -> remove (safe, left-assoc)
#   #4  L, prec(S)==prec(P), S in {+,×}   -> remove (ambiguous)
#   #5  R, prec(S)==prec(P), P in {+,×}   -> remove (ambiguous)

def can_remove(s_op, p_op, is_left, ambiguous=False):
    """Check if parentheses around subexpression with root op s_op,
    child of parent with op p_op, can be removed."""
    if p_op is None:
        return True  # Root node
    sp, pp = PREC[s_op], PREC[p_op]
    if sp > pp:
        return True  # Rules #1, #2
    if sp == pp:
        if is_left:
            if s_op in ('-', '÷'):
                return True  # Rule #3
            if s_op in ('+', '×') and ambiguous:
                return True  # Rule #4
        else:
            if p_op in ('+', '×') and ambiguous:
                return True  # Rule #5
    return False

def infix_reduced(t, p_op=None, is_left=None, ambiguous=False):
    """Generate infix with parentheses removed according to rules.
    根据规则生成去除了括号的中缀表达式。"""
    if is_leaf(t):
        return t[1]
    op = t[2]
    l = infix_reduced(t[1], op, True, ambiguous)
    r = infix_reduced(t[3], op, False, ambiguous)
    inner = f"{l}{op}{r}"
    if p_op is None:
        return inner
    if can_remove(op, p_op, is_left, ambiguous):
        return inner
    return f"({inner})"


def count_prec_switch(q):
    """Q 中相邻运算符优先级不同的次数（去括号中缀难度度量）。

    提取 q 的运算符序列（跳过数字/空格/括号），统计相邻优先级不同的相邻对。
    例：'2+3×5'（+,×）→ 1；'2+3+5'（+,+）→ 0。
    与 experiment/material_adapter._count_prec_switch 同源（均基于 PREC）。"""
    ops = [c for c in q if c in PREC]
    return sum(1 for a, b in zip(ops, ops[1:]) if PREC[a] != PREC[b])


# ============================ Evaluation ============================

def evaluate(t, vals):
    """Evaluate tree with letter->value mapping."""
    if is_leaf(t):
        return vals[t[1]]
    a = evaluate(t[1], vals)
    b = evaluate(t[3], vals)
    op = t[2]
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '×':
        return a * b
    if op == '÷':
        return a // b

def evaluate_num(t):
    """Evaluate numeric tree (leaves are string numbers)."""
    if is_leaf(t):
        return int(t[1])
    a = evaluate_num(t[1])
    b = evaluate_num(t[3])
    op = t[2]
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '×':
        return a * b
    if op == '÷':
        return a // b

# ============================ Value Generation ============================

def div_ok(node, lv, rv):
    """除法约束：整除、右值非 0；右子树为叶子时右值 ≥2（避免平凡 ÷1）。
    复合右子树不强求 ≥2，否则约束过严会大幅抬高失败率。"""
    if rv == 0 or lv % rv != 0:
        return False
    return rv >= 2 if is_leaf(node[3]) else True

def _solve_lincong(a, c, m):
    """解 a·x ≡ c (mod m)：返回 (m1, c1) 表示 x ≡ c1 (mod m1)；无解返回 None。"""
    if m == 1:
        return (1, 0)
    g = gcd(a, m)
    if c % g:
        return None
    m1 = m // g
    if m1 == 1:
        return (1, 0)
    inv = pow((a // g) % m1, -1, m1)          # gcd(a/g, m1)=1，可逆
    return (m1, (c // g) * inv % m1)

def _leaf_cands(lo, hi, m, c):
    """叶子候选：[max(lo,1), min(hi,LEAF_MAX)] 中满足 v ≡ c (mod m) 的值。"""
    lo2, hi2 = max(lo, 1), min(hi, LEAF_MAX)
    if lo2 > hi2:
        return []
    if m == 1:
        return list(range(lo2, hi2 + 1))
    return [v for v in range(lo2, hi2 + 1) if v % m == c]

def _gen_vals(node, rng, vals, lo, hi, m, c, budget):
    """构造式赋值：返回满足 lo<=v<=hi 且 v≡c (mod m) 的节点值，叶子写入 vals；
    失败返回 None。注意：失败沿路径上溯时每层会重试（≤24 次），嵌套最坏 24^深度，
    budget 阀门将其硬性封顶（单次下推 ≤2 万步）。"""
    if is_leaf(node):
        cands = _leaf_cands(lo, hi, m, c)
        if not cands:
            return None
        v = rng.choice(cands)
        vals[node[1]] = v
        return v
    op = node[2]
    for _ in range(24):
        budget[0] -= 1
        if budget[0] <= 0:
            return None
        if op == '+':
            lv = _gen_vals(node[1], rng, vals, 1, hi - 1, 1, 0, budget)
            if lv is None:
                return None
            rv = _gen_vals(node[3], rng, vals,
                           max(1, lo - lv), hi - lv, m, (c - lv) % m, budget)
            if rv is not None:
                return lv + rv
        elif op == '-':
            rv = _gen_vals(node[3], rng, vals, 1, hi - lo, 1, 0, budget)
            if rv is None:
                return None
            lv = _gen_vals(node[1], rng, vals,
                           lo + rv, hi + rv, m, (c + rv) % m, budget)
            if lv is not None:
                return lv - rv
        elif op == '×':
            heff = min(hi, MUL_CAP)
            if lo > heff:
                return None
            rv = _gen_vals(node[3], rng, vals, 1, heff, 1, 0, budget)
            if rv is None:
                return None
            llo, lhi = max(1, -(-lo // rv)), heff // rv
            if llo > lhi:
                continue
            sub = _solve_lincong(rv, c, m)
            if sub is None:
                continue
            lv = _gen_vals(node[1], rng, vals, llo, lhi, sub[0], sub[1], budget)
            if lv is not None:
                return lv * rv
        else:  # '÷'
            rlo = 2 if is_leaf(node[3]) else 1
            rv = _gen_vals(node[3], rng, vals, rlo, DIV_RV_CAP, 1, 0, budget)
            if rv is None:
                return None
            mm = m * rv
            lv = _gen_vals(node[1], rng, vals,
                           lo * rv, hi * rv, mm, (c * rv) % mm, budget)
            if lv is not None:
                return lv // rv
    return None


def gen_values(t, rng, max_retries=60):
    for _ in range(max_retries):
        vals = {}
        # 预算阀门：单次下推最多 2 万步，硬性封顶嵌套失败重试（24^深度）的最坏情况
        if _gen_vals(t, rng, vals, 1, 10 ** 9, 1, 0, [20000]) is not None:
            return vals

# ============================ Canonical Evaluation (Verification) ============================

def eval_canonical(q_str):
    """Evaluate Q string canonically (left-associative, standard precedence).
    Raises ValueError if any division is non-exact.
    规范地评估Q字符串（左结合，标准优先级）。
    如果任何除法不是精确除法，则抛出ValueError异常。
    """
    tokens = [c for c in q_str if c.strip()]
    output = []
    op_stack = []
    for tok in tokens:
        if tok == '(':
            op_stack.append(tok)
        elif tok == ')':
            while op_stack[-1] != '(':
                output.append(op_stack.pop())
            op_stack.pop()
        elif tok in PREC:
            while (op_stack and op_stack[-1] in PREC
                   and PREC[op_stack[-1]] >= PREC[tok]):
                output.append(op_stack.pop())
            op_stack.append(tok)
        else:
            output.append(tok)
    while op_stack:
        output.append(op_stack.pop())
    stack = []
    for tok in output:
        if tok not in PREC:
            stack.append(int(tok))
        else:
            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            elif tok == '×':
                stack.append(a * b)
            elif tok == '÷':
                if b == 0 or a % b != 0:
                    raise ValueError("non-exact division")
                stack.append(a // b)
    return stack[0]

# ============================ Base Data Generation ============================

def generate_base(rng):
    """Generate all (tree, n, ic, bucket) samples with Zipf-1 length distribution.
    按 Zipf-1 长度分布生成所有（树、n、ic、桶）样本：T(n) = T(6)*6/n。"""
    samples = []
    # n=5 全枚举数 = 树结构 × 算符指派（随 OPS 自适应：4 算符 3584 / 3 算符 1134）
    base_count = num_trees(5) * (len(OPS) ** 4)

    for n in range(2, 21):
        if n <= 5:
            # Full enumeration
            for ts in enumerate_trees(n):
                for ops in itertools.product(OPS, repeat=n - 1):
                    tree = assign_ops(assign_leaves(ts), list(ops))
                    ic = colless(tree)
                    bk = colless_bucket(ic, n)
                    samples.append({'tree': tree, 'n': n, 'ic': ic, 'bk': bk})
            cnt = sum(1 for s in samples if s['n'] == n)
            print(f"  n={n}: {cnt} samples (full enum, {num_trees(n)} trees × {len(OPS)**(n-1)} ops)")
        else:
            # Zipf-1 length distribution: T(n) = T(6) * 6 / n
            # 一条 n 叶样本的 pre/post/stack 内嵌其 n-1 个子式的完整解析监督，
            # 故 T(n)*n = const 使每档「子式练习量」与 token 预算均相等；
            # n=6..20 各档均 >= 1075 条，统计需求自动满足，FLOOR 仅为
            # 将来扩大 n 范围时兜底。
            FLOOR = 500
            target = max(FLOOR, round(base_count * 6 / n))
            seen = set()
            count = 0

            # 每个 n 都强制注入近平衡树：桶 0 在 Catalan 均匀采样下指数稀有，
            # 需保证每个长度档的平衡桶测试集 >= 5 棵（50 棵 × 8:2 ≈ 10 棵）。
            # ic/bk 按实际结构计算：非 2 的幂的 n 上贪心平衡树的 Ic > 0。
            FORCE_BALANCED = 50
            bt = assign_leaves(make_balanced_tree(n))
            bt_ic = colless(bt)
            bt_bk = colless_bucket(bt_ic, n)
            forced = 0
            fattempts = 0
            while forced < min(FORCE_BALANCED, target) and fattempts < FORCE_BALANCED * 20:
                ops = [rng.choice(OPS) for _ in range(n - 1)]
                tree = assign_ops(bt, ops)
                fattempts += 1
                key = tree_key(tree)
                if key not in seen:
                    seen.add(key)
                    samples.append({'tree': tree, 'n': n, 'ic': bt_ic, 'bk': bt_bk})
                    forced += 1
            count += forced

            attempts = 0
            max_attempts = target * 50
            while count < target and attempts < max_attempts:
                ts = sample_tree(n, rng)
                ops = [rng.choice(OPS) for _ in range(n - 1)]
                tree = assign_ops(assign_leaves(ts), ops)
                key = tree_key(tree)
                if key not in seen:
                    seen.add(key)
                    ic = colless(tree)
                    bk = colless_bucket(ic, n)
                    samples.append({'tree': tree, 'n': n, 'ic': ic, 'bk': bk})
                    count += 1
                attempts += 1
            print(f"  n={n}: {count} samples (target={target}, sampled from {num_trees(n)} trees)")
    return samples

# ============================ Splitting ============================

def split_by_bucket(samples, rng):
    """Split 8:2 within each Colless bucket."""
    by_bk = defaultdict(list)
    for s in samples:
        by_bk[s['bk']].append(s)
    for bk in by_bk:
        bk_s = by_bk[bk]
        rng.shuffle(bk_s)
        si = int(len(bk_s) * 0.8)
        for i, s in enumerate(bk_s):
            s['sp'] = 'train' if i < si else 'test'

def split_by_q(samples, rng):
    """Split 8:2 by Q group (same Q -> same split)."""
    by_q = defaultdict(list)
    for s in samples:
        by_q[s['Q']].append(s)
    q_keys = list(by_q.keys())
    rng.shuffle(q_keys)
    total = len(samples)
    target_train = int(total * 0.8)
    train_count = 0
    for qk in q_keys:
        group = by_q[qk]
        if train_count < target_train:
            for s in group:
                s['sp'] = 'train'
            train_count += len(group)
        else:
            for s in group:
                s['sp'] = 'test'

# ============================ Dataset Generation ============================

def make_dataset_a(samples):
    """Dataset A: letters, fully parenthesized infix."""
    result = []
    for s in samples:
        t = s['tree']
        q = infix_no_outer(t)
        result.append({
            'n': s['n'],
            'Q': q,
            'pre': prefix(t),
            'post': postfix(t),
            'stack': stack_eval_sym(t),
            'ANS': q,  # Symbolic ANS = Q
            'Ic': s['ic'],
            'bk': s['bk'],
            'sp': s['sp']
        })
    return result

def make_dataset_b(samples):
    """Dataset B: letters, rules #1-#3 (safe paren removal)."""
    result = []
    for s in samples:
        t = s['tree']
        q = infix_reduced(t, ambiguous=False)
        ans = infix_no_outer(t)  # Canonical form
        result.append({
            'n': s['n'],
            'Q': q,
            'pre': prefix(t),
            'post': postfix(t),
            'stack': stack_eval_sym(t),
            'ANS': ans,
            'Ic': s['ic'],
            'bk': s['bk'],
            'sp': s['sp']
        })
    return result

# ============================ Value Checks ============================

def check_tree(t, vals):
    """校验树在给定叶子值下全部中间结果合法（单遍后序 O(n)，替代旧 O(n²) 版）：
    '-': 左>右（中间结果恒正）；'×': 乘积≤MUL_CAP；'÷': 见 div_ok。"""
    def go(node):
        if is_leaf(node):
            return vals[node[1]]
        a, b = go(node[1]), go(node[3])
        if a is None or b is None:
            return None
        op = node[2]
        if op == '+':
            return a + b
        if op == '-':
            return a - b if a > b else None
        if op == '×':
            return a * b if a * b <= MUL_CAP else None
        return a // b if div_ok(node, a, b) else None          # '÷'
    return go(t) is not None

def count_div(t):
    """树中 ÷ 节点个数，用于挑选组内约束最严的主树。"""
    if is_leaf(t):
        return 0
    return count_div(t[1]) + count_div(t[3]) + (1 if t[2] == '÷' else 0)

# ============================ Dataset Generation ============================

def make_dataset_c(samples, rng):
    """Dataset C: numbers, rules #1-#5, multi-solution groups keyed by flat Q.

    关键改动：填数前先按「字母压平串」分组（中缀压平保持叶子从左到右顺序），
    同组树共用一套数字 -> Q 相同、ANS 由结合律恒等 -> split_by_q 真正生效。
    生成策略：先用回溯生成器喂组内约束最严的主树，再用同一套数字校验组内其余树。
    """
    # 1) 按字母压平串分组（含算符，如 "a+b-c"）
    groups = defaultdict(list)
    for s in samples:
        gq = infix_reduced(s['tree'], ambiguous=True)
        groups[(s['n'], gq)].append(s)

    result = []
    skipped = 0      # 完全失败的样本数
    dropped = 0      # 降级组被丢弃的姊妹树数
    degraded = 0     # 降级组数
    multi = 0        # 多解组数
    total = len(groups)

    for gi, ((n, gq), grp) in enumerate(groups.items()):
        if gi % 1000 == 0:
            print(f"    Processing group {gi}/{total}...")
        trees = [s['tree'] for s in grp]
        if len(trees) > 1:
            multi += 1

        # 2) 主树 = ÷ 最多的树（约束最严）；USE_DIV=False 时 count_div 恒 0，等效取第一棵
        mi = max(range(len(trees)), key=lambda i: count_div(trees[i]))
        main = trees[mi]

        # 3) 整组共用一套数字，组内每棵树联合校验
        emitted = None
        for _ in range(30):
            vals = gen_values(main, rng)          # 构造式生成，失败≈该树不可行
            if vals is None:
                break
            if not all(check_tree(t, vals) for t in trees):
                continue
            cand = []
            ok = True
            for t in trees:
                nt = replace_leaves(t, vals)
                q = infix_reduced(nt, ambiguous=True)
                ans = evaluate_num(nt)
                try:
                    if eval_canonical(q) != ans:
                        ok = False
                        break
                except (ValueError, IndexError):
                    ok = False
                    break
                cand.append((nt, q, ans))
            # 防御：组内 Q/ANS 必须恒等（理论上由结合律保证）
            if ok and len({(q, a) for _, q, a in cand}) == 1:
                emitted = cand
                break

        # 4) 整组失败 -> 降级：只保留主树（gen_values 已保证其合法，不算 skip）
        if emitted is None and len(trees) > 1:
            degraded += 1
            dropped += len(trees) - 1
            trees = [main]
            grp = [grp[mi]]

        if emitted is None:
            vals = gen_values(trees[0], rng)
            if vals is not None and check_tree(trees[0], vals):
                nt = replace_leaves(trees[0], vals)
                q = infix_reduced(nt, ambiguous=True)
                ans = evaluate_num(nt)
                try:
                    if eval_canonical(q) == ans:
                        emitted = [(nt, q, ans)]
                except (ValueError, IndexError):
                    pass
        if emitted is None:
            skipped += len(grp)
            continue

        # 5) 组内每棵树各出一条记录：同 Q 同 ANS，不同 pre/post/stack
        for s, (nt, q, ans) in zip(grp, emitted):
            result.append({
                'n': n,
                'ops': ops_of_tree(s['tree']),
                'gid': gi,
                'tree': infix_full(s['tree']),
                'Q': q,
                'pre': prefix(nt),
                'post': postfix(nt),
                'stack': stack_eval_num(nt),
                'ANS': ans,
                'prec_switch': count_prec_switch(q),
                'ans_digits': len(str(abs(ans))),
                'Ic': s['ic'],
                'bk': s['bk'],
                'sp': ''
            })

    print(f"    Groups: {total} | multi-solution: {multi} | degraded: {degraded} "
          f"(dropped {dropped}) | skipped: {skipped}")
    split_by_q(result, rng)
    return result

# ============================ Statistics ============================

def compute_stats(samples, ds_a, ds_b, ds_c):
    stats = {
        'total_base': len(samples),
        'dataset_A': len(ds_a),
        'dataset_B': len(ds_b),
        'dataset_C': len(ds_c),
        'by_n': {},
        'by_n_C': {},
        'by_bucket': {},
        'split_A': {},
        'split_C': {},
        'bucket_0_test_A': 0,
        'unique_Q_C': 0,
    }
    for s in samples:
        n = str(s['n'])
        bk = str(s['bk'])
        stats['by_n'][n] = stats['by_n'].get(n, 0) + 1
        stats['by_bucket'][bk] = stats['by_bucket'].get(bk, 0) + 1
        sp = s['sp']
        stats['split_A'][sp] = stats['split_A'].get(sp, 0) + 1
        if bk == '0' and sp == 'test':
            stats['bucket_0_test_A'] += 1
    for d in ds_c:
        sp = d['sp']
        stats['split_C'][sp] = stats['split_C'].get(sp, 0) + 1
        nc = str(d['n'])
        stats['by_n_C'][nc] = stats['by_n_C'].get(nc, 0) + 1
    stats['unique_Q_C'] = len(set(d['Q'] for d in ds_c))
    stats['multi_solution_samples_C'] = len(ds_c) - stats['unique_Q_C']
    return stats

# ============================ Save ============================

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================ Main ============================

def main():
    import argparse
    global LEAF_MAX, MUL_CAP, OUT_DIR

    ap = argparse.ArgumentParser(description='生成 A/B/C/D 解析数据集（可选多位数变体）')
    ap.add_argument('--leaf-max', type=int, default=LEAF_MAX,
                    help='叶子操作数上限（默认 9=一位数；99=两位数操作数数据集）')
    ap.add_argument('--mul-cap', type=int, default=MUL_CAP,
                    help='× 节点值上限（默认 9999；leaf-max=99 时必须提到 999999，'
                         '否则嵌套 × 大量不可行 → 约束求解重试爆炸）')
    ap.add_argument('--out-dir', default=OUT_DIR,
                    help='输出目录（默认 parse/dataset；变体建议指向新目录）')
    args = ap.parse_args()
    LEAF_MAX, MUL_CAP, OUT_DIR = args.leaf_max, args.mul_cap, args.out_dir
    print(f"[config] LEAF_MAX={LEAF_MAX}  MUL_CAP={MUL_CAP}  OUT_DIR={OUT_DIR}")

    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 幂等：输出文件已齐全则跳过生成（避免重复跑耗时的 base/C 生成）
    out_files = ['dataset_A.jsonl', 'dataset_B.jsonl', 'dataset_C.jsonl',
                 'dataset_D.jsonl', 'stats.json']
    if all(os.path.exists(os.path.join(OUT_DIR, f)) for f in out_files):
        print(f"[skip] 数据集已存在：{OUT_DIR}（A/B/C/D + stats.json 齐全），跳过生成。")
        print(f"  如需重新生成，请删除 {OUT_DIR} 下对应文件后重跑。")
        return

    print("=" * 60)
    print("LLM Attention Evaluation Dataset Generator")
    print("=" * 60)

    print("\n[1/5] Generating base data (trees + operators)...")
    samples = generate_base(rng)
    print(f"  Total base samples: {len(samples)}")

    # Split by Colless bucket (for A and B)
    split_by_bucket(samples, rng)

    bk0_test = sum(1 for s in samples if s['bk'] == 0 and s['sp'] == 'test')
    bk0_total = sum(1 for s in samples if s['bk'] == 0)
    print(f"  Bucket 0 (balanced): {bk0_total} total, {bk0_test} test samples")

    print("\n[2/5] Dataset A: letters, full parens...")
    ds_a = make_dataset_a(samples)
    save_jsonl(ds_a, f'{OUT_DIR}/dataset_A.jsonl')
    print(f"  Saved {len(ds_a)} samples -> dataset_A.jsonl")

    print("\n[3/5] Dataset B: letters, rules #1-#3...")
    ds_b = make_dataset_b(samples)
    save_jsonl(ds_b, f'{OUT_DIR}/dataset_B.jsonl')
    print(f"  Saved {len(ds_b)} samples -> dataset_B.jsonl")

    print("\n[4/5] Dataset C: numbers, rules #1-#5, Q-grouped...")
    rng2 = random.Random(SEED + 1)
    ds_c = make_dataset_c(samples, rng2)
    save_jsonl(ds_c, f'{OUT_DIR}/dataset_C.jsonl')
    print(f"  Saved {len(ds_c)} samples -> dataset_C.jsonl")

    print("\n[5/5] Dataset D: same as C (length distribution applied)...")
    save_jsonl(ds_c, f'{OUT_DIR}/dataset_D.jsonl')
    print(f"  Saved {len(ds_c)} samples -> dataset_D.jsonl")

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    stats = compute_stats(samples, ds_a, ds_b, ds_c)
    save_json(stats, f'{OUT_DIR}/stats.json')
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Done! Output files:")
    print(f"  {OUT_DIR}/dataset_A.jsonl")
    print(f"  {OUT_DIR}/dataset_B.jsonl")
    print(f"  {OUT_DIR}/dataset_C.jsonl")
    print(f"  {OUT_DIR}/dataset_D.jsonl")
    print(f"  {OUT_DIR}/stats.json")

if __name__ == '__main__':
    main()
