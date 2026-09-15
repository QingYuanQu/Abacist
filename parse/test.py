from fractions import Fraction
from math import comb
import random

# T(n)：n 个叶子节点的满二叉树结构数，即 C_{n-1}
def T(n):
    if n <= 0:
        return 0
    k = n - 1
    return comb(2 * k, k) // (k + 1)

class Node:
    __slots__ = ("op", "val", "left", "right")
    def __init__(self):
        self.op = None
        self.val = None
        self.left = None
        self.right = None

def is_leaf(node):
    return node.left is None and node.right is None

# 均匀采样树结构
def sample_tree(n):
    node = Node()
    if n == 1:
        return node

    weights = [T(i) * T(n - i) for i in range(1, n)]
    i = random.choices(range(1, n), weights=weights)[0]

    node.left = sample_tree(i)
    node.right = sample_tree(n - i)
    return node

# 分配数字与运算符
def assign(tree, numbers, ops):
    if is_leaf(tree):
        tree.val = numbers.pop()
        return

    tree.op = random.choice(ops)
    assign(tree.left, numbers, ops)
    assign(tree.right, numbers, ops)

def to_infix(node, sym):
    if is_leaf(node):
        return str(node.val)
    return f"({to_infix(node.left, sym)} {sym[node.op]} {to_infix(node.right, sym)})"

def to_prefix(node, sym):
    if is_leaf(node):
        return str(node.val)
    return f"{sym[node.op]} {to_prefix(node.left, sym)} {to_prefix(node.right, sym)}"

def to_postfix(node, sym):
    if is_leaf(node):
        return str(node.val)
    return f"{to_postfix(node.left, sym)} {to_postfix(node.right, sym)} {sym[node.op]}"

def eval_tree(node):
    if is_leaf(node):
        return Fraction(node.val)

    l = eval_tree(node.left)
    r = eval_tree(node.right)

    if node.op == "+":
        return l + r
    if node.op == "-":
        return l - r
    if node.op == "*":
        return l * r
    if node.op == "/":
        if r == 0:
            raise ZeroDivisionError
        return l / r

    raise ValueError(f"unknown op: {node.op}")

def postfix_tokens(node):
    if is_leaf(node):
        return [str(node.val)]
    return postfix_tokens(node.left) + postfix_tokens(node.right) + [node.op]

def fmt_stack(stack):
    return "[" + ", ".join(str(x) for x in stack) + "]"

def stack_eval_trace(tokens, sym):
    stack = []
    trace = []

    for tok in tokens:
        if tok in ("+", "-", "*", "/"):
            b = stack.pop()
            a = stack.pop()

            if tok == "+":
                v = a + b
            elif tok == "-":
                v = a - b
            elif tok == "*":
                v = a * b
            else:
                if b == 0:
                    raise ZeroDivisionError
                v = a / b

            stack.append(v)
            trace.append(
                f"{sym[tok]}: pop {b}, pop {a} -> push {v}; stack={fmt_stack(stack)}"
            )
        else:
            v = Fraction(int(tok))
            stack.append(v)
            trace.append(f"push {v}; stack={fmt_stack(stack)}")

    return stack[0], trace

def generate_sample(n, number_range=(1, 20), ops=("+", "-", "*", "/")):
    sym = {
        "+": "+",
        "-": "-",
        "*": "×",
        "/": "÷",
    }

    numbers = random.sample(range(number_range[0], number_range[1] + 1), n)
    tree = sample_tree(n)
    assign(tree, numbers[:], list(ops))

    ans = eval_tree(tree)

    infix = to_infix(tree, sym)
    prefix = to_prefix(tree, sym)
    postfix = to_postfix(tree, sym)

    _, trace = stack_eval_trace(postfix_tokens(tree), sym)

    return {
        "Q": infix,
        "pre": prefix,
        "post": postfix,
        "stack": " | ".join(trace),
        "ANS": str(ans),
    }

def structure_key(node):
    if node.left is None and node.right is None:
        return "L"
    return f"({structure_key(node.left)},{structure_key(node.right)})"

# 示例
# random.seed(0)
# for n in [3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
#     print(generate_sample(n))

from collections import Counter
from math import comb
import random
from scipy import stats  # 需要 scipy，用于卡方检验


# 前面定义的 T(n) 和 Node, sample_tree 保持不变
# ...
import pytest
@pytest.mark.parametrize("n", [14])
def test_uniform(n, num_samples=100000, seed=42):
    random.seed(seed)

    counter = Counter()
    for _ in range(num_samples):
        tree = sample_tree(n)
        key = structure_key(tree)
        counter[key] += 1

    # 理论结构总数
    total_structures = T(n)
    expected_count = num_samples / total_structures

    print(f"n = {n}, 树结构数量 = {total_structures}")
    print(f"样本数 = {num_samples}, 每个结构期望出现 {expected_count:.1f} 次\n")

    # 按结构签名排序输出
    for key in sorted(counter):
        obs = counter[key]
        print(f"{key:20s} 实际 {obs:6d}  期望 {expected_count:6.1f}  偏差 {obs / expected_count - 1:+.3%}")

    # 卡方检验
    observed = [counter[k] for k in sorted(counter)]
    expected = [expected_count] * len(observed)

    chi2, p_value = stats.chisquare(observed, expected)
    print(f"\n卡方统计量 = {chi2:.3f}")
    print(f"p 值 = {p_value:.4f}")

    if p_value < 0.05:
        print("结论：拒绝均匀分布假设（存在显著偏差）")
    else:
        print("结论：不能拒绝均匀分布假设（采样看起来是均匀的）")