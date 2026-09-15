"""精确计算 n 个叶子节点时的表达式组合数量。

组合数 = 树结构数 C_{n-1}  ×  运算符分配 4^{n-1}  ×  数字排列 n!
（不含：去括号后的 Q 去重、除零剔除、数值等价合并）

同时用采样实测每条记录的字节数，估算落盘体积。
"""

from math import comb, factorial
import random

OPS = "+-*/"


def catalan(k: int) -> int:
    """第 k 个卡特兰数"""
    return comb(2 * k, k) // (k + 1)


def trees(n: int) -> int:
    """n 个叶子的满二叉树结构数 = C_{n-1}"""
    return catalan(n - 1) if n >= 1 else 0


def combos(n: int, n_ops: int = 4) -> int:
    """树结构 × 运算符分配"""
    return trees(n) * n_ops ** (n - 1)


def total(n: int, n_ops: int = 4) -> int:
    """再乘上数字的全排列"""
    return combos(n, n_ops) * factorial(n)


# ---------- 采样实测：全括号中缀表达式 + 答案 的记录长度 ----------

def random_infix(n: int, rng: random.Random) -> str:
    """随机生成一棵 n 叶子的表达式树，返回全括号中缀串与值（用 Fraction 精确求值）。"""
    from fractions import Fraction

    def build(k):
        if k == 1:
            return str(rng.randint(1, 9)), Fraction(rng.randint(1, 9))
        left_size = rng.randint(1, k - 1)
        ls, lv = build(left_size)
        rs, rv = build(k - left_size)
        op = rng.choice(OPS)
        if op == "+":
            v = lv + rv
        elif op == "-":
            v = lv - rv
        elif op == "*":
            v = lv * rv
        else:
            v = lv / rv if rv != 0 else Fraction(0)
        return f"({ls} {op} {rs})", v

    s, v = build(n)
    return s, v


def measure_record_bytes(n: int, samples: int = 20000, seed: int = 0) -> float:
    """实测平均记录字节数（'Q\\tANS\\n' 的 UTF-8 长度）"""
    rng = random.Random(seed)
    acc = 0
    for _ in range(samples):
        q, v = random_infix(n, rng)
        rec = f"{q}\t{v}\n"
        acc += len(rec.encode("utf-8"))
    return acc / samples


def human(num: float) -> str:
    for unit in ("", "K", "M", "G", "T", "P"):
        if abs(num) < 1000:
            return f"{num:.2f}{unit}" if unit else f"{num:.0f}"
        num /= 1000
    return f"{num:.2f}E"


def size_str(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if nbytes < 1024:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} EB"


def main(n_max: int = 12):
    print(f"{'n':>3} {'树结构 C(n-1)':>14} {'4^(n-1)':>12} "
          f"{'树×运算符':>16} {'×n! 总表达式':>20} {'平均B/条':>9} {'落盘体积':>12}")
    print("-" * 94)

    cum = 0
    cum_bytes = 0.0
    for n in range(2, n_max + 1):
        t = trees(n)
        p = 4 ** (n - 1)
        c = combos(n)
        tot = total(n)
        avg_b = measure_record_bytes(n)
        nb = tot * avg_b
        cum += tot
        cum_bytes += nb
        print(f"{n:>3} {t:>14,} {p:>12,} {c:>16,} {tot:>20,} "
              f"{avg_b:>9.1f} {size_str(nb):>12}")

    print("-" * 94)
    print(f"n=2..{n_max} 累计表达式数：{cum:,}（约 {human(cum)}）")
    print(f"n=2..{n_max} 累计落盘体积：{size_str(cum_bytes)}（未压缩，按 'Q\\tANS\\n' 计）")

    print("\n>>> 只枚举到 n<=6：")
    c6 = sum(total(n) for n in range(2, 7))
    print(f"    n=2..6 合计 {c6:,} 条（约 {human(c6)}）")
    print(f"    n=6 单条平均 {measure_record_bytes(6):.1f} B，"
          f"落盘约 {size_str(sum(total(n) * measure_record_bytes(n) for n in range(2, 7)))}")


if __name__ == "__main__":
    main(12)
