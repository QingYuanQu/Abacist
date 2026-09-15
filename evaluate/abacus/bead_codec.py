"""珠态编解码器 —— 数值 ↔ AbacusState ↔ 珠态文本。

旧文本格式（原 datagen/eval/abacus_model.py，2026-09-04 阶段 1 迁入新内核）：

    [L4+U1|L2+U1] = 97

  - 高位在左，档间 '|'，整体方括号；负数前缀 '-'；零 = [空]
  - 单档：'空' | 'L<n>' | 'U<n>' | 'L<n>+U<n>'（下珠在前）
  - L = 靠梁下珠颗数（每颗 ×1），U = 靠梁上珠颗数（每颗 ×5）
  - 紧凑形式默认裁剪前导空档，0 仅一档 [空]

消费方：datagen/gen_bead.py（课程珠态课）、model_vm/gen.py（看图读珠态）。
文本编码函数与旧实现逐字节兼容（doctest 与旧版一致）。
"""

from evaluate.abacus.domain import (
    Abacus,
    AbacusState,
    BeadPosition,
    BeadType,
    UnsupportedOperation,
    num_to_state,
)


# ==================== 纯文本编码（与旧 _num_to_bead / _bead_to_num 严格兼容） ====================

# base → **默认**上珠颗数（每颗面值 5）。纯文本编码层在没有显式形制时的回退表。
# 调用方若能拿到 AbacusSpec，应显式传 upper_count —— 那才是形制贯通的正路
# （内核状态桥 num_to_state 就是这么做的）。表本身不是硬编码的替身，只是默认值。
_UPPER_COUNT_BY_BASE = {10: 1, 16: 2}


def _default_upper_count(base: int) -> int:
    """base → 默认上珠颗数。二五珠 base=10 → 1；十六进制上二下五 → 2。

    表里没有的进制显式报错，避免静默按"上1下4"处理而产出越界珠态
    （旧实现写作 `2 if base == 16 else 1`，任何非 16 进制都悄悄按 1 处理）。
    """
    try:
        return _UPPER_COUNT_BY_BASE[base]
    except KeyError:
        raise UnsupportedOperation(
            f"珠态文本无 base={base} 的默认形制（已注册 {sorted(_UPPER_COUNT_BY_BASE)}）；"
            f"请显式传 upper_count，或走 num_to_state / state_to_bead_text") from None


def _digit_bead_state(d: int, base: int = 10,
                      upper_count: int | None = None) -> tuple[int, int]:
    """单个数位 → (上珠颗数, 下珠颗数)。上珠优先（每颗 =5），受物理颗数上限约束。

    upper_count 显式给出上珠颗数时优先（形制贯通）；省略时按 base 查默认形制表。

    base=10（上1下4）上珠最多 1 颗；base=16（上2下5）最多 2 颗：
      d=10,16 → (2,0)=U2；d=15,16 → (2,5)=L5+U2。
    """
    if d == 0:
        return 0, 0
    max_upper = _default_upper_count(base) if upper_count is None else upper_count
    upper = min(d // 5, max_upper)
    return upper, d - 5 * upper


def _digit_to_bead(d: int, base: int = 10, upper_count: int | None = None) -> str:
    """单个数位 → 珠态字符串。"""
    upper, lower = _digit_bead_state(d, base, upper_count)
    if upper == 0 and lower == 0:
        return "空"
    if upper == 0:
        return f"L{lower}"
    if lower == 0:
        return f"U{upper}"
    return f"L{lower}+U{upper}"


def _bead_to_digit(bead: str) -> int:
    """珠态字符串 → 单个数位。"""
    bead = bead.strip()
    if bead == "空":
        return 0
    total = 0
    for part in bead.split("+"):
        part = part.strip()
        if part.startswith("L"):
            total += int(part[1:])
        elif part.startswith("U"):
            total += 5 * int(part[1:])
        else:
            raise ValueError(f"无法解析珠态: {bead}")
    return total


def num_to_bead_text(n: int, base: int = 10,
                     upper_count: int | None = None) -> str:
    """整数 → 珠态括号表示（紧凑，高位在左，负数前缀 '-'）。

    upper_count 显式给出上珠颗数时优先（形制贯通，如传 `abacus.spec.upper_count`）；
    省略时按 base 查默认形制表（见 `_default_upper_count`）。

    >>> num_to_bead_text(0)
    '[空]'
    >>> num_to_bead_text(8)
    '[L3+U1]'
    >>> num_to_bead_text(97)
    '[L4+U1|L2+U1]'
    >>> num_to_bead_text(173)
    '[L1|L2+U1|L3]'
    >>> num_to_bead_text(105)
    '[L1|空|U1]'
    >>> num_to_bead_text(-12)
    '-[L1|L2]'
    >>> num_to_bead_text(255, 16)
    '[L5+U2|L5+U2]'
    """
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if n_abs == 0:
        return f"{sign}[空]"
    digits = []
    while n_abs:
        digits.append(n_abs % base)
        n_abs //= base
    digits.reverse()  # 高位在左
    return f"{sign}[{'|'.join(_digit_to_bead(d, base, upper_count) for d in digits)}]"


def bead_text_to_num(bracket: str, base: int = 10) -> int:
    """珠态括号表示 → 整数（num_to_bead_text 的逆）。

    >>> bead_text_to_num('[空]')
    0
    >>> bead_text_to_num('[L4+U1|L2+U1]')
    97
    >>> bead_text_to_num('-[L1|L2]')
    -12
    """
    s = bracket.strip()
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("-").strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"珠态文本须为 [...] 形式: {bracket}")
    inner = s[1:-1]
    if inner == "空":
        return 0
    parts = inner.split("|")
    total = 0
    for i, part in enumerate(parts):
        total += _bead_to_digit(part) * (base ** (len(parts) - 1 - i))
    return sign * total


# ==================== 内核状态桥（AbacusState ↔ 文本） ====================
# （数值 → AbacusState 的唯一实现 num_to_state 已下沉到 domain.state，此处复用）


def _rod_to_bead_text(rod_state) -> str:
    """单档快照 → 珠态字符串（与 _digit_to_bead 一致的格式）。"""
    upper = sum(1 for b in rod_state.upper_beads if b.position.is_active)
    lower = sum(1 for b in rod_state.lower_beads if b.position.is_active)
    if upper == 0 and lower == 0:
        return "空"
    if upper == 0:
        return f"L{lower}"
    if lower == 0:
        return f"U{upper}"
    return f"L{lower}+U{upper}"


def state_to_bead_text(state: AbacusState, strip_leading: bool = True) -> str:
    """珠态快照 → 珠态文本（高位在左；strip_leading=True 裁前导空档，与旧 to_bead() 一致）。

    >>> state_to_bead_text(num_to_state(97, Abacus.standard()))
    '[L4+U1|L2+U1]'
    """
    parts = [_rod_to_bead_text(rs) for rs in state.rods_left_to_right]
    if strip_leading:
        while len(parts) > 1 and parts[0] == "空":
            parts.pop(0)
    return "[" + "|".join(parts) + "]"


def bead_text_to_state(text: str, abacus: Abacus) -> AbacusState:
    """珠态文本 → 珠态快照（bead_text_to_num + num_to_state）。"""
    return num_to_state(bead_text_to_num(text, base=abacus.spec.base), abacus)
