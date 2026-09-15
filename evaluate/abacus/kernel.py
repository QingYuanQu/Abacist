# -*- coding: utf-8 -*-
"""kernel.py —— 珠算内核的行为层：状态转移 / 口诀检索 / 四则编排 / 审计链。

拆分自原 abacus_domain.py（2026-09-07）。依赖 domain 子包（数据层）与
rhymes（口诀种子，仅被接收为参数、不 import 其构建器）；不反向依赖渲染
与上层数据管线。

  TransitionEngine    唯一状态转移入口；守护前置位与下珠前缀不变式
  RhymeResolver     全函数口诀检索（真值裁判，assert_total 可自验）
  ArithmeticComposer  四则编排：加减逐位 / 乘法九九错位叠加 / 商除（含余数分区）
  CalculationStep     单步审计记录（is_valid 由差分内生推导）
  Operation           运算审计链聚合根，可确定性重放
  AbacusEnv           gym 风格运行时（外围适配层，在线 RL 落点）
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterator, Sequence

from evaluate.abacus.domain import (
    Abacus,
    AbacusError,
    AbacusSpec,
    AbacusState,
    BeadAddressing,
    BeadPosition,
    BeadState,
    BeadType,
    Carry,
    CompositeAction,
    DigitOrder,
    Rhyme,
    RhymeCategory,
    RhymeNotFound,
    InvariantViolation,
    OpType,
    Rod,
    RodState,
    StateDelta,
    UnsupportedOperation,
    AtomicAction,
    Bead,
    num_to_state,
)



# ═══════════════════ 7. 行为内核层（领域服务：无状态） ═══════════════════

class TransitionEngine:
    """唯一状态转移入口：reduce() 全有或全无；守护前置位与物理不变式（下珠前缀）。

    清盘态构造（原 initial_state）已下沉到 AbacusState.empty——那是状态层职责，
    不是行为内核职责（2026-09-07）。
    """

    @staticmethod
    def reduce(state: AbacusState, action: CompositeAction, abacus: Abacus,
               base_rod_index: int) -> tuple[AbacusState, StateDelta]:
        if not action.atomic_actions:
            raise InvariantViolation("空动作")
        # ① 顺序绑定 + 前置位校验（工作副本上模拟，全有或全无）
        positions = {bs.bead_id: bs.position for rs in state.rod_states
                     for bs in rs.upper_beads + rs.lower_beads}
        bound: list[tuple[int, AtomicAction]] = []
        for atom in action.atomic_actions:
            rod = abacus.rod_at(base_rod_index + atom.rod_offset)
            bead = TransitionEngine._resolve(rod, atom, positions)
            cur = positions[bead.bead_id]
            if cur is not atom.before_position:
                raise InvariantViolation(
                    f"[{action.name}] 珠B{bead.bead_id}(档{rod.index}) "
                    f"当前{cur.value} ≠ 前置{atom.before_position.value}")
            positions[bead.bead_id] = atom.after_position
            bound.append((bead.bead_id, atom))
        # ② 提交（with_bead 内含前缀不变式校验）
        new_state = state
        for bead_id, atom in bound:
            new_state = new_state.with_bead(bead_id, atom.after_position)
        new_state = replace(new_state, step_index=state.step_index + 1)
        # ③ 差分
        return new_state, StateDelta.diff(state, new_state)

    @staticmethod
    def _resolve(rod: Rod, atom: AtomicAction,
                 positions: dict[int, BeadPosition]) -> Bead:
        if atom.bead_type is BeadType.UPPER or atom.addressing is BeadAddressing.FIXED:
            return rod.bead(atom.bead_type, atom.ordinal)
        lowers = rod.lower_beads                      # ordinal 升序
        if atom.addressing is BeadAddressing.NEXT_INACTIVE:
            for b in lowers:
                if positions[b.bead_id] is BeadPosition.RESTING:
                    return b
            raise InvariantViolation(f"档{rod.index}下珠已满无处可推（连锁进位？）")
        for b in reversed(lowers):                    # LAST_ACTIVE
            if positions[b.bead_id] is BeadPosition.ENGAGED:
                return b
        raise InvariantViolation(f"档{rod.index}下珠已空无处可拨")

class RhymeResolver:
    """
    输入： op: OpType, operand: int, rod_state: RodState
    返回： Rhyme
    全函数口诀检索（真值裁判）。知识即种子数据；全函数性可用 assert_total() 验收。

    进制：分类判定表（`_classify`）目前只实现**十进制**（上1下4、档值 0..9）。
    进制是本类的一等属性（构造参数 `base`，默认 `BASE`）——当前只放行十进制，
    但**签名与字段已经就位**：将来接入十六进制分类表（上二下五）时，只需新增
    分类逻辑并放行 `base=16`，调用方（ArithmeticComposer 的进制一致性校验、
    `_require_matching_base`）一行都不用改。这是"预留十六进制"的结构性占位，
    而不是散落在各处的 `if base != 10` 封条。
    """

    BASE = 10          # 已实现分类判定表的进制（唯一声明点）

    def __init__(self, rhymes: Sequence[Rhyme], base: int = BASE):
        if base != self.BASE:
            raise UnsupportedOperation(
                f"口诀分类判定表目前只实现十进制（base={self.BASE}），"
                f"base={base} 待接入")
        self.base = base
        self.rhymes = tuple(rhymes)
        self._by_key: dict[tuple[OpType, RhymeCategory, int], Rhyme] = {}
        for f in self.rhymes:
            key = (f.category.op, f.category, f.trigger_digit)
            if key in self._by_key:
                raise AbacusError(f"口诀重复: {f.rhyme_text}")
            self._by_key[key] = f

    def resolve(self, op: OpType, operand: int, rod_state: RodState) -> Rhyme:
        v = rod_state.rod_value
        cat = self._classify(op, operand, v)
        f = self._by_key.get((op, cat, operand))
        if f is None:
            raise RhymeNotFound(
                f"缺口诀: 盘面{v} {op.name} {operand} → 需{cat.cn}({cat.name})，请补种子")
        return f

    @staticmethod
    def _classify(op: OpType, d: int, v: int) -> RhymeCategory:
        """口诀分类判定表（十进制、上1下4，档值 0..9），返回带中文名 cn 的类别。

        >>> RhymeResolver._classify(OpType.ADD, 2, 1).cn    # 盘面 1 加 2
        '直加'
        >>> RhymeResolver._classify(OpType.ADD, 1, 4).cn    # 盘面 4 加 1：满五
        '满五加'
        >>> RhymeResolver._classify(OpType.ADD, 9, 1).cn    # 盘面 1 加 9：进十
        '进十加'
        >>> RhymeResolver._classify(OpType.ADD, 6, 5).cn    # 盘面 5 加 6：破五进十
        '破五进十加'
        >>> RhymeResolver._classify(OpType.SUB, 3, 8).cn    # 盘面 8 减 3：直减
        '直减'
        >>> RhymeResolver._classify(OpType.SUB, 3, 6).cn    # 盘面 6 减 3：破五
        '破五减'
        >>> RhymeResolver._classify(OpType.SUB, 7, 3).cn    # 盘面 3 减 7：退十补五
        '退十补五减'
        >>> RhymeResolver._classify(OpType.SUB, 7, 6).cn    # 盘面 6 减 7：退十
        '退十减'
        """
        if op is OpType.ADD:
            r = v + d
            if r >= 10:
                return (RhymeCategory.ADD_BREAK5_CARRY10
                        if v >= 5 and d >= 6 and (v - 5) < (10 - d)
                        else RhymeCategory.ADD_CARRY10)
            if v <= 4 and d <= 4 and r >= 5:
                return RhymeCategory.ADD_FILL5
            return RhymeCategory.ADD_DIRECT
        if op is OpType.SUB:
            r = v - d
            if r < 0:                                 # 退十：左档-1，本档+(10-d)
                return (RhymeCategory.SUB_BORROW10_FILL5
                        if d >= 6 and v <= 4 and v + (10 - d) >= 5
                        else RhymeCategory.SUB_BORROW10)
            if v >= 5 and d <= 4 and (v - 5) < d:
                return RhymeCategory.SUB_BREAK5
            return RhymeCategory.SUB_DIRECT
        raise UnsupportedOperation("乘除由 ArithmeticComposer 编排为加减序列；归除另行接入")

    def assert_total(self, op: OpType = OpType.ADD) -> None:
        """全函数性断言：90 格（operand 1-9 × 盘面 0-9）每格均可检索到口诀。

        用真实的 num_to_state 构造单档盘面（取代硬编码上1下4的 _mock_rod_state），
        保证判定的盘面形态与内核实际使用的完全一致。
        """
        abacus = Abacus.standard(AbacusSpec(1, 4, 1, 10))   # 单档十进制算盘
        for v in range(10):
            rs = num_to_state(v, abacus).rod_state_at(0)
            for d in range(1, 10):
                self.resolve(op, d, rs)

class ArithmeticComposer:
    """把四则运算（加/减/乘/除）编排为珠算口诀序列，产出可审计的 Operation 审计链。

    编排是状态化游走：每一步选用的口诀取决于上一句弹完后的盘面，故产物
    即审计链 Operation（list[Rhyme] 形式的乐谱可由 steps 投影得到）。

    分工：加减为逐档直加/借位的基础运算；乘法=九九口诀错位叠加（复用加法）、
    除法=归除/商除，二者均降级为加减序列后复用同一套内核。
    """
    def __init__(self, resolver: RhymeResolver, engine: TransitionEngine | None = None,
                 digit_order: DigitOrder = DigitOrder.MSD):
        self.resolver = resolver
        self.engine = engine or TransitionEngine()
        self.digit_order = digit_order

    def _compose_binary(self, abacus, a, b, op_type, operation_id) -> Operation:
        """组合一次二元运算（加/减）的完整珠算过程。

        流程：
        1. 初始化空算盘；
        2. 拨入操作数 a（恒为直加）；
        3. 对操作数 b 逐位执行 op_type 运算；
        4. 计算预期结果，组装 Operation 并校验。
        """
        self._require_matching_base(abacus)
        state = AbacusState.empty(abacus)
        steps: list[CalculationStep] = []
        for rod_idx, digit in self._digits(a, abacus.spec.base):        # ① 拨入 a（必为直加）
            state = self._apply(abacus, state, OpType.ADD, digit, rod_idx, steps)
        for rod_idx, digit in self._digits(b, abacus.spec.base):        # ② 运算 b, 逐位运算（方向由
            state = self._apply(abacus, state, op_type, digit, rod_idx, steps)  #   digit_order 决定）
        expected = a + b if op_type is OpType.ADD else a - b
        op = Operation(operation_id, op_type, a, b, expected, abacus, steps, state)
        op.verify()
        return op

    def _require_matching_base(self, abacus: Abacus) -> None:
        """算盘进制必须与口诀裁判的进制一致。

        校验只在此处定义一处（此前在 5 个入口各贴一张"必须十进制"的封条，
        既重复又易漏）。裁判在构造时已声明自己支持的进制，故这里只比对一致性——
        将来裁判接入十六进制分类表后，本方法无需改动。
        """
        if abacus.spec.base != self.resolver.base:
            raise UnsupportedOperation(
                f"算盘 base={abacus.spec.base} 与口诀裁判 base={self.resolver.base} 不一致")

    def compose(self, op_type: OpType, abacus: Abacus, a: int, b: int,
                operation_id: int = 0) -> Operation:
        if op_type is OpType.ADD:
            return self._compose_binary(abacus, a, b, OpType.ADD, operation_id)
        if op_type is OpType.SUB:
            # 走 compose_subtraction 而非直接 _compose_binary：后者无 a<b 前置检查，
            # 负差会退位至档越界并抛 InvariantViolation，掩盖真正原因。
            return self.compose_subtraction(abacus, a, b, operation_id)
        if op_type is OpType.MUL:
            return self.compose_multiplication(abacus, a, b, operation_id)
        if op_type is OpType.DIV:
            return self.compose_division(abacus, a, b, operation_id)
        raise UnsupportedOperation(f"未知运算类型: {op_type}")

    def compose_addition(self, abacus, a, b, operation_id=0) -> Operation:
        return self._compose_binary(abacus, a, b, OpType.ADD, operation_id)

    def compose_subtraction(self, abacus, a, b, operation_id=0) -> Operation:
        if a < b:
            raise UnsupportedOperation("被减数<减数需跨档退十，暂不支持")
        return self._compose_binary(abacus, a, b, OpType.SUB, operation_id)

    def compose_multiplication(self, abacus, a, b, operation_id=0) -> Operation:
        """错位叠加：a 的第 k 位数字 d → 把九九积 (d×b) 逐位加到第 k 档起"""
        self._require_matching_base(abacus)
        state = AbacusState.empty(abacus)
        steps: list[CalculationStep] = []
        for k, d in self._digits(a, abacus.spec.base):        # 被乘数逐位：MSD=空盘前乘/LSD=破头乘
            for j, sd in self._digits(d * b, abacus.spec.base):
                state = self._apply(abacus, state, OpType.ADD, sd, k + j, steps)
        op = Operation(operation_id, OpType.MUL, a, b, a * b, abacus, steps, state)
        op.verify()
        return op

    def compose_division(self, abacus, a, b, operation_id=0) -> Operation:
        """商除法（长除法）：被除数 a 拨入高位区，商逐位立入低位区（个位在档0）。
        每位流程 = 立商（试商 q_digit 用加法口诀拨入）→ 减积（q_digit×b 用九九得积，
        再错位用减法口诀从被除数区减去，退位由 _apply 递归）。整除时被除数区归零、
        终盘=商；非整除时被除数区残留即余数，终盘=商+余数分区并排，一并过 verify。"""
        self._require_matching_base(abacus)
        if b == 0:
            raise UnsupportedOperation("除数不能为 0")
        quotient, remainder = divmod(a, b)
        q_str = str(quotient)
        h = len(q_str)                                # 被除数区起点档 = 商位数（商占 0..h-1）
        state = AbacusState.empty(abacus)
        steps: list[CalculationStep] = []
        # ① 拨入被除数 a 到高位区（个位在档 h，盘面值 = a × base^h）
        for rod_idx, digit in self._digits(a, abacus.spec.base):
            state = self._apply(abacus, state, OpType.ADD, digit, h + rod_idx, steps)
        # ② 逐位立商 + 减积（立商固定 MSD：长除法只能从高位起，与 digit_order 无关）
        for pos, ch in enumerate(q_str):
            q_digit = int(ch)
            k = len(q_str) - 1 - pos                 # 商的该位档号（个位=0）
            if q_digit == 0:
                continue
            state = self._apply(abacus, state, OpType.ADD, q_digit, k, steps)   # 立商
            for j, sd in self._digits(q_digit * b, abacus.spec.base):           # 减积
                state = self._apply(abacus, state, OpType.SUB, sd, h + k + j, steps)
        op = Operation(operation_id, OpType.DIV, a, b, quotient, abacus, steps, state,
                       remainder=remainder)
        op.verify()
        return op

    def compose_digit_stream(self, abacus, digits: Sequence[int],
                             operation_id: int = 0) -> Operation:
        """逐位累加数字流：空盘起，每位在个位档做一次加法（进位由 _apply 递归）。

        数字流 = 任意 0..base-1 序列（π 的各位、循环节、电话号码……）；
        0 无拨珠即跳过（休止不占步）。审计链语义：operand_a=0（空盘起算）、
        operand_b=各位数字和，终盘=数字和——verify/replay 天然成立。
        供 music.py 演奏数字流之用（如 π 累加的进位心跳、1/7 循环节的相位漂移）。
        """
        self._require_matching_base(abacus)
        state = AbacusState.empty(abacus)
        steps: list[CalculationStep] = []
        total = 0
        for d in digits:
            if not 0 <= d < abacus.spec.base:
                raise ValueError(f"数字流元素须 0..{abacus.spec.base - 1}，得到 {d}")
            if d == 0:                                 # 休止：无拨珠不占步
                continue
            state = self._apply(abacus, state, OpType.ADD, d, 0, steps)
            total += d
        op = Operation(operation_id, OpType.ADD, 0, total, total, abacus, steps, state)
        op.verify()
        return op

    # —— 内部 ——
    def _digits(self, n: int, base: int) -> Iterator[tuple[int, int]]:
        """按 self.digit_order 方向逐位展开：(档号, 数字)。"""
        yield from (self._digits_msd if self.digit_order is DigitOrder.MSD else self._digits_lsd)(n, base)

    @staticmethod
    def _digits_lsd(n: int, base: int = 10) -> Iterator[tuple[int, int]]:
        """Least Significant Digit 优先：从最低位（个位，档号 0）向高位逐位展开。

        返回 (档号, 数字) 序列；数字为 0 的位无拨珠动作，故跳过。
        每个循环：取当前最低位 n % base，累加器 i 即所在档号，整除 base 后进位。

        >>> list(ArithmeticComposer._digits_lsd(4708))   # 4708 = 4千7百0十8，0 位跳过
        [(0, 8), (2, 7), (3, 4)]
        >>> list(ArithmeticComposer._digits_lsd(101))
        [(0, 1), (2, 1)]
        """
        i = 0
        while n > 0:
            digit = n % base          # 当前最低位
            if digit != 0:
                yield i, digit        # i 为档号，跳过 0 位
            n //= base                 # 进位到更高一位
            i += 1

    @classmethod
    def _digits_msd(cls, n: int, base: int = 10) -> Iterator[tuple[int, int]]:
        """Most Significant Digit 优先：从最高位向最低位（个位）逐位展开。

        内部先按 lsd 顺序收集全部 (档号, 数字)，再 reverse 得到高位→低位的顺序。
        同样跳过数字为 0 的位（无拨珠动作）。

        >>> list(ArithmeticComposer._digits_msd(4708))   # 高位在前：4千→7百→8个
        [(3, 4), (2, 7), (0, 8)]
        >>> list(ArithmeticComposer._digits_msd(9999))
        [(3, 9), (2, 9), (1, 9), (0, 9)]
        """
        pairs = list(cls._digits_lsd(n, base))   # 先得到 低位→高位 序列
        yield from reversed(pairs)                # 反转即得 高位→低位

    def _apply(self, abacus, state, op_type, digit, rod_idx, steps) -> AbacusState:
        """递归调度：本档按口诀拨珠后，若产生跨档（进一/退一），对邻档递归同向 +1/−1。
        进位档的 ±1 与任意普通 ±1 走同一 resolve，天然支持满五进位、连环进位/退位。"""
        while True:
            rhyme = self.resolver.resolve(op_type, digit, state.rod_state_at(rod_idx))
            if len(rhyme.action_mapping) != 1:
                raise UnsupportedOperation("当前内核按'一句口诀=一次联拨'审计，暂不支持多复合动作口诀")
            composite = rhyme.action_mapping[0]
            new_state, delta = self.engine.reduce(state, composite, abacus, rod_idx)
            steps.append(CalculationStep(new_state.step_index, rod_idx, rhyme, composite,
                                         state, new_state, delta))
            state = new_state
            carry = self._carry(op_type, rhyme.category)
            if carry is None:
                return state
            rod_idx += carry.rod_offset
            digit = carry.digit

    @staticmethod
    def _carry(op_type: OpType, category: RhymeCategory) -> "Carry | None":
        if op_type is OpType.ADD and category in (RhymeCategory.ADD_CARRY10,
                                                  RhymeCategory.ADD_BREAK5_CARRY10):
            return Carry(rod_offset=+1, digit=1)      # 进一：左档 +1
        if op_type is OpType.SUB and category in (RhymeCategory.SUB_BORROW10,
                                                  RhymeCategory.SUB_BORROW10_FILL5):
            return Carry(rod_offset=+1, digit=1)      # 退一：向高位（左档）借 1，即 −1
        return None




# ═══════════════════ 8. 运算层（审计链：可确定性重放） ═══════════════════

@dataclass
class CalculationStep:
    """单步审计记录：is_valid 由 matches_semantic() 内生推导，非外部标注"""
    step_index: int
    base_rod_index: int
    rhyme: Rhyme
    action: CompositeAction
    before_state: AbacusState
    after_state: AbacusState
    state_delta: StateDelta
    is_valid: bool = field(init=False)

    def __post_init__(self):
        # 方案④：内生推导期望差分（本档净值 + 跨档项），与手写 expected_delta 双保险
        # 跨档项由动作的 rod_offset 原子直接推导，消除手写对账漂移。
        base = self.after_state.base          # 取自形制（AbacusSpec.base），非硬编码
        expected_net = sum(e.value_delta * (base ** e.rod_offset)
                           for e in self.rhyme.expected_delta)
        self.is_valid = self.state_delta.matches_semantic(
            expected_net, self.base_rod_index)

@dataclass
class Operation:
    """聚合根：完整运算审计链"""
    operation_id: int
    op_type: OpType
    operand_a: int
    operand_b: int
    expected_result: int
    abacus: Abacus = field(repr=False)
    steps: list[CalculationStep] = field(default_factory=list, repr=False)
    final_state: AbacusState | None = None
    remainder: int = 0                              # 仅 DIV 有效：余数（非除法则为 0）

    def verify(self) -> None:
        """审计自检：每步有效 + 终盘 = 期望结果。
        除法终盘布局 = 商（低位区）+ 余数 × base^商位数（被除数区），一并校验。

        校验通过即正常返回；任何不变量被破坏时抛 InvariantViolation。
        **不用 assert** —— `python -O` 会剔除 assert，使审计静默失效。

        （此前签名写作 `-> bool` 却永不返回 False，是骗人的 API；现已修正为 None，
        不再靠 docstring 打补丁遮盖。）
        """
        for s in self.steps:
            if not s.is_valid:
                raise InvariantViolation(
                    f"步骤 step{s.step_index} 对账失败：[{s.rhyme.rhyme_text}] "
                    f"档{s.base_rod_index} 实际净值 {s.state_delta.total_value_delta}")
        if self.final_state is None:
            raise InvariantViolation("缺终盘")
        if self.op_type is OpType.DIV:
            h = len(str(self.expected_result)) if self.expected_result else 1
            base = self.abacus.spec.base
            want = self.expected_result + self.remainder * (base ** h)
            if self.final_state.total_value != want:
                raise InvariantViolation(
                    f"除法终盘{self.final_state.total_value}≠商{self.expected_result}"
                    f"+余{self.remainder}×{base}^{h}（={want}）")
        elif self.final_state.total_value != self.expected_result:
            raise InvariantViolation(
                f"终盘{self.final_state.total_value}≠期望{self.expected_result}")

    def replay(self) -> list[StateDelta]:
        """确定性重放：从首步前置态重演，校验差分逐一致（失配抛 InvariantViolation）。"""
        engine = TransitionEngine()
        state, deltas = self.steps[0].before_state, []
        for s in self.steps:
            state, delta = engine.reduce(state, s.action, self.abacus, s.base_rod_index)
            if delta != s.state_delta:
                raise InvariantViolation(f"重放失配 @step{s.step_index}")
            deltas.append(delta)
        return deltas

# ═══════════════════ 9. 外围适配层（运行时：可整体替换，内核零感知） ═══════════════════

class AbacusEnv:
    """gym 风格运行时（应用服务）：包装 TransitionEngine，在线 RL 落点。
    obs=AbacusState；reward 可插拔；非法动作 → reward=-1 且终止。"""
    def __init__(self, abacus: Abacus,
                 step_reward: Callable[[AbacusState, StateDelta], float] | None = None,
                 episode_reward: Callable[[AbacusState, bool], float] | None = None,
                 max_steps: int = 64):
        self.abacus = abacus
        self._step_reward = step_reward or (lambda s, d: -0.01)
        self._episode_reward = episode_reward or (lambda s, done: 1.0 if done else 0.0)
        self.max_steps = max_steps
        self._state: AbacusState | None = None
        self._target, self._n = 0, 0

    def reset(self, start_value: int = 0, target: int | None = None) -> AbacusState:
        self._state = num_to_state(start_value, self.abacus)
        self._target = target if target is not None else start_value
        self._n = 0
        return self._state

    def step(self, action: CompositeAction, base_rod_index: int = 0):
        assert self._state is not None, "先 reset()"
        self._n += 1
        try:
            new_state, delta = TransitionEngine.reduce(self._state, action,
                                                       self.abacus, base_rod_index)
        except InvariantViolation as e:
            return self._state, -1.0, True, {"error": str(e)}
        done = new_state.total_value == self._target
        reward = self._step_reward(new_state, delta) \
            + (self._episode_reward(new_state, done) if done else 0.0)
        if self._n >= self.max_steps:
            done = True
        self._state = new_state
        return new_state, reward, done, {"delta": delta}
