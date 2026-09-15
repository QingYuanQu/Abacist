# -*- coding: utf-8 -*-
"""领域词汇表：异常 + 枚举（最底层，被所有其他领域模块依赖）。

拆分自原 domain_model.py（2026-09-07）。本模块零内部依赖（除枚举间互引），
是领域模型的"词汇表"：结构 / 状态 / 差分 / 动作 / 口诀层都从这里取类型。
"""
from __future__ import annotations

from enum import Enum


class AbacusError(Exception):
    """领域异常基类"""

class InvariantViolation(AbacusError):
    """物理/语义不变式被破坏（前置位不符、下珠非前缀激活、连锁进位未处理…）"""

class RhymeNotFound(AbacusError):
    """口诀检索缺口——全函数性被破坏（可直接当单测断言用）"""

class UnsupportedOperation(AbacusError):
    """暂不支持的运算（乘除编排中 / 归除待接入）"""


# ═══════════════════ 1. 枚举层 ═══════════════════

class BeadType(Enum):
    """珠类型：value 即单位面值"""
    UPPER = 5   # 上珠
    LOWER = 1   # 下珠

class BeadPosition(Enum):
    """珠的语义状态（两稳态，对上下珠正交）：ENGAGED=靠梁(激活/计入) / RESTING=靠框(未激活/不计入)"""
    ENGAGED = "ENGAGED"   # 靠梁
    RESTING = "RESTING"   # 靠框

    @property
    def is_active(self) -> bool:
        return self is BeadPosition.ENGAGED

class Finger(Enum):
    """拨珠手指（动作物理保真）：拇指推下珠靠梁 / 食指拨下珠离梁 / 中指管上珠"""
    THUMB = "THUMB"     # 拇指
    INDEX = "INDEX"     # 食指
    MIDDLE = "MIDDLE"   # 中指

class OpType(Enum):
    """四则运算类型：全系统唯一运算分类源"""
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"

class DigitOrder(Enum):
    """逐位运算推进方向（编排策略，不影响运算语义/终盘）：
    MSD=高位→低位（珠算/珠心算习惯）  Most Significant Digit（最高有效位 / 高位）
    LSD=低位→高位（竖式/计算机习惯）  Least Significant Digit（最低有效位 / 低位）
    除法立商除外——长除法数学上只能从高位起，固定 MSD。"""
    MSD = "MSD"
    LSD = "LSD"

class RhymeCategory(Enum):
    """口诀学分类（描述拨珠方式；乘法九九属数算口诀，不入此表）"""
    ADD_DIRECT         = ("直加", OpType.ADD)
    ADD_FILL5          = ("满五加", OpType.ADD)
    ADD_CARRY10        = ("进十加", OpType.ADD)
    ADD_BREAK5_CARRY10 = ("破五进十加", OpType.ADD)
    SUB_DIRECT         = ("直减", OpType.SUB)
    SUB_BREAK5         = ("破五减", OpType.SUB)
    SUB_BORROW10       = ("退十减", OpType.SUB)
    SUB_BORROW10_FILL5 = ("退十补五减", OpType.SUB)
    DIV_GUI            = ("归除", OpType.DIV)

    @property
    def cn(self) -> str:
        return self.value[0]

    @property
    def op(self) -> OpType:
        return self.value[1]

# —— 实现级补充枚举（原表内嵌于字段说明，此处显式化）——

class ActionType(Enum):
    """ENGAGE: 靠梁（下珠上推 / 上珠下拉）/ RELEASE: 离梁（下珠下拨 / 上珠上推）"""
    ENGAGE = "ENGAGE"
    RELEASE = "RELEASE"

    @property
    def before(self) -> BeadPosition:
        return BeadPosition.RESTING if self is ActionType.ENGAGE else BeadPosition.ENGAGED

    @property
    def after(self) -> BeadPosition:
        return BeadPosition.ENGAGED if self is ActionType.ENGAGE else BeadPosition.RESTING

class BeadAddressing(Enum):
    """模板内珠寻址方式（实现期新增：固定 ordinal 撑不起'三上三/进一'的跨盘面复用）"""
    FIXED = "FIXED"                  # 精确序号（口诀前提已锁定盘面，如"一下五去四"必在盘面4）
    NEXT_INACTIVE = "NEXT_INACTIVE"  # 最低未激活下珠（推：上d / 进一）
    LAST_ACTIVE = "LAST_ACTIVE"      # 最高已激活下珠（拨去：去d）
