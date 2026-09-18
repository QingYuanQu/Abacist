# -*- coding: utf-8 -*-
"""bridge/ir.py —— 底层中间表示（IR）契约。

IR 是一条表达式的「富结构」落盘格式：纸笔层（parse 产出）+ 算盘层（eval 产出），
供所有模型（model_lm / model_vm / model_vlm 及未来模型）离线投影，不重算。

设计原则：
  1. 结构化为真源（postfix 数组、steps 对象），展示串（pre/post）为便利字段；
  2. 最小充分：actions 存数值 value，珠态文本 / 盘面图由消费方调 bead_codec 派生；
  3. 纸笔与算盘分层：stack（纸笔栈）与 abacus.actions（算盘拨珠）分开存，
     不再像旧 evaluate.py 那样把口诀塞进栈追踪。

当前聚焦加减乘（+ - ×）；÷ 尚未实现，留待后续扩展。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AbacusAction:
    """abacus 口径的单步拨珠（含「摆左操作数」步 + 「运算」步）。

    oral : 口诀文本（中文数字，如「一上一」）。
    rod  : 档位（个位=0、十位=1…，与 TransitionEngine.reduce 的 base_rod_index 一致）。
    value: 该步执行后的盘面数值（空盘起算 + 单次二元运算，数值唯一确定盘面）。
    """
    oral: str
    rod: int
    value: int


@dataclass
class AbacusTrace:
    """abacus 口径的完整执行链：一次二元运算 a op b 的逐步拨珠。"""
    actions: list[AbacusAction] = field(default_factory=list)


@dataclass
class EvalStep:
    """后序序列里的一个 token 步：push（数字入纸笔栈）/ calc（二元运算）。

    push 步：只有 kind/token/stack。
    calc 步：另有 a/b 操作数 + abacus（口诀链）/ digit（逐位展开）注解。
    """
    kind: str                                # 'push' | 'calc'
    token: str                               # 数字字符串 或 运算符符号（+ - ×）
    stack: list[int] = field(default_factory=list)   # 该步执行后的纸笔栈快照
    stack_before: list[int] = field(default_factory=list)  # calc 专用：弹出 a、b 后、结果入栈前的栈（无泄漏）
    a: Optional[int] = None                  # calc 专用：次栈顶
    b: Optional[int] = None                  # calc 专用：栈顶
    abacus: Optional[AbacusTrace] = None     # calc 专用：abacus 口径注解
    digit: Optional[str] = None              # calc 专用：digit 口径逐位展开串


@dataclass
class ExpressionInstance:
    """IR 单行：一条表达式 = 元数据 + 纸笔层 + 算盘层。

    字段命名遵循 parse(ODS) 层约定：用语言学术语（infix/prefix/postfix/answer），
    不含 chat 语义的 Q/A —— Q/A 是 experiment 投影层按任务赋予的角色
    （如 infix=>postfix 任务时 Q=infix、A=postfix）。
    """
    # —— 元数据 ——
    n: int                                   # 叶子数
    ops: str                                 # 运算符组合（树中序运算符序列，如 "+-"）
    ic: int                                  # Colless 不平衡度
    bk: int                                  # Colless 桶 0-9
    sp: str                                  # train / test
    gid: int                                 # 多解组 id：去括号后同一 infix 的多棵树共享同一 gid
    tree: str                                # 完整括号中缀串（字母树，审计/去重）
    # —— 纸笔层（parse 产出）——
    infix: str                               # 去括号中缀（模型实际输入的问题表达式）
    prefix: str                              # 前序展示串
    postfix: str                             # 后序展示串
    answer: int                              # 数学答案（数值；不可求值数据集 A/B 为 0）
    # —— 算盘层（eval 产出）——
    steps: list[EvalStep] = field(default_factory=list)


def to_json(inst: ExpressionInstance) -> str:
    """IR → 单行 JSON（ensure_ascii=False，中文口诀原样保留）。"""
    return json.dumps(asdict(inst), ensure_ascii=False)


def build_instance(record: dict, steps: list[EvalStep],
                   *, gid: int = 0, tree: str = "") -> ExpressionInstance:
    """从 parse 的 record + align 的 steps 组装 IR 实例。

    record: parse/dataset_generator 的一条记录。ops/tree/gid 优先取 record 内字段
    （阶段 2 起 parse 落盘已补全），缺失时回退：ops 从 postfix 推导，tree/gid 用占位。

    字段键兼容：parse(ODS) 层已用 infix/prefix/postfix/answer；同时为兼容 expr 型
    trial 的 _generate_expr（本次暂不改动）仍写 Q/pre/post/ANS，故此处两套键都认。
    待 _generate_expr 迁移后移除旧键兼容分支。
    """
    postfix_str = record.get("postfix", record.get("post", ""))
    postfix = postfix_str.split()
    ops = record.get("ops") or "".join(t for t in postfix if t in "+-×÷")
    return ExpressionInstance(
        n=record["n"], ops=ops,
        ic=record.get("Ic", 0), bk=record.get("bk", 0),
        sp=record.get("sp", "train"),
        gid=record.get("gid", gid),
        tree=record.get("tree", tree),
        infix=record.get("infix", record.get("Q", "")),
        prefix=record.get("prefix", record.get("pre", "")),
        postfix=postfix_str,
        answer=record.get("answer", record.get("ANS", 0)),
        steps=steps,
    )
