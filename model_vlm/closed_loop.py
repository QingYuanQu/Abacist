"""closed_loop.py — VLM 端到端闭环控制器

把 VLM 从离线单步评测变成：
    Q → [PARSE] PRE/POST → 解析 POST → 数字压栈 / 运算符弹出 →
    [EVAL] 口诀链 → 算盘执行 → [READ] 读数 → 压栈

命名约定（本模块统一）：
  - ref_*     = 真值参考（只用于比对评分，不参与执行）
  - teacher_* = 教师信号（接管模型输出、直接驱动执行，即 teacher forcing）
  - pred_*    = 模型预测

支持两种模式：
  - 默认：PARSE 模型真实生成 POST
  - --teacher-post：用真值 POST 跳过 PARSE，只验证 EVAL/READ/执行链路本身

用法示例：
    python -m model_vlm.closed_loop \
        --parse-ckpt data/vlm_3task/model.pth \
        --eval-ckpt data/vlm_3task/model.pth \
        --read-ckpt data/vlm_3task/model.pth \
        --data data/vlm_3k/train.jsonl \
        --teacher-post --n-samples 20
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# 项目根加入 sys.path（支持从任意目录运行）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate.abacus import (                            # noqa: E402
    AbacusSpec,
    AbacusState,
    OpType,
    TransitionEngine,
    build_addition_rhymes,
    build_subtraction_rhymes,
)
from evaluate.abacus.render.registry import make_render  # noqa: E402
from model_vlm.parse_eval_bridge import (       # noqa: E402
    SYM_TO_OP,
    make_default_composer,
)
from model_vlm.infer import (                    # noqa: E402
    decode,
    generate,
    load_checkpoint,
    tokenize,
)


def _build_rhyme_table() -> dict[tuple[OpType, str], object]:
    """(运算类型, 口诀文本) → 指法（CompositeAction）查表。

    供「模型驱动执行」使用：模型输出口诀后，靠本表把口诀翻译成
    可在算盘上真实 reduce 的动作，而不是退回真值动作。

    注意：加/减法存在同名口诀（如「五去五」对应不同指法），
    因此必须带上运算类型作复合键。
    """
    table: dict[tuple[OpType, str], object] = {}
    for f in build_addition_rhymes() + build_subtraction_rhymes():
        if not f.action_mapping:
            continue
        table[(f.category.op, f.rhyme_text)] = f.action_mapping[0]
    return table


RHYME_TABLE = _build_rhyme_table()


def _lookup_action(rhyme_text: str, prefer_op: OpType | None = None):
    """按口诀文本查指法；prefer_op 为该步的运算类型（档位同理，均由执行计划给出）。"""
    if prefer_op is not None:
        hit = RHYME_TABLE.get((prefer_op, rhyme_text))
        if hit is not None:
            return hit
    for op in (OpType.ADD, OpType.SUB):
        hit = RHYME_TABLE.get((op, rhyme_text))
        if hit is not None:
            return hit
    return None


def parse_rhyme_output(pred: str) -> tuple[str, int | None]:
    """解析 EVAL 模型输出「口诀@档位」→ (口诀文本, 档位或 None)。

    档位坐标系与 TransitionEngine.reduce 的 base_rod_index 一致
    （个位=0、十位=1…）。旧格式输出（无 @档位 后缀）或后缀畸变时
    返回 (原文, None)，由调用方回退到执行计划的档位（向后兼容旧 checkpoint）。
    """
    if '@' not in pred:
        return pred, None
    rhyme, _, rod_s = pred.rpartition('@')
    rod_s = rod_s.strip()
    if rhyme and rod_s.isdigit():
        return rhyme, int(rod_s)
    return pred, None


def tokenize_post(post: str) -> list[str]:
    """把 POST 字符串切分为 token。

    两种输入格式：
      - 带空白分隔（真值 POST，如 '12 3 +'）→ 按空白切分，保留多位数边界
      - 紧凑串（PARSE 模型输出，如 '15+'）→ 退化为按字符切分，
        此时每个 token 均为单数字，与「词表只含 0-9 单数字」一致
    """
    raw = (post or '').strip()
    if not raw:
        return []
    if ' ' in raw:
        return [t for t in raw.split() if t and (t in SYM_TO_OP or t.isdigit())]
    tokens: list[str] = []
    for ch in raw:
        if ch in SYM_TO_OP or ch.isdigit():
            tokens.append(ch)
    return tokens


def _norm_post(post: str | None) -> str:
    """归一化 POST 用于比对：空白规整为单空格（保留操作数词法边界）。

    不能抹掉全部空格：多位数下 "12 4 +" 与 "1 24 +" 抹平后同为 "124+"，
    会产生假阳性比对。真值（parse.jsonl）与模型输出均为空格分隔格式。
    """
    return ' '.join((post or '').split())


def eval_post(post: str) -> int | None:
    """对 POST 求值（整数栈机）。无效返回 None。"""
    stack: list[int] = []
    for tok in tokenize_post(post):
        if tok not in SYM_TO_OP:
            stack.append(int(tok))
            continue
        if len(stack) < 2:
            return None
        b, a = stack.pop(), stack.pop()
        v = _apply_op(SYM_TO_OP[tok], a, b)
        if v is None:
            return None
        stack.append(v)
    return stack[0] if len(stack) == 1 else None


def _apply_op(op_type, a: int, b: int) -> int:
    """整数四则（与算盘域一致：÷ 已禁用，× 为重复加法语义）。"""
    from evaluate.abacus import OpType
    if op_type == OpType.ADD:
        return a + b
    if op_type == OpType.SUB:
        return a - b
    if op_type == OpType.MUL:
        return a * b
    if op_type == OpType.DIV:
        return a // b if b and a % b == 0 else None
    raise ValueError(f"未知运算: {op_type}")


def split_pre_post(text: str) -> tuple[str, str]:
    """切分 PARSE 模型输出：PRE...;POST...# → (pre, post)。

    命名避开 parse_* 前缀：本模块里 parse 已指「句法分析任务」本身，
    再叫旧名 parse_plan_output 会读成「解析 plan 输出」，与任务名撞车。
    """
    text = text.rstrip('#').strip()
    if ';' not in text:
        raise ValueError(f"PARSE 输出格式错误，缺少 ';': {text!r}")
    pre_part, post_part = text.split(';', 1)
    pre = pre_part[3:] if pre_part.startswith('PRE') else pre_part
    post = post_part[4:] if post_part.startswith('POST') else post_part
    return pre, post


class VLMClosedLoop:
    """端到端 VLM 闭环控制器。"""

    def __init__(self, parse_ckpt: str, eval_ckpt: str, read_ckpt: str,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 三任务模型可能共享同一个 checkpoint；patch 尺寸与渲染后端
        # 均从 checkpoint 自恢复（与训练数据生成侧保持一致，旧 ckpt
        # 无 render 字段时回退 "fixed"）。
        self.parse_enc, self.parse_gpt, self.parse_vocab, n_cols, in_channels, \
            patch_h, patch_w, render = load_checkpoint(parse_ckpt, self.device)
        self.eval_enc, self.eval_gpt, self.eval_vocab, _, _, _, _, eval_render = \
            load_checkpoint(eval_ckpt, self.device)
        self.read_enc, self.read_gpt, self.read_vocab, _, _, _, _, read_render = \
            load_checkpoint(read_ckpt, self.device)

        self.composer, self.abacus = make_default_composer(
            AbacusSpec(1, 4, 13, 10))
        self.engine = TransitionEngine()

        self.n_cols = self.abacus.spec.rod_count
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.render_name = render
        self.render_fn = make_render(render, patch_h=patch_h, patch_w=patch_w)
        if not (eval_render == render == read_render):
            print(f"[警告] 三任务 checkpoint 渲染后端不一致: "
                  f"parse={render} eval={eval_render} read={read_render}；"
                  f"以 parse 的 {render!r} 为准")
        print(f"[闭环] render={render} patch={patch_h}×{patch_w} C={in_channels}")

    # ── 内部辅助 ──
    def _render(self, state) -> torch.Tensor:
        """把算盘状态渲染为 [1, n_cols, C, patch_h, patch_w] 张量。

        arr 两种形态：[H, n_cols*W] 灰度（fixed/minimal/image_gray）或
        [H, n_cols*W, C] 彩色（image），与训练侧 to_rod_patches 同构。
        """
        arr = self.render_fn(
            state, self.abacus, patch_h=self.patch_h, patch_w=self.patch_w)
        if arr.ndim == 2:
            H, W = arr.shape
            image = arr.reshape(H, self.n_cols, W // self.n_cols) \
                       .transpose(1, 0, 2)[:, None]          # [n_cols,1,H,W]
        else:
            H, W, C = arr.shape
            image = arr.reshape(H, self.n_cols, W // self.n_cols, C) \
                       .transpose(1, 3, 0, 2)                # [n_cols,C,H,W]
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    def _zero_image(self) -> torch.Tensor:
        """parse 样本用的全零占位图（该任务无视觉输入，图位置填零）。"""
        return torch.zeros(1, self.n_cols, self.in_channels,
                           self.patch_h, self.patch_w,
                           dtype=torch.float32)

    def _parse_step(self, Q: str) -> tuple[str, str]:
        """PARSE：输入中缀串 Q，输出 (pre 前序结构, post 后序求值序)。

        post 是本轮闭环的执行计划来源（据此展开逐个二元运算），
        pre 仅作结构监督、不参与执行。

        解析失败（模型未收敛 / 输出无 ';'）时返回 ('', '')，
        由调用方记为 parse 失败，而不是让整轮评测崩溃。
        """
        image = self._zero_image().to(self.device)
        prompt = f"[PARSE] Q={Q}"
        ids = tokenize(prompt, self.parse_vocab)
        pred = generate(self.parse_enc, self.parse_gpt, image, ids,
                        self.parse_vocab, self.n_cols, max_len=128)
        try:
            return split_pre_post(pred)
        except ValueError:
            return '', ''

    def _eval_one_step(self, Q: str, post: str, stack_str: str,
                       calc_str: str, step_j: int, image: torch.Tensor) -> str:
        """EVAL：输入「当前盘面图 + 上下文」，输出单句口诀（去掉末尾 #）。

        与训练样本 prompt 严格同构：
            [EVAL] Q=.. POST=.. STACK=.. CALC=a op b STEP=j
        """
        prompt = (f"[EVAL] Q={Q} POST={post} STACK={stack_str} "
                  f"CALC={calc_str} STEP={step_j}")
        ids = tokenize(prompt, self.eval_vocab)
        pred = generate(self.eval_enc, self.eval_gpt, image, ids,
                        self.eval_vocab, self.n_cols, max_len=32)
        return pred.rstrip('#')

    def _read_value(self, image: torch.Tensor) -> int:
        """READ：输入盘面图，输出整数读数。"""
        prompt = "[READ] 读数="
        ids = tokenize(prompt, self.read_vocab)
        pred = generate(self.read_enc, self.read_gpt, image, ids,
                        self.read_vocab, self.n_cols, max_len=32)
        pred = pred.rstrip('#')
        # 读数应为纯数字
        digits = [c for c in pred if c.isdigit()]
        if not digits:
            raise ValueError(f"READ 模型输出非数字: {pred!r}")
        return int(''.join(digits))

    # ── 核心闭环 ──
    def solve(self, Q: str, ANS: int | None = None,
              ref_post: str | None = None,
              ref_pre: str | None = None,
              teacher_post: str | None = None,
              model_driven: bool = False) -> dict:
        """对单个问题做端到端闭环求解。

        参数：
          - ref_post: 真值 POST，仅用于 PARSE 命中比对（不参与执行）
          - ref_pre: 真值 PRE（仅展示，不参与执行）
          - teacher_post: 教师信号，提供则跳过 PARSE，直接用它执行（--teacher-post）
          - model_driven: True 时用 EVAL 模型预测的口诀驱动算盘执行，
            口诀错误会真实传播到最终答案（真闭环）

        返回 dict 包含：
          - final_result: 闭环最终答案
          - correct: 是否与 ANS 一致（ANS 为 None 时 None）
          - post: 实际使用的 POST（模型 PARSE 结果或教师信号）
          - parse_ok: POST 是否与 ref_post 一致（教师信号模式下为 None）
          - eval_hits: EVAL 口诀链命中次数 / 总次数
          - rod_hits: 档位命中次数 / 有档位输出的总次数（旧模型无 @档位 时为 0/0）
          - read_hits: READ 读数命中次数 / 总次数
          - exec_ok: 模型驱动执行是否全程成功
          - trace: 每步 calc 的详细记录
        """
        # ① PARSE：中缀串 → 语法树（PRE 结构 + POST 求值序）
        if teacher_post is not None:            # 教师信号：跳过 PARSE
            post = teacher_post
            pre = ref_pre or ""
            parse_ok = None
            parse_semantic_ok = None
        else:
            pre, post = self._parse_step(Q)
            parse_ok = (_norm_post(post) == _norm_post(ref_post)) \
                if ref_post is not None else None
            # 语义比对：POST 求值结果与真值求值一致即算对
            # （加法交换律会产生很多字符串不同但等价的 POST）
            if ref_post is not None:
                pv, gv = eval_post(post), eval_post(ref_post)
                parse_semantic_ok = (pv is not None and pv == gv)
            else:
                parse_semantic_ok = None

        tokens = tokenize_post(post)
        # EVAL 训练样本的 POST 是真值分词格式（'1 5 +'，带空格），
        # 而 PARSE 模型输出紧凑串（'15+'）。这里统一成训练分布，否则
        # 默认模式（不经教师信号 POST）下 EVAL 会因分布偏移大幅掉点。
        post_eval = ' '.join(tokens)
        stack: list[int] = []
        trace: list[dict] = []
        eval_hits = 0
        eval_total = 0
        rod_hits = 0
        rod_total = 0
        read_hits = 0
        read_total = 0
        op_id = 0
        exec_ok = True
        exec_error = None

        for tok in tokens:
            if tok not in SYM_TO_OP:                       # 数字：压栈
                stack.append(int(tok))
                continue

            # ② 运算符：弹出两数（栈不足说明 POST 无效，记失败继续）
            if len(stack) < 2:
                exec_ok = False
                exec_error = f"POST 无效：遇到 {tok!r} 时栈只有 {len(stack)} 个数"
                break
            op_type = SYM_TO_OP[tok]
            b, a = stack.pop(), stack.pop()

            # ③ 用 composer 得到真值执行计划（驱动执行 + 作为 EVAL 的评分基准）
            try:
                op = self.composer.compose(op_type, self.abacus, a, b,
                                            operation_id=op_id)
            except Exception as exc:                 # 档越界 / 不变量违反
                exec_ok = False
                exec_error = f"compose 失败（{a}{tok}{b} 超出算盘容量？）: {exc}"
                break
            op_id += 1
            ref_rhyme = '；'.join(s.rhyme.rhyme_text for s in op.steps)

        # ④ EVAL：逐步「看当前盘面 → 预测单句口诀」
        empty_state = AbacusState.empty(self.abacus)
        stack_str = ','.join(str(x) for x in stack)
        calc_str = f"{a}{tok}{b}"
        state = empty_state
        pred_steps: list[str] = []
        for j, plan_step in enumerate(op.steps):
            cur_image = self._render(state).to(self.device)
            pred = self._eval_one_step(Q, post_eval, stack_str, calc_str, j,
                                        cur_image)
            pred_steps.append(pred)

            # 「口诀@档位」双通道：口诀文本 + 基准档位
            rhyme_text, pred_rod = parse_rhyme_output(pred)

            eval_total += 1
            if rhyme_text == plan_step.rhyme.rhyme_text:
                eval_hits += 1
            if pred_rod is not None:
                rod_total += 1
                if pred_rod == plan_step.base_rod_index:
                    rod_hits += 1

            # ⑤ 算盘执行（真实物理执行）
            #    model_driven=False → 走执行计划的动作（EVAL 只做旁观评测，
            #                         模型预测对错不影响盘面与最终答案）
            #    model_driven=True  → 口诀查表得指法、@档位给基准档（真闭环），
            #                         档位缺失（旧模型）时回退执行计划档位；
            #                         盘面由模型自己的执行结果推进，错误会累积
            action = plan_step.action
            if model_driven:
                mapped = _lookup_action(rhyme_text, plan_step.rhyme.category.op)
                if mapped is None:
                    exec_error = f"第{j}步未知口诀: {rhyme_text!r}"
                    break
                action = mapped
            rod = pred_rod if (model_driven and pred_rod is not None) \
                else plan_step.base_rod_index
            try:
                state, _ = self.engine.reduce(state, action, self.abacus, rod)
            except Exception as exc:                      # 物理不变式违反
                exec_error = f"第{j}步执行失败: {pred!r} → {exc}"
                break

            pred_rhyme = '；'.join(pred_steps)

            if exec_error is not None:            # 执行链断裂：本条样本判定失败
                exec_ok = False
                trace.append({
                    'op': tok, 'a': a, 'b': b,
                    'ref_value': None, 'pred_value': None,
                    'ref_rhyme': ref_rhyme, 'pred_rhyme': pred_rhyme,
                    'exec_error': exec_error,
                })
                break

            # ⑥ READ：模型读终盘
            final_image = self._render(state).to(self.device)
            pred_value = self._read_value(final_image)
            read_total += 1
            if pred_value == state.total_value:
                read_hits += 1

            # ⑦ 压栈
            stack.append(pred_value)

            trace.append({
                'op': tok,
                'a': a,
                'b': b,
                'ref_value': state.total_value,
                'pred_value': pred_value,
                'ref_rhyme': ref_rhyme,
                'pred_rhyme': pred_rhyme,
            })

        final_result = stack[0] if exec_ok and stack else None
        return {
            'Q': Q,
            'pre': pre,
            'post': post,
            'final_result': final_result,
            'ANS': ANS,
            'correct': final_result == ANS if ANS is not None else None,
            'exec_ok': exec_ok,
            'exec_error': exec_error,
            'parse_ok': parse_ok,
            'parse_semantic_ok': parse_semantic_ok,
            'eval_hits': (eval_hits, eval_total),
            'rod_hits': (rod_hits, rod_total),
            'read_hits': (read_hits, read_total),
            'trace': trace,
        }


def _load_records(jsonl_path: str, n_samples: int | None = None) -> list[dict]:
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if n_samples is not None:
        records = records[:n_samples]
    return records


def main():
    ap = argparse.ArgumentParser(description='VLM 端到端闭环评测')
    ap.add_argument('--parse-ckpt', required=True, help='PARSE 任务 checkpoint')
    ap.add_argument('--eval-ckpt', required=True, help='EVAL 任务 checkpoint')
    ap.add_argument('--read-ckpt', required=True, help='READ 任务 checkpoint')
    ap.add_argument('--data', required=True, help='parse 样本 jsonl（含 Q/ANS/pre/post）')
    ap.add_argument('--n-samples', type=int, default=None, help='只测前 N 条')
    ap.add_argument('--teacher-post', '--use-gold-post', dest='teacher_post',
                    action='store_true',
                    help='用真值 POST 跳过 PARSE，只测 EVAL/READ/执行链路'
                         '（--use-gold-post 为旧名，已弃用）')
    ap.add_argument('--model-driven', action='store_true',
                    help='用 EVAL 模型预测的口诀驱动算盘执行（真闭环，'
                         '口诀出错会传导到最终答案）')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    records = _load_records(args.data, args.n_samples)
    # 过滤出 parse 样本（一条记录含完整 Q/ANS/pre/post）
    parse_records = [r for r in records if r.get('task') == 'parse']
    if not parse_records:
        # 如果没有 task 字段，尝试全部当 parse 用
        parse_records = records

    loop = VLMClosedLoop(args.parse_ckpt, args.eval_ckpt, args.read_ckpt,
                         device=args.device)

    total = correct = parse_ok = parse_total = exec_ok = 0
    parse_semantic_ok = 0
    eval_hits = eval_total = 0
    rod_hits = rod_total = 0
    read_hits = read_total = 0

    for i, rec in enumerate(parse_records):
        Q = rec['Q']
        ANS = rec.get('ANS')
        ref_pre = rec.get('PRE') or rec.get('pre')
        ref_post = rec.get('POST') or rec.get('post')

        result = loop.solve(
            Q, ANS=ANS,
            ref_post=ref_post,
            ref_pre=ref_pre,
            teacher_post=ref_post if args.teacher_post else None,
            model_driven=args.model_driven,
        )

        total += 1
        if result['correct']:
            correct += 1
        if result['exec_ok']:
            exec_ok += 1
        if result['parse_ok'] is not None:
            parse_total += 1
            if result['parse_ok']:
                parse_ok += 1
        if result.get('parse_semantic_ok'):
            parse_semantic_ok += 1
        eh, et = result['eval_hits']
        rh, rt = result['read_hits']
        dh, dt = result['rod_hits']
        eval_hits += eh
        eval_total += et
        read_hits += rh
        read_total += rt
        rod_hits += dh
        rod_total += dt

        if i < 5:
            print(f"[{i}] Q={Q!r} ANS={ANS} final={result['final_result']} "
                  f"correct={result['correct']} parse_ok={result['parse_ok']}")

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} = {a/b:.2%}" if b else "—"

    print("\n[端到端闭环统计]")
    print(f"  模式: {'模型驱动（真闭环）' if args.model_driven else '真值执行（EVAL 旁观评测）'}"
          f"{' + 教师 POST（跳过 PARSE）' if args.teacher_post else ''}")
    print(f"  样本数: {total}")
    print(f"  端到端正确: {pct(correct, total)}")
    print(f"  PARSE POST 正确: {pct(parse_ok, parse_total)}"
          f"（语义等价: {pct(parse_semantic_ok, parse_total)}）")
    print(f"  EVAL 口诀命中: {pct(eval_hits, eval_total)}")
    print(f"  档位命中: {pct(rod_hits, rod_total)}"
          f"（无 @档位 输出时为 —，旧模型/未重训）")
    print(f"  READ 读数命中: {pct(read_hits, read_total)}")
    if args.model_driven:
        print(f"  执行链完整: {pct(exec_ok, total)}")


if __name__ == '__main__':
    main()
