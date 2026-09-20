"""拨珠动画时间轴：一步拨珠 → 按 fps 采样出的**帧指令**（纯数据，零 matplotlib）。

职责边界：`vertical.py` 决定"帧长什么样"，本模块决定"帧在什么时刻、珠在哪个中间位置"。
分开是因为"像不像人"几乎全在时序里：

1. 等间隔硬切（旧路径每步 5~6 帧、全局 400ms）＝幻灯片感，与画质无关；
2. **珠不随手动**——旧路径整段拨动都渲染 before 状态、最后一帧瞬移到 after，
   手在动而珠不动。这是最大的不自然点，由本模块的 `FrameSpec.bead_motion`
   （覆盖珠心的中间坐标）解决：珠必须由手推着走。

拟人依据（人手上肢运动的实证结论，不是随手编的曲线）：

- 伸手与拨动遵循**最小急动度（minimum jerk）**模型：位置 p(t)=10t³−15t⁴+6t⁵，
  速度呈钟形、起止速度为 0。这是人手伸展运动被反复验证的形态，
  比 ease-in-out（三次贝塞尔）更接近真人。
- 动作之间存在**停顿**：定位（看题/找档）→ 伸手 → 拨动 → 撞梁回弹 → 收手 → 看结果。
  把这些停顿压成 0，动作就"不像人"而"像程序"。
- **换档比同档慢**（手要横向移动），**多珠同拨有几十毫秒错峰**（手指不完全同步）。
- 珠撞到梁/框会**回弹**一下再停住——真实拨珠的"嗒"就是这一下。

产出 `FrameSpec` 列表（纯数据），可单测；渲染由 vertical.py 消费。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evaluate.abacus.domain import StateDelta
from evaluate.abacus.render.geometry import bead_centers

if TYPE_CHECKING:
    from evaluate.abacus.domain import AbacusState


@dataclass(frozen=True)
class MovePlan:
    """一次拨珠动作的时间预算（毫秒）与节奏参数。

    默认值按**教学短视频**标定：比真人熟练拨珠慢（让人看清），
    但保留真人的动作结构与停顿比例。
    """
    fps: int = 30
    hold: float = 220          # 定位：看题 / 找档（手不在场）
    approach: float = 170      # 手伸入并触到珠（珠未动）
    stroke: float = 210        # 拨动：珠随手动，末端撞梁回弹
    settle: float = 130        # 珠已到位、稳住（手开始撤）
    recover: float = 110       # 手撤离画面
    step_gap: float = 150      # 步骤间隔（同档）
    traverse_ms: float = 70    # 每跨一档的额外加时（手要横向移动）
    lead_in: float = 800       # 开场：展示题目
    lead_out: float = 1200     # 收尾：展示结果
    rebound: float = 0.05      # 撞梁回弹幅度（数据单位；7 档 1080 宽≈8px）
    stagger_ms: float = 45     # 多珠同拨的错峰（手指不完全同步）
    # 手入场/撤离相对珠位的偏移（数据坐标）：从右下方伸入、向右下方撤出
    entry_dx: float = 0.85
    entry_dy: float = -0.85
    exit_dx: float = 0.55
    exit_dy: float = -0.55
    # 拟人旋钮
    tempo: float = 1.0        # 节奏倍率：1.0 = 教学（默认）；0.55 ≈ 熟练演示
    complexity: float = 0.15  # 每多拨一颗珠，stroke 加时的比例（复杂口诀更慢）
    tremor: float = 0.010     # 手部微抖幅度（数据单位），0 = 关闭
    tremor_hz: float = 7.0    # 微抖频率（真人手震颤量级）

    def scaled(self) -> "MovePlan":
        """按 tempo 缩放全部时长：tempo<1 = 熟练（快），>1 = 更慢的教学。

        fps **不变**——节奏变化体现在帧数上；降帧率只会让画面一顿一顿。
        """
        k = self.tempo
        return MovePlan(
            fps=self.fps, tempo=1.0,
            hold=self.hold * k, approach=self.approach * k, stroke=self.stroke * k,
            settle=self.settle * k, recover=self.recover * k,
            step_gap=self.step_gap * k, traverse_ms=self.traverse_ms * k,
            lead_in=self.lead_in * k, lead_out=self.lead_out * k,
            stagger_ms=self.stagger_ms * k,
            rebound=self.rebound, complexity=self.complexity,
            tremor=self.tremor, tremor_hz=self.tremor_hz,
            entry_dx=self.entry_dx, entry_dy=self.entry_dy,
            exit_dx=self.exit_dx, exit_dy=self.exit_dy,
        )


@dataclass(frozen=True)
class FrameSpec:
    """一帧的**内容指令**（不含绘制）：画什么状态、珠在哪、手在哪、写什么字。"""
    state: AbacusState
    caption: str | None = None
    rhyme: str | None = None
    step: tuple[int, int] | None = None
    hand: tuple[float, float] | None = None
    bead_motion: dict[int, tuple[float, float]] | None = field(default=None)
    focus_col: int | None = field(default=None)   # 正在被拨的档（渲染列，0-based）


@dataclass(frozen=True)
class HitEvent:
    """一次撞击（珠撞梁 / 撞框）——**音效的时刻锚点**。

    at_ms 是撞击瞬间，不是动作开始：真人是珠撞到的那一刻才"嗒"，
    把声音放在动作起点会明显不同步（人耳对声先于画尤其敏感）。
    """
    at_ms: float
    bead_id: int
    bead_type: str        # "upper"（低沉） / "lower"（清脆）
    velocity: float       # 归一化撞击力度（0.25~1.4）→ 音量与"亮度"
    col: int              # 渲染列（0-based），留给声像/定位
    count: int = 1        # 同时撞击的珠数（同一档被一次推动的那一束）

    # 为什么同档多珠是**一声**而不是 count 声：
    # "四上四"是一次推动四颗下珠，四颗同时撞到梁，真人听到的是一记更厚更钝的响；
    # 若给每颗珠各发一声（哪怕只错开 45ms），听感就成了手指连敲四下——那是机器。
    # 反例是**跨档**：进位要换一档再拨，那是两个动作，必须是两声（故按档分组）。


# 撞击发生在 stroke 进度的这一点（= 珠冲到目标、即将回弹的瞬间），
# 与 rebound_ease 的 hit 参数同源——**音效的时刻锚点，音画同步靠它对齐**。
HIT_AT = 0.85
# 参考撞击速度（数据单位/秒）：最长行程（0.8）在默认 stroke（0.21s）下的峰值速度。
# 最小急动度曲线的峰值速度 = 1.875 × 行程 / 时长。
V_REF = 7.0

# ── 缓动 ──
def minimum_jerk(t: float) -> float:
    """最小急动度位置曲线 p(t)=10t³−15t⁴+6t⁵（t∈[0,1]）。

    速度钟形、起止速度为 0 —— 人手伸展运动的实证形态。
    """
    t = min(1.0, max(0.0, t))
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def rebound_ease(t: float, overshoot: float = 0.05, hit: float = HIT_AT) -> float:
    """主曲线 + 末端阻尼回弹：珠冲过目标一点点，撞梁后弹回并停住。

    t<hit：冲程（在 hit 处达到 1+overshoot）；
    t≥hit：阻尼振荡收敛到 1（(1−u)·cos(πu) 保证端点连续：u=0 → 1+ov，u=1 → 1）。
    """
    t = min(1.0, max(0.0, t))
    if overshoot <= 0:
        return minimum_jerk(t)
    if t <= hit:
        return (1.0 + overshoot) * minimum_jerk(t / hit)
    u = (t - hit) / (1.0 - hit)
    return 1.0 + overshoot * (1.0 - u) * math.cos(math.pi * u)


def hand_tremor(t_s: float, plan: MovePlan = MovePlan()) -> tuple[float, float]:
    """手部微抖（真人的手不可能完全静止）：两个异相正弦叠加，x/y 幅度不同。

    **只作用于空中阶段**（伸手 / 撤离）：拨动时手压着珠，必须严格同步，
    否则手会脱离珠。确定性（无随机数），同一 plan 每次出片完全一致。
    """
    a = plan.tremor
    if a <= 0:
        return (0.0, 0.0)
    w = 2 * math.pi * plan.tremor_hz
    return (a * math.sin(w * t_s), 0.6 * a * math.sin(1.37 * w * t_s + 1.1))


def _n(ms: float, fps: int) -> int:
    """毫秒 → 帧数（至少 1 帧，保证短阶段也出现）。"""
    return max(1, int(round(ms / 1000.0 * fps)))


def stroke_ms_for(plan: MovePlan, n_beads: int) -> float:
    """stroke 时长：口诀越复杂（同时拨的珠越多）真人越慢，按珠数放大。"""
    return plan.stroke * (1.0 + plan.complexity * max(0, n_beads - 1))


def bead_stroke_s(stroke_s: float, n_beads: int, stagger_s: float) -> float:
    """单颗珠的实际拨动时长：总时长扣掉错峰占用的部分（至少保留一半）。"""
    return max(stroke_s - stagger_s * max(0, n_beads - 1), stroke_s * 0.5)


def _staggered_progress(t_global: float, order: int, n_beads: int,
                        stagger_s: float, stroke_s: float) -> float:
    """多珠同拨的错峰进度：第 order 颗延后 stagger 启动，各自时长相应缩短。"""
    shift = stagger_s * min(order, max(0, n_beads - 1))
    dur = bead_stroke_s(stroke_s, n_beads, stagger_s)
    return min(1.0, max(0.0, (t_global * stroke_s - shift) / dur))


def hit_velocity(distance: float, dur_s: float) -> float:
    """归一化撞击力度：最小急动度曲线的峰值速度 = 1.875 × 行程 / 时长。

    拨得越快、行程越长 → 撞得越响，这是真人的物理，不是音量随机数。
    """
    v = 1.875 * distance / max(dur_s, 1e-3)
    return min(1.4, max(0.25, v / V_REF))


def plan_operation(op, plan: MovePlan = MovePlan(), *,
                   hits: list[HitEvent] | None = None) -> list[FrameSpec]:
    """把一次 Operation 编译成逐帧指令。

    每步结构：hold → approach → **stroke（珠随手动）** → settle → recover → 间隔；
    开场 lead_in 展示题目，收尾 lead_out 展示结果。

    hits: 传入列表则**顺便收集撞击事件**（音效的时刻锚点）。事件与帧在同一处
          计算，是为了杜绝"画面时钟"与"声音时钟"两份逻辑分叉——分叉的直接
          后果就是音画不同步（人耳对"声先于画"尤其敏感）。
    """
    plan = plan.scaled()          # tempo 先落到各阶段时长上
    abacus = op.abacus
    fps = plan.fps
    sym = {"ADD": "+", "SUB": "-", "MUL": "×", "DIV": "÷"}[op.op_type.name]
    caption = f"{op.operand_a} {sym} {op.operand_b} = ?"
    total = len(op.steps)

    # 先收集每步的"谁在动、从哪到哪、主珠在哪一档"，供跨档加时使用
    moves: list[dict] = []
    prev = op.steps[0].before_state
    for step in op.steps:
        after = step.after_state
        delta = StateDelta.diff(prev, after)
        moved = [bd for rd in delta.rod_deltas for bd in rd.changed_beads]
        starts = bead_centers(prev, abacus)
        ends = bead_centers(after, abacus)
        ids = [bd.bead_id for bd in moved]
        main_col = round(ends[ids[0]][0]) if ids else None
        moves.append({"before": prev, "after": after, "rhyme": step.rhyme.rhyme_text,
                      "ids": ids, "starts": starts, "ends": ends, "main_col": main_col})
        prev = after

    specs: list[FrameSpec] = []
    first = moves[0]["before"] if moves else op.steps[0].before_state

    # 开场：展示题目（无口诀、无手）
    specs.extend(FrameSpec(first, caption) for _ in range(_n(plan.lead_in, fps)))

    for i, mv in enumerate(moves, start=1):
        tag = (i, total)
        rhyme, ids, starts, ends = mv["rhyme"], mv["ids"], mv["starts"], mv["ends"]
        before, after = mv["before"], mv["after"]

        if not ids:                       # 无珠变化（罕见）：静止看完本步
            specs.extend(FrameSpec(after, caption, rhyme, tag)
                         for _ in range(_n(plan.hold + plan.settle, fps)))
            prev_col = mv["main_col"]
            _append_gap(specs, after, caption, rhyme, tag, plan, fps,
                        prev_col, moves[i]["main_col"] if i < total else None)
            continue

        mx, my = starts[ids[0]]
        col = round(mx) - 1                    # 目标档（渲染列，0-based；x = col+1）
        entry = (mx + plan.entry_dx, my + plan.entry_dy)

        # 1) hold：定位（手不在场，目标档高亮——人的视线先落在要拨的那一档）
        specs.extend(FrameSpec(before, caption, rhyme, tag, focus_col=col)
                     for _ in range(_n(plan.hold, fps)))

        # 2) approach：手从场外伸入并触到珠（珠不动；空中阶段带手部微抖）
        n = _n(plan.approach, fps)
        for k in range(1, n + 1):
            e = minimum_jerk(k / n)
            hx = entry[0] + (mx - entry[0]) * e
            hy = entry[1] + (my - entry[1]) * e
            dx, dy = hand_tremor(len(specs) / fps, plan)
            specs.append(FrameSpec(before, caption, rhyme, tag,
                                   hand=(hx + dx, hy + dy), focus_col=col))

        # 3) stroke：珠随手动（末端撞梁回弹）；多珠错峰 + 复杂度加时
        #    口诀越复杂（同时拨的珠越多）真人越慢，故按珠数放大 stroke 时长
        stroke_ms = stroke_ms_for(plan, len(ids))
        n = _n(stroke_ms, fps)
        stroke_s = stroke_ms / 1000.0
        stagger_s = plan.stagger_ms / 1000.0
        dur_s = bead_stroke_s(stroke_s, len(ids), stagger_s)
        pending = dict.fromkeys(ids, True)          # 尚未撞到梁/框的珠
        raw: list[tuple[int, float, float, str, int]] = []   # (档, 时刻, 力度, 珠型, id)
        for k in range(1, n + 1):
            t = k / n
            frame_ms = len(specs) / fps * 1000.0    # 本帧时刻（append 前的帧索引）
            motion: dict[int, tuple[float, float]] = {}
            for j, bid in enumerate(ids):
                local = _staggered_progress(t, j, len(ids), stagger_s, stroke_s)
                p = rebound_ease(local, plan.rebound)
                sx, sy = starts[bid]
                ex, ey = ends[bid]
                motion[bid] = (sx + (ex - sx) * p, sy + (ey - sy) * p)
                # 珠冲到目标的瞬间 = 撞击（其后 15% 是回弹）——音效的时刻锚点
                if pending[bid] and local >= HIT_AT:
                    pending[bid] = False
                    if hits is not None:
                        bead = abacus.get_bead_by_id(bid)
                        dist = math.hypot(ex - sx, ey - sy)
                        raw.append((round(sx) - 1, frame_ms,
                                    hit_velocity(dist, dur_s),
                                    bead.bead_type.name.lower(), bid))
            # 手跟随主珠（第一颗）当前位置：拨动时手压着珠，严格同步不加抖动
            specs.append(FrameSpec(before, caption, rhyme, tag, hand=motion[ids[0]],
                                   bead_motion=motion, focus_col=col))

        # 撞击事件：同档的珠是"一次推动的一束"，合并成**一声**；
        # 不同档（如进位要换一档再拨）是两个动作，必须是两声。
        if hits is not None and raw:
            groups: dict[int, list] = {}
            for item in raw:
                groups.setdefault(item[0], []).append(item)
            for col_j, group in groups.items():
                at = sum(g[1] for g in group) / len(group)          # 束的能量重心
                vel = sum(g[2] for g in group) / len(group)
                hits.append(HitEvent(
                    at_ms=at, bead_id=group[0][4], bead_type=group[0][3],
                    # k 颗同时撞击：声能叠加，振幅 ∝ √k（不是 k 倍，也不是连响 k 下）
                    velocity=min(1.4, vel * math.sqrt(len(group))),
                    col=col_j, count=len(group)))

        # 4) settle：珠已到位稳住，手开始撤（盘面切到 after）
        ex, ey = ends[ids[0]]
        n = _n(plan.settle, fps)
        for k in range(1, n + 1):
            e = minimum_jerk(k / n) * 0.6
            specs.append(FrameSpec(after, caption, rhyme, tag,
                                   hand=(ex + plan.exit_dx * e, ey + plan.exit_dy * e)))

        # 5) recover：手继续外撤，后段离场
        n = _n(plan.recover, fps)
        for k in range(1, n + 1):
            t = k / n
            if t < 0.65:
                e = 0.6 + minimum_jerk(t / 0.65) * 0.7
                dx, dy = hand_tremor(len(specs) / fps, plan)   # 撤离也是空中阶段
                hand = (ex + plan.exit_dx * e + dx, ey + plan.exit_dy * e + dy)
            else:
                hand = None
            specs.append(FrameSpec(after, caption, rhyme, tag, hand=hand))

        # 步骤间隔：跨档加时
        next_col = moves[i]["main_col"] if i < total else None
        _append_gap(specs, after, caption, rhyme, tag, plan, fps, mv["main_col"], next_col)

    # 收尾：展示结果
    last = moves[-1]["after"] if moves else first
    specs.extend(FrameSpec(last, caption) for _ in range(_n(plan.lead_out, fps)))
    return specs


def _append_gap(specs: list, after, caption, rhyme, tag, plan: MovePlan,
                fps: int, col: int | None, next_col: int | None) -> None:
    """步骤间隔：同档 step_gap，每跨一档加 traverse_ms（手要横向移动）。"""
    if next_col is None:
        return
    traverse = 0 if col is None else abs(next_col - col)
    ms = plan.step_gap + plan.traverse_ms * traverse
    specs.extend(FrameSpec(after, caption, rhyme, tag)
                 for _ in range(_n(ms, fps)))


def plan_hits(op, plan: MovePlan = MovePlan()) -> list[HitEvent]:
    """取一次 Operation 的撞击事件序列（时刻 / 珠型 / 力度 / 档位）——音轨的输入。

    与 `plan_operation` 共用同一处计算（通过 hits 参数回传），不存在第二份时钟。
    """
    hits: list[HitEvent] = []
    plan_operation(op, plan, hits=hits)
    hits.sort(key=lambda h: h.at_ms)     # 跨档事件可能逆序插入，音效层要求时刻单调
    return hits


def plan_duration_ms(plan: MovePlan = MovePlan(), n_steps: int = 1) -> float:
    """预估总时长（毫秒）：用于出片前先确认时长是否合适。"""
    per_step = plan.hold + plan.approach + plan.stroke + plan.settle + plan.recover
    gaps = plan.step_gap * max(0, n_steps - 1)
    return plan.lead_in + per_step * n_steps + gaps + plan.lead_out
