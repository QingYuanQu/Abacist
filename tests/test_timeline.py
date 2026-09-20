"""拨珠时间轴（render/timeline.py）回归测试：缓动形态 + 拟人结构。

最关键的断言是「珠随手动」：旧路径整段拨动都渲染 before 状态、最后一帧瞬移到
after，手在动而珠不动。若哪天 `FrameSpec.bead_motion` 被弄丢，这里会立刻红。
"""
import pytest

from evaluate.abacus import (Abacus, AbacusSpec, ArithmeticComposer, OpType,
                             RhymeResolver, build_addition_rhymes,
                             build_subtraction_rhymes)
from evaluate.abacus.render.timeline import (MovePlan, hand_tremor, minimum_jerk,
                                             plan_duration_ms, plan_hits,
                                             plan_operation, rebound_ease,
                                             stroke_ms_for)


def _op(rod_count: int = 7, a: int = 47, b: int = 38):
    ab = Abacus.standard(AbacusSpec(rod_count=rod_count))
    composer = ArithmeticComposer(RhymeResolver(
        build_addition_rhymes() + build_subtraction_rhymes()))
    return composer.compose(OpType.ADD, ab, a, b)


def test_minimum_jerk_shape():
    """端点固定、单调、起止速度为 0（钟形速度曲线）。"""
    assert minimum_jerk(0.0) == pytest.approx(0.0)
    assert minimum_jerk(1.0) == pytest.approx(1.0)
    assert minimum_jerk(0.5) == pytest.approx(0.5)
    vals = [minimum_jerk(i / 20) for i in range(21)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[1] - vals[0] < vals[10] - vals[9]      # 起步慢于中段


def test_rebound_overshoots_then_settles():
    """珠冲过目标 → 回弹 → 停住（撞梁的"嗒"就是这一下）。"""
    hit = 0.85
    assert rebound_ease(hit, 0.05, hit) > 1.0          # 冲过目标
    assert rebound_ease(0.95, 0.05, hit) < 1.0         # 回弹略过
    assert rebound_ease(1.0, 0.05, hit) == pytest.approx(1.0)
    assert rebound_ease(0.5, 0.0) == pytest.approx(minimum_jerk(0.5))  # 关闭即纯 jerk


def test_frame_count_matches_plan_duration():
    op = _op()
    plan = MovePlan()
    specs = plan_operation(op, plan)
    base = plan_duration_ms(plan, len(op.steps)) / 1000 * plan.fps
    assert len(specs) >= base                          # 跨档加时只会更长
    assert len(specs) < base * 1.5


def test_beads_move_with_hand():
    """stroke 阶段：珠有中间位置，且手正压在主珠上（珠被手推着走）。"""
    op = _op()
    specs = plan_operation(op, MovePlan())
    moving = [s for s in specs if s.bead_motion]
    assert len(moving) >= len(op.steps) * 3
    for s in moving:
        assert s.hand is not None
        assert s.hand in s.bead_motion.values()


def test_lead_in_and_lead_out_are_still():
    """开场只展示题目、收尾只展示结果：无手、无口诀。"""
    op = _op()
    plan = MovePlan(fps=10, lead_in=1000, lead_out=1000)
    specs = plan_operation(op, plan)
    assert specs[0].hand is None and specs[0].rhyme is None and specs[0].caption
    assert specs[-1].hand is None


def test_step_tags_are_1_based_and_bounded():
    op = _op()
    specs = plan_operation(op, MovePlan())
    tags = {s.step for s in specs if s.step}
    assert tags == {(i, len(op.steps)) for i in range(1, len(op.steps) + 1)}


def test_tempo_scales_durations_not_fps():
    """tempo 只缩放时长（帧数变少），不降帧率——降 fps 只会让画面一顿一顿。"""
    base, fast = MovePlan(fps=30), MovePlan(fps=30, tempo=0.5)
    scaled = fast.scaled()
    assert scaled.fps == 30
    assert scaled.hold == pytest.approx(base.hold * 0.5)
    assert scaled.lead_out == pytest.approx(base.lead_out * 0.5)
    assert scaled.tempo == 1.0                      # 缩放只做一次，避免重复相乘
    op = _op()
    assert len(plan_operation(op, fast)) == pytest.approx(
        len(plan_operation(op, base)) / 2, rel=0.1)


def test_complexity_lengthens_multi_bead_strokes():
    """同拨珠数越多，stroke 越长（真人遇复杂口诀会慢一点）。"""
    p = MovePlan(complexity=0.15)
    assert stroke_ms_for(p, 1) == pytest.approx(p.stroke)
    assert stroke_ms_for(p, 3) == pytest.approx(p.stroke * 1.3)


def test_hand_tremor_is_bounded_and_deterministic():
    """微抖幅度受控、持续变化（不是静止），且可关闭。"""
    p = MovePlan(tremor=0.01)
    assert hand_tremor(0.0, MovePlan(tremor=0)) == (0.0, 0.0)      # 可关闭
    vals = [hand_tremor(i / 30, p) for i in range(60)]
    assert all(abs(x) <= p.tremor + 1e-9 and abs(y) <= p.tremor + 1e-9 for x, y in vals)
    assert len({(round(x, 9), round(y, 9)) for x, y in vals}) > 30  # 确实在动
    assert hand_tremor(0.5, p) == hand_tremor(0.5, p)               # 确定性


def test_one_rod_one_hit_not_per_bead():
    """同档多珠是**一声**："四上四"一次推四颗下珠，不是连响四下。

    这是真人听感与"机器连敲"的分界线——若退化成每珠一声，这里会红。
    """
    op = _op()
    hits = plan_hits(op, MovePlan())
    assert len(hits) == len(op.steps)        # 47+38 每步都是同档一束（无跨档复合）
    assert max(h.count for h in hits) >= 4   # 存在四珠同拨的那一步
    assert all(h.count >= 1 for h in hits)


def test_focus_col_marks_the_rod_being_played():
    """高亮档 = 该步主珠所在列；开场与收尾不高亮。"""
    from evaluate.abacus.domain import StateDelta
    from evaluate.abacus.render.geometry import bead_centers

    op = _op()
    specs = plan_operation(op, MovePlan())
    assert any(s.focus_col is not None for s in specs)
    assert specs[0].focus_col is None and specs[-1].focus_col is None

    prev = op.steps[0].before_state
    main_cols = []
    for st in op.steps:
        delta = StateDelta.diff(prev, st.after_state)
        moved = [bd for rd in delta.rod_deltas for bd in rd.changed_beads]
        main_cols.append(round(bead_centers(prev, op.abacus)[moved[0].bead_id][0]) - 1
                         if moved else None)
        prev = st.after_state
    for s in specs:
        if s.focus_col is not None and s.step:
            assert s.focus_col == main_cols[s.step[0] - 1]
