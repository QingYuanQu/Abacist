# -*- coding: utf-8 -*-
"""music.py —— 算盘乐器化：把运算审计链渲染成可演奏的乐谱（sonification）。

依据：AbacusSpec 的上/下珠数是参数（"同一内核可跑二五珠/一四珠"），故

    一四珠 = 5珠/档 ↔ 东方五声（宫商角徵羽）
    二五珠 = 7珠/档 ↔ 西方七声（上珠恰补出 la/ti 两变声）
    13 档  ↔ 13 弦（恰是筝的弦数）；音色默认 GM #108 Kalimba，可配 107 回十三弦筝

计算本身自带时序（AtomicAction.duration_ms 是"物理时间唯一存放处"）、
自带指法（Finger）、自带歌词（Rhyme.rhyme_text），故一次运算 = 一首小品：

    数值即和声   盘面靠梁的珠 = 正在发声的音（sustain 模式：档上数字 d
                 的靠梁珠 = 宫调前缀和弦，d≥5 再加一记羽；终盘=尾和弦）
    口诀即歌词   Operation.steps 的 rhyme_text 序列就是唱词（Score.lyrics）
    指法即节奏   联拨 = 琶音（strum 错峰），duration_ms = 音长，指别 = 力度
    进位即转调   连环进位 = 音区爬升的花彩经过句

两种演奏模式（MusicConfig.mode）：
    sustain  盘面即和声：ENGAGE=note-on、RELEASE=note-off，珠靠梁多久音就
             响多久；拨入操作数 = 开场扫弦，终盘 = 收束和弦。
    pluck    指法即韵律：每次拨珠 = 一颗音（音长 = 该原子动作的 duration_ms），
             听的是手指的节奏而非和声——拨去为幽灵弱音。

七珠筝（AbacusSpec(2,5,13,10)）不能跑算术内核（口诀分类表只实现十进制
一四珠），但 TransitionEngine 与本模块对珠数完全中立——用 MelodyPlayer 可
直接在其上演奏旋律。

产物：Score（内存谱面，to_text() 人读）→ write_midi() 落盘标准 MIDI 文件
（format 0，纯标准库实现，无第三方依赖；GM 音色默认 108 = Kalimba 卡林巴，
另发 CC91 混响补空间感——GM 合成器是音色天花板，DAW 换源才是谱面真实水平）。

用法（从仓库根）：
    python -m evaluate.abacus.demos music    # 运算演奏 + 七珠筝旋律 → output/music/
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from evaluate.abacus.domain import (
    AbacusSpec,
    AbacusState,
    ActionType,
    AtomicAction,
    BeadAddressing,
    BeadPosition,
    BeadType,
    CompositeAction,
    Finger,
    InvariantViolation,
)
from evaluate.abacus.kernel import Operation, TransitionEngine


# ═══════════════════ 1. 音律层：音阶与定弦 ═══════════════════

@dataclass(frozen=True)
class Scale:
    """音阶：半音偏移 + 音级名（东方五声 / 西方七声）。"""
    name: str
    cn: str
    semitones: tuple[int, ...]
    degree_names: tuple[str, ...]

    def __post_init__(self):
        if len(self.semitones) != len(self.degree_names):
            raise ValueError("semitones 与 degree_names 必须等长")
        if not self.semitones or self.semitones[0] != 0:
            raise ValueError("音阶必须从 0 半音（主音）起")

PENTATONIC = Scale("pentatonic", "五声（宫商角徵羽）", (0, 2, 4, 7, 9),
                   ("宫", "商", "角", "徵", "羽"))
HEPTATONIC = Scale("heptatonic", "七声（do re mi fa sol la ti）", (0, 2, 4, 5, 7, 9, 11),
                   ("do", "re", "mi", "fa", "sol", "la", "ti"))

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    """MIDI 音高 → 科学音名。

    >>> note_name(60), note_name(69)
    ('C4', 'A4')
    """
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class Tuning:
    """定弦：统一调——全盘所有音高都落在同一个音阶里（跨八度）。

    全局音级 g = 档号 + 珠音级（下珠从 0 数起，上珠续接）：

        pitch(r, bead) = root + scale[g % n] + 12 * (g // n)

    于是**任意盘面、任意和弦都在同一调内**：五声调内不存在小二度/三全音/大七度，
    任意组合皆协和——盘面再大也不刺耳。（这是从筝式移调定弦改过来的教训：
    移调定弦一档一调，古筝一次只拨一根弦没事，但盘面同时发声时跨档会撞出
    三全音。）弦号越高音越高、品号越高音越高，方向性保留；品 0 的音高恰为
    scale[r % n] + 12 * (r // n)——与旧定弦完全一致，旋律指法不受影响。

    - degree_of：下珠#k → 音级 k，上珠#k → 下珠数 + k。于是档上数字 d 的
      靠梁珠 = 同调内一段五声窗（d<5 为前缀；d≥5 上珠跳到第 5 级再叠前缀）
      ——**数值即和声**：数字 d 决定窗口的宽度与形状。

    >>> t = Tuning(AbacusSpec(1, 4, 13, 10), PENTATONIC)
    >>> [t.degree_of(BeadType.LOWER, k) for k in range(4)]
    [0, 1, 2, 3]
    >>> t.degree_of(BeadType.UPPER, 0)
    4
    >>> t.pitch_of(0, BeadType.LOWER, 0), t.pitch_of(1, BeadType.LOWER, 0)
    (60, 62)
    >>> t.pitch_of(1, BeadType.LOWER, 2)      # 仍落在 C 五声内（G4），不随弦移调
    67
    """
    spec: AbacusSpec
    scale: Scale
    root_midi: int = 60          # C4

    def __post_init__(self):
        beads = self.spec.upper_count + self.spec.lower_count
        if len(self.scale.semitones) != beads:
            raise ValueError(
                f"音阶数({len(self.scale.semitones)})须等于每档珠数({beads})："
                f"一四珠配五声、二五珠配七声（收到 {self.scale.cn}）")
        if not 0 <= self.root_midi <= 115:
            raise ValueError(f"root_midi 须 0..115，得到 {self.root_midi}")

    def degree_of(self, bead_type: BeadType, ordinal: int) -> int:
        """珠的计数序 = 音级：下珠从 0 数起，上珠续接。"""
        return ordinal if bead_type is BeadType.LOWER \
            else self.spec.lower_count + ordinal

    def pitch_of(self, rod_index: int, bead_type: BeadType, ordinal: int) -> int:
        n = len(self.scale.semitones)
        g = rod_index + self.degree_of(bead_type, ordinal)
        return self.root_midi + self.scale.semitones[g % n] + 12 * (g // n)


# ═══════════════════ 2. 谱面层：音符与总谱 ═══════════════════

@dataclass(frozen=True)
class NoteEvent:
    """单音事件（channel 由 rod_index 派生：一弦一通道）。"""
    onset_ms: float
    duration_ms: float
    pitch: int                   # MIDI 音高
    velocity: int                # MIDI 力度（指别派生）
    rod_index: int
    label: str                   # 人读注记，如 "宫(C4)"


@dataclass
class Score:
    """总谱：事件表 + 元数据（口诀即歌词）。"""
    title: str
    events: list[NoteEvent]
    scale: Scale
    mode: str
    program: int                 # GM 音色号
    bpm: int
    rod_count: int
    lyrics: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return max((e.onset_ms + e.duration_ms for e in self.events), default=0.0)

    def pitch_range(self) -> tuple[int, int] | None:
        if not self.events:
            return None
        return (min(e.pitch for e in self.events),
                max(e.pitch for e in self.events))

    def to_text(self) -> str:
        """人读谱面：标题/定弦/歌词 + 事件表。"""
        mode_cn = {"sustain": "盘面即和声", "pluck": "指法即韵律"}.get(self.mode, self.mode)
        rng = self.pitch_range()
        lines = [
            f"《{self.title}》—— 珠算演奏谱",
            f"音阶：{self.scale.cn}    模式：{mode_cn}    "
            f"音色：GM {self.program}    声部：{self.rod_count} 弦    速度：{self.bpm} bpm",
        ]
        if self.lyrics:
            lines.append("歌词（口诀）：" + "；".join(self.lyrics))
        lines.append(f"{'onset':>7} {'dur':>6}  {'弦':>3}  {'音':<10} vel")
        for ev in sorted(self.events, key=lambda e: (e.onset_ms, e.rod_index)):
            lines.append(f"{ev.onset_ms:>7.0f} {ev.duration_ms:>6.0f}  "
                         f"{ev.rod_index:>3}  {ev.label:<10} {ev.velocity:>3}")
        if rng:
            lines.append(f"时长 {self.duration_ms / 1000:.1f}s    事件 {len(self.events)}    "
                         f"音域 {note_name(rng[0])}..{note_name(rng[1])}")
        return "\n".join(lines)


# ═══════════════════ 3. 演奏层：Sonifier / MelodyPlayer ═══════════════════

FINGER_VELOCITY = {Finger.THUMB: 100, Finger.MIDDLE: 88, Finger.INDEX: 80}
ACCENT_VELOCITY = 18            # 拍点重音加成（每 accent_every 步）
GHOST_VELOCITY_DROP = 24         # 拨去=幽灵音的力度衰减


@dataclass(frozen=True)
class MusicConfig:
    """乐器化配置（可听性默认值经试听校准）。"""
    scale: Scale = PENTATONIC
    root_midi: int = 60          # C4
    mode: str = "sustain"        # sustain=盘面即和声 / pluck=指法即韵律
    strum_ms: int = 25           # 联拨内琶音错峰（联拨物理上非严格同时）
    gap_ms: int = 60             # 步间呼吸
    sustain_tail_ms: int = 1200  # 终盘余音
    sustain_max_ms: int = 2000   # 单音持续封顶=制音上限：防大数盘面音簇浑浊
    accent_every: int = 4        # 每 N 步一记重音（拍点感；0=关）
    phrase_every: int = 0        # 每 N 步插一句读休止（乐句呼吸；0=关）
    phrase_gap_ms: int = 420    # 句读休止时长
    pluck_min_ms: int = 200      # pluck 音长下限：音要成形的最低时长
    program: int = 108           # GM #108 = Kalimba 卡林巴（五声亲和；Koto=107 可配）
    bpm: int = 100

    def __post_init__(self):
        if self.mode not in ("sustain", "pluck"):
            raise ValueError(f"mode 须 sustain/pluck，得到 {self.mode!r}")
        nums = (self.strum_ms, self.gap_ms, self.sustain_tail_ms, self.sustain_max_ms,
                self.accent_every, self.phrase_every, self.phrase_gap_ms,
                self.pluck_min_ms, self.bpm)
        if min(nums) < 0:
            raise ValueError("各时长/周期参数须非负")
        if not 0 <= self.program <= 127:
            raise ValueError(f"program 须 0..127，得到 {self.program}")


class Sonifier:
    """运算审计链（Operation）→ 乐谱（Score）。

    事件来源 = steps 的模板动作（带指法/时长，经 _bind 绑定到具体珠），
    并与 state_delta 的变化珠数逐步对账——谱面与审计链互为印证。
    """

    def __init__(self, config: MusicConfig | None = None):
        self.config = config or MusicConfig()

    def sonify(self, operation: Operation, title: str | None = None) -> Score:
        cfg = self.config
        abacus = operation.abacus
        tuning = Tuning(abacus.spec, cfg.scale, cfg.root_midi)
        movements, end_ms = self._collect_movements(operation, tuning)
        events = self._sustain(movements, end_ms, tuning) if cfg.mode == "sustain" \
            else self._pluck(movements, tuning)
        return Score(title or self._default_title(operation), events, cfg.scale,
                     cfg.mode, cfg.program, cfg.bpm, abacus.spec.rod_count,
                     [s.rhyme.rhyme_text for s in operation.steps])

    # —— 内部 ——

    @staticmethod
    def _default_title(operation: Operation) -> str:
        sym = {"ADD": "+", "SUB": "−", "MUL": "×", "DIV": "÷"}[operation.op_type.name]
        title = f"{operation.operand_a} {sym} {operation.operand_b} = {operation.expected_result}"
        if operation.op_type.name == "DIV" and operation.remainder:
            title += f" 余{operation.remainder}"
        return title

    def _collect_movements(self, operation: Operation,
                           tuning: Tuning) -> tuple[list[tuple], float]:
        """逐步绑定珠并铺时间轴。返回 ((onset, rod_index, bead, atom, step_no) 列表, 末端 ms)。"""
        cfg = self.config
        movements: list[tuple] = []
        t = 0.0
        for step_no, step in enumerate(operation.steps):
            bindings = self._bind(operation.abacus, step)
            n_changed = sum(len(rd.changed_beads) for rd in step.state_delta.rod_deltas)
            if len(bindings) != n_changed:
                raise InvariantViolation(
                    f"step{step.step_index} 绑定 {len(bindings)} 珠 ≠ 差分 {n_changed} 珠（审计链失配）")
            bindings.sort(key=lambda b: tuning.pitch_of(b[0].index, b[1].bead_type, b[1].ordinal))
            for k, (rod, bead, atom) in enumerate(bindings):
                movements.append((t + k * cfg.strum_ms, rod.index, bead, atom, step_no))
            t += step.action.duration_ms + cfg.gap_ms
            if cfg.phrase_every and (step_no + 1) % cfg.phrase_every == 0:
                t += cfg.phrase_gap_ms               # 句读休止：乐句呼吸
        return movements, t

    @staticmethod
    def _bind(abacus, step) -> list[tuple]:
        """模板原子动作 → 具体珠（与 TransitionEngine._resolve 的绑定语义逐行同构；
        引擎绑定是过程性的不落盘、审计链只存模板，故此处复刻）。"""
        positions = {bs.bead_id: bs.position
                     for rs in step.before_state.rod_states
                     for bs in rs.upper_beads + rs.lower_beads}
        bound = []
        for atom in step.action.atomic_actions:
            rod = abacus.rod_at(step.base_rod_index + atom.rod_offset)
            if atom.bead_type is BeadType.UPPER or atom.addressing is BeadAddressing.FIXED:
                bead = rod.bead(atom.bead_type, atom.ordinal)
            elif atom.addressing is BeadAddressing.NEXT_INACTIVE:
                bead = next((b for b in rod.lower_beads
                             if positions[b.bead_id] is BeadPosition.RESTING), None)
                if bead is None:
                    raise InvariantViolation(f"档{rod.index}下珠已满无处可推（绑定失配）")
            else:                                  # LAST_ACTIVE
                bead = next((b for b in reversed(rod.lower_beads)
                             if positions[b.bead_id] is BeadPosition.ENGAGED), None)
                if bead is None:
                    raise InvariantViolation(f"档{rod.index}下珠已空无处可拨（绑定失配）")
            positions[bead.bead_id] = atom.after_position
            bound.append((rod, bead, atom))
        return bound

    def _sustain(self, movements: list[tuple], end_ms: float,
                 tuning: Tuning) -> list[NoteEvent]:
        """盘面即和声：靠梁=起音，离梁=收音；终盘靠梁珠=尾和弦。
        单音持续封顶 sustain_max_ms——大数盘面的长音等于自然制音，音簇不糊。"""
        cfg = self.config
        events: list[NoteEvent] = []
        open_notes: dict[tuple[int, int], tuple] = {}
        for onset, rod_index, bead, atom, step_no in movements:
            key = (rod_index, bead.bead_id)
            pitch = tuning.pitch_of(rod_index, bead.bead_type, bead.ordinal)
            deg_name = tuning.scale.degree_names[tuning.degree_of(bead.bead_type, bead.ordinal)]
            vel = self._velocity(atom.finger, step_no)
            if atom.action_type is ActionType.ENGAGE:
                if key in open_notes:
                    raise InvariantViolation("同珠两次靠梁而未离梁，审计链不合法")
                open_notes[key] = (onset, pitch, vel, rod_index, deg_name)
            else:
                rec = open_notes.pop(key, None)
                if rec is None:                    # 防御：正常审计链从空盘起算，不会出现
                    continue
                o, p, v, r, deg = rec
                events.append(NoteEvent(o, min(max(1.0, onset - o), cfg.sustain_max_ms),
                                        p, v, r, f"{deg}({note_name(p)})"))
        close_at = end_ms + cfg.sustain_tail_ms   # 终盘余音
        for o, p, v, r, deg in open_notes.values():
            events.append(NoteEvent(o, min(max(1.0, close_at - o), cfg.sustain_max_ms),
                                    p, v, r, f"{deg}({note_name(p)})"))
        return events

    def _pluck(self, movements: list[tuple], tuning: Tuning) -> list[NoteEvent]:
        """指法即韵律：每次拨珠=一颗音；拨去=幽灵弱音。
        音长 = max(物理时长, pluck_min_ms)——120ms 的咔嗒不成音，留足成形时间。"""
        cfg = self.config
        events: list[NoteEvent] = []
        for onset, rod_index, bead, atom, step_no in movements:
            pitch = tuning.pitch_of(rod_index, bead.bead_type, bead.ordinal)
            deg_name = tuning.scale.degree_names[tuning.degree_of(bead.bead_type, bead.ordinal)]
            vel = self._velocity(atom.finger, step_no)
            if atom.action_type is ActionType.RELEASE:
                vel = max(1, vel - GHOST_VELOCITY_DROP)
            events.append(NoteEvent(onset, max(atom.duration_ms, cfg.pluck_min_ms), pitch,
                                    vel, rod_index, f"{deg_name}({note_name(pitch)})"))
        return events

    def _velocity(self, finger: Finger, step_no: int) -> int:
        """指法力度 + 拍点重音（每 accent_every 步），封顶 127。"""
        vel = FINGER_VELOCITY[finger]
        if self.config.accent_every and step_no % self.config.accent_every == 0:
            vel += ACCENT_VELOCITY
        return min(vel, 127)


class MelodyPlayer:
    """手动演奏器：在任意形制的算盘上逐音演奏（含七珠筝）。

    每个音 = 一次靠梁 + 一次离梁，全程走 TransitionEngine（物理合法、弹完归零）。
    notes: (档号, 音级, 时值ms) 序列——音级即珠的计数序（下珠从 0 数起，上珠续接）。
    """

    def __init__(self, abacus, config: MusicConfig | None = None):
        self.abacus = abacus
        self.config = config or MusicConfig()
        self.tuning = Tuning(abacus.spec, self.config.scale, self.config.root_midi)
        self.state: AbacusState | None = None        # 末次演奏终态（弹完归零可断言）

    def _bead(self, rod_index: int, degree: int):
        spec = self.abacus.spec
        size = spec.upper_count + spec.lower_count
        if not 0 <= degree < size:
            raise ValueError(f"音级须 0..{size - 1}，得到 {degree}")
        rod = self.abacus.rod_at(rod_index)
        if degree < spec.lower_count:
            return rod.bead(BeadType.LOWER, degree)
        return rod.bead(BeadType.UPPER, degree - spec.lower_count)

    def play(self, notes: Sequence[tuple[int, int, float]],
             gap_ms: int | None = None, title: str = "旋律") -> Score:
        cfg = self.config
        gap = cfg.gap_ms if gap_ms is None else gap_ms
        state = AbacusState.empty(self.abacus)
        events: list[NoteEvent] = []
        t = 0.0
        for rod_index, degree, note_ms in notes:
            if note_ms <= 0:
                raise ValueError(f"时值须 > 0，得到 {note_ms}")
            bead = self._bead(rod_index, degree)
            pitch = self.tuning.pitch_of(rod_index, bead.bead_type, bead.ordinal)
            deg_name = self.tuning.scale.degree_names[degree]
            finger = Finger.MIDDLE if bead.bead_type is BeadType.UPPER else Finger.THUMB
            events.append(NoteEvent(t, note_ms, pitch, FINGER_VELOCITY[finger],
                                    rod_index, f"{deg_name}({note_name(pitch)})"))
            for action_type, f in ((ActionType.ENGAGE, finger),
                                   (ActionType.RELEASE,
                                    Finger.MIDDLE if bead.bead_type is BeadType.UPPER
                                    else Finger.INDEX)):
                atom = AtomicAction(action_type, f, bead.bead_type, bead.ordinal)
                state, _ = TransitionEngine.reduce(
                    state, CompositeAction(action_type.value, "play", (atom,)),
                    self.abacus, rod_index)
            t += note_ms + gap
        self.state = state
        return Score(title, events, cfg.scale, "melody", cfg.program, cfg.bpm,
                     self.abacus.spec.rod_count)


# ═══════════════════ 4. MIDI 写出层（纯标准库） ═══════════════════

def _vlq(value: int) -> bytes:
    """MIDI 变长数（big-endian 7bit 组）。

    >>> _vlq(0)
    b'\\x00'
    >>> _vlq(128)
    b'\\x81\\x00'
    """
    if value < 0:
        raise ValueError("VLQ 须非负")
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(chunks))


def _channel(rod_index: int) -> int:
    """档号 → MIDI 通道（一弦一通道，方便分弦静音；跳过 9 号鼓通道）。"""
    if not 0 <= rod_index <= 13:
        raise ValueError(f"一弦一通道最多支持 14 弦（跳过鼓通道后），档号越界: {rod_index}")
    return rod_index if rod_index < 9 else rod_index + 1


def write_midi(score: Score, path: str | Path) -> Path:
    """总谱 → 标准 MIDI 文件（format 0，480 ticks/四分音符）。纯标准库。"""
    tpb = 480
    ms2tick = tpb * score.bpm / 60_000.0
    raw: list[tuple[int, int, bytes]] = []
    for ev in score.events:
        ch = _channel(ev.rod_index)
        if not 0 <= ev.pitch <= 127:
            raise ValueError(f"音高越界 {ev.pitch}（定弦根音过高？）")
        if not 1 <= ev.velocity <= 127:
            raise ValueError(f"力度越界 {ev.velocity}")
        on = round(ev.onset_ms * ms2tick)
        off = max(round((ev.onset_ms + ev.duration_ms) * ms2tick), on + 1)
        raw.append((on, 2, bytes((0x90 | ch, ev.pitch, ev.velocity))))
        raw.append((off, 1, bytes((0x80 | ch, ev.pitch, 64))))

    us_per_qn = round(60_000_000 / score.bpm)
    head: list[tuple[int, int, bytes]] = [
        (0, 0, b"\xff\x51\x03" + us_per_qn.to_bytes(3, "big")),      # tempo
    ]
    name = score.title[:40].encode("utf-8")
    head.append((0, 0, b"\xff\x03" + _vlq(len(name)) + name))        # track name
    for ch in sorted({_channel(e.rod_index) for e in score.events}):
        head.append((0, 0, bytes((0xC0 | ch, score.program))))       # program change
        head.append((0, 0, bytes((0xB0 | ch, 91, 90))))             # CC91 混响：补空间感

    track = bytearray()
    last = 0
    for tick, _order, msg in sorted(head + raw, key=lambda e: (e[0], e[1], e[2])):
        track += _vlq(tick - last) + msg
        last = tick
    track += b"\x00\xff\x2f\x00"                                      # end of track

    data = (b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb)
            + b"MTrk" + struct.pack(">I", len(track)) + track)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ═══════════════════ 5. 演示 ═══════════════════

def pi_digits(n: int) -> str:
    """π 的前 n 位数字串（含整数位 3）。Machin 公式 + decimal 高精度，纯标准库。

    供数字流实验用（累加/拨入）；数字源本身零依赖、确定性可复现。

    >>> pi_digits(10)
    '3141592653'
    """
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = n + 15
        one = Decimal(1)

        def atan_inv(x: int) -> Decimal:           # atan(1/x) 交错级数
            x2 = Decimal(x) * Decimal(x)
            term = one / Decimal(x)
            total, k = term, 1
            while term > one.scaleb(-ctx.prec + 2):
                term /= x2
                total += (-term if k % 2 else term) / (2 * k + 1)
                k += 1
            return total

        pi = 16 * atan_inv(5) - 4 * atan_inv(239)   # Machin: π = 16atan(1/5) − 4atan(1/239)
        return str(pi).replace(".", "")[:n]


def _demo_operations(out: Path) -> list[Path]:
    """东方五声（一四珠）：算算术，顺便出唱片。"""
    from evaluate.abacus import (
        Abacus, AbacusSpec, ArithmeticComposer, OpType, RhymeResolver,
        build_addition_rhymes, build_subtraction_rhymes,
    )
    abacus = Abacus.standard(AbacusSpec(1, 4, 13, 10))
    composer = ArithmeticComposer(
        RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes()))
    tag = {OpType.ADD: "add", OpType.SUB: "sub", OpType.MUL: "mul", OpType.DIV: "div"}
    files: list[Path] = []
    for op_type, a, b in [(OpType.ADD, 47, 38), (OpType.ADD, 99, 1),
                          (OpType.MUL, 12, 34), (OpType.DIV, 100, 7)]:
        operation = composer.compose(op_type, abacus, a, b)
        for mode in ("sustain", "pluck"):
            score = Sonifier(MusicConfig(mode=mode)).sonify(operation)
            files.append(write_midi(score, out / f"{tag[op_type]}_{a}_{b}_{mode}.mid"))
        if (op_type, a, b) == (OpType.ADD, 47, 38):
            score = Sonifier().sonify(operation)
            (out / "score_add_47_38.txt").write_text(score.to_text(), encoding="utf-8")
            files.append(out / "score_add_47_38.txt")
            print(score.to_text(), end="\n\n")
    return files


def _demo_digit_streams(out: Path) -> list[Path]:
    """数字流实验：π 逐位累加 vs 1/7 循环节 vs 拨入 π（和弦级数）。

    韵律不在数字里（π 的各位猜想正规=最大熵），在累加过程里：
    进位是心跳（均值 4.5/位 → 平均每 ~2.2 位一次进位）、密度随「数字×盘面」起伏、
    个位档做 mod-10 随机游走。1/7 循环节位和 27 ≡ 7 (mod 10)，个位乐句相位
    每节漂 7、十节回归，而进位缓慢抬高高档——永不精确重复（Reich 式 phase）。
    """
    from collections import Counter
    from evaluate.abacus import (
        Abacus, AbacusSpec, ArithmeticComposer, RhymeResolver,
        build_addition_rhymes, build_subtraction_rhymes,
    )
    abacus = Abacus.standard(AbacusSpec(1, 4, 13, 10))
    composer = ArithmeticComposer(
        RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes()))
    files: list[Path] = []

    # ① π 前 100 位逐位累加——韵律住在累加里，不在 π 的数字里；每 10 位一句读
    pi = pi_digits(100)
    op = composer.compose_digit_stream(abacus, [int(c) for c in pi])
    cats = Counter(s.rhyme.category.cn for s in op.steps)
    print(f"π 前100位累加：和={op.expected_result}，{len(op.steps)} 步，口诀分布={dict(cats)}")
    for mode in ("sustain", "pluck"):
        score = Sonifier(MusicConfig(mode=mode, phrase_every=10)).sonify(
            op, title=f"π×100位 累加（和{op.expected_result}）")
        files.append(write_midi(score, out / f"pi_accum_{mode}.mid"))
        if mode == "sustain":
            (out / "score_pi_accum.txt").write_text(score.to_text(), encoding="utf-8")
            files.append(out / "score_pi_accum.txt")

    # ② 1/7 循环节 ×10：个位乐句相位每节漂 7、十节回归；每循环节一句读
    op7 = composer.compose_digit_stream(abacus, [int(c) for c in "142857" * 10])
    for mode in ("sustain", "pluck"):
        score = Sonifier(MusicConfig(mode=mode, phrase_every=6)).sonify(
            op7, title=f"1/7 循环节×10（和{op7.expected_result}）")
        files.append(write_midi(score, out / f"seventh_loop_{mode}.mid"))

    # ③ 拨入 π：不累加，π 的数字串直接成为 13 弦上的和弦级数（π+0=π 本身）
    pi13 = int(pi[:13])
    op_dial = composer.compose_addition(abacus, pi13, 0)
    score = Sonifier().sonify(op_dial, title=f"拨入 π（前13位）")
    files.append(write_midi(score, out / "pi_dial_sustain.mid"))
    return files


def _demo_zither(out: Path) -> list[Path]:
    """西方七声（二五珠筝）：不算术，纯演奏。"""
    from evaluate.abacus import Abacus, AbacusSpec
    zither = Abacus.standard(AbacusSpec(2, 5, 13, 10))
    player = MelodyPlayer(zither, MusicConfig(scale=HEPTATONIC))
    # 《茉莉花》起句 mi mi sol la do' do' la | sol sol la sol（弦号即音，级 0）
    tune = [(2, 0, 420), (2, 0, 420), (4, 0, 560), (5, 0, 560),
            (7, 0, 420), (7, 0, 420), (5, 0, 700),
            (4, 0, 420), (4, 0, 420), (5, 0, 700), (4, 0, 900)]
    return [write_midi(player.play(tune, title="茉莉花（七珠筝）"),
                       out / "heptatonic_melody.mid")]


def demo() -> list[Path]:
    """端到端演示：运算演奏 + 数字流实验 + 七珠筝旋律 → output/music/。"""
    out = Path(__file__).resolve().parent / "output" / "music"
    out.mkdir(parents=True, exist_ok=True)
    files = _demo_operations(out)        # 五声：算算术，顺便出唱片
    files += _demo_digit_streams(out)    # π 累加 / 1/7 循环节 / 拨入 π
    files += _demo_zither(out)           # 七声：七珠筝旋律
    return files


if __name__ == "__main__":
    demo()
