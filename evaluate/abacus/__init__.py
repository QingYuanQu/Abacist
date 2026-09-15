"""珠算领域内核包（Abacus 模拟器）。

  domain/          数据层子包：enums（异常/枚举）+ structure（结构）+ state（状态）
                            + delta（差分）+ action（动作/口诀数据类）——纯数据，零逻辑
  kernel.py        行为层：TransitionEngine / RhymeResolver / ArithmeticComposer
                          CalculationStep / Operation / AbacusEnv
  rhymes.py      口诀种子：cn_number / build_addition_rhymes / build_subtraction_rhymes
  bead_codec.py    珠态编解码（数值 ↔ AbacusState ↔ 珠态文本，旧格式兼容层）
  render/          渲染子包（唯一官方入口 = render.registry.make_render）
  music.py         算盘乐器化：审计链 → 乐谱/MIDI（数值即和声、口诀即歌词）；
                   含数字流实验（pi_digits 累加/拨入、1/7 循环节相位）
  demo.py          端到端冒烟演示（人工验收；自动化测试见 tests/）

依赖单向：domain ← rhymes ← kernel；music.py 依赖 domain + kernel；
包内模块一律从具体模块导入，不从本包根导入（否则触发包初始化循环）。

公共 API 统一在此 re-export，外部推荐 `from evaluate.abacus import ...` 短路径。
分层设计与类体系总表见 abacus/README.md。
"""
from evaluate.abacus.domain import (
    # 异常层
    AbacusError, InvariantViolation, RhymeNotFound, UnsupportedOperation,
    # 枚举层
    BeadType, BeadPosition, Finger, OpType, DigitOrder, RhymeCategory,
    ActionType, BeadAddressing,
    # 结构层
    AbacusSpec, Beam, Frame, Bead, Rod, Abacus,
    # 状态层
    BeadState, RodState, AbacusState,
    # 数值 → 状态构造（唯一实现）
    num_to_state,
    # 差分层
    BeadStateDelta, RodStateDelta, StateDelta,
    # 动作层
    AtomicAction, CompositeAction,
    # 口诀层（数据类）
    ExpectedRodDelta, Carry, Rhyme,
)
from evaluate.abacus.kernel import (
    # 行为内核层
    TransitionEngine, RhymeResolver,
    # 运算层
    ArithmeticComposer, CalculationStep, Operation,
    # 外围适配层
    AbacusEnv,
)
from evaluate.abacus.rhymes import (
    # 口诀层：中文数字词表 + 52 句加减口诀种子
    cn_number, build_addition_rhymes, build_subtraction_rhymes,
)

# 渲染：唯一官方入口 = render.registry（make_render / default_patch）
from evaluate.abacus.render.text import TextRenderer

# 珠态编解码器（数值 ↔ AbacusState ↔ 珠态文本，旧格式兼容层）
from evaluate.abacus.bead_codec import (
    num_to_bead_text, bead_text_to_num,
    state_to_bead_text, bead_text_to_state,
)

# 算盘乐器化（审计链 → 乐谱/MIDI：数值即和声、口诀即歌词、指法即节奏）
from evaluate.abacus.music import (
    MusicConfig, MelodyPlayer, NoteEvent, Scale, Score, Sonifier, Tuning,
    HEPTATONIC, PENTATONIC, note_name, pi_digits, write_midi,
)

__all__ = [
    "AbacusError", "InvariantViolation", "RhymeNotFound", "UnsupportedOperation",
    "BeadType", "BeadPosition", "Finger", "OpType", "DigitOrder", "RhymeCategory",
    "ActionType", "BeadAddressing",
    "AbacusSpec", "Beam", "Frame", "Bead", "Rod", "Abacus",
    "BeadState", "RodState", "AbacusState",
    "BeadStateDelta", "RodStateDelta", "StateDelta",
    "AtomicAction", "CompositeAction",
    "ExpectedRodDelta", "Carry", "Rhyme",
    "TransitionEngine", "RhymeResolver",
    "ArithmeticComposer", "CalculationStep", "Operation",
    "TextRenderer", "AbacusEnv",
    "MusicConfig", "MelodyPlayer", "NoteEvent", "Scale", "Score", "Sonifier",
    "Tuning", "HEPTATONIC", "PENTATONIC", "note_name", "pi_digits", "write_midi",
    "cn_number", "build_addition_rhymes", "build_subtraction_rhymes",
    "num_to_bead_text", "bead_text_to_num",
    "num_to_state", "state_to_bead_text", "bead_text_to_state",
]
