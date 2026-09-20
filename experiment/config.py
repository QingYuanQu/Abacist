"""`config.yaml` 的唯一读写入口 —— 实验配置的格式、校验与错误报告。

为什么单独一个模块
------------------
旧格式把配置拆成 `material.csv` + `method.csv` + `config/*.json` 共 6~8 个文件，
再按 `id` 拼成一个扁平字符串命名空间交给 `Trial.from_row` 用"探测式"读取。
后果是**没有 schema，拼错的键永远静默**：`heads_per_layer` / `head_dims_per_layer`
/ `rope_theta_per_head` 三列躺了很久，不报错、不生效、无日志。

因此本模块的硬性约定：

1. **YAML 键 = dataclass 字段名**，未登记的键一律报错而非忽略。
2. **只此一处认识文件格式**。`experiment/domain.py` 是纯数据定义，`model.domain` 是纯模型参数，
   二者都不碰 YAML；`loader.py` 只负责磁盘布局与聚合根装配。
3. **校验一次性收集全部错误再抛出**，避免"改一处、跑一次"。
4. 作用域严格两级，不可混淆：
      实验级  name / data_seed / brain / pos_emb / train / eval
      trial 级 name / heads / head_dims / theta_per_head / data / method
   trial 级旋钮可写在 `trial_defaults` 作公共基值，在 `trials[]` 里写差量；
   trial 内出现实验级键（或反之）→ 报错。

头结构的三条正交轴
------------------
    heads          逐层头数，长度 = num_hidden_layers（每层维度 = hidden_size // heads[l]）—— 已实现
    head_dims      逐层、逐头的维度，长度 = num_hidden_layers（如 [[32,16,8,8], ...]，每层求和 = hidden_size）
    theta_per_head 逐头 RoPE 基频，长度 = 每层头数（如 [1e7, 1e6, 1e3, 1e2]），要求各层头数一致

后两条是"同层位置编码有粗有细"的另外两种实现路线（异质头宽 / 同宽异频），
模型层尚未支持，因此**允许写进配置但拒绝加载**（UnsupportedFeature）：
配置是研究意图的忠实记录，宁可响亮失败，也不能像旧格式那样静默跑出错误结论。
"""

import os
import re
import types
from dataclasses import dataclass, fields
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from experiment.domain import EvalConfig, Data, Method, TrainConfig
from model.domain import BrainConfig, PosEmbConfig
# 只取"名字清单"（torch 无关）。校验一份 YAML 不该拉起 torch：
# 既慢，又会把 OpenMP 运行库带进进程，导致此后 matplotlib 保存图片直接中止。
from model.registry_names import ATTN_TYPES, POS_EMB_TYPES, ROPE_TYPES

# ==================== 键白名单 ====================

SPEC_FIELDS = ("name", "data_seed", "brain", "pos_emb", "train", "eval",
               "trial_defaults", "trials")
TRIAL_FIELDS = ("name", "heads", "head_dims", "theta_per_head", "data", "method")
TRIAL_DEFAULT_FIELDS = ("heads", "head_dims", "theta_per_head", "data", "method")
SECTION_CLASSES = {"brain": BrainConfig, "pos_emb": PosEmbConfig,
                   "train": TrainConfig, "eval": EvalConfig}
# 实验级 section 中禁止出现的字段（它们属于 trial 级旋钮）
SECTION_EXCLUDE = {"brain": {"heads_per_layer"}}

# ==================== 枚举 ====================

DATA_TYPES = ("dataset", "expr", "bead")
INPUT_FORMATS = ("infix", "prefix", "postfix")
PARSE_MODES = ("direct", "fixed", "pre", "post", "prepost", "prepost_stack", "none")
EVAL_MODES = ("none", "digit", "abacus", "no_ans")
ABACUS_STYLES = ("song", "ming")
ACC_MODES = ("acc", "acc_ans", "acc_think")
# dataset 走 project_structure_a()，它只认 pre / 其余一律当 post —— 故显式收紧
DATASET_PARSE_MODES = ("pre", "post")

# 各 data.type 的必填字段（其余字段按枚举校验）
REQUIRED_BY_TYPE = {
    "dataset": ("source",),
    "expr": ("ops", "repeat", "start", "end"),
    "bead": ("start", "end"),
}

# 已声明但模型层尚未支持的能力 → 加载即拒绝，避免静默跑出错误结论
UNSUPPORTED_AXES = {
    "head_dims": "逐头维度（异质头宽）：需要 attention 支持每头独立的 q/k/v 投影与 RMSNorm",
    "theta_per_head": "逐头 RoPE 基频：需要 RopeEmbedding 支持每个头各自一套频率",
}


class ConfigError(Exception):
    """配置校验失败 —— 一次性携带全部错误。"""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("配置校验失败：\n  - " + "\n  - ".join(self.errors))


class UnsupportedFeature(Exception):
    """配置合法，但依赖的能力尚未实现（已声明未实现）。"""


class _Errors:
    """错误累加器：所有问题一次性收集，避免改一处跑一次。"""

    def __init__(self, where: str):
        self.where = where
        self.items: list[str] = []

    def add(self, msg: str) -> None:
        self.items.append(f"{self.where}: {msg}")

    def scoped(self, sub: str) -> "_Errors":
        """派生一个子作用域（错误仍汇总到同一个列表）。"""
        child = _Errors(f"{self.where}/{sub}")
        child.items = self.items
        return child


# ==================== 规格对象 ====================

@dataclass
class TrialSpec:
    """单个 trial 的配置（尚未装配磁盘路径，因此与 Trial 区分开）。"""
    id: int
    name: str
    heads: list[int]
    data: Data
    method: Method


@dataclass
class ExperimentSpec:
    """一份 config.yaml 的完整内容（纯数据，可脱离文件系统构造）。"""
    name: str
    data_seed: int
    brain: BrainConfig
    pos_emb: PosEmbConfig
    train: TrainConfig
    eval: EvalConfig
    trials: list[TrialSpec]


# ==================== 校验工具 ====================

def _check_keys(errs: _Errors, raw: Any, allowed: tuple[str, ...]) -> bool:
    """检查映射的键是否在白名单内；返回是否是合法的映射。"""
    if not isinstance(raw, dict):
        errs.add(f"应为映射（mapping），实际是 {type(raw).__name__}")
        return False
    for k in raw:
        if k not in allowed:
            errs.add(f"未知键 {k!r}（可用: {', '.join(allowed)}）")
    return True


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---- 标量类型归一 ----
#
# 为什么必须做：dataclass **不强制**字段类型，`PosEmbConfig(theta="1.0e6")` 能构造成功，
# 然后在 RoPE 里以 `str ** Tensor` 的形式炸掉几十层之下的调用栈。
# 而 YAML 恰好很容易产出字符串：PyYAML 实现的是 YAML 1.1，浮点指数**必须带符号**，
# 于是 `theta: 1.0e6` 是字符串，`theta: 1.0e-3` 才是浮点。这类"看起来是数字的字符串"
# 必须在配置入口一次性拦住或转正，不能让它逃逸到运行期。

_FAIL = object()
_HINT_CACHE: dict[type, dict[str, Any]] = {}
_YAML_TIP = ("（PyYAML 按 YAML 1.1 解析：1e6 / 1.0e6 会被当成字符串，"
             "请写 1000000.0 或 1.0e+6）")


def _hint_name(hint: Any) -> str:
    """把类型注解渲染成可读名（用于错误信息）。"""
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return " | ".join(_hint_name(a) for a in get_args(hint))
    if origin is not None:
        inner = ", ".join(_hint_name(a) for a in get_args(hint))
        return f"{getattr(origin, '__name__', origin)}[{inner}]"
    return getattr(hint, "__name__", str(hint))


def _unwrap_optional(hint: Any) -> tuple[Any, bool]:
    """Optional[X] → (X, True)；其余 → (hint, False)。"""
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == len(args) - 1:
            return non_none[0], True
    return hint, False


def _coerce_value(v: Any, hint: Any) -> Any:
    """按注解把 YAML 标量归一到声明类型；不可归一 → _FAIL。"""
    core, allow_none = _unwrap_optional(hint)
    if v is None:
        return None if allow_none else _FAIL
    if core is bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return v.strip().lower() == "true"
        return _FAIL
    if core is int:
        if isinstance(v, bool):
            return _FAIL
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                return _FAIL
        return _FAIL
    if core is float:
        if isinstance(v, bool):
            return _FAIL
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                return _FAIL
        return _FAIL
    if core is str:
        return v if isinstance(v, str) else _FAIL
    return v          # list / dict 等复合类型由各自的专项校验负责


def _field_types(cls) -> dict[str, Any]:
    if cls not in _HINT_CACHE:
        _HINT_CACHE[cls] = get_type_hints(cls)
    return _HINT_CACHE[cls]


def _coerce_fields(errs: _Errors, cls, data: dict) -> dict:
    """按 dataclass 字段类型归一 data；未知键丢弃（已由 _check_keys 报错）。"""
    hints = _field_types(cls)
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k not in hints:
            continue
        new = _coerce_value(v, hints[k])
        if new is _FAIL:
            tip = _YAML_TIP if isinstance(v, str) and hints[k] in (int, float) else ""
            errs.add(f"{k}: 应为 {_hint_name(hints[k])}，实际 {v!r}{tip}")
            continue
        out[k] = new
    return out


def _int_list(raw: Any, errs: _Errors, what: str, min_value: int = 1) -> list[int] | None:
    if not isinstance(raw, list) or not raw:
        errs.add(f"{what}: 应为非空整数列表")
        return None
    for i, v in enumerate(raw):
        if not _is_int(v) or v < min_value:
            errs.add(f"{what}[{i}]: 应为 >= {min_value} 的整数，实际 {v!r}")
            return None
    return list(raw)


def _parse_section(errs: _Errors, raw: dict, key: str, cls, required: bool):
    """解析 brain / pos_emb / train / eval 四个实验级 section。

    失败时返回 None（错误已记入 errs）。**不能**回退成 `cls()`：像 EvalConfig 这种
    无默认值的类根本构造不出来，会在报错路径上再抛一个无关的 TypeError。
    """
    sub = raw.get(key)
    if sub is None:
        if required:
            errs.add(f"{key}: 必填")
        return None
    child = errs.scoped(key)
    allowed = tuple(f.name for f in fields(cls))
    excluded = SECTION_EXCLUDE.get(key, set())
    allowed = tuple(k for k in allowed if k not in excluded)
    if not _check_keys(child, sub, allowed):
        return None
    for k in excluded & set(sub):
        child.add(f"{k} 是 trial 级旋钮，请写在 trials[].{k}")
    try:
        return cls(**_coerce_fields(child, cls, sub))
    except TypeError as e:  # 必填字段缺失等
        child.add(f"{key} 组装失败: {e}")
        return None


def _parse_data(errs: _Errors, raw: Any) -> Data | None:
    allowed = tuple(f.name for f in fields(Data))
    if not _check_keys(errs, raw, allowed):
        return None
    m = _coerce_fields(errs, Data, raw)
    type_ = m.get("type")
    if type_ not in DATA_TYPES:
        errs.add(f"type={type_!r} 未知（可用: {', '.join(DATA_TYPES)}）")
    else:
        for k in REQUIRED_BY_TYPE[type_]:
            if k not in m:
                errs.add(f"{type_} 类型缺少必填字段 {k!r}")

    if m.get("input_format", "infix") not in INPUT_FORMATS:
        errs.add(f"input_format={m.get('input_format')!r} 未知（可用: {', '.join(INPUT_FORMATS)}）")
    parse = m.get("parse")
    if parse is not None and parse not in PARSE_MODES:
        errs.add(f"parse={parse!r} 未知（可用: {', '.join(PARSE_MODES)}）")
    elif type_ == "dataset" and parse not in DATASET_PARSE_MODES:
        errs.add(f"parse={parse!r} 不合法：dataset 走纯结构投影，只支持 "
                 f"{' | '.join(DATASET_PARSE_MODES)}（其余值会被静默当成 post）")
    eval_ = m.get("eval")
    if eval_ is not None and eval_ not in EVAL_MODES:
        errs.add(f"eval={eval_!r} 未知（可用: {', '.join(EVAL_MODES)}）")
    if m.get("abacus_style", "song") not in ABACUS_STYLES:
        errs.add(f"abacus_style={m.get('abacus_style')!r} 未知（可用: {', '.join(ABACUS_STYLES)}）")
    if m.get("source") and type_ != "dataset":
        errs.add(f"source 仅 dataset 类型使用（当前 type={type_!r}）")
    try:
        return Data(**m)
    except TypeError as e:  # 必填字段缺失等（类型错误已逐个报过）
        errs.add(f"data 组装失败: {e}")
        return None


def _parse_method(errs: _Errors, raw: Any) -> Method | None:
    allowed = tuple(f.name for f in fields(Method))
    if not _check_keys(errs, raw, allowed):
        return None
    for k in allowed:
        if k not in raw:
            errs.add(f"method 缺少必填字段 {k!r}")
    try:
        return Method(**_coerce_fields(errs, Method, raw))
    except TypeError as e:
        errs.add(f"method 组装失败: {e}")
        return None


def _resolve_heads(errs: _Errors, where: str, heads_raw: Any, head_dims_raw: Any,
                   brain: BrainConfig) -> list[int] | None:
    """把 heads / head_dims 两条互斥写法归一为逐层头数列表。"""
    num_layers = brain.num_hidden_layers
    hidden = brain.hidden_size

    if heads_raw is not None and head_dims_raw is not None:
        errs.add("heads 与 head_dims 互斥，只能给一个（head_dims 是 heads 的逐头展开写法）")
        return None

    if head_dims_raw is not None:
        if not isinstance(head_dims_raw, list) or len(head_dims_raw) != num_layers:
            errs.add(f"head_dims: 应为长度 {num_layers}（= num_hidden_layers）的列表，"
                     f"实际长度 {len(head_dims_raw) if isinstance(head_dims_raw, list) else '非法'}")
            return None
        heads = []
        for l, dims in enumerate(head_dims_raw):
            sub = errs.scoped(f"head_dims[{l}]")
            if not isinstance(dims, list) or not dims:
                sub.add("应为非空整数列表（该层每个头的维度）")
                return None
            for h in dims:
                if not _is_int(h) or h < 1:
                    sub.add(f"头维度应为 >= 1 的整数，实际 {h!r}")
                    return None
            if sum(dims) != hidden:
                sub.add(f"各头维度之和 {sum(dims)} != hidden_size {hidden}")
                return None
            heads.append(len(dims))
        return heads

    if heads_raw is None:
        return None          # 交给调用方取均匀默认
    heads = _int_list(heads_raw, errs, "heads")
    if heads is None:
        return None
    if len(heads) != num_layers:
        errs.add(f"heads: 长度 {len(heads)} != num_hidden_layers {num_layers}")
        return None
    for l, h in enumerate(heads):
        if hidden % h != 0:
            errs.add(f"heads[{l}]={h} 不能整除 hidden_size={hidden}（head_dim 会被截断）")
    return heads


# ==================== 解析入口 ====================

def parse_spec(raw: Any, where: str) -> ExperimentSpec:
    """把已加载的 YAML 映射解析为 ExperimentSpec（渲染键名校验错误）。"""
    errs = _Errors(where)
    if not _check_keys(errs, raw, SPEC_FIELDS):
        raise ConfigError(errs.items)

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        errs.add("name: 必填且为非空字符串")
    elif name != where:
        errs.add(f"name={name!r} 与目录名 {where!r} 不一致（配置与产物必须能对上）")

    brain = _parse_section(errs, raw, "brain", BrainConfig, required=True) or BrainConfig()
    pos_emb = _parse_section(errs, raw, "pos_emb", PosEmbConfig, required=False) or PosEmbConfig()
    train = _parse_section(errs, raw, "train", TrainConfig, required=True)
    eval_cfg = _parse_section(errs, raw, "eval", EvalConfig, required=True)

    # 枚举校验（三个注册表是唯一事实来源，不在这里另抄一份清单）
    if brain.attn_type not in ATTN_TYPES:
        errs.add(f"brain.attn_type={brain.attn_type!r} 未知（可用: {', '.join(sorted(ATTN_TYPES))}）")
    if pos_emb.pos_type not in POS_EMB_TYPES:
        errs.add(f"pos_emb.pos_type={pos_emb.pos_type!r} 未知（可用: {', '.join(sorted(POS_EMB_TYPES))}）")
    if pos_emb.rope_type not in ROPE_TYPES:
        errs.add(f"pos_emb.rope_type={pos_emb.rope_type!r} 未知（可用: {', '.join(sorted(ROPE_TYPES))}）")
    if isinstance(eval_cfg, EvalConfig) and eval_cfg.acc_mode not in ACC_MODES:
        errs.add(f"eval.acc_mode={eval_cfg.acc_mode!r} 未知（可用: {', '.join(ACC_MODES)}）")

    data_seed = raw.get("data_seed", 42)
    if not _is_int(data_seed):
        errs.add(f"data_seed 应为整数，实际 {data_seed!r}")
        data_seed = 42

    # ---- trial_defaults ----
    defaults = raw.get("trial_defaults") or {}
    if not _check_keys(errs.scoped("trial_defaults"), defaults, TRIAL_DEFAULT_FIELDS):
        defaults = {}

    # ---- trials ----
    trials_raw = raw.get("trials")
    trials: list[TrialSpec] = []
    if not isinstance(trials_raw, list) or not trials_raw:
        errs.add("trials: 必填且为非空列表")
        trials_raw = []

    seen_names: dict[str, int] = {}
    unsupported: list[str] = []
    for i, t in enumerate(trials_raw):
        w = f"trials[{i}]"
        sub = errs.scoped(f"trials[{i}]")
        if not _check_keys(sub, t, TRIAL_FIELDS):
            continue

        tname = t.get("name")
        if not isinstance(tname, str) or not tname.strip():
            sub.add("name: 必填且为非空字符串")
            tname = None
        else:
            if tname in seen_names:
                sub.add(f"name={tname!r} 与 trials[{seen_names[tname]}] 重复（name 必须唯一，"
                        f"它就是 report.csv 的标签与配置指纹）")
            else:
                seen_names[tname] = i
            if re.search(r"[\\/\s]", tname):
                sub.add(f"name={tname!r} 含路径分隔符或空白，会污染产物路径")

        # 头结构三轴
        heads = _resolve_heads(sub, w, t.get("heads", defaults.get("heads")),
                               t.get("head_dims", defaults.get("head_dims")), brain)
        if heads is None:
            heads = [brain.num_attention_heads] * brain.num_hidden_layers
        theta = t.get("theta_per_head", defaults.get("theta_per_head"))
        if theta is not None:
            # 逐头基频按「头序」对齐，与层数无关 → 要求各层头数一致
            per_layer = set(heads)
            if len(per_layer) != 1:
                sub.add(f"theta_per_head 要求各层头数一致（当前 {heads}），否则头序无法对齐")
            elif not isinstance(theta, list) or len(theta) != heads[0]:
                sub.add(f"theta_per_head: 长度应与每层头数 {heads[0]} 一致"
                        f"（逐头，不是层数 {len(heads)}）")
            else:
                nums = [_coerce_value(v, float) for v in theta]
                if any(n is _FAIL or n <= 0 for n in nums):
                    sub.add("theta_per_head: 每项应为正数" + _YAML_TIP)
            unsupported.append(f"{w} 声明了 theta_per_head → {UNSUPPORTED_AXES['theta_per_head']}")
        if t.get("head_dims", defaults.get("head_dims")) is not None:
            unsupported.append(f"{w} 声明了 head_dims → {UNSUPPORTED_AXES['head_dims']}")

        # data / method：trial_defaults 作基值，trial 写差量（扁平结构浅合并）。
        # 仅当 trial 与 defaults 都提供时合并；否则以提供的那一侧为准。
        data = None
        mraw = _merge_table(defaults.get("data"), t.get("data"))
        if mraw is None:
            sub.add("缺 data，且 trial_defaults 未提供")
        else:
            data = _parse_data(sub.scoped("data"), mraw)
        method = None
        thraw = _merge_table(defaults.get("method"), t.get("method"))
        if thraw is None:
            sub.add("缺 method，且 trial_defaults 未提供")
        else:
            method = _parse_method(sub.scoped("method"), thraw)

        if tname and data and method:
            trials.append(TrialSpec(id=i, name=tname, heads=heads,
                                    data=data, method=method))

    if errs.items:
        raise ConfigError(errs.items)
    if unsupported:
        raise UnsupportedFeature(
            f"{where}: 配置里声明了模型层尚未支持的能力，已拒绝加载（不是静默忽略）：\n  - "
            + "\n  - ".join(unsupported)
        )
    return ExperimentSpec(name=name, data_seed=data_seed, brain=brain, pos_emb=pos_emb,
                          train=train, eval=eval_cfg, trials=trials)


def _merge_table(defaults, trial):
    """data/method 合并：trial_defaults 作基值，trial 写差量。

    两侧都提供时浅合并（trial 同名键覆盖 defaults；data 为扁平结构，浅合并足够），
    两侧都缺失返回 None（交由调用方报「缺 data」）。每次返回新 dict，不污染 defaults，
    故多个 trial 共享同一份 defaults 也不会串味。
    """
    if trial is None:
        return defaults
    if defaults is None:
        return trial
    return {**defaults, **trial}


def load_spec(path: str) -> ExperimentSpec:
    """读取并校验一个 config.yaml。目录名即实验名（作为校验上下文）。"""
    where = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if not os.path.isfile(path):
        raise ConfigError([f"{path}: 不存在（实验配置只有这一个输入文件）"])
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ConfigError([f"{where}: 配置为空"])
    if not isinstance(raw, dict):
        raise ConfigError([f"{where}: 顶层应为映射（mapping），实际是 {type(raw).__name__}"])
    return parse_spec(raw, where)


def clone_text(path: str, new_name: str) -> str:
    """基于已有配置生成克隆文本 —— 只替换顶层 name，保留全部注释与排版。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new_text, n = re.subn(r"^name:[^\n]*$", f"name: {new_name}", text,
                          count=1, flags=re.MULTILINE)
    if n == 0:
        raise ConfigError([f"{path}: 找不到顶层 name: 行，无法克隆"])
    return new_text


# ==================== 模板 ====================

def skeleton(name: str) -> str:
    """生成新实验的 config.yaml 模板（新实验从这里开始改）。"""
    return f"""# 实验配置 —— 唯一输入文件。
# 所有键名 = experiment/domain.py / model/domain.py 的 dataclass 字段名，写错会直接报错。
# 作用域：brain / pos_emb / train / eval 是实验级；data / method / heads 是 trial 级。
name: {name}
data_seed: 42

brain:
  hidden_size: 64
  num_attention_heads: 4
  num_hidden_layers: 5
  attn_type: sdpa

pos_emb:
  # 注意：1.0e6 在 YAML 1.1 里是「字符串」（指数必须带符号），写成 1.0e+6 或 1000000.0
  theta: 1000000.0
  pos_type: rope
  rope_type: default
  max_position_embeddings: 256

train:
  enable_validation: true
  early_stop_patience: 5
  dropout: 0.0
  train_seed: 42

eval:
  eval_batch_size: 1024
  print_interval: 1000
  acc_mode: acc
  pass_threshold: 0.95

# trial 级旋钮的公共基值；trials[] 里只写差量
trial_defaults:
  method:
    epochs: 25
    repeat_factor: 1
    batch_size: 64
    learning_rate: 1.0e-3
    shuffle: true

# id 由列表位置派生（0..N-1），不要手写
trials:
  - name: baseline
    heads: [4, 4, 4, 4, 4]
    data:
      type: expr
      ops: "+-"
      repeat: 1
      start: 0
      end: 9
      allow_negative: false
      input_format: infix
      parse: post
      eval: none
"""
