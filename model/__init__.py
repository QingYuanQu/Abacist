"""model 包 —— 对外收口导出。

包外一律从本文件取公共 API（`from model import GPT` 等），
不写 `from model.gpt import ...`，这样内部模块再拆分重组时外部无需改动。

依赖方向单向无环：
    domain / norm / attn_fn / pos_emb ← registry
    attention ← domain, norm, registry
    block     ← attention, norm, domain, registry
    gpt       ← block, norm, domain, registry

为什么是懒加载（PEP 562）
------------------------
torch 只被 gpt/block/attention 这条装配链需要，**配置层不需要**。若此处顶层
`from model.gpt import GPT`，则任何 `import model.domain`（→ 触发本文件）都会
把 torch 拖进进程。后果不只是慢：torch 会把自带 OpenMP 运行库载入进程，此后
`matplotlib.savefig` 会直接以 `OMP: Error #15` 中止——于是"读 config.yaml 画张图"
这种纯配置/分析工作也被迫跟 torch 绑在一起。
故改为延迟解析：`from model import GPT` 的写法一字不变，torch 只在真正要建模时才出现。
（同一模式见 evaluate/abacus/render/__init__.py。）
"""
__all__ = ["Attention", "Block", "FeedForward", "GPT", "repeat_kv"]

_LAZY = {
    "Attention": "model.attention",
    "repeat_kv": "model.attention",
    "Block": "model.block",
    "FeedForward": "model.block",
    "GPT": "model.gpt",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + list(_LAZY)))
