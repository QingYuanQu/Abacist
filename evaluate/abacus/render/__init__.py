"""渲染子包：统一函数式渲染接口 + 四种输出形态（纯库，无 __main__）。

渲染的唯一官方入口 = registry.make_render()：按后端名解析出统一签名
`render_fn(state, abacus, patch_h, patch_w) -> np.ndarray`，使数据生成 / 训练 /
推理 / 闭环各环节用同一调用方式切换后端（后端清单见 registry 模块 docstring）。

- text.py：TextRenderer 文本渲染器（给人看的 ASCII 盘面），零依赖，包加载时即导入；
- image.py：ImageRenderer matplotlib 渲染器（彩色 PNG/GIF，强制 Agg 离线出图）——懒加载；
- animation.py：AbacusAnimator matplotlib 实时窗口动画（常驻 ax + FuncAnimation，**不**强制 Agg）——懒加载；
- fixed.py：render_fixed 固定分辨率灰度 patch（给模型吃的机器观测，
  纯 numpy）——懒加载，VLA 数据管线无需 matplotlib；
- minimal.py：render_minimal 一珠一像素极简盘面（信息论下限的消融
  对照，纯 numpy）——懒加载；
- registry.py：make_render / default_patch / backend_names 后端注册表——懒加载；
- geometry.py：bead_centers 共享渲染几何（image/fixed 珠位同源）。

历史：曾以 render/port.py 的 Renderer Protocol 作为渲染抽象（对象式
`render(state, abacus, view)`），2026-09-07 移除——真实消费方（model_vlm、
model_vm、evaluate.bridge）全走 registry 函数式接口，Protocol 无生产实现、
无生产调用方，属于悬空抽象，删除以消除「两套渲染接口」的歧义。

示例入口（薄壳，输出统一到 abacus/output/<name>/）：
    python -m evaluate.abacus.demos text       # 逐步 ASCII 帧   → output/text/
    python -m evaluate.abacus.demos image      # PNG + GIF       → output/image/
    python -m evaluate.abacus.demos fixed      # 逐步灰度 patch  → output/fixed/
    python -m evaluate.abacus.demos minimal    # 逐步极简盘面    → output/minimal/
    python -m evaluate.abacus.demos all        # 依次跑全部后端
"""
# 注意：image/fixed/minimal/registry 一律走下面的 __getattr__ 懒加载，
# 勿在此处直接 import（否则 import 本包就会拉起 matplotlib，破坏"VLA 管线零 matplotlib"）。
from evaluate.abacus.render.text import TextRenderer

__all__ = ["TextRenderer", "ImageRenderer", "Style",
           "render_fixed", "render_minimal"]

_LAZY = {"ImageRenderer": "evaluate.abacus.render.image",
         "Style": "evaluate.abacus.render.style",
         "render_fixed": "evaluate.abacus.render.fixed",
         "render_minimal": "evaluate.abacus.render.minimal"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + list(_LAZY)))
