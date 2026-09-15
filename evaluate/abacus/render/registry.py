"""渲染后端注册表：按名字解析渲染函数，统一调用签名。

统一签名：render_fn(state, abacus, patch_h=None, patch_w=None) -> np.ndarray
适配器负责吞掉不适用的参数（如 minimal 没有 patch_h 概念），
使数据生成 / 训练 / 推理 / 闭环各环节可用同一调用方式切换后端。

内置后端：
  - "minimal" → render_minimal（一珠一像素，默认 5×1 二值位图）★ 默认
  - "fixed"   → render_fixed（默认 80×28，含梁/杆/留白的灰度 patch）
  - "image"       → ImageRenderer 真实算盘彩色图 RGB（默认 128×20，tight 逐档对齐）
  - "image_gray"  → 同上灰度对照（单通道，与 fixed 同通道不同外观）

全局默认后端由 DEFAULT_RENDER 控制（当前为 "minimal"）。

自定义注册：register_render("my", fn, default_patch=(h, w))，
fn 只需支持 (state, abacus, patch_h=None, patch_w=None) 关键字调用。

用法：
    from evaluate.abacus.render.registry import make_render
    render_fn = make_render()                    # 默认后端 minimal
    render_fn = make_render("fixed", patch_h=64)  # 覆盖尺寸
    arr = render_fn(state, abacus)                # shape (h, n_cols*w)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from evaluate.abacus.domain import Abacus, AbacusState

RenderFn = Callable[..., np.ndarray]

# 全局默认渲染后端：极简盘面（一珠一像素）。
# 切换回高清 patch 只需改这里，或在各环节显式传 render='fixed'。
DEFAULT_RENDER = "minimal"

# name -> (解析函数, 默认 patch 尺寸描述)
_BACKENDS: dict[str, dict] = {}


def _resolve_patch(default_h: int | None, default_w: int | None,
                   patch_h: int | None, patch_w: int | None) -> tuple[int | None, int | None]:
    """解析最终 patch 尺寸——**默认值与合法性校验的唯一定义处**。

    所有后端共用，避免"默认尺寸在 factory 与 register_builtin 两处各写一遍、
    校验在三个后端各写一遍"这类重复。

    - `None` 表示"取后端默认"；默认值为 `None` 的维度表示该后端无此概念
      （如 minimal 没有 patch_h），跳过该维度的校验。
    - 非法尺寸（< 1）一律 ValueError，不静默退化——此前 `patch or 默认` 会把
      0 吞成默认值，使底层的参数校验永远不可达。
    """
    h = default_h if patch_h is None else patch_h
    w = default_w if patch_w is None else patch_w
    if (h is not None and h < 1) or (w is not None and w < 1):
        raise ValueError(f"patch_h / patch_w 须 ≥1，得到 {h}×{w}")
    return h, w


def _backend_fixed(h: int, w: int) -> RenderFn:
    from evaluate.abacus.render.fixed import render_fixed

    def fn(state, abacus, patch_h=None, patch_w=None):
        hh, ww = _resolve_patch(h, w, patch_h, patch_w)
        return render_fixed(state, abacus, patch_h=hh, patch_w=ww)
    return fn


def _backend_minimal(h: int, w: int) -> RenderFn:
    from evaluate.abacus.render.minimal import render_minimal

    def fn(state, abacus, patch_h=None, patch_w=None):
        # minimal 没有 patch_h 概念（一档 = 上珠数+下珠数 行），故该维度传 None 跳过校验
        _, ww = _resolve_patch(None, w, None, patch_w)
        return render_minimal(state, abacus, patch_w=ww)
    return fn


def _backend_image(h: int, w: int, gray: bool = False) -> RenderFn:
    """image 后端：matplotlib 真实算盘图。

    gray=False → 彩色 RGB 三通道 [hh, n_cols*ww, 3]（默认）；
    gray=True  → 灰度对照 [hh, n_cols*ww]（与 fixed 同通道，仅外观不同）。

    默认 patch_w = patch_h/6.4：精确匹配真实算盘宽高比（n_cols : n_upper+n_lower+1.4）。
    128/6.4=20 恰为整数，resize 完全等比，珠子保持圆形不模糊。
    """
    from PIL import Image
    from evaluate.abacus.render.image import ImageRenderer

    renderer = ImageRenderer(seed=0)  # 固定 seed，颜色抖动可复现

    def fn(state, abacus, patch_h=None, patch_w=None):
        hh, ww = _resolve_patch(h, w, patch_h, patch_w)
        n_cols = abacus.spec.rod_count
        # tight 模式：裁掉文字区与留白，x 取杆对称范围，缩放后逐档对齐
        rgb = renderer.render(state, abacus, tight=True)
        if gray:
            arr = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0  # [H, W]
            resized = Image.fromarray((arr * 255).astype(np.uint8), mode="L") \
                .resize((n_cols * ww, hh), Image.LANCZOS)
            return np.asarray(resized, dtype=np.float32) / 255.0  # [hh, n_cols*ww]
        arr = np.asarray(rgb, dtype=np.float32) / 255.0             # [H, W, 3]
        resized = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB") \
            .resize((n_cols * ww, hh), Image.LANCZOS)
        return np.asarray(resized, dtype=np.float32) / 255.0        # [hh, n_cols*ww, 3]
    return fn


def _backend_image_gray(h: int, w: int) -> RenderFn:
    """image 灰度对照后端（= _backend_image(gray=True)）。"""
    return _backend_image(h, w, gray=True)


register_builtin = {
    "fixed": {"factory": _backend_fixed, "default_patch": (80, 28),
              "doc": "固定分辨率灰度 patch（render_fixed，默认 80×28）"},
    "minimal": {"factory": _backend_minimal, "default_patch": (5, 1),
                "doc": "一珠一像素二值位图（render_minimal，默认 5×1）"},
    "image": {"factory": _backend_image, "default_patch": (128, 20),
              "doc": "matplotlib 真实算盘彩色图 RGB（ImageRenderer，默认 128×20）"},
    "image_gray": {"factory": _backend_image_gray, "default_patch": (128, 20),
                   "doc": "matplotlib 真实算盘灰度图（灰度对照，默认 128×20）"},
}


def register_render(name: str, fn: RenderFn,
                    default_patch: tuple[int, int] = (80, 28)) -> None:
    """注册自定义渲染后端（fn 需支持 patch_h/patch_w 关键字）。

    自定义后端与内置后端共用同一套默认值填充与尺寸校验规则（`_resolve_patch`），
    不会因为是外部注册就绕过校验。
    """
    def factory(h: int, w: int) -> RenderFn:
        def wrapped(state, abacus, patch_h=None, patch_w=None):
            hh, ww = _resolve_patch(h, w, patch_h, patch_w)
            return fn(state, abacus, patch_h=hh, patch_w=ww)
        return wrapped

    _BACKENDS[name] = {"factory": factory, "default_patch": default_patch,
                       "doc": fn.__doc__ or ""}


def make_render(name: str = DEFAULT_RENDER, patch_h: int | None = None,
                patch_w: int | None = None) -> RenderFn:
    """按名字取统一签名的渲染函数（缺省后端 DEFAULT_RENDER）。

    patch 尺寸可覆盖后端默认；None 表示用后端自带默认。
    """
    if name is None:  # 显式传 None 也回退默认（默认参数不吞 None，此处统一防御）
        name = DEFAULT_RENDER
    spec = register_builtin.get(name) or _BACKENDS.get(name)
    if spec is None:
        raise ValueError(
            f"未知渲染后端 {name!r}，可用: {', '.join(backend_names())}")
    dh, dw = spec["default_patch"]                       # 默认尺寸的唯一来源
    h, w = _resolve_patch(dh, dw, patch_h, patch_w)      # 构造期即校验，非法尺寸立刻暴露
    return spec["factory"](h, w)


def default_patch(name: str = DEFAULT_RENDER) -> tuple[int, int]:
    """取后端的默认 (patch_h, patch_w)。

    ⚠️ 无参调用返回的是**全局 DEFAULT_RENDER** 的尺寸，与你实际使用的后端无关。
    取"我这个后端该用什么尺寸"请用 `resolve_patch(name, ...)`，
    否则会出现"用 minimal 的 5×1 去处理 fixed 的 80×28 数据"。
    """
    spec = register_builtin.get(name) or _BACKENDS.get(name)
    if spec is None:
        raise ValueError(f"未知渲染后端 {name!r}")
    return spec["default_patch"]


def resolve_patch(name: str = DEFAULT_RENDER, patch_h: int | None = None,
                  patch_w: int | None = None) -> tuple[int, int]:
    """解析最终 patch 尺寸——**取尺寸的统一入口**。

    patch 尺寸是**后端的属性**，不是全局常量：None 取该后端的默认，并统一校验。

    >>> resolve_patch("fixed")
    (80, 28)
    >>> resolve_patch("fixed", patch_h=64)
    (64, 28)
    >>> resolve_patch("minimal")
    (5, 1)
    """
    dh, dw = default_patch(name)
    return _resolve_patch(dh, dw, patch_h, patch_w)


def backend_names() -> list[str]:
    return sorted(set(register_builtin) | set(_BACKENDS))


def describe_backends() -> str:
    rows = [f"{'*' if n == DEFAULT_RENDER else ' '} {n:<10} "
            f"{spec['default_patch'][0]}×{spec['default_patch'][1]}  {spec['doc']}"
            for n, spec in sorted({**register_builtin, **_BACKENDS}.items())]
    return "\n".join(rows)


if __name__ == "__main__":
    print(f"可用渲染后端（* = 默认 {DEFAULT_RENDER}；name 默认patch 说明）:")
    print(describe_backends())
