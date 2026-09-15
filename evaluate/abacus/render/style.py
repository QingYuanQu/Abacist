"""渲染配色（纯配置，零 matplotlib 依赖）。

独立成模块的原因：image.py 顶部 `matplotlib.use("Agg")` 是离线出图（GIF/PNG）所需，
但会把全局后端锁死成 Agg；若交互式动画（render/animation.py）从 image.py 导入 Style，
会连带强制 Agg、弹不出窗口。把纯配色 dataclass 拆到这里，动画模块只依赖本模块，
既不触发 Agg，也与 image.py 共用同一套视觉常量。
"""
from dataclasses import dataclass


@dataclass
class Style:
    """视觉样式（可选随机抖动做数据增强）"""
    frame: str = "#8B5A2B"         # 外框/梁/杆：深棕
    bead_active: str = "#C0392B"   # 靠梁珠：暗红
    bead_rest: str = "#F5CBA7"     # 离梁珠：暖米色
    edge: str = "#7B241C"          # 珠描边：深棕红
