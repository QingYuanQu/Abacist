"""拨珠小手图元：image（横条版式）与 vertical（竖屏版式）共用的唯一手势定义。

为什么单独成模块：两个版式的画布比例完全不同（前者约 2.5:1 横条，后者 9:16），
但"手相对珠位的几何"是同一个东西——掌心在珠右侧、三指朝左捏向珠、腕向右延伸。
各写一份必然导致手势样式分叉（改一处忘一处）。

线宽口径：基准值是 image.py 现状的磅数（掌心 1.0 / 手指 3.0 / 腕 4.0），
`lw_scale=1.0` 时绘制结果与抽离前**逐像素一致**；画布放大时调用方传入
`lw_scale = px_per_unit / 50`（50 = image.py 13 档非 tight 画布的 px/数据单位），
使手的粗细随算盘等比放大，而不是被固定磅数困成细线。

不在 registry 的机器观测链路上（fixed/minimal 无 matplotlib），
模块顶部不设置后端，后端由调用方决定（image.py 已强制 Agg）。
"""
from __future__ import annotations

SKIN = "#F1C27D"
OUTLINE = "#C07A3A"

# 手的几何（数据坐标，相对珠心）
PALM_DX, PALM_R = 0.55, 0.28
FINGER_DY = (-0.14, 0.0, 0.14)
FINGER_X0, FINGER_X1 = 0.42, 0.18
WRIST_X0, WRIST_X1 = 0.72, 0.95

# 基准线宽（磅），对应 image.py 抽离前的取值
_PALM_LW = 1.0
_FINGER_LW = 3.0
_WRIST_LW = 4.0

# image.py 13 档非 tight 画布的 px/数据单位：竖屏等大画布据此换算 lw_scale
BASE_PX_PER_UNIT = 50.0


def draw_hand(ax, x: float, y: float, *, lw_scale: float = 1.0,
              zorder: int = 7) -> list:
    """在珠位 (x, y) 旁画一只简笔小手（从右侧伸入捏珠），返回加入的 artist 列表。

    lw_scale: 线宽倍率（1.0 = image.py 现状磅数）；
    zorder:   需高于珠子（珠为 3、横梁为 5），故默认 7。
    """
    from matplotlib.patches import Circle

    artists = []
    palm = Circle((x + PALM_DX, y), PALM_R, facecolor=SKIN, edgecolor=OUTLINE,
                  linewidth=_PALM_LW * lw_scale, zorder=zorder)
    ax.add_patch(palm)
    artists.append(palm)

    for dy in FINGER_DY:
        ln, = ax.plot([x + FINGER_X0, x + FINGER_X1], [y + dy, y + dy],
                      color=SKIN, linewidth=_FINGER_LW * lw_scale,
                      solid_capstyle="round", zorder=zorder)
        artists.append(ln)

    wrist, = ax.plot([x + WRIST_X0, x + WRIST_X1], [y, y], color=SKIN,
                     linewidth=_WRIST_LW * lw_scale, solid_capstyle="round",
                     zorder=zorder)
    artists.append(wrist)
    return artists
