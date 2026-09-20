"""渲染示例统一入口（人工验收用，非自动化测试）。

用法（从**仓库根**运行）：

    python -m evaluate.abacus.demos text       # 逐步 ASCII 帧    → output/text/
    python -m evaluate.abacus.demos image      # PNG + GIF        → output/image/
    python -m evaluate.abacus.demos fixed      # 逐步灰度 patch   → output/fixed/
    python -m evaluate.abacus.demos minimal    # 逐步极简盘面      → output/minimal/
    python -m evaluate.abacus.demos music      # 运算演奏 + 数字流实验 → output/music/
    python -m evaluate.abacus.demos vertical   # 竖屏短视频 9:16   → output/vertical/
    python -m evaluate.abacus.demos animate    # 交互式窗口动画    （需 GUI 后端）
    python -m evaluate.abacus.demos all        # 依次跑全部后端
    python -m evaluate.abacus.demos            # 同 all（默认参数）

内核端到端演示（四则全链路 + 双模式自检）见 `python -m evaluate.abacus.demo`。

为什么是一个入口而不是四个薄壳：此前 `demo_text.py` / `demo_image.py` /
`demo_fixed.py` / `demo_minimal.py` 除了一行 import 和一行 print 外完全相同，
且各自复制了一份 sys.path 上样板——四处重复才是需要消除的东西，
而不是把四份重复搬进同一个文件夹。合并后用法只差一个参数。
"""
import argparse
import importlib

# name -> (模块路径, demo 函数名, 产物说明)
_BACKENDS = {
    "text": ("evaluate.abacus.render.text", "demo", "逐步 ASCII 帧"),
    "image": ("evaluate.abacus.render.image", "demo", "PNG + GIF"),
    "fixed": ("evaluate.abacus.render.fixed", "demo", "逐步灰度 patch"),
    "minimal": ("evaluate.abacus.render.minimal", "demo", "逐步极简盘面"),
    "music": ("evaluate.abacus.music", "demo", "运算演奏 + 数字流实验 + 七珠筝旋律 MIDI"),
    # 竖屏 9:16：7 档 + 拟人时间轴 → 1080×1920 MP4（逐帧出图较慢）
    "vertical": ("evaluate.abacus.render.vertical", "demo", "竖屏短视频 9:16（7 档 MP4）"),
    "animate": ("evaluate.abacus.render.animation", "demo", "交互式窗口动画"),
}


def run(name: str) -> list:
    """运行单个后端的示例渲染，返回写出的文件路径列表（animate 无文件产出，返回 []）。"""
    module_path, func_name, kind = _BACKENDS[name]
    demo = getattr(importlib.import_module(module_path), func_name)
    files = demo()
    if files:
        print(f"[{name}] {len(files)} 个文件（{kind}）→ {files[0].parent}")
    else:
        print(f"[{name}] 交互式动画已播放（{kind}）")
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluate.abacus.demos",
        description="渲染示例入口（产物写入 evaluate/abacus/output/<name>/）")
    parser.add_argument("backend", nargs="?", default="animate",
                        choices=[*_BACKENDS, "all"],
                        help="渲染后端名；all = 依次跑全部（默认）")
    args = parser.parse_args(argv)
    names = list(_BACKENDS) if args.backend == "all" else [args.backend]
    for name in names:
        run(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
