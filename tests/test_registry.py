"""注册名声明与真实注册表的一致性 + 配置校验不依赖 torch。

`model/registry_names.py` 是"名字"的第二份副本，存在的唯一理由是让配置校验
不必加载 torch（torch 会把 OpenMP 运行库带进进程，此后 matplotlib 保存图片会
直接 OMP Error #15 中止）。副本必须被强制对齐，否则会出现
"配置写了合法名字却被拒绝"或"声明了却没实现"这类静默失配。
"""
import os
import subprocess
import sys

from model.registry_names import ATTN_TYPES, POS_EMB_TYPES, ROPE_TYPES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_attn_types_match_registry():
    from model.registry import ATTN_REGISTRY
    assert set(ATTN_TYPES) == set(ATTN_REGISTRY)


def test_pos_emb_types_match_registry():
    from model.registry import POS_EMB_REGISTRY
    assert set(POS_EMB_TYPES) == set(POS_EMB_REGISTRY)


def test_rope_types_match_rope_init():
    from model.pos_emb import ROPE_INIT
    assert set(ROPE_TYPES) == set(ROPE_INIT)


def test_schema_import_does_not_pull_torch():
    """配置校验链路（experiment.config / loader）必须保持 torch 无关。"""
    code = ("import sys\n"
            "import experiment.config, experiment.tools.loader\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'numpy')]\n"
            "print(bad)\n"
            "sys.exit(1 if bad else 0)\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, \
        f"配置校验链路引入了重型依赖: {proc.stdout.strip()}{proc.stderr.strip()}"
