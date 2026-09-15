"""pytest 全局配置：确保仓库根在 sys.path 中。

放在仓库根，pytest 加载 conftest 时会把本目录插入 sys.path（prepend 导入模式），
于是 `python -m pytest` 从任意 cwd 运行都能导入 evaluate / model_* 等包。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
