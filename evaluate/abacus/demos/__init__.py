"""渲染示例（人工验收用，非自动化测试）。

入口是 `__main__.py`，一个参数化脚本取代了原先四个内容重复的薄壳：

    python -m evaluate.abacus.demos {text|image|fixed|minimal|all}

产物统一写到 `evaluate/abacus/output/<name>/`。
内核端到端演示（四则全链路 + 双模式自检）另见 `python -m evaluate.abacus.demo`。

需从**仓库根**运行（依赖 `evaluate` 包可导入）；本包不做 sys.path hack——
此前 4 个薄壳各复制一份 `parents[N]` 上样板（且层级还数错了一级），
与其修四份，不如合一份入口、把运行方式收敛到标准做法。
"""
