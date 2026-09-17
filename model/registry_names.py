"""注册名清单（torch 无关）—— 只声明"有哪些名字可选"，不含任何实现。

为什么单独一份
--------------
`model/registry.py` 与 `model/pos_emb/` 顶层都 `import torch`，而配置校验
（`experiment/schema.py`）只需要名字。若校验直接查注册表，则**校验一份 YAML 也要
拉起整个深度学习框架**，代价有二：
  1. 慢：改一个配置键就要加载 torch；
  2. 脏：torch 会把自带 OpenMP 运行库带进进程，此后 `matplotlib.savefig` 会直接以
     `OMP: Error #15` 中止（Windows 实测）。于是任何"读 config.yaml + 画图"的分析
     脚本都被迫依赖 torch 的加载顺序。

因此名字在此声明，实现在各自模块注册。两者必须一致，由 `tests/test_registry.py`
强制：不一致时测试立即失败——要的是响亮失败，不是静默失配。
"""
# 注意力纯函数后端（实现：model/attn_fn.py，注册于 model/registry.py）
ATTN_TYPES = ("sdpa", "eager")

# 位置编码类型（实现：model/pos_emb/*，注册于 model/registry.py）
POS_EMB_TYPES = ("rope", "alibi", "learned", "sinusoidal", "none")

# RoPE 子变体（实现：model/pos_emb/rope.py 的 ROPE_INIT 路由表）
ROPE_TYPES = ("default", "yarn")
