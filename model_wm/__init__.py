"""世界模型子系统：盘面 + 口诀@档位 → 下一盘面珠态。

把珠算引擎的**状态转移函数**蒸馏进模型：给定当前盘面与一步口诀，预测执行后的盘面珠态。
学成后模型「脑中拨珠」，可替代确定性引擎做前瞻（planning / rollout）。

状态：未开始（本目录目前只有设计说明，暂无代码）。

落地约定（避免重复造轮子）：
  - 渲染 / 编解码复用 model_vm（gen.py 的渲染后端 + bead_codec 的珠态文本）
  - 转移函数真值来自 evaluate/abacus/（TransitionEngine.reduce），不手写第二套规则
  - 依赖方向单向：model_wm → model → evaluate（禁止反向 import）
  - 产物放 model_wm/data/（数据）与 model_wm/ckpt/（权重），均已 gitignore
"""
