"""VLA 子系统：视觉-语言-动作（看图 + 指令 → 口诀 + 指法动作）。

方案 4（VLM，见 model_vlm/）的 VLA 版本：模型不只输出「口诀@档位」，
还输出驱动算盘的**动作**（指法），即把「口诀 → 指法」的查表过程也交给模型学。

状态：未开始（本目录目前只有设计说明，暂无代码）。

落地约定（避免重复造轮子）：
  - 复用 model_vlm 的数据与桥接：model_vlm/dataset.py、model_vlm/parse_eval_bridge.py
  - 只增量加两件事：动作词表（指法 token） + 指法解码/执行对接
  - 原型参考：model_vlm/closed_loop.py（PARSE→EVAL→执行→READ→压栈 闭环）
  - 依赖方向单向：model_vla → model_vlm → model（禁止反向 import）
  - 珠算引擎（口诀查表 RHYME_TABLE → TransitionEngine.reduce 真实拨珠）在 evaluate/abacus/，
    是确定性的、不学习，只作为执行器与评分基准。
"""
