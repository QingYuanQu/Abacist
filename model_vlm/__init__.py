"""VLM 子系统：视觉-语言-动作（看图 + 指令 → 口诀动作）。

主线模块：
  - train.py   三任务（parse/eval/read）训练入口
  - dataset.py 数据集与 collate
  - infer.py       推理（greedy decode + 准确率统计，checkpoint 自恢复渲染后端）
  - closed_loop.py 端到端闭环控制器（PARSE→EVAL→执行→READ→压栈）
  - vocab.py   遗留：早期硬编码的 VLM 词表构造器（build_vlm_vocab）。
                   现词表由 `model/vocab.py` 从训练 jsonl 自动构建
                   （train_vlm.build_vocab），本文件暂无调用者。

通用依赖在 `model/`：`model/vocab.py`（通用词表库）与 `model/vision.py`
（VisionEncoder）被本子系统与主线课程训练（`python -m exp`）共用。

运行：
  python -m model_vlm.train_vlm --jsonl ... --image-dir ... --out ...
  python -m model_vlm.closed_loop --parse-ckpt ... --eval-ckpt ... --read-ckpt ...
"""
