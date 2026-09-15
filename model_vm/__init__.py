"""独立 vision 子系统：看图读珠态（阶段 3.3，主线之外的对照实验）。

  - gen.py     视觉→珠态数据生成（PNG 图像目录 + jsonl）
  - dataset.py 数据集与 collate
  - vocab.py   珠态词表构造器
  - train.py   训练入口
  - eval.py           识数推理入口（三张默认示例图，可 --img 指定）
  - test_vision.py    端到端测试（含收敛率断言）

与主线的关系：本子系统的「读盘」能力已被主线 VLA 的 read 任务吸收覆盖，
两者只共享 model/vision.py 的 VisionEncoder，互不依赖。

运行：
  python -m model_vm.test_vision
  python -m model_vm.train_vision
  python -m model_vm.eval
"""
