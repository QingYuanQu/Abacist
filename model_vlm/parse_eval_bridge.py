# -*- coding: utf-8 -*-
"""parse ↔ eval 桥接层（VLM 三任务数据导出）。

阶段 4 起：对齐/注解逻辑已上提到底层 bridge/（IR 契约），本模块只负责
「IR（ExpressionInstance）→ VLM 三任务样本」的投影 + 盘面渲染，不再自算对齐。

closed_loop 依赖本模块的 SYM_TO_OP / make_default_composer（保留，含 ÷ 推理路径）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluate.abacus import (                        # noqa: E402
    Abacus,
    AbacusSpec,
    ArithmeticComposer,
    RhymeResolver,
    OpType,
    build_addition_rhymes,
    build_subtraction_rhymes,
)
from evaluate.abacus import num_to_state
from evaluate.bridge.align import align as bridge_align
from evaluate.bridge.ir import build_instance, ExpressionInstance
from evaluate.digit import make_digit_fn

__all__ = [
    'SYM_TO_OP', 'OP_TO_SYM', 'make_default_composer', 'emit_vlm_samples',
]

# 运算符符号（统一 ×，÷ 随 USE_DIV 同步启用；closed_loop 推理侧保留 ÷ 路径）
SYM_TO_OP = {'+': OpType.ADD, '-': OpType.SUB, '×': OpType.MUL, '÷': OpType.DIV}
OP_TO_SYM = {v: k for k, v in SYM_TO_OP.items()}


def make_default_composer(spec: AbacusSpec | None = None):
    """默认装配：13 档二五珠算盘 + 加减口诀真值裁判 + 编排器。

    closed_loop 依赖本函数（传 spec）。
    """
    abacus = Abacus.standard(spec or AbacusSpec(1, 4, 13, 10))
    resolver = RhymeResolver(build_addition_rhymes() + build_subtraction_rhymes())
    return ArithmeticComposer(resolver), abacus


def _default_vlm_abacus():
    """默认算盘（渲染后端见 abacus.render.registry，纯 numpy、无 matplotlib）。"""
    return Abacus.standard(AbacusSpec(1, 4, 13, 10))


def _save_png(arr: np.ndarray, path: Path) -> None:
    """渲染数组落 PNG。支持 [H, W] 灰度与 [H, W, C] 彩色（image 后端）。

    不传 mode 参数：Pillow 按 shape 自动判 L/RGB；显式 mode 会触发
    Pillow 13 的弃用告警（mode 用于改变 dtype 已弃用）。
    """
    from PIL import Image
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def _to_rod_patches(arr: np.ndarray, n_cols: int) -> np.ndarray:
    """整幅盘面 → 逐档 patch [n_cols, C, H, W]（C=1 灰度 / C=3 彩色）。

    与 model_vlm/dataset.py 的 to_rod_patches 同构（npy 存储
    与 PNG 加载两条路径的布局必须逐元素一致）；此处独立实现以保持
    本模块零 torch 依赖。
    """
    if arr.ndim == 2:                                   # 灰度
        h, w = arr.shape
        return arr.reshape(h, n_cols, w // n_cols).transpose(1, 0, 2)[:, None]
    if arr.ndim == 3:                                   # 彩色（末维通道）
        h, w, c = arr.shape
        return arr.reshape(h, n_cols, w // n_cols, c).transpose(1, 3, 0, 2)
    raise ValueError(f"无法切分的图像维度: {arr.shape}")


class _NpyImageStore:
    """盘面渲染去重栈：同一 rod_states 只渲染一次，最终整体存单个 npy。

    全量生成时避免数十万张 PNG 小文件（会拖垮 IDE / 文件索引）；
    此模式下样本的 'image' 字段由相对路径改为数组下标（int），
    训练侧通过 vlm_dataset 的 npy_path + 下标加载。
    """

    def __init__(self):
        self.arrays: list[np.ndarray] = []   # 每张 [n_cols, C, H, W] uint8
        self._index: dict = {}               # rod_states → 数组下标

    def get(self, state, abacus, render_fn, patch_h, patch_w) -> int:
        key = state.rod_states   # frozen dataclass 元组，可哈希；渲染仅依赖珠位
        idx = self._index.get(key)
        if idx is None:
            arr = render_fn(state, abacus, patch_h=patch_h, patch_w=patch_w)
            n_cols = abacus.spec.rod_count
            arr = _to_rod_patches(arr, n_cols)
            self.arrays.append(np.ascontiguousarray((arr * 255).astype(np.uint8)))
            idx = len(self.arrays) - 1
            self._index[key] = idx
        return idx

    def save(self, path: Path) -> int:
        """存为单个 images.npy（uint8 压缩体积），返回去重后图像数。"""
        np.save(path, np.stack(self.arrays))
        return len(self.arrays)


def emit_vlm_samples(inst: ExpressionInstance,
                     image_dir: Path | str,
                     *,
                     sample_idx: int = 0,
                     render_fn=None,
                     abacus: Abacus | None = None,
                     patch_h: int | None = None,
                     patch_w: int | None = None,
                     reuse_images: bool = False,
                     image_store: _NpyImageStore | None = None) -> list[dict]:
    """从 IR（ExpressionInstance）投影 VLM 三任务样本。

    输入 inst 已含对齐注解（steps：push 栈快照 / calc 的 abacus 口诀链），
    本函数只做投影 + 渲染，不再重算对齐。盘面图由数值 value 经 num_to_state
    重建（空盘起算 + 单次二元运算下，盘面由数值唯一确定）。

    输出（每条记录 1 parse + n 组 eval/read）:
      - task='parse': Q / PRE / POST / A（双遍历）
      - task='eval':  Q / POST / 空盘图 / stack / A（口诀@档位，逐步）
      - task='read':  图 / value（含空盘 0 与每步终盘）

    image_store 非 None 时为 npy 模式：'image' 字段存数组下标。
    """
    if image_store is None:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

    # render_fn 缺省用全局默认后端（DEFAULT_RENDER = minimal）；
    # patch 尺寸 None → 取后端自带默认（minimal 5×1 / fixed 80×28）。
    if render_fn is None:
        from evaluate.abacus.render.registry import (
            DEFAULT_RENDER, make_render, resolve_patch)
        # 后端与 patch 尺寸必须**同源**：都取全局默认后端，尺寸按该后端解析。
        # （此前 make_render() 与 default_patch() 都是无参调用，靠"恰好都是
        #   DEFAULT_RENDER"维持一致——改任意一处就会脱节，故显式写出后端名。）
        render_fn = make_render(DEFAULT_RENDER, patch_h=patch_h, patch_w=patch_w)
        patch_h, patch_w = resolve_patch(DEFAULT_RENDER, patch_h, patch_w)

    samples: list[dict] = []

    # ── parse 样本：中缀串 Q → 语法树的两种线性化 ──
    samples.append({
        'task': 'parse',
        'idx': sample_idx,
        'prompt': f"[PARSE] Q={inst.infix}",
        'Q': inst.infix,
        'PRE': inst.prefix,
        'POST': inst.postfix,
        'ANS': inst.answer,
        'A': f"PRE{inst.prefix};POST{inst.postfix}#",
    })

    if abacus is None:
        abacus = _default_vlm_abacus()

    calc_i = 0
    for step in inst.steps:
        if step.kind != 'calc':
            continue
        trace = step.abacus
        actions = trace.actions
        op_sym = step.token

        # 空盘（0）+ 终盘（最后一个 action 的盘面数值）→ 数值重建 state
        empty_value = 0
        final_value = actions[-1].value
        empty_state = num_to_state(empty_value, abacus)
        final_state = num_to_state(final_value, abacus)

        if image_store is not None:
            rel_empty = image_store.get(empty_state, abacus, render_fn,
                                        patch_h, patch_w)
            rel_final = image_store.get(final_state, abacus, render_fn,
                                        patch_h, patch_w)
        else:
            empty_name = f"{sample_idx:06d}_step{calc_i}_empty.png"
            final_name = f"{sample_idx:06d}_step{calc_i}_final.png"
            empty_path = image_dir / empty_name
            final_path = image_dir / final_name
            for path, state in ((empty_path, empty_state),
                                (final_path, final_state)):
                if reuse_images and path.exists():
                    continue
                arr = render_fn(state, abacus, patch_h=patch_h, patch_w=patch_w)
                _save_png(arr, path)
            rel_empty = empty_path.relative_to(image_dir).as_posix()
            rel_final = final_path.relative_to(image_dir).as_posix()

        # ── eval 样本（逐步）：第 j 步执行前的真实盘面 → 单句口诀 ──
        stack_before_str = ','.join(str(x) for x in step.stack_before)
        for j, action in enumerate(actions):
            before_value = 0 if j == 0 else actions[j - 1].value
            before_state = num_to_state(before_value, abacus)
            if j == 0:
                sub_rel = rel_empty                      # 与空盘图复用
            elif image_store is not None:
                sub_rel = image_store.get(before_state, abacus, render_fn,
                                          patch_h, patch_w)
            else:
                sub_name = f"{sample_idx:06d}_step{calc_i}_sub{j}.png"
                sub_path = image_dir / sub_name
                if not (reuse_images and sub_path.exists()):
                    sub_arr = render_fn(before_state, abacus, patch_h=patch_h,
                                        patch_w=patch_w)
                    _save_png(sub_arr, sub_path)
                sub_rel = sub_path.relative_to(image_dir).as_posix()

            eval_prompt = (f"[EVAL] Q={inst.infix} POST={inst.postfix} "
                           f"STACK={stack_before_str} "
                           f"CALC={step.a}{op_sym}{step.b} STEP={j}")
            samples.append({
                'task': 'eval',
                'idx': sample_idx,
                'step': calc_i,
                'sub': j,
                'Q': inst.infix,
                'POST': inst.postfix,
                'prompt': eval_prompt,
                'image': sub_rel,
                'stack': stack_before_str,
                'A': f"{action.oral}@{action.rod}#",
                'rod': action.rod,
            })

        # ── read 样本：空盘读 0 ──
        samples.append({
            'task': 'read',
            'idx': sample_idx,
            'step': calc_i,
            'stage': 'empty',
            'prompt': "[READ] 读数=",
            'image': rel_empty,
            'value': empty_value,
            'A': str(empty_value) + '#',
        })

        # ── read 样本：终盘读结果 ──
        samples.append({
            'task': 'read',
            'idx': sample_idx,
            'step': calc_i,
            'stage': 'final',
            'prompt': "[READ] 读数=",
            'image': rel_final,
            'value': final_value,
            'A': str(final_value) + '#',
        })

        calc_i += 1

    return samples


def _emit_vlm_cli(out_dir: Path, n_samples: int,
                  dataset_path: Path | None = None,
                  reuse_images: bool = False,
                  render: str | None = None,
                  patch_h: int | None = None,
                  patch_w: int | None = None) -> None:
    """生成 VLM 三任务样本到 out_dir；n_samples<=0 表示全量。

    流程：读 dataset_C → bridge.align 注解 → build_instance 组装 IR →
    emit_vlm_samples 投影三任务。对齐在底层 bridge 完成，本函数只做编排。

    n_samples != 0（抽样）：图片输出 PNG（out_dir/images）；n_samples <= 0（全量）：
    图片去重单文件 out_dir/images.npy，样本 image 字段存数组下标。
    """
    import json

    dataset_path = dataset_path or (Path(__file__).resolve().parent.parent
                                    / 'parse' / 'dataset' / 'dataset_C.jsonl')
    if not dataset_path.exists():
        print(f"[skip] 数据集不存在: {dataset_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / 'images'

    from evaluate.abacus.render.registry import DEFAULT_RENDER, make_render, default_patch
    render = render or DEFAULT_RENDER
    render_fn = make_render(render, patch_h=patch_h, patch_w=patch_w)
    # 默认 patch 尺寸必须按实际后端名查（fixed → 80×28）；
    # 无参 default_patch() 查的是 DEFAULT_RENDER（minimal 5×1），勿用。
    if patch_h is None or patch_w is None:
        _dh, _dw = default_patch(render)
        patch_h = patch_h if patch_h is not None else _dh
        patch_w = patch_w if patch_w is not None else _dw
    print(f"[VLM] 渲染后端: {render} (patch_h={patch_h or '默认'}, "
          f"patch_w={patch_w or '默认'})")
    abacus = Abacus.standard(AbacusSpec(1, 4, 13, 10))

    npy_mode = n_samples <= 0
    image_store = _NpyImageStore() if npy_mode else None
    print(f"[VLM] 图像输出: {'images.npy（去重单文件）' if npy_mode else 'PNG 目录 images/'}")

    composer, _ = make_default_composer(abacus.spec)
    digit_fn = make_digit_fn()

    files = {
        'parse': open(out_dir / 'parse.jsonl', 'w', encoding='utf-8'),
        'eval':  open(out_dir / 'eval.jsonl',  'w', encoding='utf-8'),
        'read':  open(out_dir / 'read.jsonl',  'w', encoding='utf-8'),
        'train': open(out_dir / 'train.jsonl', 'w', encoding='utf-8'),
    }
    counts = {k: 0 for k in files}
    max_stack_depth = 0
    max_prompt_len = 0
    max_a_len = 0

    with open(dataset_path, encoding='utf-8') as f:
        lines = f.readlines()
    total = len(lines) if n_samples <= 0 else min(n_samples, len(lines))

    for i, line in enumerate(lines[:total]):
        rec = json.loads(line)
        # record → IR（底层 bridge 对齐注解）
        steps, _ = bridge_align(rec.get('postfix', rec['post']).split(), composer, abacus,
                                digit_fn=digit_fn, expected_ans=rec.get('answer', rec['ANS']))
        inst = build_instance(rec, steps)
        samples = emit_vlm_samples(inst, image_dir, sample_idx=i,
                                   render_fn=render_fn, abacus=abacus,
                                   patch_h=patch_h, patch_w=patch_w,
                                   reuse_images=reuse_images,
                                   image_store=image_store)
        for s in samples:
            files[s['task']].write(json.dumps(s, ensure_ascii=False) + '\n')
            counts[s['task']] += 1
            files['train'].write(json.dumps(s, ensure_ascii=False) + '\n')
            counts['train'] += 1
            max_prompt_len = max(max_prompt_len, len(s.get('prompt', '')))
            max_a_len = max(max_a_len, len(str(s.get('A', ''))))
            if s['task'] == 'eval':
                stack = s['stack'].split(',') if s['stack'] else []
                max_stack_depth = max(max_stack_depth, len(stack))

    for f in files.values():
        f.close()

    print(f"[VLM] 生成完成：{out_dir}")
    print(f"  样本数 {total} | parse={counts['parse']} eval={counts['eval']} read={counts['read']} train={counts.get('train', 0)}")
    if image_store is not None:
        n_imgs = image_store.save(out_dir / 'images.npy')
        print(f"  图像（去重）{n_imgs} 张 → 单文件 {out_dir / 'images.npy'}"
              f"（样本 image 字段=数组下标）")
    else:
        print(f"  图片目录 {image_dir}")
    print(f"  最大栈深度 {max_stack_depth}")
    print(f"  最大 prompt 长度 {max_prompt_len}")
    print(f"  最大 A 长度 {max_a_len}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='VLM 三任务样本生成（IR 投影）')
    ap.add_argument('n', nargs='?', type=int, default=0,
                    help='抽检/生成条数（0=全量。n>0 抽样输出 PNG，n=0 全量输出 images.npy）')
    ap.add_argument('--vlm-out-dir', type=Path, default="model_vlm/data",
                    help='生成 VLM 三任务样本到此目录')
    ap.add_argument('--dataset', type=Path, default=None,
                    help='dataset_C 路径（默认 parse/dataset/dataset_C.jsonl）')
    ap.add_argument('--reuse-images', action='store_true',
                    help='已存在的图片跳过重渲染（仅改标签格式重生成时用）')
    ap.add_argument('--render', default="minimal",
                    choices=sorted({'minimal', 'fixed', 'image', 'image_gray'}),
                    help='渲染后端：minimal / fixed / image / image_gray')
    ap.add_argument('--patch-h', type=int, default=None,
                    help='覆盖渲染后端默认 patch 高（minimal 5 / fixed 80）')
    ap.add_argument('--patch-w', type=int, default=None,
                    help='覆盖渲染后端默认 patch 宽（minimal 1 / fixed 28）')
    args = ap.parse_args()
    _emit_vlm_cli(args.vlm_out_dir, args.n, dataset_path=args.dataset,
                  reuse_images=args.reuse_images, render=args.render,
                  patch_h=args.patch_h, patch_w=args.patch_w)
