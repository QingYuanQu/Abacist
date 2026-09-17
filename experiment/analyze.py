# -*- coding: utf-8 -*-
"""experiment/analyze.py —— 实验结果可视化分析（分析工具，不参与训练主链路）。

为什么合并成一个模块
--------------------
原 `experiment/studies/PATTERN_in2post/visualize.py` 与
`PATTERN_in2pre/visualize.py` 是**逐字节相同**的两份拷贝，且各自硬编码了：
  - `HIDDEN = 64`（从 brain.json 抄来的常量）
  - "全部 15 课"（实际 19 课，早已过期）
  - 数据来源 `material.csv` + `config/head_patterns.json`（两者均已随配置迁移删除）
同一份分析代码复制到每个实验目录，既会各自腐化，也让配置改动同时打断多处。
故收敛为单一入口，元信息一律**从 config.yaml 经 loader 读取**（唯一事实来源）。

数据来源（全部走 TrialPaths，不跨模块手拼路径）
-----------------------------------------------
  - config.yaml            逐层头数 heads / hidden_size（经 loader.load_experiment）
  - logs/trial_{id}_train.jsonl   逐 epoch 训练记录（train_loss / acc）
  - bucket_report.csv      按难度桶聚合的结构正确率（由 experiment.bucket_report --csv 产出）

输出（写到实验目录，被 .gitignore 排除）
----------------------------------------
  fig_training.png   训练动态（loss + acc 双面板）
  fig_headdim.png    head_dim 主因分析（散点 + 多重集分组）
  fig_buckets.png    难度结构与分布形状对比（热力图 + 逐 trial 柱状）

用法
----
    python -m experiment.analyze --experiment PATTERN_in2post
    python -m experiment.analyze --experiment PATTERN_in2post --buckets some.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from experiment.loader import load_experiment

# 中文字体（Windows 常见；找不到则回退默认）
for _cand in ("Microsoft YaHei", "SimHei", "Noto Sans SC"):
    if any(_cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _cand
        break
plt.rcParams["axes.unicode_minus"] = False

# 已知分布形状的固定配色；其余家族按 tab20 顺序取色
KNOWN_COLORS = {
    "regular":   "#1f77b4",
    "focus":     "#ff7f0e",
    "expansion": "#2ca02c",
    "balanced":  "#d62728",
    "balanced2": "#9467bd",
}
_FALLBACK = plt.get_cmap("tab20").colors


def family_of(name: str) -> str:
    """分析期家族 = 名字里最后一个 `-数字` 之前的部分。

    配置层已取消 family/variant（trial name 就是唯一自由标签），这里只做**分析期的
    配色分组**：`regular-1`、`focus-16` → `regular`、`focus`；无该后缀
    （如 `none`、`prepost_stack`）则整体即家族。
    """
    base, _, tail = name.rpartition("-")
    return base if base and tail.isdigit() else name


@dataclass
class TrialView:
    """绘图用的 trial 视图（trial 元信息 + 由 heads 派生的容量指标）。"""
    id: int
    name: str
    family: str
    heads: list[int]
    head_dims: list[int]        # 每层 head_dim = hidden_size // heads[l]

    @property
    def hd_multiset(self) -> tuple[int, ...]:
        return tuple(sorted(self.head_dims))

    @property
    def mean_hd(self) -> float:
        return float(np.mean(self.head_dims))


def build_views(exp) -> list[TrialView]:
    """从 Experiment 装配绘图视图（heads 来自 config.yaml，不再读 material.csv）。"""
    hidden = exp.brain.hidden_size
    views = []
    for t in exp.trials:
        dims = [hidden // h for h in t.heads]
        assert all(hidden % h == 0 for h in t.heads), \
            f"{exp.name}/{t.name}: heads={t.heads} 不能整除 hidden_size={hidden}"
        views.append(TrialView(id=t.id, name=t.name, family=family_of(t.name),
                               heads=list(t.heads), head_dims=dims))
    return views


def palette(views: list[TrialView]) -> dict[str, str]:
    """家族 → 颜色（已知家族用固定色，其余按出现顺序取 tab20）。"""
    families = list(OrderedDict.fromkeys(v.family for v in views))
    colors, extra = {}, 0
    for fam in families:
        if fam in KNOWN_COLORS:
            colors[fam] = KNOWN_COLORS[fam]
        else:
            colors[fam] = "#%02x%02x%02x" % tuple(
                int(c * 255) for c in _FALLBACK[extra % len(_FALLBACK)][:3])
            extra += 1
    return colors


# ==================== 数据读取 ====================

def _load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_logs(exp, views: list[TrialView]) -> dict[int, list[dict]]:
    """trial_id → 逐 epoch 训练记录（路径取自 TrialPaths.train_log）。"""
    return {v.id: _load_jsonl(exp.trials[v.id].paths.train_log) for v in views}


def _read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_buckets(path: str) -> dict[str, dict[tuple, list[int]]]:
    """bucket_report.csv → {trial 标签: {(table,row,col): [correct,total]}}。"""
    agg: dict[str, dict[tuple, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    for r in _read_csv(path):
        cell = agg[r["trial"]][(r["table"], r["row"], r["col"])]
        cell[0] += int(r["correct"])
        cell[1] += int(r["total"])
    return agg


class Buckets:
    """bucket 聚合的查询封装（标签约定 = bucket_report 写入的 `L{id}_{name}`）。"""

    def __init__(self, agg: dict):
        self._agg = agg

    def overall(self, view: TrialView, table: str | None = None) -> float:
        cs = ts = 0
        for (tb, _row, _col), (c, t) in self._agg.get(self._label(view), {}).items():
            if table and tb != table:
                continue
            cs += c
            ts += t
        return (cs / ts * 100.0) if ts else float("nan")

    def nband(self, view: TrialView, band: str, min_t: int = 30) -> float:
        cs = ts = 0
        for (tb, row, _col), (c, t) in self._agg.get(self._label(view), {}).items():
            if tb == "n×bk" and row == band:
                cs += c
                ts += t
        return (cs / ts * 100.0) if ts >= min_t else float("nan")

    @staticmethod
    def _label(view: TrialView) -> str:
        return f"L{view.id}_{view.name}"


# ==================== 图 1：训练动态 ====================

def fig_training(exp, views, logs, colors, out_dir) -> str | None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plotted = 0
    for v in views:
        log = logs[v.id]
        if not log:
            continue
        ep = [d.get("epoch") for d in log]
        loss = [d.get("train_loss") for d in log]
        acc = [None if d.get("acc") is None else d["acc"] * 100 for d in log]
        col = colors[v.family]
        axes[0].plot(ep, loss, color=col, alpha=0.8, lw=1.2)
        axes[1].plot(ep, acc, color=col, alpha=0.8, lw=1.2,
                     label=f"L{v.id} {v.name} (hd={v.mean_hd:.0f})")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None

    n = len(views)
    axes[0].set_title(f"训练损失 (train_loss) — 全部 {n} 个 trial", fontsize=13)
    axes[1].set_title(f"验证集 acc（严格匹配）— 全部 {n} 个 trial", fontsize=13)
    for ax, ylab in zip(axes, ("loss", "acc (%)")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_training.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ==================== 图 2：head_dim 主因 ====================

def fig_headdim(views, buckets: Buckets, colors, out_dir) -> str | None:
    """左：整体正确率 vs 平均 head_dim；右：按 head_dim 多重集分组（位置无关性）。"""
    values = {v.id: buckets.overall(v) for v in views}
    if all(np.isnan(x) for x in values.values()):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：基准族连成曲线（它是 head_dim 扫描），其余家族散点
    base_fam = views[0].family
    bx = [v.mean_hd for v in views if v.family == base_fam]
    by = [values[v.id] for v in views if v.family == base_fam]
    order = np.argsort(bx)
    bx, by = np.array(bx)[order], np.array(by)[order]
    axes[0].plot(bx, by, "-o", color=colors[base_fam], lw=2,
                 label=f"{base_fam} 扫描 (hd 一致)")
    seen = set()
    for v in views:
        if v.family == base_fam:
            continue
        axes[0].scatter(v.mean_hd, values[v.id], color=colors[v.family], s=70, zorder=5,
                        label=v.family if v.family not in seen else None)
        seen.add(v.family)
        axes[0].annotate(f"L{v.id}", (v.mean_hd, values[v.id]),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xscale("log")
    axes[0].set_xticks(bx)
    axes[0].set_xticklabels([f"{x:.0f}" for x in bx])
    axes[0].set_xlabel("head_dim（每头维度，log 刻度）")
    axes[0].set_ylabel("整体结构正确率 (%)")
    axes[0].set_title(f"head_dim 是主因：{base_fam} 单调下滑；分布形状同量级", fontsize=12)
    axes[0].minorticks_off()
    axes[0].grid(alpha=0.3, which="major")
    axes[0].legend(fontsize=8)

    # 右：按 head_dim 多重集分组
    grp = defaultdict(list)
    for v in views:
        grp[v.hd_multiset].append(v)
    labels, means, errs = [], [], []
    for ms in sorted(grp, key=lambda k: -np.mean(k)):
        vals = [values[v.id] for v in grp[ms]]
        labels.append("{" + ",".join(str(x) for x in ms) + "}")
        means.append(np.nanmean(vals))
        errs.append(np.nanstd(vals))
    x = np.arange(len(labels))
    axes[1].bar(x, means, yerr=errs, capsize=4, color="#4c72b0")
    for xi, m in zip(x, means):
        axes[1].text(xi, m + 0.4, f"{m:.1f}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("整体结构正确率 (%)")
    axes[1].set_title("相同 head_dim 多重集（顺序不同）→ 表现一致", fontsize=13)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_headdim.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ==================== 图 3：难度结构 + 分布形状对比 ====================

def fig_buckets(views, buckets: Buckets, colors, out_dir) -> str | None:
    bands = ["2-5", "6-8", "9-12", "13-20"]
    if not buckets._agg:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：热力图 trials × n-band
    M = np.array([[buckets.nband(v, b) for b in bands] for v in views], dtype=float)
    im = axes[0].imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    axes[0].set_xticks(range(len(bands)))
    axes[0].set_xticklabels(bands)
    axes[0].set_yticks(range(len(views)))
    axes[0].set_yticklabels([f"L{v.id} {v.name}" for v in views], fontsize=7)
    axes[0].set_xlabel("n 难度带")
    axes[0].set_title("结构正确率 × n 难度带（n×bk 桶，t≥30）", fontsize=13)
    for i in range(len(views)):
        for j in range(len(bands)):
            if not np.isnan(M[i, j]):
                axes[0].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                             fontsize=6, color="white" if M[i, j] < 55 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # 右：逐 trial 整体正确率。不用家族均值——基准族实为 head_dim 扫描（容量变量），
    # 与分布形状不可比，平均会误导。
    vals = [buckets.overall(v) for v in views]
    base_fam = views[0].family
    axes[1].bar(np.arange(len(views)), vals, color=[colors[v.family] for v in views])
    for xi, v in enumerate(views):
        axes[1].text(xi, vals[xi] + 0.3, f"{vals[xi]:.1f}", ha="center", fontsize=7.5)
        if v.family == base_fam:
            axes[1].text(xi, vals[xi] + 2.0, f"hd={v.mean_hd:.0f}", ha="center",
                         fontsize=6.5, color=colors[base_fam], rotation=90)
    axes[1].set_xticks(np.arange(len(views)))
    axes[1].set_xticklabels([f"L{v.id}\n{v.name}" for v in views], fontsize=6.5)
    axes[1].set_ylabel("整体结构正确率 (%)")
    axes[1].set_title(f"逐 trial 整体结构正确率（颜色=家族；{base_fam} 标注 head_dim）", fontsize=12)
    axes[1].grid(alpha=0.3, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f])
               for f in OrderedDict.fromkeys(v.family for v in views)]
    axes[1].legend(handles, list(OrderedDict.fromkeys(v.family for v in views)), fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_buckets.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ==================== CLI ====================

def main() -> None:
    ap = argparse.ArgumentParser(description="实验结果可视化（读 config.yaml + logs + bucket_report.csv）")
    ap.add_argument("--experiment", required=True, help="实验名（experiment/studies/<name>）")
    ap.add_argument("--buckets", default=None,
                    help="bucket_report.csv 路径（默认 <实验目录>/bucket_report.csv）")
    ap.add_argument("--out-dir", default=None, help="图片输出目录（默认实验目录）")
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    out_dir = args.out_dir or exp.root_dir
    os.makedirs(out_dir, exist_ok=True)
    views = build_views(exp)
    colors = palette(views)
    logs = load_logs(exp, views)
    buckets = Buckets(load_buckets(args.buckets or os.path.join(exp.root_dir, "bucket_report.csv")))

    outs = [fig_training(exp, views, logs, colors, out_dir),
            fig_headdim(views, buckets, colors, out_dir),
            fig_buckets(views, buckets, colors, out_dir)]
    made = [o for o in outs if o]
    if not made:
        print("[分析] 无可绘制的数据：需要 logs/trial_<id>_train.jsonl 或 bucket_report.csv。"
              "先运行训练，再用 python -m experiment.bucket_report --experiment ... --csv ... 生成分桶表。")
        return
    print("已生成：")
    for o in made:
        print("  ", o)

    print(f"\n{'id':>3}  {'name':<16} {'heads':<24} {'hd_multiset':<22} overall%  mean_hd")
    for v in views:
        ov = buckets.overall(v)
        print(f"{v.id:>3}  {v.name:<16} {str(v.heads):<24} {str(v.hd_multiset):<22} "
              f"{'-' if np.isnan(ov) else format(ov, '6.1f')}   {v.mean_hd:.0f}")


if __name__ == "__main__":
    main()
