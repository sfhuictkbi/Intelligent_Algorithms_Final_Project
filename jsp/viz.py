"""可视化：甘特图（含关键路径）与收敛曲线。"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# 中文字体（Windows 优先微软雅黑 / SimHei，找不到则退化为默认字体）
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")


def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def _colors(n: int) -> List[Tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return [cmap(i / max(1, n)) for i in range(n)]


def plot_gantt(inst, sched, critical: Optional[Sequence[int]] = None,
               title: str = "", save: Optional[str] = None, show: bool = False):
    """绘制甘特图：横轴时间、纵轴机器，按工件着色，关键路径描红边。"""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(12, 0.42 * inst.m + 2.2))
    colors = _colors(inst.n)
    crit = set(critical or [])

    for M in range(inst.m):
        for o in sched.machine_seq[M]:
            j = o // inst.m
            s, e = sched.start[o], sched.end[o]
            ax.barh(M, e - s, left=s, height=0.62, align="center",
                    color=colors[j], edgecolor="black", linewidth=0.5)
            ax.text((s + e) / 2, M, f"J{j + 1}", ha="center", va="center",
                    fontsize=7, color="white" if sum(colors[j][:3]) < 1.6 else "black")
            if o in crit:
                ax.barh(M, e - s, left=s, height=0.62, align="center",
                        facecolor="none", edgecolor="red", linewidth=2.0)

    ax.set_yticks(range(inst.m))
    ax.set_yticklabels([f"M{M + 1}" for M in range(inst.m)])
    ax.set_xlabel("时间")
    ax.set_ylabel("机器")
    ax.set_title(title or f"{inst.name} 调度甘特图（makespan = {sched.makespan}）")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_xlim(0, sched.makespan * 1.02)
    handles = [mpatches.Patch(color=colors[j], label=f"工件 {j + 1}") for j in range(inst.n)]
    if crit:
        handles.append(mpatches.Patch(facecolor="none", edgecolor="red",
                                      linewidth=2, label="关键路径工序"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7, ncol=max(1, inst.n // 12 + 1))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save


def plot_convergence(curves: Dict[str, Tuple[Sequence[int], Sequence[float]]],
                     title: str = "", xlabel: str = "评价次数",
                     ylabel: str = "当前最优 makespan", save: Optional[str] = None,
                     show: bool = False):
    """绘制收敛曲线对比图。curves[标签] = (横轴, 纵轴)。"""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, linewidth=1.6, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "收敛曲线对比")
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save


def plot_bars(labels: Sequence[str], values: Sequence[float], ylabel: str = "",
              title: str = "", save: Optional[str] = None, show: bool = False,
              err: Optional[Sequence[float]] = None):
    """柱状图（用于参数敏感性、初始解对比等）。"""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(labels))
    ax.bar(list(x), values, yerr=err, capsize=4, color="#4C72B0", edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save


def plot_lines(levels: Sequence[float], series: Dict[str, Sequence[float]],
               xlabel: str = "", ylabel: str = "", title: str = "",
               save: Optional[str] = None, show: bool = False):
    """折线图（参数敏感性：横轴为参数水平）。"""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, ys in series.items():
        ax.plot(list(levels), list(ys), marker="o", linewidth=1.6, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save
