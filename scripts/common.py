"""各实验脚本的公共小工具：路径、CSV 写出、并行执行、统计与检验。"""

from __future__ import annotations

import csv
import os
import statistics
from typing import Callable, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, "results", "csv")
FIG_DIR = os.path.join(ROOT, "results", "fig")
LOG_DIR = os.path.join(ROOT, "results")


def ensure_dirs() -> None:
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def csv_path(name: str) -> str:
    ensure_dirs()
    return os.path.join(CSV_DIR, name)


def fig_path(name: str) -> str:
    ensure_dirs()
    return os.path.join(FIG_DIR, name)


def write_csv(path: str, header: Sequence, rows: Sequence[Sequence]) -> str:
    """写出 CSV（UTF-8 with BOM，方便 Excel 直接打开）。"""
    ensure_dirs()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))
    return path


def run_many(job_fn: Callable, jobs: Sequence, processes: int = 0) -> List:
    """并行执行一批独立任务。jobs 是可 pickle 的参数元组列表。"""
    if processes <= 0:
        processes = min(12, os.cpu_count() or 4)
    if processes == 1 or len(jobs) == 1:
        return [job_fn(j) for j in jobs]
    with __import__("multiprocessing").Pool(processes) as pool:
        return pool.map(job_fn, list(jobs))


def mean(xs: Sequence[float]) -> float:
    return statistics.mean(xs)


def stdev(xs: Sequence[float]) -> float:
    """样本标准差（n-1）。"""
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def rpd(makespan: float, optimum: int) -> float:
    """相对百分比偏差 RPD(%) = (求得值 - 文献最优) / 文献最优 × 100。"""
    return (makespan - optimum) / optimum * 100.0


def wilcoxon(a: Sequence[float], b: Sequence[float]):
    """Wilcoxon 符号秩检验（配对，双侧）。返回 (统计量, p 值)。"""
    from scipy.stats import wilcoxon as _w
    try:
        stat, p = _w(a, b, zero_method="wilcox")
    except ValueError:
        return float("nan"), float("nan")
    return float(stat), float(p)


def paired_t(a: Sequence[float], b: Sequence[float]):
    """配对 t 检验。返回 (t 统计量, p 值)。"""
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(a, b)
    return float(stat), float(p)


def fmt_p(p: float) -> str:
    """p 值的紧凑写法。"""
    if p != p:
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"
