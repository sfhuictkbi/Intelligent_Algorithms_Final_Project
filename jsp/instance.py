"""JSP 基准实例解析与数据结构（OR-Library 标准格式）。"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# 默认数据集根目录，可用环境变量 JSP_DATA 覆盖
DEFAULT_DATA_ROOT = r"C:\Users\占\Desktop\作业\2026-2027学年第一学期\智能算法\dazuoye\JSP基准数据集"

# 文献报告最优 makespan（来源：OR-Library / job-shop-lib 基准库索引表）
FALLBACK_OPTIMUM = {
    "ft06": 55, "ft10": 930, "ft20": 1165,
    "la01": 666, "la11": 1222, "la21": 1046,
    "ta01": 1231, "ta11": 1357,
    "la06": 926, "la16": 945, "la26": 1218, "la31": 1784, "la36": 1268,
}


def data_root() -> str:
    return os.environ.get("JSP_DATA", DEFAULT_DATA_ROOT)


@dataclass
class Instance:
    """作业车间调度实例。

    工序全局编号采用 o = j * m + k（第 j 个工件的第 k 道工序）。
    本数据集所有实例均为"每工件恰有 m 道工序"的方形实例。
    """

    name: str
    n: int                      # 工件数
    m: int                      # 机器数
    jobs: List[List[Tuple[int, int]]]   # jobs[j][k] = (机器号0-based, 加工时间)
    optimum: int | None = None  # 文献报告最优 makespan

    # 扁平化加速数组
    mach: List[int] = field(default_factory=list)
    dur: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.mach = [self.jobs[j][k][0] for j in range(self.n) for k in range(self.m)]
        self.dur = [self.jobs[j][k][1] for j in range(self.n) for k in range(self.m)]

    @property
    def n_ops(self) -> int:
        return self.n * self.m

    def __repr__(self) -> str:
        return f"<Instance {self.name} {self.n}x{self.m} opt={self.optimum}>"


def load_optima(root: str | None = None) -> Dict[str, int]:
    """从实例索引表.csv 读取全部文献最优值。"""
    root = root or data_root()
    path = os.path.join(root, "实例索引表.csv")
    opt: Dict[str, int] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                v = (row.get("文献最优值") or "").strip()
                if v:
                    try:
                        opt[row["实例名"].strip()] = int(float(v))
                    except ValueError:
                        pass
    opt.update({k: v for k, v in FALLBACK_OPTIMUM.items() if k not in opt})
    return opt


def load_instance(name: str, root: str | None = None) -> Instance:
    """解析 OR-Library 格式的单个实例文件。

    第 1 行：n m；随后 n 行，每行 m 对「机器号(1-based) 加工时间」。
    """
    root = root or data_root()
    path = os.path.join(root, name[0:2] if name[:2] in
                        ("ab", "ft", "la", "or", "sw", "ta", "yn") else "", f"{name}.txt")
    if not os.path.exists(path):
        # 兜底：递归查找
        for d in os.listdir(root):
            cand = os.path.join(root, d, f"{name}.txt")
            if os.path.isfile(cand):
                path = cand
                break
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到实例文件: {name} (根目录 {root})")

    tokens: List[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens.extend(int(x) for x in line.split())

    n, m = tokens[0], tokens[1]
    body = tokens[2:]
    if len(body) < 2 * m * n:
        raise ValueError(f"{name}: 数据长度不足，期望 {2*m*n} 个数，实得 {len(body)}")

    jobs: List[List[Tuple[int, int]]] = []
    pos = 0
    for _ in range(n):
        seq: List[Tuple[int, int]] = []
        for _ in range(m):
            machine = body[pos] - 1   # 转为 0-based
            dur = body[pos + 1]
            pos += 2
            if not (0 <= machine < m):
                raise ValueError(f"{name}: 机器号越界 {machine + 1}")
            seq.append((machine, dur))
        jobs.append(seq)

    optima = load_optima(root)
    return Instance(name=name, n=n, m=m, jobs=jobs, optimum=optima.get(name))


# 大作业规定使用的基准实例
BENCHMARK = ["ft06", "ft10", "ft20", "la01", "la11", "la21"]
