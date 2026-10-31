"""调度解码、makespan 评价、关键路径与可行性检验。

编码约定（基于工序的置换编码 operation-based encoding）
------------------------------------------------------
解是一个长度 L = n*m 的整数序列 perm，工件号 j 出现 m 次，
第 k 次出现代表工件 j 的第 k 道工序。

例：n=2, m=2 时 perm = [0, 1, 1, 0] 表示
    工件0工序1 → 工件1工序1 → 工件1工序2 → 工件0工序2

解码规则（半主动调度）
--------------------
按 perm 的顺序依次安排工序：工序的开工时刻 =
    max(该工件上一道工序的完工时刻, 该机器上一道工序的完工时刻)
这同时保证了：
    * 工艺路线约束（同一工件工序先后顺序）——由编码中"第 k 次出现"天然保证；
    * 机器独占约束（同一机器工序不重叠）——由"机器上一道工序完工时刻"保证。
因此解的可行性由编码结构自动保证，评价函数不需要惩罚项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .instance import Instance


@dataclass
class Schedule:
    """一次解码得到的完整调度结果。"""

    start: List[int]              # start[o] = 工序 o 的开工时刻
    end: List[int]                # end[o]   = 工序 o 的完工时刻
    machine_seq: List[List[int]]  # machine_seq[M] = 机器 M 上的加工顺序（工序号列表）
    makespan: int                 # 最大完工时间
    perm: List[int]               # 产生该调度的工序置换

    def job_of(self, inst: Instance, o: int) -> int:
        """工序 o 属于哪个工件。"""
        return o // inst.m


def decode(inst: Instance, perm: Sequence[int]) -> Schedule:
    """把工序置换解码为调度。

    工序全局编号 o = j*m + k（工件 j 的第 k 道工序）。
    """
    n, m = inst.n, inst.m
    mach, dur = inst.mach, inst.dur

    start = [0] * (n * m)
    end = [0] * (n * m)
    job_ready = [0] * n          # 每个工件当前的最早可开工时刻
    mach_free = [0] * m          # 每台机器当前的最早可开工时刻
    machine_seq: List[List[int]] = [[] for _ in range(m)]
    cnt = [0] * n                # cnt[j] = 工件 j 已经安排了几道工序

    for gene in perm:
        j = gene
        k = cnt[j]
        cnt[j] += 1
        o = j * m + k            # 本次安排的工序
        M = mach[o]
        s = max(job_ready[j], mach_free[M])   # 最早可行开工时刻
        e = s + dur[o]
        start[o] = s
        end[o] = e
        job_ready[j] = e
        mach_free[M] = e
        machine_seq[M].append(o)

    return Schedule(start, end, machine_seq, max(job_ready), list(perm))


def makespan(inst: Instance, perm: Sequence[int]) -> int:
    """只算 makespan 的快速入口（不构造 Schedule，省去对象开销）。"""
    n, m = inst.n, inst.m
    mach, dur = inst.mach, inst.dur
    job_ready = [0] * n
    mach_free = [0] * m
    cnt = [0] * n
    for gene in perm:
        j = gene
        k = cnt[j]
        cnt[j] += 1
        o = j * m + k
        M = mach[o]
        s = job_ready[j] if job_ready[j] > mach_free[M] else mach_free[M]
        e = s + dur[o]
        job_ready[j] = e
        mach_free[M] = e
    return max(job_ready)


# --------------------------------------------------------------------------
# 关键路径（析取图最长路径）
# --------------------------------------------------------------------------
def critical_path(inst: Instance, sched: Schedule) -> List[int]:
    """返回关键路径上的工序序列（按加工先后排列）。

    从完工最晚的工序出发，每步回溯到"完工时刻恰好等于本工序开工时刻"的前驱
    （工件前驱或机器前驱），直到没有紧前工序为止。
    """
    m = inst.m

    # 每台机器上各工序的前驱（按机器加工顺序）
    mach_prev: dict = {}
    for M in range(m):
        prev: Optional[int] = None
        for o in sched.machine_seq[M]:
            mach_prev[o] = prev
            prev = o

    tail = max(range(inst.n_ops), key=lambda o: sched.end[o])
    path: List[int] = []
    o: Optional[int] = tail
    while o is not None:
        path.append(o)
        j, k = o // m, o % m
        cand: List[Tuple[int, int]] = []
        if k > 0:                                    # 同一工件的前一道工序
            jp = j * m + k - 1
            if sched.end[jp] == sched.start[o]:
                cand.append((sched.end[jp], jp))
        mp = mach_prev.get(o)                        # 同一机器的前一道工序
        if mp is not None and sched.end[mp] == sched.start[o]:
            cand.append((sched.end[mp], mp))
        if not cand:
            break
        cand.sort(key=lambda t: -t[0])
        o = cand[0][1]
    path.reverse()
    return path


def critical_jobs(inst: Instance, sched: Schedule) -> List[int]:
    """关键路径上工序所属的工件号（用于关键路径偏向变异）。"""
    return [o // inst.m for o in critical_path(inst, sched)]


# --------------------------------------------------------------------------
# 可行性检验（任务一要求）
# --------------------------------------------------------------------------
def check_permutation(inst: Instance, perm: Sequence[int]) -> Tuple[bool, str]:
    """检验工序置换是否合法（可作为评价函数的输入）。

    合法条件：
      1. 长度等于 n*m；
      2. 元素均为合法工件号 [0, n)；
      3. 每个工件号恰好出现 m 次。
    """
    n, m = inst.n, inst.m
    if len(perm) != n * m:
        return False, f"长度错误：期望 {n * m}，实得 {len(perm)}"
    counts = [0] * n
    for gene in perm:
        if not isinstance(gene, (int,)) or gene < 0 or gene >= n:
            return False, f"非法工件号：{gene}（合法范围 0~{n - 1}）"
        counts[gene] += 1
    for j, c in enumerate(counts):
        if c != m:
            return False, f"工件 {j + 1} 出现 {c} 次，应为 {m} 次"
    return True, "合法"


def check_schedule(inst: Instance, sched: Schedule) -> Tuple[bool, str]:
    """检验解码结果是否满足工艺路线约束、机器独占约束与 makespan 一致性。"""
    m = inst.m
    for j in range(inst.n):                       # 工艺路线约束
        for k in range(1, m):
            o1, o2 = j * m + k - 1, j * m + k
            if sched.start[o2] < sched.end[o1]:
                return False, f"工件 {j + 1} 工序 {k + 1} 早于工序 {k} 完工"
    for M in range(m):                            # 机器独占约束
        seq = sorted(sched.machine_seq[M], key=lambda o: sched.start[o])
        for a, b in zip(seq, seq[1:]):
            if sched.start[b] < sched.end[a]:
                return False, f"机器 {M + 1} 上工序重叠"
    if sched.makespan != max(sched.end):          # makespan 一致性
        return False, f"makespan 与最大完工时刻不一致：{sched.makespan} vs {max(sched.end)}"
    return True, "满足全部约束"
