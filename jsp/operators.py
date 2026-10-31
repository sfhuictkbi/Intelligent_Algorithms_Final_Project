"""遗传算子与初始解构造。

内容
----
1. 初始解构造：随机置换、SPT 优先规则、MWR 优先规则（任务书 (五) 要求至少两种）
2. PSO 连续位置 → 工序置换 的映射：SPV / LOV
3. GA 交叉算子：IPOX、POX（保持工序链约束）
4. GA 变异算子：互换 swap、插入 insertion、逆转 inverse、关键路径偏向变异（加分项）
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from .instance import Instance
from .schedule import critical_path, decode


# --------------------------------------------------------------------------
# 1. 初始解构造
# --------------------------------------------------------------------------
def init_random(inst: Instance, rng: random.Random) -> List[int]:
    """随机置换：把 [0]*m + [1]*m + ... 随机打乱。"""
    perm = [j for j in range(inst.n) for _ in range(inst.m)]
    rng.shuffle(perm)
    return perm


def init_by_rule(inst: Instance, rng: random.Random, rule: str) -> List[int]:
    """用 Giffler-Thompson 活性调度构造 + 优先规则生成置换。

    每一步：
      1) 在所有待安排工序中找最早完工时刻 C* 及其所在机器 M*；
      2) 冲突集 = 机器 M* 上能在 C* 之前开工的工序；
      3) 按优先规则从冲突集中选一个工序安排：
            "spt"：加工时间最短（Shortest Processing Time）
            "mwr"：所属工件剩余加工时间最长（Most Work Remaining）
            "random"：随机优先规则（任务书提到的随机优先规则组合）
    平局用随机数打破，因此同一种子也能产生不同个体。
    """
    n, m = inst.n, inst.m
    mach, dur = inst.mach, inst.dur
    next_op = [0] * n                     # 每个工件下一道待安排工序的序号
    job_ready = [0] * n                   # 工件最早可开工时刻
    mach_free = [0] * m                   # 机器最早可开工时刻
    remaining = [sum(inst.dur[j * m + k] for k in range(m)) for j in range(n)]
    perm: List[int] = []

    for _ in range(n * m):
        # ---- 1) 最早完工时刻 C* 与对应机器 M* ----
        best_c = None
        best_M = None
        for j in range(n):
            if next_op[j] >= m:
                continue
            o = j * m + next_op[j]
            M = mach[o]
            st = job_ready[j] if job_ready[j] > mach_free[M] else mach_free[M]
            c = st + dur[o]
            if best_c is None or c < best_c:
                best_c, best_M = c, M

        # ---- 2) 机器 M* 上的冲突工序集 ----
        conflict: List[int] = []
        for j in range(n):
            if next_op[j] >= m:
                continue
            o = j * m + next_op[j]
            if mach[o] != best_M:
                continue
            st = job_ready[j] if job_ready[j] > mach_free[best_M] else mach_free[best_M]
            if st < best_c:
                conflict.append(j)
        if not conflict:                   # 兜底：冲突集为空时取任一待安排工序
            conflict = [j for j in range(n) if next_op[j] < m]

        # ---- 3) 按优先规则挑选 ----
        if rule == "random":
            j = conflict[rng.randrange(len(conflict))]
        else:
            best_val = None
            best: List[int] = []
            for j in conflict:
                o = j * m + next_op[j]
                val = float(dur[o]) if rule == "spt" else -float(remaining[j])
                val += rng.random() * 1e-9
                if best_val is None or val < best_val:
                    best_val, best = val, [j]
                elif val == best_val:
                    best.append(j)
            j = best[rng.randrange(len(best))]

        # ---- 安排该工序 ----
        o = j * m + next_op[j]
        M = mach[o]
        st = job_ready[j] if job_ready[j] > mach_free[M] else mach_free[M]
        e = st + dur[o]
        job_ready[j] = e
        mach_free[M] = e
        remaining[j] -= dur[o]
        next_op[j] += 1
        perm.append(j)
    return perm


def make_initial(inst: Instance, rng: random.Random, method: str = "random") -> List[int]:
    """统一入口：method ∈ {"random", "spt", "mwr", "gt_random"}。

    "random"     = 随机置换
    "spt"/"mwr"  = Giffler-Thompson 活性调度 + 对应优先规则
    "gt_random"  = Giffler-Thompson 活性调度 + 随机优先规则
    """
    if method == "random":
        return init_random(inst, rng)
    if method in ("spt", "mwr", "gt_random"):
        return init_by_rule(inst, rng, "random" if method == "gt_random" else method)
    raise ValueError(f"未知初始解构造方法：{method}")


# --------------------------------------------------------------------------
# 2. PSO 连续位置 → 工序置换 的映射
# --------------------------------------------------------------------------
def multiset(inst: Instance) -> List[int]:
    """置换编码对应的工件号多重集：[0]*m + [1]*m + ..."""
    return [j for j in range(inst.n) for _ in range(inst.m)]


def spv_map(inst: Instance, x: Sequence[float]) -> List[int]:
    """SPV（Smallest Position Value）映射。

    位置向量分量越小，该"槽位"越早被填入多重集中的工件号。
    由于只比较相对大小，映射结果一定满足"每个工件恰好出现 m 次"，即一定可行。
    """
    L = inst.n * inst.m
    order = sorted(range(L), key=lambda i: x[i])     # 按位置值升序
    ms = multiset(inst)
    perm = [0] * L
    for rank, pos in enumerate(order):
        perm[pos] = ms[rank]
    return perm


def lov_map(inst: Instance, x: Sequence[float]) -> List[int]:
    """LOV（Largest Order Value）映射：位置值越大优先级越高（越早被填入）。"""
    L = inst.n * inst.m
    order = sorted(range(L), key=lambda i: -x[i])    # 按位置值降序
    ms = multiset(inst)
    perm = [0] * L
    for rank, pos in enumerate(order):
        perm[pos] = ms[rank]
    return perm


def perm_to_position(inst: Instance, perm: Sequence[int]) -> List[float]:
    """SPV 的逆映射：由置换反推出一个位置向量 x，使 spv_map(x) == perm。

    做法：按多重集顺序给各工件占用的槽位分配递增的排名值（归一化到 [0,1]），
    使位置向量与速度裁剪区间保持同一量纲。
    """
    L = inst.n * inst.m
    ms = multiset(inst)
    slots: List[List[int]] = [[] for _ in range(inst.n)]
    for idx, gene in enumerate(perm):
        slots[gene].append(idx)
    x = [0.0] * L
    used = [0] * inst.n
    denom = max(1, L - 1)
    for rank in range(L):
        j = ms[rank]
        x[slots[j][used[j]]] = rank / denom
        used[j] += 1
    return x


# --------------------------------------------------------------------------
# 3. 交叉算子
# --------------------------------------------------------------------------
def _fill_child(keep_from: Sequence[int], fill_from: Sequence[int],
                keep_jobs: set) -> List[int]:
    """保留 keep_from 中属于 keep_jobs 的基因（位置与相对顺序不变），
    空位按 fill_from 中不属于 keep_jobs 的基因顺序填入。"""
    L = len(keep_from)
    child: List[int] = [-1] * L
    for i, gene in enumerate(keep_from):
        if gene in keep_jobs:
            child[i] = gene
    rest = [g for g in fill_from if g not in keep_jobs]
    it = iter(rest)
    for i in range(L):
        if child[i] == -1:
            child[i] = next(it)
    return child


def ipox(inst: Instance, p1: Sequence[int], p2: Sequence[int],
         rng: random.Random) -> Tuple[List[int], List[int]]:
    """IPOX（Improved Precedence Operation Crossover）。

    随机把工件集划分为 J1 / J2：
      子代1 = 保留父代1中 J1 的基因 + 按父代2顺序填入 J2 的基因；
      子代2 = 保留父代2中 J1 的基因 + 按父代1顺序填入 J2 的基因。
    同一工件的基因在父代中的相对顺序被原样保留，故子代一定满足工序链约束。
    """
    j1 = {j for j in range(inst.n) if rng.random() < 0.5}
    if not j1:
        j1 = {rng.randrange(inst.n)}
    return _fill_child(p1, p2, j1), _fill_child(p2, p1, j1)


def pox(inst: Instance, p1: Sequence[int], p2: Sequence[int],
        rng: random.Random) -> Tuple[List[int], List[int]]:
    """POX（Precedence Operation Crossover）：与 IPOX 思路相同，
    区别在于子代2保留互补组 J2，使两个子代差异更大。"""
    j1 = {j for j in range(inst.n) if rng.random() < 0.5}
    if not j1:
        j1 = {rng.randrange(inst.n)}
    j2 = set(range(inst.n)) - j1
    return _fill_child(p1, p2, j1), _fill_child(p2, p1, j2)


# --------------------------------------------------------------------------
# 4. 变异算子
# --------------------------------------------------------------------------
def mutate_swap(inst: Instance, perm: Sequence[int], rng: random.Random) -> List[int]:
    """互换变异：随机交换两个不同位置上的基因。"""
    L = len(perm)
    child = list(perm)
    i = rng.randrange(L)
    j = rng.randrange(L)
    while j == i:
        j = rng.randrange(L)
    child[i], child[j] = child[j], child[i]
    return child


def mutate_insertion(inst: Instance, perm: Sequence[int], rng: random.Random) -> List[int]:
    """插入变异：随机取出一个基因，插入到另一个位置。"""
    L = len(perm)
    child = list(perm)
    gene = child.pop(rng.randrange(L))
    child.insert(rng.randrange(L), gene)
    return child


def mutate_inverse(inst: Instance, perm: Sequence[int], rng: random.Random) -> List[int]:
    """逆转变异：随机取一段区间并反转。"""
    L = len(perm)
    child = list(perm)
    i = rng.randrange(L)
    j = rng.randrange(L)
    if i > j:
        i, j = j, i
    child[i:j + 1] = reversed(child[i:j + 1])
    return child


def _position_of(inst: Instance, perm: Sequence[int], op: int) -> int:
    """工序 op = j*m + k 在置换 perm 中的下标（即工件 j 的第 k 次出现）。"""
    j, k = op // inst.m, op % inst.m
    cnt = -1
    for idx, gene in enumerate(perm):
        if gene == j:
            cnt += 1
            if cnt == k:
                return idx
    raise ValueError("工序不在置换中")


def mutate_critical(inst: Instance, perm: Sequence[int], rng: random.Random) -> List[int]:
    """关键路径偏向变异（加分项）。

    先解码求关键路径，再随机交换关键路径上相邻两道工序在染色体中的位置。
    只有改动关键路径上的工序才可能缩短 makespan，因此比随机变异更有针对性。
    """
    path = critical_path(inst, decode(inst, perm))
    if len(path) < 2:
        return list(perm)
    i = rng.randrange(len(path) - 1)
    o1, o2 = path[i], path[i + 1]
    if o1 // inst.m == o2 // inst.m:      # 同一工件的两道工序不能换序
        return list(perm)
    child = list(perm)
    p1 = _position_of(inst, child, o1)
    p2 = _position_of(inst, child, o2)
    child[p1], child[p2] = child[p2], child[p1]
    return child


MUTATIONS = {
    "swap": mutate_swap,
    "insertion": mutate_insertion,
    "inverse": mutate_inverse,
    "critical": mutate_critical,
}

CROSSOVERS = {"ipox": ipox, "pox": pox}

DEFAULT_MUTATIONS = ("swap", "insertion", "inverse")


def mutate(inst: Instance, perm: Sequence[int], rng: random.Random,
           kinds: Sequence[str] = DEFAULT_MUTATIONS) -> List[int]:
    """从给定的几种变异中随机选一种执行。"""
    return MUTATIONS[rng.choice(list(kinds))](inst, perm, rng)


def crossover(inst: Instance, p1: Sequence[int], p2: Sequence[int], rng: random.Random,
              kind: str = "ipox") -> Tuple[List[int], List[int]]:
    """交叉算子统一入口：kind ∈ {"ipox", "pox"}。"""
    return CROSSOVERS[kind](inst, p1, p2, rng)
