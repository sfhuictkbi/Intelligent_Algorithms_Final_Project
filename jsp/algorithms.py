"""五套算法：纯 PSO、纯 GA、方案一、方案二、方案三c（自适应参数）。

统一接口
--------
    result = run(算法名, 实例, 参数)
    result.best_makespan / best_perm / history / time_sec / n_evals

参数取值依据（任务书 四(七) 参数设计与调优）
-------------------------------------------
    种群规模 N     = 100     （建议 50～200）
    最大迭代数     = 1000    （建议 500～2000）
    惯性权重 w     = 0.7     （建议 0.4～0.9）
    加速系数 c1,c2 = 2.0     （建议 1.5～2.5）
    交叉率 pc      = 0.8     （建议 0.7～0.95）
    变异率 pm      = 0.1     （建议 0.05～0.2）
    速度上限 vmax  = 0.5     （任务书未列；位置分量取值 [0,1]，取其一半）

方案三c 的自适应参数同样严格落在上述建议区间内：
    w ∈ [0.4, 0.9]   c1,c2 ∈ [1.5, 2.5]   pc ∈ [0.7, 0.95]   pm ∈ [0.05, 0.2]
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .instance import Instance
from .operators import crossover, make_initial, mutate, perm_to_position, spv_map
from .schedule import makespan


# --------------------------------------------------------------------------
# 参数与结果
# --------------------------------------------------------------------------
@dataclass
class Params:
    """算法的全部可调参数（默认值即主实验使用的固定值）。"""

    seed: int = 1
    pop_size: int = 100          # 种群规模 N
    max_iter: int = 1000         # 最大迭代代数（终止条件）

    # PSO
    w: float = 0.7               # 惯性权重
    c1: float = 2.0              # 个体认知系数
    c2: float = 2.0              # 社会认知系数
    vmax: float = 0.5            # 速度上限

    # GA
    pc: float = 0.8              # 交叉率
    pm: float = 0.1              # 变异率
    tournament: int = 3          # 锦标赛规模
    elite: int = 2               # 精英保留个数
    crossover_kind: str = "ipox"      # "ipox" / "pox"
    mutation_kinds: Tuple[str, ...] = ("swap", "insertion", "inverse", "critical")

    # 公共
    init_method: str = "random"  # 初始解构造："random" / "spt" / "mwr"

    # 方案一专用
    directed_partner: str = "none"   # 定向交叉对象："none"=随机个体 / "pbest" / "gbest"
    # 方案二专用
    exchange_every: int = 10         # 双种群精英交换周期（代）

    # 方案三c 专用
    adaptive: bool = False           # True = 启用自适应参数
    stag_max: int = 50               # 停滞代数归一化上限

    def copy(self, **kw) -> "Params":
        """返回改了若干字段后的新参数对象。"""
        return replace(self, **kw)


@dataclass
class Result:
    """一次运行的输出（任务二要求：最优调度、收敛曲线数据、运行时间）。"""

    algo: str
    instance: str
    seed: int
    best_perm: List[int]
    best_makespan: int
    history: List[Tuple[int, int]] = field(default_factory=list)  # (代数, 截至该代最优)
    time_sec: float = 0.0
    n_evals: int = 0                # makespan 评价次数
    n_iter: int = 0                 # 实际迭代代数


# --------------------------------------------------------------------------
# 公用小工具
# --------------------------------------------------------------------------
def initial_population(inst: Instance, rng: random.Random, p: Params) -> List[List[int]]:
    """生成初始种群。"""
    return [make_initial(inst, rng, p.init_method) for _ in range(p.pop_size)]


def tournament(fits: Sequence[int], rng: random.Random, k: int = 3) -> int:
    """锦标赛选择：随机取 k 个个体，返回其中最优（makespan 最小）者的下标。"""
    best = rng.randrange(len(fits))
    for _ in range(k - 1):
        i = rng.randrange(len(fits))
        if fits[i] < fits[best]:
            best = i
    return best


def perm_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """两个置换的归一化距离：不同位置所占的比例，取值 [0, 1]。"""
    return sum(1 for i in range(len(a)) if a[i] != b[i]) / len(a)


def diversity(pop: Sequence[Sequence[int]], gbest: Sequence[int]) -> float:
    """种群多样性：所有个体与全局最优解的平均归一化距离，取值 [0, 1]。"""
    if not pop:
        return 0.0
    return sum(perm_distance(x, gbest) for x in pop) / len(pop)


def adapt(p: Params, div: float, stag: int) -> Tuple[float, float, float, float, float]:
    """方案三c：按种群多样性 div 与停滞代数 stag 自适应调整参数。

    返回 (w, c1, c2, pc, pm)，全部落在任务书建议区间内：
        w  = 0.4 + 0.5*(1-div)        ∈ [0.4, 0.9]
        c1 = 1.5 + 1.0*(1-div)        ∈ [1.5, 2.5]  多样性低 → 加强个体认知（促进分散）
        c2 = 1.5 + 1.0*div            ∈ [1.5, 2.5]  多样性高 → 加强社会认知（加快收敛）
        pc = 0.7 + 0.25*div           ∈ [0.7, 0.95] 多样性高 → 多重组
        pm = 0.05 + 0.15*(1-div)      ∈ [0.05, 0.2] 多样性低 → 多变异（跳出局部最优）
        停滞越久变异率越高：pm 再叠加 0.05*min(1, stag/stag_max)，仍不超过 0.2
    """
    stag_n = min(1.0, stag / max(1, p.stag_max))
    w = 0.4 + 0.5 * (1.0 - div)
    c1 = 1.5 + 1.0 * (1.0 - div)
    c2 = 1.5 + 1.0 * div
    pc = 0.7 + 0.25 * div
    pm = 0.05 + 0.15 * (1.0 - div) + 0.05 * stag_n
    return w, c1, c2, pc, min(pm, 0.2)


# --------------------------------------------------------------------------
# 1. 纯 PSO：连续位置向量 + SPV 映射
# --------------------------------------------------------------------------
def run_pso(inst: Instance, p: Params) -> Result:
    """纯 PSO。

    每个粒子是一个 L = n*m 维连续位置向量 x，用 SPV 映射成工序置换后评价。
    标准速度—位置更新：
        v ← w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)，并裁剪到 ±vmax
        x ← x + v
    pbest 是粒子自身历史最优位置，gbest 是全体历史最优位置。
    """
    rng = random.Random(p.seed)
    npr = np.random.default_rng(p.seed)
    N, L = p.pop_size, inst.n_ops

    # 初始化：随机置换 → 反推位置向量 → 加小扰动打散
    X = np.array([perm_to_position(inst, make_initial(inst, rng, p.init_method))
                  for _ in range(N)], dtype=float)
    X += npr.uniform(-0.5 / L, 0.5 / L, size=X.shape)
    V = npr.uniform(-p.vmax, p.vmax, size=(N, L))

    pbest_x = X.copy()
    pbest_fit = [10 ** 9] * N
    gbest_x = X[0].copy()
    gbest_fit = 10 ** 9

    t0 = time.perf_counter()
    n_evals = 0
    history: List[Tuple[int, int]] = []

    for it in range(1, p.max_iter + 1):
        # --- 速度—位置更新（numpy 向量化，一次更新全部粒子）---
        r1 = npr.random((N, 1))
        r2 = npr.random((N, 1))
        V = p.w * V + p.c1 * r1 * (pbest_x - X) + p.c2 * r2 * (gbest_x - X)
        np.clip(V, -p.vmax, p.vmax, out=V)
        X = X + V

        # --- 评价 ---
        for i in range(N):
            perm = spv_map(inst, X[i])
            f = makespan(inst, perm)
            n_evals += 1
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest_x[i] = X[i].copy()
                if f < gbest_fit:
                    gbest_fit = f
                    gbest_x = X[i].copy()
        history.append((it, gbest_fit))

    return Result("纯 PSO", inst.name, p.seed, spv_map(inst, gbest_x), gbest_fit,
                  history, time.perf_counter() - t0, n_evals, p.max_iter)


# --------------------------------------------------------------------------
# 2. 纯 GA：锦标赛选择 + IPOX 交叉 + 变异 + 精英保留
# --------------------------------------------------------------------------
def run_ga(inst: Instance, p: Params) -> Result:
    """纯 GA。

    每代：精英保留 → 锦标赛选择双亲 → 以概率 pc 做 IPOX 交叉 → 以概率 pm 变异
          → 组成新一代种群。解的可行性由编码结构与算子自动保证。
    """
    rng = random.Random(p.seed)
    N = p.pop_size

    pop = initial_population(inst, rng, p)
    fits = [makespan(inst, x) for x in pop]
    n_evals = N
    gi = min(range(N), key=lambda i: fits[i])
    gbest_perm, gbest_fit = list(pop[gi]), fits[gi]

    t0 = time.perf_counter()
    history: List[Tuple[int, int]] = []

    for it in range(1, p.max_iter + 1):
        order = sorted(range(N), key=lambda i: fits[i])
        new_pop = [list(pop[i]) for i in order[:p.elite]]        # 精英保留
        new_fit = [fits[i] for i in order[:p.elite]]

        while len(new_pop) < N:
            i = tournament(fits, rng, p.tournament)
            j = tournament(fits, rng, p.tournament)
            if rng.random() < p.pc:
                a, b = crossover(inst, pop[i], pop[j], rng, p.crossover_kind)
            else:
                a, b = list(pop[i]), list(pop[j])
            if rng.random() < p.pm:
                a = mutate(inst, a, rng, p.mutation_kinds)
            if rng.random() < p.pm:
                b = mutate(inst, b, rng, p.mutation_kinds)
            for c in (a, b):
                if len(new_pop) >= N:
                    break
                f = makespan(inst, c)
                n_evals += 1
                new_pop.append(c)
                new_fit.append(f)
                if f < gbest_fit:
                    gbest_fit, gbest_perm = f, list(c)

        pop, fits = new_pop, new_fit
        history.append((it, gbest_fit))

    return Result("纯 GA", inst.name, p.seed, gbest_perm, gbest_fit,
                  history, time.perf_counter() - t0, n_evals, p.max_iter)


# --------------------------------------------------------------------------
# 3. 方案一：GA 框架 + PSO 引导交叉
# --------------------------------------------------------------------------
def run_hybrid1(inst: Instance, p: Params) -> Result:
    """方案一：GA 框架 + PSO 引导交叉。

    主框架与纯 GA 完全一致，两处由 PSO 的"速度—位置"思想驱动：

    (1) 交叉概率由粒子速度决定。每个个体携带一个速度标量 v_i：
            v_i ← w*v_i + c1*r1*d(x_i, pbest_i) + c2*r2*d(x_i, gbest)
            drive = |v_i| / v_ref，v_ref = (c1*d_p + c2*d_g)/(1-w)   （速度的理论稳态上界）
            pc_i  = 0.7 + 0.25*drive               （落在任务书建议区间 [0.7, 0.95]）
        速度越大 ⇒ 个体越远离 pbest / gbest ⇒ 越需要重组 ⇒ 交叉概率越高；
        个体已收敛到精英附近时速度变小，交叉概率随之降低（转而依靠变异）。
    (2) 可选的定向交叉：directed_partner 指定交叉对象为 "pbest"（个体历史最优）、
        "gbest"（全局最优）或 "none"（随机个体，默认）。默认取 "none"，即只用速度
        调制交叉概率——实验表明过早地全部与精英交叉会迅速损失多样性（见任务四）。
    """
    rng = random.Random(p.seed)
    N = p.pop_size

    pop = initial_population(inst, rng, p)
    fits = [makespan(inst, x) for x in pop]
    n_evals = N
    pbest_perm = [list(x) for x in pop]
    pbest_fit = list(fits)
    gi = min(range(N), key=lambda i: fits[i])
    gbest_perm, gbest_fit = list(pop[gi]), fits[gi]
    v = [0.0] * N

    t0 = time.perf_counter()
    history: List[Tuple[int, int]] = []

    for it in range(1, p.max_iter + 1):
        order = sorted(range(N), key=lambda i: fits[i])
        new_pop = [list(pop[i]) for i in order[:p.elite]]
        new_fit = [fits[i] for i in order[:p.elite]]

        while len(new_pop) < N:
            i = tournament(fits, rng, p.tournament)
            # --- PSO 速度更新 → 决定本次交叉概率 ---
            d_p = perm_distance(pop[i], pbest_perm[i])
            d_g = perm_distance(pop[i], gbest_perm)
            v[i] = p.w * v[i] + p.c1 * rng.random() * d_p + p.c2 * rng.random() * d_g
            v_ref = (p.c1 * d_p + p.c2 * d_g) / max(1e-9, 1.0 - p.w)  # 速度稳态上界
            drive = min(1.0, abs(v[i]) / max(1e-9, v_ref))
            pc_i = 0.7 + 0.25 * drive

            if rng.random() < pc_i:
                # --- pbest / gbest 引导的定向交叉 ---
                if p.directed_partner == "gbest":
                    partner = gbest_perm
                elif p.directed_partner == "pbest":
                    partner = pbest_perm[tournament(pbest_fit, rng, p.tournament)]
                else:                                   # "none"：与随机个体交叉
                    partner = pop[tournament(fits, rng, p.tournament)]
                a, b = crossover(inst, pop[i], partner, rng, p.crossover_kind)
            else:
                a, b = list(pop[i]), list(pop[i])
            if rng.random() < p.pm:
                a = mutate(inst, a, rng, p.mutation_kinds)
            if rng.random() < p.pm:
                b = mutate(inst, b, rng, p.mutation_kinds)

            for c in (a, b):
                if len(new_pop) >= N:
                    break
                f = makespan(inst, c)
                n_evals += 1
                new_pop.append(c)
                new_fit.append(f)
                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_perm[i] = list(c)
                if f < gbest_fit:
                    gbest_fit, gbest_perm = f, list(c)

        pop, fits = new_pop, new_fit
        history.append((it, gbest_fit))

    name = "方案一" if not p.adaptive else "方案三c"
    return Result(name, inst.name, p.seed, gbest_perm, gbest_fit,
                  history, time.perf_counter() - t0, n_evals, p.max_iter)


# --------------------------------------------------------------------------
# 4. 方案二：PSO 全局搜索 + GA 局部精修（双种群协同 + 定期交换精英）
# --------------------------------------------------------------------------
def run_hybrid2(inst: Instance, p: Params) -> Result:
    """方案二：PSO 全局搜索 + GA 局部精修（双种群协同 + 定期交换精英）。

        种群 A（GA ，规模 N）：锦标赛选择 + IPOX 交叉 + 变异 + 精英保留 → 局部精修
        种群 B（PSO，规模 N）：连续位置 + SPV 映射 + 速度—位置更新     → 全局探索
    每 exchange_every 代双向交换精英个体：
        B 的历史最优 → 替换 A 中最差个体；A 的最优 → 替换 B 中最差粒子（并重置其速度）。
    方案三c（p.adaptive=True）在此基础上让 pc / pm / w / c1 / c2 随多样性与停滞代数变化。
    """
    rng = random.Random(p.seed)
    npr = np.random.default_rng(p.seed)
    N, L = p.pop_size, inst.n_ops
    w, c1, c2 = p.w, p.c1, p.c2
    pc, pm = p.pc, p.pm
    T = max(1, p.exchange_every)

    # ---- 种群 A：GA ----
    pop_a = initial_population(inst, rng, p)
    fit_a = [makespan(inst, x) for x in pop_a]
    n_evals = N

    # ---- 种群 B：PSO ----
    NB = N                                           # 两个种群同规模
    X = np.array([perm_to_position(inst, make_initial(inst, rng, p.init_method))
                  for _ in range(NB)], dtype=float)
    X += npr.uniform(-0.5 / L, 0.5 / L, size=X.shape)
    V = npr.uniform(-p.vmax, p.vmax, size=(NB, L))
    pbest_x = X.copy()
    pbest_fit = [10 ** 9] * NB

    gi = min(range(N), key=lambda i: fit_a[i])
    gbest_perm, gbest_fit = list(pop_a[gi]), fit_a[gi]

    t0 = time.perf_counter()
    history: List[Tuple[int, int]] = []
    stag = 0

    for it in range(1, p.max_iter + 1):
        # ---------- 种群 A：GA 一代（局部精修）----------
        order = sorted(range(N), key=lambda i: fit_a[i])
        new_a = [list(pop_a[i]) for i in order[:p.elite]]
        new_fa = [fit_a[i] for i in order[:p.elite]]
        while len(new_a) < N:
            i = tournament(fit_a, rng, p.tournament)
            j = tournament(fit_a, rng, p.tournament)
            if rng.random() < pc:
                a, b = crossover(inst, pop_a[i], pop_a[j], rng, p.crossover_kind)
            else:
                a, b = list(pop_a[i]), list(pop_a[j])
            if rng.random() < pm:
                a = mutate(inst, a, rng, p.mutation_kinds)
            if rng.random() < pm:
                b = mutate(inst, b, rng, p.mutation_kinds)
            for c in (a, b):
                if len(new_a) >= N:
                    break
                f = makespan(inst, c)
                n_evals += 1
                new_a.append(c)
                new_fa.append(f)
                if f < gbest_fit:
                    gbest_fit, gbest_perm, stag = f, list(c), 0
        pop_a, fit_a = new_a, new_fa

        # ---------- 种群 B：PSO 一代（全局探索）----------
        gb = np.array(perm_to_position(inst, gbest_perm))[None, :]
        r1 = npr.random((NB, 1))
        r2 = npr.random((NB, 1))
        V = w * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gb - X)
        np.clip(V, -p.vmax, p.vmax, out=V)
        X = X + V
        for i in range(NB):
            perm = spv_map(inst, X[i])
            f = makespan(inst, perm)
            n_evals += 1
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest_x[i] = X[i].copy()
                if f < gbest_fit:
                    gbest_fit, gbest_perm, stag = f, list(perm), 0

        # ---------- 定期交换精英个体 ----------
        if it % T == 0:
            bi = min(range(NB), key=lambda i: pbest_fit[i])
            ai = min(range(N), key=lambda i: fit_a[i])
            wa = max(range(N), key=lambda i: fit_a[i])
            wb = max(range(NB), key=lambda i: pbest_fit[i])
            if pbest_fit[bi] < fit_a[wa]:                    # B 的精英 → A
                pop_a[wa] = spv_map(inst, pbest_x[bi])
                fit_a[wa] = pbest_fit[bi]
            if fit_a[ai] < pbest_fit[wb]:                    # A 的精英 → B
                pbest_x[wb] = np.array(perm_to_position(inst, pop_a[ai]))
                pbest_fit[wb] = fit_a[ai]
                X[wb] = pbest_x[wb] + npr.uniform(-0.5 / L, 0.5 / L, L)
                V[wb] = npr.uniform(-p.vmax, p.vmax, L)

        # ---------- 方案三c：自适应参数 ----------
        if p.adaptive:
            w, c1, c2, pc, pm = adapt(p, diversity(pop_a, gbest_perm), stag)
        stag += 1
        history.append((it, gbest_fit))

    name = "方案二" if not p.adaptive else "方案三c"
    return Result(name, inst.name, p.seed, gbest_perm, gbest_fit,
                  history, time.perf_counter() - t0, n_evals, p.max_iter)


# --------------------------------------------------------------------------
# 5. 方案三c：自适应参数混合（= 方案二 + 自适应 pc/pm/w/c1/c2）
# --------------------------------------------------------------------------
def run_hybrid3(inst: Instance, p: Params) -> Result:
    """方案三c：在方案二框架上启用自适应参数，与固定参数版本直接对比。"""
    return run_hybrid2(inst, p.copy(adaptive=True))


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
ALGORITHMS: Dict[str, str] = {
    "pso": "纯 PSO",
    "ga": "纯 GA",
    "hybrid1": "方案一",
    "hybrid2": "方案二",
    "hybrid3": "方案三c",
}

ALGO_ORDER = ["pso", "ga", "hybrid1", "hybrid2", "hybrid3"]


def run(algo: str, inst: Instance, p: Params) -> Result:
    """运行指定算法。algo ∈ {"pso", "ga", "hybrid1", "hybrid2", "hybrid3"}。"""
    table = {
        "pso": run_pso,
        "ga": run_ga,
        "hybrid1": run_hybrid1,
        "hybrid2": run_hybrid2,
        "hybrid3": run_hybrid3,
    }
    if algo not in table:
        raise ValueError(f"未知算法：{algo}，可选 {list(table)}")
    return table[algo](inst, p)
