"""正确性验证（任务一）：可行性检验测试用例 + ft06 精确求解对比。

1. feasibility_cases / run_feasibility_tests：合法与非法工序序列各若干条，
   验证 check_permutation 能正确判别（任务书要求"合法、非法工序序列各若干"）。
2. exact_cpsat：用 OR-Tools 的 CP-SAT 精确求解小规模实例（ft06），
   仅用于验证解码与 makespan 评价的正确性，主实例求解不使用（任务书明确禁止）。
3. verify_decoder：随机生成合法置换，验证解码结果与快速 makespan 评价一致。
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .instance import Instance
from .operators import init_by_rule, init_random
from .schedule import check_permutation, check_schedule, decode, makespan


# --------------------------------------------------------------------------
# 1. 可行性检验测试用例
# --------------------------------------------------------------------------
def feasibility_cases(inst: Instance) -> List[Tuple[str, List[int], bool]]:
    """返回 (用例名, 工序序列, 期望是否合法)。"""
    rng = random.Random(2026)
    n, m = inst.n, inst.m
    block = [j for j in range(n) for _ in range(m)]

    cases: List[Tuple[str, List[int], bool]] = [
        ("随机置换", init_random(inst, rng), True),
        ("SPT 规则构造", init_by_rule(inst, rng, "spt"), True),
        ("MWR 规则构造", init_by_rule(inst, rng, "mwr"), True),
        ("按工件分块顺序", block, True),
        ("逆序分块", [j for j in range(n - 1, -1, -1) for _ in range(m)], True),
        ("长度不足（少 1 个基因）", block[:-1], False),
        ("长度超出（多 1 个基因）", block + [0], False),
        ("工件号越界（= n）", [n if i == 0 else g for i, g in enumerate(block)], False),
        ("工件号为负", [-1] + block[1:], False),
        ("工件计数不平衡", [0] + block[1:-1] + [1], False),
        ("仅含单一工件", [0] * (n * m), False),
    ]
    return cases


def run_feasibility_tests(inst: Instance, verbose: bool = True) -> Tuple[int, int]:
    """执行可行性检验用例，返回 (通过数, 总数)，并打印表格。"""
    passed = 0
    rows = []
    for name, seq, expect in feasibility_cases(inst):
        ok, msg = check_permutation(inst, seq)
        good = (ok == expect)
        passed += int(good)
        rows.append((name, len(seq), "合法" if ok else "非法", msg,
                     "通过" if good else "失败"))
    if verbose:
        print(f"\n=== {inst.name} 可行性检验测试用例 ===")
        print(f"{'用例':<22}{'长度':>6}{'判定':>6}  说明 / 结果")
        for name, ln, verdict, msg, res in rows:
            print(f"{name:<22}{ln:>6}{verdict:>6}  {msg}  [{res}]")
        print(f"通过 {passed}/{len(rows)}")
    return passed, len(rows)


# --------------------------------------------------------------------------
# 2. 精确求解（CP-SAT）
# --------------------------------------------------------------------------
def exact_cpsat(inst: Instance, time_limit: float = 120.0) -> Optional[int]:
    """用 OR-Tools CP-SAT 精确求解最小 makespan；未安装 ortools 则返回 None。

    模型：每道工序一个区间变量；同一工件相邻工序加先后约束；
          同一机器上所有区间加 NoOverlap 约束；最小化 max(各工件末工序完工时刻)。
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    model = cp_model.CpModel()
    horizon = sum(inst.dur)
    starts, ends, by_machine = {}, {}, {M: [] for M in range(inst.m)}

    for j in range(inst.n):
        for k in range(inst.m):
            M, p = inst.jobs[j][k]
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            model.Add(e == s + p)
            starts[(j, k)] = s
            ends[(j, k)] = e
            by_machine[M].append(model.NewIntervalVar(s, p, e, f"i_{j}_{k}"))
            if k > 0:
                model.Add(s >= ends[(j, k - 1)])          # 工艺路线约束

    for M in range(inst.m):
        model.AddNoOverlap(by_machine[M])                  # 机器独占约束

    cmax = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(cmax, [ends[(j, inst.m - 1)] for j in range(inst.n)])
    model.Minimize(cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        return int(solver.ObjectiveValue())
    if status == cp_model.FEASIBLE:
        return int(solver.Value(cmax))
    return None


# --------------------------------------------------------------------------
# 3. 解码—评价一致性验证
# --------------------------------------------------------------------------
def verify_decoder(inst: Instance, n_random: int = 200, seed: int = 7) -> bool:
    """随机生成合法置换，验证三点：
       1) check_permutation 全部通过；
       2) decode 得到的调度满足工艺路线与机器独占约束；
       3) 快速入口 makespan() 与 Schedule.makespan 一致。
    """
    rng = random.Random(seed)
    for _ in range(n_random):
        perm = init_random(inst, rng)
        ok, msg = check_permutation(inst, perm)
        if not ok:
            print("可行性检验失败:", msg)
            return False
        sched = decode(inst, perm)
        ok, msg = check_schedule(inst, sched)
        if not ok:
            print("约束违反:", msg)
            return False
        if makespan(inst, perm) != sched.makespan:
            print("makespan 不一致")
            return False
    return True
