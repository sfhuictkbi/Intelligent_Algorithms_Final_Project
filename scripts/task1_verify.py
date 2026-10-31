"""任务一：实例解析、甘特图可视化与正确性验证。

产出
----
results/csv/task1_instances.csv      解析得到的 6 个基准实例信息（对应表 1）
results/csv/task1_ft06_exact.csv     ft06 精确求解对比（对应表 3）
results/csv/task1_feasibility.csv    可行性检验测试用例结果
results/fig/task1_gantt_random.png   随机可行调度的甘特图
results/fig/task1_gantt_best.png     算法求得最优调度的甘特图（标出关键路径）
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import csv_path, fig_path, rpd, write_csv               # noqa: E402
from jsp import BENCHMARK, critical_path, decode, load_instance     # noqa: E402
from jsp import makespan, plot_gantt                                # noqa: E402
from jsp.algorithms import ALGO_ORDER, Params, run                  # noqa: E402
from jsp.operators import init_random                               # noqa: E402
from jsp.schedule import check_permutation                          # noqa: E402
from jsp.verify import (exact_cpsat, feasibility_cases,             # noqa: E402
                        run_feasibility_tests, verify_decoder)


def main() -> None:
    print("=" * 72)
    print("任务一　实例解析、甘特图与正确性验证")
    print("=" * 72)

    # ---------- 1. 实例解析 ----------
    print("\n[1] OR-Library 实例解析")
    rows = []
    for name in BENCHMARK:
        inst = load_instance(name)
        rows.append([name, f"{inst.n}×{inst.m}", inst.n_ops, inst.optimum, sum(inst.dur)])
        print(f"  {name:<6} {inst.n}×{inst.m}  工序数 {inst.n_ops:3d}  "
              f"文献最优 {inst.optimum}  总加工时间 {sum(inst.dur)}")
    write_csv(csv_path("task1_instances.csv"),
              ["实例", "规模(工件×机器)", "工序数", "文献最优 makespan", "总加工时间"], rows)

    # ---------- 2. ft06 精确求解 + 评价函数验证 ----------
    print("\n[2] ft06 精确求解（OR-Tools CP-SAT）与评价函数验证")
    inst = load_instance("ft06")
    t0 = time.perf_counter()
    exact = exact_cpsat(inst, time_limit=120.0)
    t_exact = time.perf_counter() - t0
    if exact is None:
        # ortools 未安装（或求解失败）：退化为用文献最优值做对照，不影响其余验证
        print("  CP-SAT 不可用（未安装 ortools 或求解失败）——改用文献最优值"
              f" {inst.optimum} 作对照。如需精确求解，请执行: pip install ortools")
        exact = inst.optimum
    print(f"  ft06 最优 makespan 参照值 = {exact}   （精确求解耗时 {t_exact:.2f}s）")

    rng = random.Random(7)
    best_rand = min(makespan(inst, init_random(inst, rng)) for _ in range(2000))
    ok_decode = verify_decoder(inst, n_random=200, seed=7)
    print(f"  随机采样 2000 个可行解的最小 makespan = {best_rand}（应 ≥ {exact}）")
    print(f"  解码—评价一致性验证（200 个随机解）：{'通过' if ok_decode else '失败'}")

    ft06_rows = []
    for algo in ALGO_ORDER:
        r = run(algo, inst, Params(seed=1))
        ft06_rows.append([r.algo, exact, r.best_makespan, r.best_makespan - exact,
                          round(rpd(r.best_makespan, exact), 2),
                          "一致" if r.best_makespan >= exact else "异常"])
        print(f"  {r.algo:<7} 求得 {r.best_makespan}  偏差 {r.best_makespan - exact}")
    write_csv(csv_path("task1_ft06_exact.csv"),
              ["算法", "精确最优 makespan", "算法求得最优", "偏差", "RPD(%)", "结论"],
              ft06_rows)

    # ---------- 3. 可行性检验测试用例 ----------
    print("\n[3] 可行性检验测试用例")
    passed, total = run_feasibility_tests(inst)
    case_rows = []
    for name, seq, expect in feasibility_cases(inst):
        ok, msg = check_permutation(inst, seq)
        case_rows.append([name, len(seq), "合法" if expect else "非法",
                          "合法" if ok else "非法", msg, "通过" if ok == expect else "失败"])
    write_csv(csv_path("task1_feasibility.csv"),
              ["用例", "序列长度", "期望判定", "实际判定", "说明", "结果"], case_rows)
    print(f"  用例通过 {passed}/{total}")

    # ---------- 4. 甘特图 ----------
    print("\n[4] 甘特图")
    perm_rand = init_random(inst, random.Random(2026))
    sched_rand = decode(inst, perm_rand)
    plot_gantt(inst, sched_rand, critical_path(inst, sched_rand),
               title=f"ft06 随机可行调度甘特图（makespan = {sched_rand.makespan}）",
               save=fig_path("task1_gantt_random.png"))
    print(f"  随机调度 makespan = {sched_rand.makespan} → fig/task1_gantt_random.png")

    best = min((run(a, inst, Params(seed=1)) for a in ALGO_ORDER),
               key=lambda r: r.best_makespan)
    sched_best = decode(inst, best.best_perm)
    cp = critical_path(inst, sched_best)
    plot_gantt(inst, sched_best, cp,
               title=f"ft06 最优调度甘特图（{best.algo}，makespan = {sched_best.makespan}，"
                     f"关键路径 {len(cp)} 道工序）",
               save=fig_path("task1_gantt_best.png"))
    print(f"  {best.algo} 最优调度 makespan = {sched_best.makespan}"
          f"（关键路径 {len(cp)} 道工序）→ fig/task1_gantt_best.png")

    print("\n任务一完成。")


if __name__ == "__main__":
    main()
