"""任务二：五套算法在 ft06 上的一致性测试（通过后再开展基准实验）。

检查内容
--------
1. 输出的编码是合法工序置换（每个工件恰好出现 m 次）；
2. 解码得到的调度满足工艺路线约束与机器独占约束；
3. 输出的 makespan 与重新解码的 makespan 一致；
4. 输出的最优调度各机器工序序列可复现同一 makespan。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import csv_path, write_csv                              # noqa: E402
from jsp import check_permutation, check_schedule, decode           # noqa: E402
from jsp import load_instance, makespan                             # noqa: E402
from jsp.algorithms import ALGORITHMS, ALGO_ORDER, Params, run      # noqa: E402

N_SEEDS = 5


def main() -> None:
    print("=" * 72)
    print("任务二　算法实现与混合框架 —— ft06 一致性测试")
    print("=" * 72)

    inst = load_instance("ft06")
    print(f"\n实例：{inst.name}  {inst.n}×{inst.m}  文献最优 {inst.optimum}\n")

    rows = []
    all_ok = True
    for algo in ALGO_ORDER:
        hits = 0
        line = [ALGORITHMS[algo]]
        for seed in range(1, N_SEEDS + 1):
            r = run(algo, inst, Params(seed=seed))
            ok_perm, msg_perm = check_permutation(inst, r.best_perm)
            sched = decode(inst, r.best_perm)
            ok_sch, msg_sch = check_schedule(inst, sched)
            same = (makespan(inst, r.best_perm) == r.best_makespan == sched.makespan)
            ok = ok_perm and ok_sch and same
            all_ok = all_ok and ok
            hits += int(r.best_makespan == inst.optimum)
            line.append(f"seed{seed}: {r.best_makespan}{'' if ok else '✗'}")
            rows.append([ALGORITHMS[algo], seed, r.best_makespan,
                         "通过" if ok_perm else f"失败({msg_perm})",
                         "通过" if ok_sch else f"失败({msg_sch})",
                         "一致" if same else "不一致",
                         "通过" if ok else "失败", r.time_sec, r.n_evals])
        line.append(f"命中最优 {hits}/{N_SEEDS}")
        print("  ".join(f"{x:<12}" for x in line))

    write_csv(csv_path("task2_consistency.csv"),
              ["算法", "种子", "求得 makespan", "编码合法性", "调度约束",
               "makespan 一致性", "结论", "运行时间(s)", "评价次数"], rows)

    print("\n一致性测试结论：" + ("全部通过，可开展基准实验"
                                 if all_ok else "存在未通过项，需修正"))
    print("明细 → results/csv/task2_consistency.csv")


if __name__ == "__main__":
    main()
