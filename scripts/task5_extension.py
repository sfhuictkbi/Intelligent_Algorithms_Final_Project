"""任务五（选做·扩展 a）：Taillard 大规模实例上的扩展研究。

内容
----
在 ta01（15×15）、ta11（20×15）上运行全部五套算法，参数与主实验一致
（N=100、1000 代、w=0.7、c1=c2=2.0、pc=0.8、pm=0.1），每个组合重复 10 次，
报告 makespan、RPD 与运行时间，并分析算法随规模扩展的退化程度。

产出
----
results/csv/task5_taillard.csv    表 7 扩展研究结果
results/fig/task5_conv_ta01.png   ta01 收敛曲线对比
results/fig/task5_gantt_ta01.png  ta01 最优调度甘特图
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (csv_path, fig_path, mean, rpd, run_many, stdev,  # noqa: E402
                    write_csv)
from jsp import critical_path, decode, load_instance, plot_convergence  # noqa: E402
from jsp import plot_gantt                                             # noqa: E402
from jsp.algorithms import ALGO_ORDER, ALGORITHMS, Params, run         # noqa: E402

INSTANCES = ["ta01", "ta11"]
N_REPEAT = 10
SEEDS = list(range(1, N_REPEAT + 1))


def _job(args):
    name, algo, seed = args
    inst = load_instance(name)
    r = run(algo, inst, Params(seed=seed))
    return dict(instance=name, algo=algo, algo_cn=ALGORITHMS[algo], seed=seed,
                makespan=r.best_makespan, time=r.time_sec, evals=r.n_evals,
                perm=r.best_perm, history=r.history)


def main() -> None:
    print("=" * 72)
    print(f"任务五（扩展 a）　Taillard 大规模实例（{N_REPEAT} 次重复）")
    print("=" * 72)

    jobs = [(name, algo, seed) for name in INSTANCES for algo in ALGO_ORDER
            for seed in SEEDS]
    print(f"\n共 {len(jobs)} 次独立运行，并行执行中……")
    results = run_many(_job, jobs)

    by = defaultdict(list)
    for r in results:
        by[(r["instance"], r["algo"])].append(r)

    rows = []
    for name in INSTANCES:
        inst = load_instance(name)
        opt = inst.optimum
        print(f"\n=== {name}  {inst.n}×{inst.m}  工序数 {inst.n_ops}  "
              f"文献最优 {opt} ===")
        print(f"  {'算法':<8}{'最优':>7}{'平均':>9}{'标准差':>8}{'RPD(平均)%':>11}"
              f"{'平均时间(s)':>12}")
        for algo in ALGO_ORDER:
            rs = by[(name, algo)]
            ms = [r["makespan"] for r in rs]
            avg, sd = mean(ms), stdev(ms)
            tm = mean([r["time"] for r in rs])
            rows.append([name, f"{inst.n}×{inst.m}", ALGORITHMS[algo], min(ms),
                         round(avg, 2), round(sd, 2), round(rpd(avg, opt), 2),
                         round(tm, 2)])
            print(f"  {ALGORITHMS[algo]:<8}{min(ms):>7}{avg:>9.1f}{sd:>8.2f}"
                  f"{rpd(avg, opt):>11.2f}{tm:>12.2f}")
    write_csv(csv_path("task5_taillard.csv"),
              ["实例", "规模", "算法", "最优 makespan", "平均 makespan", "标准差",
               "RPD(平均)%", "平均运行时间(s)"], rows)

    # 收敛曲线与甘特图（ta01）
    name = "ta01"
    curves = {}
    for algo in ALGO_ORDER:
        rs = by[(name, algo)]
        n_pt = min(len(r["history"]) for r in rs)
        step = max(1, n_pt // 100)
        xs = [rs[0]["history"][i][0] for i in range(0, n_pt, step)]
        ys = [mean([r["history"][i][1] for r in rs]) for i in range(0, n_pt, step)]
        curves[ALGORITHMS[algo]] = (xs, ys)
    plot_convergence(curves, title=f"{name} 收敛曲线对比（{N_REPEAT} 次平均）",
                     xlabel="迭代代数", ylabel="当前最优 makespan（10 次平均）",
                     save=fig_path("task5_conv_ta01.png"))

    best = min(by[(name, "hybrid3")], key=lambda r: r["makespan"])
    inst = load_instance(name)
    sched = decode(inst, best["perm"])
    cp = critical_path(inst, sched)
    plot_gantt(inst, sched, cp,
               title=f"{name} 最优调度甘特图（方案三c，makespan = {sched.makespan}，"
                     f"文献最优 {inst.optimum}）",
               save=fig_path("task5_gantt_ta01.png"))
    print(f"\n  ta01 最优解 makespan = {sched.makespan}（文献最优 {inst.optimum}）")
    print("  图 → fig/task5_conv_ta01.png、fig/task5_gantt_ta01.png")

    # 规模退化对比：汇总各实例的方案三c RPD
    print("\n规模扩展退化情况（方案三c 的 RPD(平均)%）")
    print("  建议在报告中结合任务三的 6 个实例一并对比：")
    print("  ft06(6×6) → ft10(10×10) → la21(15×10) → ta01(15×15) → ta11(20×15)")

    print("\n任务五完成。")


if __name__ == "__main__":
    main()
