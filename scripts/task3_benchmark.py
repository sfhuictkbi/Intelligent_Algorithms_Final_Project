"""任务三：基准实验与统计对比（6 实例 × 5 算法 × 30 次独立重复）。

设置
----
实例：ft06、ft10、ft20、la01、la11、la21
算法：纯 PSO、纯 GA、方案一、方案二、方案三c
重复：30 次（随机种子 1～30）
参数：N=100，最大迭代 1000 代，w=0.7，c1=c2=2.0，pc=0.8，pm=0.1（任务书建议区间内）

产出
----
results/csv/task3_runs.csv          每次运行的原始结果
results/csv/task3_summary.csv       表 4 基准实验汇总
results/csv/task3_wilcoxon.csv      表 5 统计检验结果
results/csv/task3_convergence.csv   各实例平均收敛曲线数据
results/fig/task3_conv_*.png        各实例收敛曲线对比图
results/fig/task3_gantt_*.png       最优调度甘特图（标出关键路径）
results/csv/task3_equal_budget.csv  相同评价预算下的补充对比（纯 GA 加倍代数）
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (csv_path, fig_path, fmt_p, mean, paired_t, rpd,      # noqa: E402
                    run_many, stdev, wilcoxon, write_csv)
from jsp import BENCHMARK, critical_path, decode, load_instance     # noqa: E402
from jsp import plot_convergence, plot_gantt                        # noqa: E402
from jsp.algorithms import ALGO_ORDER, ALGORITHMS, Params, run      # noqa: E402

N_REPEAT = 30                       # 每个"算法 × 实例"重复 30 次（种子 1～30）
SEEDS = list(range(1, N_REPEAT + 1))
CONV_INSTANCES = ["ft06", "ft10", "la01", "la21"]   # 画收敛曲线的实例
GANT_INSTANCES = ["ft10", "la21"]                   # 画甘特图的实例


def _job(args):
    """一次独立运行（供并行调用）。"""
    name, algo, seed = args
    inst = load_instance(name)
    r = run(algo, inst, Params(seed=seed))
    return dict(instance=name, algo=algo, algo_cn=ALGORITHMS[algo], seed=seed,
                makespan=r.best_makespan, time=r.time_sec, evals=r.n_evals,
                perm=r.best_perm, history=r.history)


def _conv_job(args):
    """相同评价预算的补充实验：纯 GA 加倍代数，与方案二 / 方案三c 对比。"""
    name, algo, seed, max_iter = args
    inst = load_instance(name)
    r = run(algo, inst, Params(seed=seed, max_iter=max_iter))
    return dict(instance=name, algo=algo, algo_cn=f"{ALGORITHMS[algo]}（{max_iter}代）",
                seed=seed, makespan=r.best_makespan, time=r.time_sec, evals=r.n_evals)


def main() -> None:
    print("=" * 72)
    print(f"任务三　基准实验：{len(BENCHMARK)} 实例 × {len(ALGO_ORDER)} 算法 "
          f"× {N_REPEAT} 次重复")
    print("=" * 72)

    jobs = [(name, algo, seed) for name in BENCHMARK for algo in ALGO_ORDER
            for seed in SEEDS]
    print(f"\n共 {len(jobs)} 次独立运行，并行执行中……")
    results = run_many(_job, jobs)
    print(f"完成 {len(results)} 次运行")

    # ---------- 原始数据 ----------
    write_csv(csv_path("task3_runs.csv"),
              ["实例", "算法", "种子", "makespan", "运行时间(s)", "评价次数"],
              [[r["instance"], r["algo_cn"], r["seed"], r["makespan"],
                round(r["time"], 3), r["evals"]] for r in results])

    # ---------- 表 4：汇总 ----------
    by: dict = defaultdict(list)
    for r in results:
        by[(r["instance"], r["algo"])].append(r)

    instances = {name: load_instance(name) for name in BENCHMARK}
    summary = []
    print("\n表 4　基准实验汇总（30 次重复）")
    header = f"{'实例':<6}{'算法':<8}{'最优':>7}{'平均':>9}{'标准差':>8}{'RPD(平均)%':>11}" \
             f"{'RPD(最优)%':>11}{'平均时间(s)':>12}{'评价次数':>10}"
    for name in BENCHMARK:
        opt = instances[name].optimum
        print("-" * 82)
        print(header)
        for algo in ALGO_ORDER:
            rs = by[(name, algo)]
            ms = [r["makespan"] for r in rs]
            avg, sd = mean(ms), stdev(ms)
            tm = mean([r["time"] for r in rs])
            ev = mean([r["evals"] for r in rs])
            summary.append([name, ALGORITHMS[algo], min(ms), round(avg, 2), round(sd, 2),
                            round(rpd(avg, opt), 2), round(rpd(min(ms), opt), 2),
                            round(tm, 2), int(ev)])
            print(f"{name:<6}{ALGORITHMS[algo]:<8}{min(ms):>7}{avg:>9.1f}{sd:>8.2f}"
                  f"{rpd(avg, opt):>11.2f}{rpd(min(ms), opt):>11.2f}{tm:>12.2f}{int(ev):>10}")
    write_csv(csv_path("task3_summary.csv"),
              ["实例", "算法", "最优 makespan", "平均 makespan", "标准差",
               "RPD(平均)%", "RPD(最优)%", "平均运行时间(s)", "平均评价次数"],
              summary)

    # ---------- 表 5：统计检验 ----------
    print("\n表 5　统计检验（Wilcoxon 符号秩检验，α = 0.05）")
    pairs = []
    for name in BENCHMARK:
        for a, b in [("hybrid1", "ga"), ("hybrid2", "ga"), ("hybrid3", "ga"),
                     ("hybrid3", "hybrid2"), ("hybrid1", "pso"), ("hybrid2", "pso")]:
            pairs.append((name, a, b))
    test_rows = []
    for name, a, b in pairs:
        ra = {r["seed"]: r["makespan"] for r in by[(name, a)]}
        rb = {r["seed"]: r["makespan"] for r in by[(name, b)]}
        seeds = sorted(ra)
        xa = [ra[s] for s in seeds]
        xb = [rb[s] for s in seeds]
        stat, p = wilcoxon(xa, xb)
        tstat, tp = paired_t(xa, xb)
        diff = mean(xa) - mean(xb)
        sig = "显著" if p == p and p < 0.05 else "不显著"
        better = ALGORITHMS[a] if diff < 0 else (ALGORITHMS[b] if diff > 0 else "持平")
        test_rows.append([name, f"{ALGORITHMS[a]} vs {ALGORITHMS[b]}",
                          "Wilcoxon 符号秩检验", round(stat, 1), fmt_p(p),
                          "配对 t 检验", round(tstat, 3), fmt_p(tp),
                          round(diff, 2), sig, better])
        if name in ("ft10", "la21"):
            print(f"  {name}  {ALGORITHMS[a]} vs {ALGORITHMS[b]}: "
                  f"平均差 {diff:+.1f}  W = {stat:.0f}  p = {fmt_p(p)}  {sig}")
    write_csv(csv_path("task3_wilcoxon.csv"),
              ["实例", "对比", "检验方法1", "统计量 W", "p 值（Wilcoxon）",
               "检验方法2", "统计量 t", "p 值（配对 t）",
               "平均差(前者-后者)", "显著性(α=0.05)", "更优者"], test_rows)

    # ---------- 收敛曲线 ----------
    print("\n收敛曲线")
    conv_rows = []
    for name in CONV_INSTANCES:
        curves = {}
        for algo in ALGO_ORDER:
            rs = by[(name, algo)]
            n_pt = min(len(r["history"]) for r in rs)
            step = max(1, n_pt // 100)
            xs = [rs[0]["history"][i][0] for i in range(0, n_pt, step)]
            ys = [mean([r["history"][i][1] for r in rs]) for i in range(0, n_pt, step)]
            curves[ALGORITHMS[algo]] = (xs, ys)
            for x, y in zip(xs, ys):
                conv_rows.append([name, ALGORITHMS[algo], x, round(y, 2)])
        plot_convergence(curves, title=f"{name} 收敛曲线对比（{N_REPEAT} 次平均）",
                         xlabel="迭代代数", ylabel="当前最优 makespan（30 次平均）",
                         save=fig_path(f"task3_conv_{name}.png"))
        print(f"  fig/task3_conv_{name}.png")
    write_csv(csv_path("task3_convergence.csv"),
              ["实例", "算法", "迭代代数", "当前最优 makespan(30次平均)"], conv_rows)

    # ---------- 最优调度甘特图 ----------
    print("\n最优调度甘特图（标出关键路径）")
    for name in GANT_INSTANCES:
        best = min(by[(name, "hybrid3")], key=lambda r: r["makespan"])
        inst = instances[name]
        sched = decode(inst, best["perm"])
        cp = critical_path(inst, sched)
        plot_gantt(inst, sched, cp,
                   title=f"{name} 最优调度甘特图（方案三c，makespan = {sched.makespan}，"
                         f"文献最优 {inst.optimum}，关键路径 {len(cp)} 道工序）",
                   save=fig_path(f"task3_gantt_{name}.png"))
        print(f"  {name}: makespan = {sched.makespan}（文献最优 {inst.optimum}）"
              f" → fig/task3_gantt_{name}.png")

    # ---------- 补充：相同评价预算对比 ----------
    print("\n补充实验：相同评价预算下（纯 GA 加倍代数）与方案二 / 方案三c 对比")
    extra_jobs = [("ft10", "ga", s, 2000) for s in SEEDS] + \
                 [("ft10", "hybrid2", s, 1000) for s in SEEDS] + \
                 [("ft10", "hybrid3", s, 1000) for s in SEEDS]
    extra = run_many(_conv_job, extra_jobs)
    eb = defaultdict(list)
    for r in extra:
        eb[r["algo_cn"]].append(r)
    extra_rows = []
    opt = instances["ft10"].optimum
    print(f"  {'算法':<20}{'平均 makespan':>14}{'RPD%':>8}{'平均评价次数':>14}{'平均时间(s)':>12}")
    for key in ["纯 GA（2000代）", "方案二（1000代）", "方案三c（1000代）"]:
        rs = eb[key]
        ms = [r["makespan"] for r in rs]
        extra_rows.append([key, round(mean(ms), 2), round(rpd(mean(ms), opt), 2),
                           int(mean([r["evals"] for r in rs])),
                           round(mean([r["time"] for r in rs]), 2)])
        print(f"  {key:<20}{mean(ms):>14.1f}{rpd(mean(ms), opt):>8.2f}"
              f"{mean([r['evals'] for r in rs]):>14.0f}"
              f"{mean([r['time'] for r in rs]):>12.2f}")
    write_csv(csv_path("task3_equal_budget.csv"),
              ["算法（相同评价预算）", "平均 makespan", "RPD(%)", "平均评价次数",
               "平均运行时间(s)"], extra_rows)

    print("\n任务三完成。")


if __name__ == "__main__":
    main()
