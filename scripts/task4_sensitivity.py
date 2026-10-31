"""任务四：参数敏感性与算法行为分析。

实验设置（严格按任务书表 6）
----------------------------
在 ft10 上，对交叉率 pc、变异率 pm、惯性权重 w、加速系数 c1=c2 四个参数
各取 4 个水平，其余参数保持主实验默认值（N=100、1000 代），每个水平重复 10 次：
    pc    ∈ {0.7, 0.8, 0.9, 0.95}     → 纯 GA
    pm    ∈ {0.05, 0.1, 0.15, 0.2}    → 纯 GA
    w     ∈ {0.4, 0.6, 0.8, 0.9}      → 纯 PSO
    c1=c2 ∈ {1.0, 1.5, 2.0, 2.5}      → 纯 PSO

另做两组分析实验（任务书 四(四)/四(五) 要求）：
    算子组合：仅变异 / IPOX+互换 / IPOX+全变异 / POX+全变异
    初始解构造：随机置换 / SPT 规则 / MWR 规则

产出
----
results/csv/task4_sensitivity.csv   表 6 参数敏感性实验结果
results/csv/task4_operators.csv     交叉、变异算子组合对比
results/csv/task4_initial.csv       初始解构造方式对比
results/csv/task4_cost.csv          计算代价对比（评价次数与运行时间）
results/fig/task4_pc/pm/w/c1.png    四个参数的敏感性折线图
results/fig/task4_operators.png     算子组合柱状图
results/fig/task4_initial.png       初始解对比柱状图
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (csv_path, fig_path, mean, rpd, run_many, stdev,  # noqa: E402
                    write_csv)
from jsp import load_instance, plot_bars, plot_lines                # noqa: E402
from jsp.algorithms import ALGORITHMS, Params, run                  # noqa: E402

INSTANCE = "ft10"
N_REPEAT = 10
SEEDS = list(range(1, N_REPEAT + 1))

# 任务书表 6 给定的参数水平
LEVELS = {
    "pc": ("ga", [0.7, 0.8, 0.9, 0.95]),
    "pm": ("ga", [0.05, 0.1, 0.15, 0.2]),
    "w": ("pso", [0.4, 0.6, 0.8, 0.9]),
    "c1=c2": ("pso", [1.0, 1.5, 2.0, 2.5]),
}
PARAM_LABEL = {"pc": "交叉率 pc（纯 GA）", "pm": "变异率 pm（纯 GA）",
               "w": "惯性权重 w（纯 PSO）", "c1=c2": "加速系数 c1=c2（纯 PSO）"}

# 算子组合：名称 → (交叉率, 变异算子集合)
OP_COMBOS = [
    ("仅变异（不交叉）", 0.0, ("swap", "insertion", "inverse", "critical")),
    ("IPOX + 互换变异", 0.8, ("swap",)),
    ("IPOX + 全变异", 0.8, ("swap", "insertion", "inverse", "critical")),
    ("POX + 全变异", 0.8, ("swap", "insertion", "inverse", "critical")),
]
OP_CROSS = {"仅变异（不交叉）": "ipox", "IPOX + 互换变异": "ipox",
            "IPOX + 全变异": "ipox", "POX + 全变异": "pox"}


def _job(args):
    """一次运行：(算法, 种子, 参数字典)，其中 "_tag" 只是结果分组标签。"""
    name, algo, seed, kw = args
    kw = dict(kw)
    tag = kw.pop("_tag", "")
    inst = load_instance(name)
    r = run(algo, inst, Params(seed=seed, **kw))
    return dict(algo=algo, algo_cn=ALGORITHMS[algo], seed=seed, makespan=r.best_makespan,
                time=r.time_sec, evals=r.n_evals, history=r.history, extra=tag)


def main() -> None:
    print("=" * 72)
    print(f"任务四　参数敏感性与算法行为分析（{INSTANCE}，每水平 {N_REPEAT} 次重复）")
    print("=" * 72)

    inst = load_instance(INSTANCE)
    opt = inst.optimum

    # ---------- 1. 参数敏感性（表 6）----------
    print("\n[1] 参数敏感性实验（其余参数固定为主实验默认值）")
    jobs, tags = [], []
    for param, (algo, levels) in LEVELS.items():
        for lv in levels:
            for s in SEEDS:
                kw = {"pc": lv} if param == "pc" else \
                     {"pm": lv} if param == "pm" else \
                     {"w": lv} if param == "w" else {"c1": lv, "c2": lv}
                kw = dict(kw, _tag=f"{param}={lv}")
                jobs.append((INSTANCE, algo, s, kw))
    results = run_many(_job, jobs)

    grouped = defaultdict(list)
    for r in results:
        grouped[r["extra"]].append(r)

    sens_rows = []
    curves = {}
    for param, (algo, levels) in LEVELS.items():
        xs, ys_avg, ys_best = [], [], []
        print(f"\n  {PARAM_LABEL[param]}")
        print(f"    {'取值':>8}{'平均 makespan':>14}{'最优 makespan':>14}"
              f"{'RPD(平均)%':>12}{'平均时间(s)':>12}")
        for lv in levels:
            rs = grouped[f"{param}={lv}"]
            ms = [r["makespan"] for r in rs]
            tm = mean([r["time"] for r in rs])
            sens_rows.append([param, lv, round(mean(ms), 2), min(ms),
                              round(rpd(mean(ms), opt), 2), round(tm, 2)])
            print(f"    {lv:>8}{mean(ms):>14.1f}{min(ms):>14}{rpd(mean(ms), opt):>12.2f}"
                  f"{tm:>12.2f}")
            xs.append(lv)
            ys_avg.append(round(mean(ms), 1))
            ys_best.append(min(ms))
        curves[param] = (xs, ys_avg, ys_best)
        plot_lines(xs, {"平均 makespan": ys_avg, "最优 makespan": ys_best},
                   xlabel=PARAM_LABEL[param], ylabel="makespan",
                   title=f"{INSTANCE} 参数敏感性：{PARAM_LABEL[param]}"
                         f"（{N_REPEAT} 次重复）",
                   save=fig_path(f"task4_{param.replace('=', '')}.png"))
    write_csv(csv_path("task4_sensitivity.csv"),
              ["参数", "取值", "平均 makespan", "最优 makespan", "RPD(平均)%",
               "平均运行时间(s)"], sens_rows)
    print("\n  折线图 → fig/task4_pc.png、task4_pm.png、task4_w.png、task4_c1c2.png")

    # ---------- 2. 算子组合 ----------
    print("\n[2] 交叉、变异算子组合对解质量的贡献（纯 GA，10 次重复）")
    jobs = []
    for label, pc, mk in OP_COMBOS:
        for s in SEEDS:
            jobs.append((INSTANCE, "ga", s,
                         {"pc": pc, "mutation_kinds": mk,
                          "crossover_kind": OP_CROSS[label], "_tag": label}))
    op_res = defaultdict(list)
    for r in run_many(_job, jobs):
        op_res[r["extra"]].append(r)
    op_rows = []
    print(f"  {'算子组合':<18}{'平均 makespan':>14}{'最优':>8}{'RPD(平均)%':>12}"
          f"{'平均时间(s)':>12}")
    for label, _, _ in OP_COMBOS:
        rs = op_res[label]
        ms = [r["makespan"] for r in rs]
        op_rows.append([label, round(mean(ms), 2), min(ms), round(rpd(mean(ms), opt), 2),
                        round(mean([r["time"] for r in rs]), 2)])
        print(f"  {label:<18}{mean(ms):>14.1f}{min(ms):>8}{rpd(mean(ms), opt):>12.2f}"
              f"{mean([r['time'] for r in rs]):>12.2f}")
    write_csv(csv_path("task4_operators.csv"),
              ["算子组合", "平均 makespan", "最优 makespan", "RPD(平均)%",
               "平均运行时间(s)"], op_rows)
    plot_bars([r[0] for r in op_rows], [r[1] for r in op_rows],
              ylabel="平均 makespan", title=f"{INSTANCE} 算子组合对比（10 次重复）",
              save=fig_path("task4_operators.png"))

    # ---------- 3. 初始解构造 ----------
    print("\n[3] 初始解构造方式对最终解质量的影响（10 次重复）")
    jobs = [(INSTANCE, algo, s, {"init_method": m, "_tag": f"{ALGORITHMS[algo]}|{m}"})
            for algo in ("ga", "hybrid3")
            for m in ("random", "spt", "mwr", "gt_random") for s in SEEDS]
    init_res = defaultdict(list)
    for r in run_many(_job, jobs):
        init_res[r["extra"]].append(r)
    init_rows = []
    print(f"  {'算法':<8}{'初始解':<12}{'平均 makespan':>14}{'最优':>8}{'RPD(平均)%':>12}")
    for algo in ("ga", "hybrid3"):
        for m in ("random", "spt", "mwr", "gt_random"):
            rs = init_res[f"{ALGORITHMS[algo]}|{m}"]
            ms = [r["makespan"] for r in rs]
            init_rows.append([ALGORITHMS[algo], m, round(mean(ms), 2), min(ms),
                              round(rpd(mean(ms), opt), 2)])
            print(f"  {ALGORITHMS[algo]:<8}{m:<12}{mean(ms):>14.1f}{min(ms):>8}"
                  f"{rpd(mean(ms), opt):>12.2f}")
    write_csv(csv_path("task4_initial.csv"),
              ["算法", "初始解构造", "平均 makespan", "最优 makespan", "RPD(平均)%"],
              init_rows)

    # 初始解自身的初始 makespan（对比"初始解质量"与"最终解质量"）
    import random as _rnd
    from jsp import makespan
    from jsp.operators import make_initial
    first_rows = []
    for m in ("random", "spt", "mwr", "gt_random"):
        vals = [makespan(inst, make_initial(inst, _rnd.Random(s), m)) for s in SEEDS]
        first_rows.append([m, round(mean(vals), 1), min(vals)])
    print(f"\n  初始解本身的 makespan（种子 1~{N_REPEAT}）：" +
          "  ".join(f"{a}: 平均 {b:.0f}/最优 {c}" for a, b, c in first_rows))
    write_csv(csv_path("task4_initial_quality.csv"),
              ["初始解构造", "初始平均 makespan", "初始最优 makespan"], first_rows)

    # ---------- 3b. 方案一定向交叉对象（混合增益来源分析）----------
    print("\n[3b] 方案一定向交叉对象对比（10 次重复）")
    jobs = [(INSTANCE, "hybrid1", s, {"directed_partner": dp, "_tag": dp})
            for dp in ("none", "pbest", "gbest") for s in SEEDS]
    dp_res = defaultdict(list)
    for r in run_many(_job, jobs):
        dp_res[r["extra"]].append(r)
    dp_rows = []
    label = {"none": "随机个体（仅速度调制 pc）", "pbest": "pbest 定向", "gbest": "gbest 定向"}
    print(f"  {'交叉对象':<24}{'平均 makespan':>14}{'最优':>8}{'RPD(平均)%':>12}")
    for dp in ("none", "pbest", "gbest"):
        rs = dp_res[dp]
        ms = [r["makespan"] for r in rs]
        dp_rows.append([label[dp], round(mean(ms), 2), min(ms), round(rpd(mean(ms), opt), 2)])
        print(f"  {label[dp]:<24}{mean(ms):>14.1f}{min(ms):>8}{rpd(mean(ms), opt):>12.2f}")
    write_csv(csv_path("task4_directed.csv"),
              ["方案一交叉对象", "平均 makespan", "最优 makespan", "RPD(平均)%"], dp_rows)

    # ---------- 4. 计算代价 ----------
    print("\n[4] 计算代价对比（每代评价次数与运行时间）")
    cost_jobs = [(INSTANCE, a, s, {"_tag": a}) for a in
                 ("pso", "ga", "hybrid1", "hybrid2", "hybrid3") for s in range(1, 6)]
    cost_res = defaultdict(list)
    for r in run_many(_job, cost_jobs):
        cost_res[r["algo_cn"]].append(r)
    cost_rows = []
    print(f"  {'算法':<8}{'平均评价次数':>14}{'平均时间(s)':>12}{'平均 makespan':>14}")
    for a in ("pso", "ga", "hybrid1", "hybrid2", "hybrid3"):
        rs = cost_res[ALGORITHMS[a]]
        ev = mean([r["evals"] for r in rs])
        tm = mean([r["time"] for r in rs])
        ms = mean([r["makespan"] for r in rs])
        cost_rows.append([ALGORITHMS[a], int(ev), round(tm, 2), round(ms, 2)])
        print(f"  {ALGORITHMS[a]:<8}{int(ev):>14}{tm:>12.2f}{ms:>14.1f}")
    write_csv(csv_path("task4_cost.csv"),
              ["算法", "平均评价次数", "平均运行时间(s)", "平均 makespan（5 次）"],
              cost_rows)

    print("\n任务四完成。")


if __name__ == "__main__":
    main()
