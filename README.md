# 基于粒子群与遗传混合算法的作业车间调度（JSP）求解实验

智能算法分析与设计 · 期末课程大作业 · 完整可运行源码与实验数据

---

## 1. 环境依赖

- Python 3.8 及以上
- 依赖包：`numpy`、`matplotlib`、`scipy`（统计检验）；`ortools`（可选，仅用于任务一 ft06 的 CP-SAT 精确验证，主实例求解不使用）

```bash
pip install -r requirements.txt      # numpy matplotlib scipy ortools
```

数据集：`JSP基准数据集/`（OR-Library 标准格式），默认路径写在
`jsp/instance.py::DEFAULT_DATA_ROOT`，可用环境变量 `JSP_DATA` 覆盖：

```bash
set JSP_DATA=C:\path\to\JSP基准数据集
```

## 2. 目录结构

```
├── jsp/                    核心库（约 900 行，注释完整）
│   ├── instance.py         OR-Library 实例解析 + 文献最优值索引
│   ├── schedule.py         解码、makespan 评价、关键路径、可行性检验
│   ├── operators.py        初始解构造（随机 / GT 活性调度 + SPT/MWR/随机规则）、
│                           SPV/LOV 映射、IPOX/POX 交叉、互换/插入/逆转/关键路径变异
│   ├── algorithms.py       五套算法 + 统一参数类 Params 与入口 run()
│   ├── verify.py           可行性测试用例、CP-SAT 精确求解、解码一致性验证
│   └── viz.py              甘特图（含关键路径）、收敛曲线、柱状/折线图
├── scripts/
│   ├── common.py           公共工具：路径、CSV 写出、并行执行、统计与检验
│   ├── task1_verify.py     任务一：实例解析、甘特图、可行性用例、ft06 精确验证
│   ├── task2_smoke.py      任务二：五套算法在 ft06 上的一致性测试
│   ├── task3_benchmark.py  任务三：6 实例 × 5 算法 × 30 次重复 + Wilcoxon 检验
│   ├── task4_sensitivity.py任务四：参数敏感性 / 算子组合 / 初始解构造 / 计算代价
│   └── task5_extension.py  任务五（选做·扩展 a）：ta01、ta11 大规模实例
├── results/
│   ├── csv/                全部数据表（对应报告表 1～表 7）
│   └── fig/                全部图（甘特图、收敛曲线、敏感性折线图）
└── README.md
```

## 3. 编码、解码与算子（任务书 四(三)(四)）

**编码**：基于工序的置换编码。解是长度 `L = n*m` 的整数序列，工件号 j 出现 m 次，
第 k 次出现代表工件 j 的第 k 道工序。该编码天然保证同一工件工序顺序可行。

**解码**：按序列顺序逐个安排工序，开工时刻 =
`max(该工件上一道工序完工时刻, 该机器上一道工序完工时刻)`。
机器独占约束由此自动满足，评价函数无需惩罚项。

**PSO 映射**：粒子是 L 维连续位置向量，用 SPV（Smallest Position Value）映射成置换——
位置值越小越早被填入多重集中的工件号；因只比较相对大小，映射结果必然可行。
逆映射 `perm_to_position` 用于由置换反推位置向量（排名值归一化到 [0,1]）。

**GA 算子**：
- 交叉：IPOX（默认）、POX —— 随机划分工件集，保留父代中一组工件的基因及其相对顺序，
  空位按另一父代顺序填入，因此子代一定满足工序链约束。
- 变异：互换 swap、插入 insertion、逆转 inverse、关键路径偏向变异 critical（加分项，
  只交换关键路径上相邻工序在染色体中的位置）。

**初始解构造**：随机置换；Giffler-Thompson 活性调度构造 + 优先规则（SPT / MWR / 随机规则，
任务书要求至少两种并对比，任务四给出了四种初始解的对比结果）。

## 4. 五套算法（任务书 四(二)）

| 算法 | 设计要点 |
|---|---|
| 纯 PSO | L 维连续位置 + SPV 映射，标准速度—位置更新，pbest/gbest 引导 |
| 纯 GA | 锦标赛选择 + IPOX 交叉 + 变异 + 精英保留 |
| 方案一 | GA 框架 + PSO 引导交叉：个体携带速度标量 v，`v ← w·v + c1·r1·d(x,pbest) + c2·r2·d(x,gbest)`，速度越大交叉概率越高（`pc = 0.7 + 0.25·drive ∈ [0.7,0.95]`）；另可选 pbest / gbest 定向交叉（默认关闭，任务四给出三种取值的对比） |
| 方案二 | PSO 全局搜索 + GA 局部精修：双种群协同（GA 种群 + PSO 种群，规模均为 N），每 10 代双向交换精英个体 |
| 方案三c | 在方案二上启用自适应参数：pc / pm / w / c1 / c2 按种群多样性与停滞代数动态调整，取值仍严格落在任务书建议区间内 |

## 5. 参数设置（任务书 四(七)）

主实验统一参数，全部取自任务书建议区间：

| 参数 | 取值 | 任务书建议 |
|---|---|---|
| 种群规模 N | 100 | 50～200 |
| 最大迭代数 | 1000 | 500～2000 |
| 惯性权重 w | 0.7 | 0.4～0.9 |
| 加速系数 c1、c2 | 2.0 | 1.5～2.5 |
| 交叉率 pc | 0.8 | 0.7～0.95 |
| 变异率 pm | 0.1 | 0.05～0.2 |
| 速度上限 vmax | 0.5 | 任务书未列；位置分量取值 [0,1]，取其一半 |
| 锦标赛规模 / 精英保留 | 3 / 2 | — |

方案三c 的自适应规则（同样落在建议区间内）：

```
w  = 0.4 + 0.5*(1-div)      ∈ [0.4, 0.9]      多样性低 → 减小惯性、加强局部开发
c1 = 1.5 + 1.0*(1-div)      ∈ [1.5, 2.5]      多样性低 → 加强个体认知（促进分散）
c2 = 1.5 + 1.0*div          ∈ [1.5, 2.5]      多样性高 → 加强社会认知（加快收敛）
pc = 0.7 + 0.25*div         ∈ [0.7, 0.95]     多样性高 → 多重组
pm = 0.05 + 0.15*(1-div) + 0.05*停滞比  ∈ [0.05, 0.2]  多样性低/停滞久 → 多变异
```

其中 `div` 为种群个体与全局最优解的平均归一化距离（[0,1]），停滞比 = 无改进代数 / 50。

## 6. 运行方式

```bash
python scripts/task1_verify.py       # 任务一：解析、甘特图、ft06 精确验证（约 10 秒）
python scripts/task2_smoke.py        # 任务二：ft06 一致性测试（约 10 秒）
python scripts/task3_benchmark.py    # 任务三：6×5×30 = 900 次运行（并行，约 10 分钟）
python scripts/task4_sensitivity.py  # 任务四：敏感性 + 算子组合 + 初始解（约 5 分钟）
python scripts/task5_extension.py    # 任务五：ta01 / ta11（约 5 分钟）
```

实验默认使用多进程并行（最多 12 进程），可在 `scripts/common.py::run_many` 中调整。

## 7. 结果索引（报告表 1～表 7 对应文件）

| 报告表格 | 文件 |
|---|---|
| 表 1 基准实例与文献最优值 | `results/csv/task1_instances.csv` |
| 表 3 正确性验证结果 | `results/csv/task1_ft06_exact.csv`、`task1_feasibility.csv` |
| 表 4 基准实验汇总 | `results/csv/task3_summary.csv`（原始数据在 `task3_runs.csv`） |
| 表 5 统计检验结果 | `results/csv/task3_wilcoxon.csv` |
| 表 6 参数敏感性 | `results/csv/task4_sensitivity.csv` |
| 表 7 扩展研究 | `results/csv/task5_taillard.csv` |
| 收敛曲线数据 | `results/csv/task3_convergence.csv` |
| 相同预算补充对比 | `results/csv/task3_equal_budget.csv` |
| 算子组合 / 初始解 / 计算代价 | `results/csv/task4_operators.csv`、`task4_initial.csv`、`task4_initial_quality.csv`、`task4_directed.csv`、`task4_cost.csv` |

主要图件：
- `results/fig/task1_gantt_best.png`：ft06 最优调度甘特图（红框为关键路径）
- `results/fig/task3_conv_*.png`：各实例五算法平均收敛曲线
- `results/fig/task3_gantt_*.png`：最优调度甘特图（标出关键路径）
- `results/fig/task4_pc.png`、`task4_pm.png`、`task4_w.png`、`task4_c1c2.png`：参数敏感性折线
- `results/fig/task5_conv_ta01.png`、`task5_gantt_ta01.png`：大规模实例结果

## 8. 常见问题

**提示 `CP-SAT 不可用（未安装 ortools ...）`？**
说明当前 Python 环境缺少 ortools。脚本会自动退化为用文献最优值（ft06 = 55）作对照，
其余功能不受影响；如需精确求解，在**运行脚本的同一个 Python** 里执行：

```
python -m pip install -r requirements.txt
```

可用 `python -c "import ortools; print(ortools.__version__)"` 确认安装成功。
