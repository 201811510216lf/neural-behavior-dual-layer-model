# Data profile: source_data/fig_d_random_window_null.csv

**Shape:** 4500 rows × 8 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `target` | ordinal | 4500 | 0 | 3 levels: 2(1500), 3(1500), 5(1500); min_group_n=1500 |
| `target_label` | categorical | 4500 | 0 | 3 levels: L2(1500), L3(1500), L5(1500); min_group_n=1500 |
| `outcome` | categorical | 4500 | 0 | 3 levels: Stage-2 macro recall(1500), Hesitating recall(1500), Walking recall(1500); min_group_n=1500 |
| `fold` | categorical | 4500 | 0 | 3 levels: fold01(1500), fold02(1500), fold03(1500); min_group_n=1500 |
| `repeat` | continuous | 4500 | 0 | mean=250, sd=144, range=[1, 500], skew=0.00 (approximately symmetric); -> log axis |
| `random_delta` | continuous | 4500 | 0 | mean=-3.53, sd=3.57, range=[-12.4, 2.5], skew=-0.17 (approximately symmetric) |
| `observed_maximum_delta` | continuous | 4500 | 0 | mean=-2.89, sd=3.5, range=[-8.79, 2.5], skew=-0.20 (approximately symmetric) |
| `empirical_one_sided_p` | continuous | 4500 | 0 | mean=0.859, sd=0.202, range=[0.451, 1], skew=-1.12 (highly skewed); outliers=500 (IQR) |

## Group structure
- Grouped by: `target`, `fold`
- Number of groups: 9
- Group size: min=500, median=500, max=500

## Correlations (Pearson, sorted by |r|)
- `random_delta` ↔ `observed_maximum_delta` : r = 0.937 (very strong)
- `observed_maximum_delta` ↔ `empirical_one_sided_p` : r = 0.750 (very strong)
- `random_delta` ↔ `empirical_one_sided_p` : r = 0.625 (strong)
- `repeat` ↔ `random_delta` : r = -0.003 (negligible)
- `repeat` ↔ `empirical_one_sided_p` : r = 0.000 (negligible)
- `repeat` ↔ `observed_maximum_delta` : r = 0.000 (negligible)

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['repeat', 'random_delta', 'observed_maximum_delta', 'empirical_one_sided_p']）或 pairplot 散点矩阵
- 分类维度组合数 = 81（target, target_label, outcome, fold 全交叉），**一张图塞不下**——建议按某一维拆成多面板，或选择子集。
- repeat 跨数个量级（1 ~ 500）→ 用对数 y 轴
- empirical_one_sided_p 高度偏态（skew=-1.12）→ 考虑对数变换或小提琴图代替均值柱图

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
