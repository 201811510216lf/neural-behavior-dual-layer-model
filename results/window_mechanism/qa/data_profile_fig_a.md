# Data profile: source_data/fig_a_window_behavior_delta_recall.csv

**Shape:** 75 rows × 8 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `fold` | categorical | 75 | 0 | 3 levels: fold01(25), fold02(25), fold03(25); min_group_n=25 |
| `window` | categorical | 75 | 0 | 5 levels: L1(15), L2(15), L3(15), L4(15), L5(15); min_group_n=15 |
| `target_raw_label` | ordinal | 75 | 0 | 5 levels: 1(15), 2(15), 3(15), 4(15), 5(15); min_group_n=15 |
| `behavior` | categorical | 75 | 0 | 5 levels: Hole Exploration(15), Wrong Hole Exploration(15), Hesitating(15), Changing Direction(15), Walking(15); min_group_n=15 |
| `metric` | categorical | 75 | 0 | 1 levels: recall(75); min_group_n=75 |
| `baseline` | continuous | 75 | 0 | mean=75.6, sd=8.16, range=[58.7, 85], skew=-0.59 (moderately skewed) |
| `perturbed` | continuous | 75 | 0 | mean=74.4, sd=8.57, range=[55.7, 85], skew=-0.48 (approximately symmetric) |
| `delta_perturb_minus_baseline` | continuous | 75 | 0 | mean=-1.15, sd=2.14, range=[-8.79, 2.91], skew=-1.30 (highly skewed); outliers=5 (IQR) |

## Group structure
- Grouped by: `window`, `behavior`, `fold`
- Number of groups: 75
- Group size: min=1, median=1, max=1
- **WARN**: at least one group has n<3 — statistics unreliable; must show all raw points.

## Correlations (Pearson, sorted by |r|)
- `baseline` ↔ `perturbed` : r = 0.969 (very strong)
- `perturbed` ↔ `delta_perturb_minus_baseline` : r = 0.311 (moderate)
- `baseline` ↔ `delta_perturb_minus_baseline` : r = 0.065 (negligible)

## Chart suggestions (preliminary)
- 分类 vs 连续，小样本（每组 n<10）→ **箱线图/小提琴图 + stripplot 叠加原始点**；**避免**只画均值柱状图，会掩盖分布。
- ≥3 个连续变量 → 相关性热力图（['baseline', 'perturbed', 'delta_perturb_minus_baseline']）或 pairplot 散点矩阵
- 分类维度组合数 = 375（fold, window, target_raw_label, behavior, metric 全交叉），**一张图塞不下**——建议按某一维拆成多面板，或选择子集。
- delta_perturb_minus_baseline 高度偏态（skew=-1.30）→ 考虑对数变换或小提琴图代替均值柱图

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
