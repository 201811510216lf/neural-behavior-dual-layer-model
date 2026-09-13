# Data profile: source_data/fig_b_architecture_perturbation_fold.csv

**Shape:** 180 rows × 20 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `architecture` | categorical | 180 | 0 | 4 levels: full(45), no_conv(45), no_lstm(45), no_attention(45); min_group_n=45 |
| `architecture_label` | categorical | 180 | 0 | 4 levels: Full(45), NoConv(45), NoLSTM(45), NoAttention(45); min_group_n=45 |
| `target` | ordinal | 180 | 0 | 3 levels: 2(60), 3(60), 5(60); min_group_n=60 |
| `target_label` | categorical | 180 | 0 | 3 levels: L2(60), L3(60), L5(60); min_group_n=60 |
| `fold` | categorical | 180 | 0 | 3 levels: fold01(60), fold02(60), fold03(60); min_group_n=60 |
| `condition` | categorical | 180 | 0 | 5 levels: baseline(36), maximum(36), attention_shuffle(36), attention_zero(36), random_window(36); min_group_n=36 |
| `accuracy` | continuous | 180 | 0 | mean=69.6, sd=3.62, range=[60.3, 76.9], skew=-0.14 (approximately symmetric) |
| `recall` | continuous | 180 | 0 | mean=68.4, sd=3.25, range=[61.5, 74.7], skew=-0.04 (approximately symmetric) |
| `f1` | continuous | 180 | 0 | mean=68, sd=3.57, range=[57.4, 74.8], skew=-0.20 (approximately symmetric) |
| `mcc` | continuous | 180 | 0 | mean=51.8, sd=5.45, range=[38.5, 62.4], skew=-0.29 (approximately symmetric) |
| `auc` | continuous | 180 | 0 | mean=79.5, sd=3.66, range=[71.4, 85.5], skew=-0.30 (approximately symmetric) |
| `aupr` | continuous | 180 | 0 | mean=69.1, sd=4.44, range=[59.6, 77.5], skew=-0.11 (approximately symmetric) |
| `c0` | continuous | 180 | 0 | mean=80.5, sd=2.97, range=[74.6, 84.3], skew=-0.34 (approximately symmetric) |
| `c1` | continuous | 180 | 0 | mean=72.4, sd=7.14, range=[53.6, 85], skew=-0.25 (approximately symmetric) |
| `c2` | continuous | 180 | 0 | mean=74.5, sd=7.1, range=[63.4, 90], skew=0.45 (approximately symmetric) |
| `c3` | continuous | 180 | 0 | mean=68.5, sd=7.08, range=[46.9, 81.5], skew=-0.57 (moderately skewed); outliers=4 (IQR) |
| `c4` | continuous | 180 | 0 | mean=64, sd=10.5, range=[38.6, 89.5], skew=0.58 (moderately skewed) |
| `true_probability` | continuous | 180 | 0 | mean=63.8, sd=5.52, range=[53.8, 74.2], skew=0.06 (approximately symmetric) |
| `logit_margin` | continuous | 180 | 0 | mean=2.64, sd=0.948, range=[0.841, 4.06], skew=-0.18 (approximately symmetric) |
| `stage2_macro_recall` | continuous | 180 | 0 | mean=68.3, sd=5.36, range=[50.1, 77.4], skew=-0.51 (moderately skewed); outliers=2 (IQR) |

## Group structure
- Grouped by: `architecture`, `condition`, `target`, `fold`
- Number of groups: 180
- Group size: min=1, median=1, max=1
- **WARN**: at least one group has n<3 — statistics unreliable; must show all raw points.

## Correlations (Pearson, sorted by |r|)
- `accuracy` ↔ `mcc` : r = 0.979 (very strong)
- `f1` ↔ `stage2_macro_recall` : r = 0.952 (very strong)
- `mcc` ↔ `true_probability` : r = 0.951 (very strong)
- `accuracy` ↔ `f1` : r = 0.947 (very strong)
- `f1` ↔ `mcc` : r = 0.947 (very strong)
- `accuracy` ↔ `true_probability` : r = 0.946 (very strong)
- `auc` ↔ `aupr` : r = 0.941 (very strong)
- `recall` ↔ `mcc` : r = 0.924 (very strong)
- `recall` ↔ `f1` : r = 0.921 (very strong)
- `true_probability` ↔ `logit_margin` : r = 0.909 (very strong)
- ... +81 more pairs

## Chart suggestions (preliminary)
- 分类 vs 连续，小样本（每组 n<10）→ **箱线图/小提琴图 + stripplot 叠加原始点**；**避免**只画均值柱状图，会掩盖分布。
- ≥3 个连续变量 → 相关性热力图（['accuracy', 'recall', 'f1', 'mcc', 'auc']）或 pairplot 散点矩阵
- 分类维度组合数 = 2160（architecture, architecture_label, target, target_label, fold, condition 全交叉），**一张图塞不下**——建议按某一维拆成多面板，或选择子集。

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
