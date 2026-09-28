# 单时间点神经数据消融 × 模块消融：结果摘要

## QA 与协议

- `Summary_P_long` 严格筛选得到 **24** 个唯一显著时间点，与预设清单完全一致。
- 未训练模型、未重新划分数据。Full、NoConv、NoLSTM、NoAttention 均使用同一组 saved Run1/Run2 三折 test indices。
- legacy 协议保持为 5 s、10 ms/bin、500 bins、250-bin step、stage-label-constant selection。
- point-wise mask 将每个行为 onset 映射回 recording，只替换 `[center-1 s, center+1 s)`；fill 是每个神经通道在该原始 5 s window 内的 maximum。
- clean Full 预测逐样本复现旧保存结果；结构 clean 指标复现 coherent 三折表。详细证据见 `qa/`。

## 数据消融和模块消融的效应

- 绝对值最大的单个 ΔRecall 是 `tp24_L5_p0p70` 对 `L5`：-6.313 pp（定义始终为 ablated − baseline）。
- 绝对值最大的模块 ΔRecall 是 `NoConv` 对 `L4`：-14.810 pp。
- 所有原始数值、符号、逐折值和 stage-wise 样本级转移均已保存；符号不是原始数值的替代品。

## Correlation / similarity（相关与相似性）

- 5-behavior ΔRecall 向量相似性中，绝对 Spearman 最大的组合是 `tp01_L1_m0p69` × `NoConv`，ρ=0.821。
- pooled Run1/Run2 key 的 CW damage-set Jaccard 最大组合是 `tp05_L2_p0p06` × `NoLSTM`，J=0.065。
- 相同 OOF 样本上的 ΔPtrue/Δlogit-margin 关联按 stage 计算；最强绝对 Spearman 为 `tp12_L4_m0p56` × `NoConv`、run2、true_class_probability，ρ=0.154。
- 这些量只表示两类干预在输出效应上的一致程度。它们不证明某个模块“编码了”某个生理时间点。

## Double-ablation interaction（双消融交互）

- interaction 定义为 `(MaskFull − CleanFull) − (MaskNoM − CleanNoM)`。
- 绝对值最大的 5-behavior recall interaction 是 `tp19_L5_p0p08` × `NoAttention` × `L5`：+4.261 pp。
- 非零 interaction 表明数据掩码效应依赖模型结构背景；它与简单相关/相似性是不同证据，不应互相替代。

## Mask overlap 与统计限制

- Run1 中 Jaccard ≥ 0.90 的非对角 point-mask 配对记录数为 52。相邻 L4/L5 显著点的 ±1 s 区间高度重叠，因此不能把每个点解释为独立的时间机制。
- 500-bin windows 以 250 bins 步长重叠。样本不是独立生物重复。本分析不提供把重叠窗口当独立样本的显著性 p 值。
- ΔPtrue/Δmargin 的相关 CI 使用 recording-file block bootstrap；跨 recording 或时间不连续的 legacy window 不进入 CI，但保留在点估计和逐样本 source-data 中。
- t-stat 与消融效应只作为探索性系数，见 `G_ttest_vs_data_ablation_correlations.csv`，不作因果或独立重复推断。

## Causal biological interpretation（因果生物学解释边界）

本结果支持的是 frozen classifier 在既定 legacy protocol 下的功能依赖与结构依赖。输入替换、模块结构消融和双消融交互均是对模型的干预，不是对动物神经回路的实验性因果操纵。即使某个时间点与某个模块呈高相关、高 Jaccard 或强 interaction，也只能表述为模型层面的相容性/依赖性证据。要提出生物学因果结论，仍需独立动物/recording 级验证和直接神经操纵实验。

## 文件导航

- `source_data/condition_predictions_stagewise.csv.gz`：四种结构、clean 与 24 个 point-mask 的逐样本输出。
- `source_data/paired_predictions_stagewise.csv.gz`：baseline/ablated 成对概率、margin、CC/CW/WC/WW、mask hit 与替换 bin 数。
- `tables/`：A–G 关联、ΔFP/ΔFN、confusion、interaction 和指标表。
- `figures/`：SVG/PDF/TIFF/PNG 论文图。
- `qa/`：baseline 复现、fold 对齐、checkpoint hash、mask overlap、路径与环境记录。
