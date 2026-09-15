# 实验来源与一致性核查

## 结论先行

本轮已经把结构消融、最大值掩码和注意力干预统一到同一套三折、同一批
Run1/Run2 测试索引和同一组 Full checkpoint 上，并新增逐窗口成对预测审计。
在这个统一口径内，重算值与既有 `fig_b_architecture_perturbation_fold.csv`
的最大绝对误差为 `7.11e-15`，可视为完全复现。

旧版结构消融 Table 5 不能再作为当前掩码实验的对照表。它不是来自一次可追溯
的统一实验：可找到的单元格来源显示，均值、标准差、折数和模型名称发生了混合。
建议论文与后续 PPT 改用本目录的三折一致性结果，并把旧 Table 5 标记为历史材料。

## 为什么“结构消融”和“数据掩码”的全体准确率不一致

1. 旧结构消融总体表的 Full accuracy 是 `74.50 ± 3.33`，当前三折掩码基线是
   `74.70 ± 0.70`。两者不是同一次运行，也没有共享可核对的逐样本预测。
2. 差异在分类别结果上非常大：旧 Full 与当前三折 Full 的 L2、L3、L4、L5
   分别相差 `+10.38`、`+9.43`、`-6.03`、`+17.16` pp。总准确率相近只是抵消，
   不能证明两个基线相同。
3. 当前系统是双层层级分类器：Run1 识别 L1、L3 和 L2/L4/L5 粗门控；Run2
   在另一套窗口中细分 L2/L4/L5。旧代码报告的“五类准确率”是两个阶段的指标级
   合并，不是同一批样本的一次五分类预测。
4. Run1 有 979 个窗口，Run2 有 214 个窗口；两套窗口的构造、标签和索引不同。
   因此不能把两阶段预测强行拼成一张样本级五分类混淆矩阵。

## 旧 Table 5 的具体来源问题

对公开表 `results/model/structural_ablation_per_class_ppt.csv` 的每个单元格，已与
历史十折文本和五折 summary CSV 进行精确匹配。完整证据见
`qa/public_table5_cell_source_matches.csv`。

- `No Attention`：五个均值来自十折 `ConvNoFeatureAttentionLSTM`，五个 SD 来自
  同名模型的五折 summary。均值与 SD 不是同一组折。
- `No Conv`：五个均值来自十折 `ConvFeatureAttentionNoTimeLSTM`，但 SD 来自五折
  `NoConvFeatureAttentionLSTM`。该行同时混用了不同模型和不同折数。
- `No LSTM`：五个均值来自十折 `NoConvFeatureAttentionLSTM`，但 SD 来自五折
  `ConvFeatureTimeAttentionClassifier`。该行同样混用了不同模型和不同折数。
- `Our Proposed / Full model`：在可用历史五折/十折原始 summary 中没有找到整行
  精确来源；数值也不同于当前三折 Full 基线。

旧模型命名本身也容易误读：`ConvFeatureAttentionNoTimeLSTM` 是去除时间注意力，
并不是去除卷积；`NoConvFeatureAttentionLSTM` 才是去除卷积。另外，旧 Full 模型
的 feature-attention 输出维度为 1，却在该维度做 softmax，权重恒为 1；它并不等同
于当前代码中真正对时间注意力做均匀化的 `NoAttention`。

## 本轮新增的严格成对分析

每个 stage、fold、dataset index 均保存：`y_true`、基线预测、干预预测、真实类概率、
掩码是否真正命中，以及 CC/CW/WC/WW 转移类别。14 个比较、1193 个唯一阶段窗口，
共 16702 行成对预测。

比较包括：

- 最大值掩码 L1–L5；
- attention zero：L2、L3、L5；
- attention shuffle：L2、L3、L5；
- Full vs NoConv、NoLSTM、NoAttention。

三折测试集对两个阶段都构成精确分区：无重复 test index、无缺失 test index。
模型权重的 SHA-256 已写入 `qa/checkpoint_manifest_sha256.csv`。

## 新结果如何解释

### L2 最大值掩码

- Run1 的 L2/L4/L5 粗门控：`CW=43`、`WC=8`，损伤率 `11.08%`，恢复率
  `11.76%`，净准确率变化 `-7.68 pp`。
- Run2 的 L2 细分类：`CW=2`、`WC=3`，净变化 `+1.27 pp`。

所以旧表中 L2 相关性能下降主要来自上游粗门控，而不是 Run2 对 L2 的细分类器。
这比只说“L2 recall 下降”更具体。

### L5 最大值掩码

- Run1 粗门控：`CW=30`、`WC=9`，损伤率 `7.73%`，净变化 `-4.61 pp`。
- Run2 的 L5 细分类：`CW=3`、`WC=0`，损伤率 `15.79%`，净变化 `-7.14 pp`。

L5 的效应同时出现在粗门控和细分类，但 Run2 的 L5 只有 42 个窗口，不能仅凭
三个受损窗口作强机制结论。

### 结构消融

结构模型与 Full 在完全相同的 test index 上比较。总体净准确率变化为：

- NoConv：Run1 `-3.17 pp`，Run2 `-11.68 pp`；
- NoLSTM：Run1 `-6.33 pp`，Run2 `-4.67 pp`；
- NoAttention：Run1 `+0.61 pp`，Run2 `-7.94 pp`。

结构消融会同时产生大量 CW 和 WC，因此“均值降低”不能概括全部变化；损伤率和
恢复率热图应与总体指标一起展示。

## 统计限制

本轮输出了精确 McNemar p 值和 BH-FDR q 值，但只能作为探索性结果。legacy 窗口
为 5 秒、步长 2.5 秒，相邻窗口共享 50% 原始数据，不满足独立生物学重复假设；
折划分还是窗口级而非动物/记录级，并存在训练窗与测试窗共享原始片段的风险。
论文主结论应以按动物或独立记录分组、无重叠窗口的复现实验为准。

## 论文/PPT 的统一口径

1. 当前所有掩码与结构比较统一引用 `table_e_current_structural_ablation_coherent.csv`
   和本轮成对预测表。
2. 旧结构 Table 5 不再与当前掩码 Table 并列比较；若必须展示，明确标注
   “historical, incompatible protocol”。
3. 五类汇总表仅称为“legacy metric-level merge”，不要称为单个五分类器的样本级结果。
4. 混淆矩阵、McNemar、CW/WC 和预测去向只按 Run1/Run2 分阶段报告。
5. attention shuffle 的逐样本预测来自 20 次随机打乱概率的平均值；它反映平均化后的
   决策，不代表 20 次随机重复各自的离散转移分布。

## 机器可读证据

- `source_data/paired_predictions_stagewise.csv`：逐样本成对预测（仅本地保存，不上传公开仓库）；
- `source_data/paired_confusion_matrices_stagewise.csv`：基线、干预和差值混淆矩阵；
- `tables/table_c_transition_summary_pooled.csv`：合并 OOF 的转移率和检验；
- `tables/table_c_transition_summary_by_fold.csv`：逐折转移结果；
- `tables/table_d_transition_destinations.csv`：受损后的错误去向与恢复来源；
- `tables/table_e_current_structural_ablation_coherent.csv`：同协议三折结构消融；
- `tables/table_f_baseline_protocol_comparison.csv`：旧结构基线与当前掩码基线差异；
- `tables/table_g_mask_coverage_stagewise.csv`：各掩码实际命中窗口数；
- `qa/recomputed_vs_saved_fig_b.csv`：重算与既有结果逐单元格核对；
- `qa/public_table5_cell_source_matches.csv`：旧 Table 5 单元格来源匹配；
- `qa/fold_partition_audit.csv`：折分区完整性；
- `qa/checkpoint_manifest_sha256.csv`：模型权重指纹（仅本地保存）。
