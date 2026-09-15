# 实验—代码—模型—结果对应表

## 建议作为最终打包主入口的代码

| 实验 | 主代码 | 是否训练 | 模型/数据来源 | 主要结果 |
|---|---|---:|---|---|
| 双层 Full、NoConv、NoLSTM、NoAttention；最大值、attention shuffle/zero、随机窗 | `raster/result/run_experiments.py` | 结构变体首次运行需要训练；已有权重则复用 | Full 用 `outputs_legacy`；结构变体用 `result/checkpoints`；三折 saved split | `source_data/fig_b_*`、`tables/table_a_*` |
| 同形状随机平移零分布 | `raster/result/run_shape_matched_null.py` | 否 | Full frozen checkpoints、相同三折 | `source_data/fig_d_shape_matched_null.csv` |
| 原图 A–E | `raster/result/build_figures.py` | 否 | 规范化 source_data | `figures/fig_a_*` 至 `fig_e_*` |
| 样本级 CC/CW/WC/WW、McNemar、混淆矩阵差值 | `raster/result/run_paired_transition_analysis.py` | 否，仅推理 | 24 个现有三折 checkpoint；Run1/Run2 各自配对 | `paired_predictions_stagewise.csv`、Table C/D/G |
| 历史结构消融数值溯源 | `raster/result/audit_legacy_result_consistency.py` | 否 | 历史五折 summary、十折文本、公开 Table 5 | Table E/F 与 `qa/public_table5_cell_source_matches.csv` |
| 新图 G–K | `raster/result/build_transition_figures.py` | 否 | paired transition CSV | `figures/fig_g_*` 至 `fig_k_*` |
| 全流程 | `raster/result/run_all.ps1` | 视 checkpoint 是否已存在 | 指定 conda 环境 | 依次刷新全部结果与 QA |

## 当前双层模型本体

- 配置：`raster/brain_model_regroup_1246_vs_35_fixed/config.py`
- 数据与窗口：`raster/brain_model_regroup_1246_vs_35_fixed/data_processing.py`
- 模型：`raster/brain_model_regroup_1246_vs_35_fixed/model.py`
- 训练：`raster/brain_model_regroup_1246_vs_35_fixed/train.py`
- Full 权重与三折索引：`raster/brain_model_regroup_1246_vs_35_fixed/outputs_legacy`
- 统一数据/掩码构造核心：
  `raster/brain_model_regroup_1246_vs_35_fixed/legacy_5s_50pct_significant_segment_replace/segment_replace_core.py`

Run1：979 个 5 秒窗口，识别 L1、L3、L2/L4/L5 gate。
Run2：214 个 5 秒窗口，只细分 L2、L4、L5。两个阶段使用不同窗口集合。

## 已存在实验的代码地图

| 实验族 | 目录/入口 | 模型处理 | 用途与当前地位 |
|---|---|---|---|
| 精确显著点置零 | `legacy_5s_50pct_significant_time_mask/run_legacy_5s_significant_time_mask.py` | frozen Full，无训练 | 旧的 10 ms 点掩码；用于历史对照 |
| ±1 秒段替换 max/mean/min | `legacy_5s_50pct_significant_segment_replace/run_all_replacements.py` | frozen Full，无训练 | 当前 maximum 掩码表的直接来源 |
| Attention 四类干预 | `legacy_5s_50pct_attention_perturbation/run_all.ps1` | frozen Full，无训练 | uniform、shuffle、key/non-key swap、key zero |
| LSTM 四类干预 | `legacy_5s_50pct_lstm_perturbation/run_all.ps1` | frozen Full，无训练 | block shuffle、远段交换、hidden reset、history truncation |
| Conv 四类干预 | `legacy_5s_50pct_conv_perturbation/run_all.ps1` | frozen Full，无训练 | phase randomization、colored noise、high-band、mean replacement |
| L5 全输入消融 | `label5_neural_ablation/run_label5_neural_ablation.py` | frozen Full，无训练 | 保留 L5 样本，只消融输入；已有逐预测 CSV |
| 事件对齐 1 秒重训 | `event_aligned_1s_retrain/train.py` | 重新训练三折 | 时间轴与 -1~1 s 统计表一致；是未来严谨掩码的更合适基础 |
| SCA 选窗与随机窗 | `sca_factor_guided_time_mask_analysis/run_sca_guided_time_mask.py` | SCA 训练；分类器 frozen | 分阶段报告，不构造伪五类结果；探索性 |
| 神经—模型表征关联 | `raster/legacy_5s_neural_model_association/run_all.py` | frozen Full | RSA、CKA、决策几何、重叠审计、测试时功能干预 |
| 栅格图与行为统计 | `raster/analysis_binned10_v1`、`raster/analysis_binned10_regroup_1246_vs_35`、`raster/data_label` | 无分类模型 | 事件栅格、行为组比较、显著时间点来源 |

## 历史目录的处理建议

- `multi_classification/ablationstudy`：旧单阶段消融实验。保留作溯源，不再作为当前
  论文主结果；旧 Table 5 已证实混合五折/十折和不同模型。
- `final`、`last`、`version1_26`：多个阶段的训练/对照/搜索副本。不可直接挑数拼表；
  需要引用时先固定数据版本、折数、代码 commit 和 checkpoint hash。
- `neural-behavior-dual-layer-model`：公开结果包。适合放可公开的代码、图和派生表，
  不应放原始神经数据、label 文件或模型权重。
- `raster/result`：从现在起作为论文机制实验的唯一汇总与 QA 目录。

## 收尾与打包建议

1. 只保留一个运行入口：`raster/result/run_all.ps1`。
2. 论文表格只从 `source_data`/`tables` 自动生成，不再手工抄录均值或 SD。
3. 每次结果发布同时保存：环境、折索引、模型 SHA-256、源 CSV 和图形 QA。
4. 将 `multi_classification/ablationstudy`、`final`、`last` 标记为 `archive/historical`，
   不从这些目录直接生成正文图表。
5. 下一次真正重跑模型时，优先改为按动物/记录分组、无重叠窗口，并独立划分验证集，
   解决当前窗口泄漏和 held-out fold 选 epoch 的乐观偏差。
