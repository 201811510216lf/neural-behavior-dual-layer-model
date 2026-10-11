# Multi-region neural masking × module-ablation association analysis

## 结论先行

- 全程未训练、未重划 fold；复用 frozen Full/NoConv/NoLSTM/NoAttention、原 test indices 和 legacy 5 s/10 ms/500-bin/250-bin protocol。
- Pair 采用原始 clean window 上的 Boolean union，一次 maximum replacement。共 8 个预注册 pair；最强 NoConv Pearson 为 `PAIR__L1_m0p69__L2_p0p06`，r=0.859。
- Pair 相对 constituent single 最佳 NoConv Pearson 的最大增益为 `PAIR__L3_m0p08__L2_p0p06`，gain=+0.064。
- 最强 triple `TRIPLE__L1_m0p69__L3_m0p08__L2_p0p06` 的 NoConv Pearson=0.878，gain=-0.003。
- 样本级 NoConv ΔPtrue 最大绝对 Spearman 为 `TRIPLE__L1_m0p69__L3_m0p08__L2_p0p82`、run2，rho=0.100；pooled CW Jaccard 最大为 `TRIPLE__L1_m0p69__L3_m0p08__L2_p0p06`，J=0.060。
- Size-matched null 中，NoConv Pearson percentile 最高为 `PAIR__L1_m0p69__L2_m0p39`：83.4%，经验 p=0.1876，全局 BH q=1.0000；gain percentile 最高为 `PAIR__L1_m0p69__L2_m0p39`：83.4%，经验 p=0.1876，全局 BH q=1.0000。

## Q1–Q10

1. Pair 是否更像 NoConv：8 个 pair 中仅 2 个 Pearson gain>0；最大为 `PAIR__L3_m0p08__L2_p0p06`，gain=+0.064，但其 Spearman gain=+0.000、cosine gain=+0.003，证据不跨指标一致。
2. Triple 是否继续增加：最强 triple `TRIPLE__L1_m0p69__L3_m0p08__L2_p0p06` 的 NoConv Pearson=0.878，gain=-0.003。 相对最佳 constituent single 没有进一步增加。
3. 是否 NoConv-specific：pair 的最大 Pearson gain 分别为 NoConv=+0.064、NoLSTM=-0.010、NoAttention=+0.045。短名单中正增益主要见于 NoConv（另有较小 NoAttention 增益），但 matched-null 不支持 region-specific correspondence。
4. 样本级是否同步：三模块全部条件的最大 |ΔPtrue rho|=0.106；NoConv 最大为 0.100。类别级相似没有转化为强样本级一致。
5. CW 是否同步：三模块全部条件最大 pooled Jaccard=0.070；NoConv 最大=0.060，具体受损样本重合很低。
6. 是否 additive：pair 分类计数 {"roughly_additive": 7, "sub_additive_redundancy_or_compensation": 1}；triple {"roughly_additive": 2}。
7. Synergy/redundancy：逐 behavior residual 已保存；recall 有边界，故仅为 descriptive interaction。
8. 真实 correlation 是否超过 random null：否。NoConv Pearson 最小单侧经验 p=0.1876，Pearson-family BH q≥0.9900；负结果完整保留。
9. 真实 gain 是否超过 random gain：否；最佳 NoConv gain 的经验检验未通过，完整 p/q 见 `multi_mask_random_gain_null.csv`。
10. overlap/perturbation size：主 pair 均满足 max Jaccard<0.80；每个真实条件使用 N=500 个 stage/coverage/component/union-size 匹配 null。真实结果没有超过该 null，因此更符合“扰动范围增大导致输出表型趋近模块消融”，不足以证明所选区域具有 Conv-specific correspondence。

## 解释边界

Correlation/similarity 描述输出效应指纹；additivity residual 描述 frozen-model 非加性；两者均不能证明 causal biological interpretation。每个 center 实际对应 ±1 s 区域，不是 10-ms causal node。Random control 是 within-retained-window shifted-centre geometry-matched null：保留 stage 和命中窗口，随机移动 constituent centers 后仍用 ±1 s、边界裁剪、一次 union 和原窗 per-channel maximum。
