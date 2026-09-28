# 实现前方法文献记录

本记录只用于设计与解释边界，不改变本项目已经锁定的 legacy protocol。

1. Zeiler & Fergus (ECCV 2014), *Visualizing and Understanding Convolutional Networks*. DOI: 10.1007/978-3-319-10590-1_53. 经典层/输入遮挡与网络消融工作，支持用预测变化衡量局部输入或模块贡献。
2. Adebayo et al. (NeurIPS 2018), *Sanity Checks for Saliency Maps*. 强调解释方法必须验证其对模型参数和数据关系的敏感性，不能只凭视觉吸引力。
3. Kriegeskorte, Mur & Bandettini (Frontiers in Systems Neuroscience 2008), *Representational similarity analysis—connecting the branches of systems neuroscience*. DOI: 10.3389/neuro.06.004.2008. 提供跨神经数据、行为和模型比较相似结构的框架；本项目用向量/混淆效应相似性，但不把相似性等同于因果。
4. Kornblith et al. (ICML 2019), *Similarity of Neural Network Representations Revisited*. 介绍 CKA 及表示比较的约束。当前任务聚焦输出效应和 intervention interaction，CKA 作为未来表征层扩展，不擅自加入主协议。
5. Šimić, Veas & Sabol (Scientific Reports 2025), *A comprehensive analysis of perturbation methods in explainable AI feature attribution validation for neural time series classifiers*. DOI: 10.1038/s41598-025-09538-2. 指出 time-series faithfulness 对 perturbation choice 和 region size 敏感，支持本项目固定并完整记录 replacement 与区间长度、同时报告 mask overlap。
6. Mercier et al. (2022), *Time to Focus: A Comprehensive Benchmark Using Time Series Attribution Methods*, arXiv:2202.03759. 对时序 attribution 的 perturbation/gradient 方法进行比较，强调不同评估维度没有单一万能方法。

在线来源：

- https://doi.org/10.1007/978-3-319-10590-1_53
- https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- https://doi.org/10.3389/neuro.06.004.2008
- https://proceedings.mlr.press/v97/kornblith19a.html
- https://doi.org/10.1038/s41598-025-09538-2
- https://arxiv.org/abs/2202.03759
