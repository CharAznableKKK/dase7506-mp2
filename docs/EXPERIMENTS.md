# 实验记录

最终模型已冻结并完成正式测试评分。研究设计与分析见 [完整报告](../report/REPORT.md)。

协议：[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)；复现命令：[README.md](../README.md)。

## 研究问题与固定规则

- 主问题：固定表示能否改善分歧题选择？答案共识与聚合是否有增益？表面特征能解释多少收益？
- 拟合仅用训练集；验证选择优先分歧子集正确数，再完整正确数，再小资产。
- 固定 15 次确定性岭回归拟合，33 个验证配置；种子记录为 20261024。
- 总搜索墙钟耗时 36.8005 秒；逐配置评分、命令、预测及中间模型在重跑时生成于 code/runs/validation-study-v1/，不随发布包分发；汇总证据保留在 records/experiments/。
- 完整源哈希、成本、配置、资产哈希和结果：records/experiments/validation-study-v1.json。

## 全部验证配置（按预先声明的选择规则排序）

| 实验 ID | 完整验证 % | 分歧子集 % | 拟合秒 | 预测+评分秒 | 解压资产字节 |
|---|---:|---:|---:|---:|---:|
| embedding_surface_pairwise_a0.1_none | 92.6065 | 83.3803 | 0.3839 | 0.8944 | 25538 |
| full_pairwise_a0.1_answer_mean | 92.3559 | 82.8169 | 0.3554 | 0.9407 | 25604 |
| embedding_pairwise_a0.1_none | 92.1053 | 82.2535 | 0.4672 | 0.8337 | 25242 |
| embedding_surface_pairwise_a1_none | 92.1053 | 82.2535 | 0.3616 | 0.8541 | 25538 |
| full_pairwise_a1_answer_mean | 92.1053 | 82.2535 | 0.4277 | 0.8857 | 25604 |
| full_pairwise_a0.1_none | 91.8546 | 81.6901 | 0.3554 | 0.9088 | 25597 |
| embedding_pairwise_a1_none | 91.3534 | 80.5634 | 0.3789 | 0.7987 | 25242 |
| embedding_surface_pairwise_a0.1_answer_mean | 91.3534 | 80.5634 | 0.3839 | 0.8927 | 25545 |
| full_pairwise_a1_none | 91.2281 | 80.2817 | 0.4277 | 0.8699 | 25597 |
| embedding_pointwise_a0.1_none | 90.9774 | 79.7183 | 0.5154 | 0.8643 | 25243 |
| embedding_surface_pairwise_a10_none | 90.7268 | 79.1549 | 0.3592 | 0.8861 | 25539 |
| embedding_surface_pairwise_a1_answer_mean | 90.7268 | 79.1549 | 0.3616 | 0.9460 | 25545 |
| embedding_pairwise_a0.1_answer_mean | 90.4762 | 78.5915 | 0.4672 | 0.8170 | 25249 |
| embedding_pairwise_a10_none | 90.1003 | 77.7465 | 0.3821 | 0.7832 | 25243 |
| full_pairwise_a10_answer_mean | 89.9749 | 77.4648 | 0.3707 | 0.9219 | 25605 |
| embedding_pairwise_a1_answer_mean | 89.8496 | 77.1831 | 0.3789 | 0.8090 | 25249 |
| full_pairwise_a10_none | 89.8496 | 77.1831 | 0.3707 | 0.9084 | 25598 |
| embedding_pointwise_a1_none | 89.5990 | 76.6197 | 0.5112 | 0.8091 | 25243 |
| embedding_pointwise_a0.1_answer_mean | 88.9724 | 75.2113 | 0.5154 | 0.9333 | 25250 |
| embedding_pointwise_a10_none | 88.7218 | 74.6479 | 0.4912 | 0.7808 | 25244 |
| embedding_surface_pairwise_a10_answer_mean | 88.2206 | 73.5211 | 0.3592 | 0.9269 | 25546 |
| surface_pairwise_a0.1_none | 87.8446 | 72.6761 | 0.0407 | 0.8729 | 952 |
| embedding_pairwise_a10_answer_mean | 87.5940 | 72.1127 | 0.3821 | 0.7859 | 25250 |
| embedding_pointwise_a1_answer_mean | 87.4687 | 71.8310 | 0.5112 | 0.8293 | 25250 |
| surface_pairwise_a1_none | 87.3434 | 71.5493 | 0.0431 | 0.9525 | 952 |
| surface_pairwise_a10_none | 86.9674 | 70.7042 | 0.0492 | 0.9080 | 953 |
| baseline_majority | 86.2155 | 69.0141 | 0.0000 | 0.7575 | 49 |
| surface_pairwise_a0.1_answer_mean | 85.8396 | 68.1690 | 0.0407 | 0.8906 | 959 |
| surface_pairwise_a1_answer_mean | 85.8396 | 68.1690 | 0.0431 | 0.8682 | 959 |
| embedding_pointwise_a10_answer_mean | 85.5890 | 67.6056 | 0.4912 | 0.8193 | 25251 |
| surface_pairwise_a10_answer_mean | 84.9624 | 66.1972 | 0.0492 | 0.8597 | 960 |
| baseline_first | 78.1955 | 54.9296 | 0.0000 | 0.8154 | 46 |
| baseline_shortest | 73.1830 | 39.7183 | 0.0000 | 0.7958 | 49 |

注意：同一拟合的两种 pooling 共享拟合成本，不应把表中重复的拟合耗时相加作为总成本。外部验证耗时非参考 CPU 资格认证。

## 选择与分析

- 当前选择：embedding_surface_pairwise_a0.1_none；完整 739/798，分歧 296/355。
- 比多数投票多答对 51 道分歧题：修正 74，退化 23；分歧准确率 +14.3662 个百分点。
- 表示成对模型比独立模型净增 9 题；再加表面特征净增 4 题，不能夸大这些较小开发集差异。
- 添加共识特征或对选中方法进行同答案平均均无提升；保留负结果。
- 配对分析及确定性错误示例：records/analysis/paired-comparisons.json、validation-error-examples.json。
- 重训资产字节及验证预测完全一致：records/reproduction/validation-replay.json。
- 新题库适配、无参考读取、离线与实际四线程：records/robustness/candidate-check.json。
- 独立核心代码审查：records/reviews/core-review.md，未发现阻止冻结的问题。

## 冻结与最终测试

- 冻结源提交：cf8e1ca64c8d51ae0edb04e276acebbb38c56398。
- 冻结时间：2026-10-24 22:22:54 UTC+8；测试时间：22:23:09 UTC+8。
- 完整测试：384/467，官方 accuracy = 82.22698072805139%。
- 本机新进程预测加评分：0.7545 秒；最终推理资产解压后 25,530 字节。
- 原始搜索配置为 Windows CRLF，冻结时仅将配置转为可跨平台的 LF（资产总量减少 8 字节）；模型参数和选择不变。
- 证据：records/final/freeze.json、test-score.json、evaluation.json。测试后未调参或改变算法；后续相同 checkpoint 评测仅用于复现。

## 干净环境复现（2026-10-25）

新建无全局 site-packages 的虚拟环境，从本地 wheels 离线安装固定依赖；使用 Git 冻结版本导出源码。九项测试通过；467 题 ID 与选择逐项一致；重训 checkpoint 字节完全一致；未知 ID／变长题库、仅 checkpoint 文件读取和四线程检查通过。证据：`records/reproduction/clean-environment.json`。本次测试评分只作冻结方法复现，未用于进一步选择或调参。

## 版本与记录说明

历史冻结提交标识开发来源。当前仓库中的方法与 checkpoint 由 `records/final/freeze.json` 和 `records/release/source-mapping.json` 的逐文件哈希核对。复现使用当前检出版本，在仓库根目录执行 README 命令。原始记录中的机器绝对路径表示历史运行位置，不是运行依赖。
