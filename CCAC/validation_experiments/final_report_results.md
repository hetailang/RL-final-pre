# Final Report Results Draft

本文档是期末报告的实验结果章节草稿。完整命令、日志路径和原始输出见
`validation_experiments/results.md`；本文只保留报告正文需要的核心表格和
结论。

## 1. Experimental Setup

比较对象：

- Baseline: original CCAC, `classifier_loss=bce`, `ood_mode=hard`
- Ours: `classifier_loss=focal`, `ood_mode=soft`,
  `focal_alpha=0.75`, `focal_gamma=2.0`

实验任务：

- `OfflineBallRun-v0`: policy-level comparison, seeds `0/1/2`
- `OfflineCarRun-v0`: classifier/cost-critic mechanism repeat, seeds `0/1`;
  policy-level comparison, seed `0`

训练预算：

- Test A classifier-only: `5000` steps
- Test B cost-critic separation: `10000` steps
- Test C full policy: `50000` update steps
- Policy evaluation target costs: `[1, 2, 5, 10]`, `20` episodes

## 2. Test A: OOD Classifier

Test A 检查 focal loss 是否降低 unsafe/OOD 样本被错判为 IND 的比例。这里
最重要的安全侧指标是 false negative rate (FNR)。

| Task | Seed | Method | AUROC | AUPRC | FNR | FPR | ECE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| BallRun | 0 | original | 0.94270 | 0.93778 | 0.10380 | 0.17205 | 0.02227 |
| BallRun | 0 | focal_soft | 0.93943 | 0.93326 | 0.03565 | 0.28166 | 0.10002 |
| CarRun | 0 | original | 0.92558 | 0.88536 | 0.13206 | 0.17275 | 0.03152 |
| CarRun | 0 | focal_soft | 0.92649 | 0.88579 | 0.05261 | 0.28111 | 0.13474 |
| CarRun | 1 | original | 0.92868 | 0.89186 | 0.18108 | 0.12899 | 0.01671 |
| CarRun | 1 | focal_soft | 0.92992 | 0.89303 | 0.06185 | 0.25883 | 0.12755 |

结论：

- `focal_soft` 在三个 classifier 设置中都显著降低 FNR。
- BallRun seed 0: FNR 从 `0.104` 降到 `0.036`。
- CarRun seed 0: FNR 从 `0.132` 降到 `0.053`。
- CarRun seed 1: FNR 从 `0.181` 降到 `0.062`。
- 代价也很明确：FPR 上升，ECE 变差，说明 focal loss 更保守但校准更差。
- 报告中应表述为 safety-oriented tradeoff，而不是无代价提升。

## 3. Test B: Matched-Budget Cost-Critic Separation

Test B 检查生成的 OOD 动作是否能在相同 query budget 下得到更高的 cost
critic 估计。主要指标是：

- `qc_matched_gap`: OOD `Q_c` mean minus matched IND `Q_c` mean
- `qc_selected_matched_gap`: selected OOD `Q_c` mean minus matched IND `Q_c`
  mean
- `ood_selected_rate`: 被 classifier 选为 OOD 的生成样本比例

| Task | Seed | Method | qc_matched_gap | qc_selected_matched_gap | ood_selected_rate |
| --- | ---: | --- | ---: | ---: | ---: |
| BallRun | 0 | original | -26.829 | 88.324 | 0.4186 |
| BallRun | 0 | focal_soft | 8.711 | 6.131 | 0.7555 |
| CarRun | 0 | original | 2.641 | 4.929 | 0.3106 |
| CarRun | 0 | focal_soft | 3.632 | 6.644 | 0.4868 |
| CarRun | 1 | original | 3.267 | 6.069 | 0.2267 |
| CarRun | 1 | focal_soft | 5.031 | 8.027 | 0.4255 |

结论：

- `focal_soft` 在 CarRun seed 0 和 seed 1 上都提升了 matched-budget
  `Q_c` separation。
- BallRun 上 original 的 `qc_matched_gap` 为负，说明其 cost-critic 尺度不稳定；
  `focal_soft` 给出正向 matched gap。
- `focal_soft` 会选择更多生成样本作为 OOD，这与 Test A 的更保守分类行为一致。
- Test B 支持机制层假设：更保守的 OOD 识别和 soft weighting 可以增强 OOD
  cost signal。

## 4. Test C: Full Policy Evaluation

Policy 结果不是单调正向。报告应明确区分：

- BallRun: `focal_soft` 有有用改进，但存在 seed/budget 失败。
- CarRun: `focal_soft` 提高 reward，但没有降低 cost，是明确负结果。

### 4.1 Compact Policy Summary

`safe targets` 表示 4 个 target cost 中有多少个满足 `real_cost <= target_cost`。

| Task | Seed | Method | Avg reward | Avg real cost | Safe targets |
| --- | ---: | --- | ---: | ---: | ---: |
| BallRun | 0 | original | 347.274 | 0.000 | 4/4 |
| BallRun | 0 | focal_soft | 422.771 | 0.000 | 4/4 |
| BallRun | 1 | original | 464.176 | 19.000 | 3/4 |
| BallRun | 1 | focal_soft | 420.406 | 3.250 | 4/4 |
| BallRun | 2 | original | 409.096 | 0.000 | 4/4 |
| BallRun | 2 | focal_soft | 443.294 | 9.650 | 3/4 |
| CarRun | 0 | original | 859.456 | 181.200 | 0/4 |
| CarRun | 0 | focal_soft | 913.301 | 183.213 | 0/4 |

### 4.2 BallRun Key Observations

| Seed | Main observation |
| ---: | --- |
| 0 | `focal_soft` improves reward across all target costs while keeping zero cost. |
| 1 | `focal_soft` fixes the strict-budget failure: original has cost `74` at target `1`, while `focal_soft` has cost `0`. It loses reward at target `5/10`. |
| 2 | `focal_soft` improves reward at target `2/5` and remains safe there, but fails badly at target `1` with cost `33.6`. |

BallRun 结论：

- `focal_soft` 可以提升部分 budget 下的 reward，并在 seed 1 修复 strict-budget
  safety failure。
- 但它不是 uniformly safer。seed 2 target cost `1` 是明确失败案例。
- 最合理表述是：该变体能改善部分小规模 policy 结果，但 budget-conditioned
  policy stability 仍不足。

### 4.3 CarRun Policy Result

| Method | Target 1 cost | Target 5 cost | Avg reward | Avg real cost | Safe targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 181.00 | 181.25 | 859.456 | 181.200 | 0/4 |
| focal_soft | 183.05 | 183.55 | 913.301 | 183.213 | 0/4 |

CarRun 结论：

- original CCAC 和 `focal_soft` 都严重超出所有 target cost。
- `focal_soft` reward 更高，但 cost 没有改善。
- 这应作为负结果报告：机制层指标改善不能直接保证 policy 层安全。

## 5. Final Result Statement

可以在最终报告中使用如下结论：

> We propose a lightweight modification to CCAC that replaces the original BCE
> OOD classifier loss with focal loss and uses the OOD score as a soft weight in
> the OOD cost-critic update. On classifier-only diagnostics, the variant
> consistently reduces false negative rates on BallRun and CarRun, meaning fewer
> unsafe/OOD samples are mistakenly treated as in-distribution. On matched-budget
> cost-critic diagnostics, it improves OOD-vs-IND cost separation on CarRun
> across two seeds. However, full policy results are mixed: the method improves
> reward or strict-budget safety in several BallRun settings, but fails on
> BallRun seed 2 target cost 1 and does not solve CarRun policy safety. Thus, the
> method is best viewed as a useful diagnostic and small-scale extension rather
> than a robust new offline safe RL algorithm.

中文报告可以写成：

> 本实验基于 CCAC 做了一个轻量级扩展：使用 focal loss 训练 OOD
> classifier，并在 OOD cost-critic 更新中使用 soft OOD score。机制实验表明，
> 该方法在 BallRun 和 CarRun 上都能显著降低 unsafe/OOD 样本的漏检率，并在
> CarRun 两个 seed 上提升 matched-budget 的 OOD cost-critic 分离度。但完整
> policy 实验结果是混合的：BallRun 上若干 budget/seed 下 reward 或严格预算
> 安全性得到改善，但仍存在 seed 2 严格预算失败；CarRun 上两种方法都严重超
> 预算。因此，该扩展更适合作为一个有效的诊断性和小规模改进，而不能声称为
> 稳定的新 offline safe RL 方法。

## 6. Recommended Figures And Tables

最终报告建议放 3 张主表：

1. Classifier metrics table: 使用本文第 2 节表格。
2. Cost-critic separation table: 使用本文第 3 节表格。
3. Policy summary table: 使用本文第 4.1 节表格。

可选图：

- Test A OOD score histogram: 展示 focal loss 带来的 conservative shift。
- Test B `qc_gap_curve.png`: 展示 cost-critic separation 训练趋势。

报告中不建议加入更多长实验。当前结果已经能支撑一个完整、诚实、可复现的
期末作业故事。
