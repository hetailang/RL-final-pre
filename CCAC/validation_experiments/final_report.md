# 基于 OOD 分类器改进的 CCAC 小规模扩展实验

## 摘要

本文基于已有的 CCAC（Constraint-Conditioned Actor-Critic）离线安全强化
学习项目，设计并验证了一个轻量级扩展：使用 focal loss 替代原始 BCE
训练 OOD classifier，并在 OOD cost-critic 更新中使用 soft OOD score 作为
权重。实验目标不是提出完整的新算法，而是在一到两天内完成一个可复现、
可解释的期末作业级扩充实验。

实验分为三部分。Test A 只训练 OOD classifier，检查 unsafe/OOD 样本的漏检
率。Test B 训练简化的 cost critic，检查相同 query budget 下 OOD 样本是否
得到更高的 cost estimate。Test C 运行 50k update 的完整 policy 训练，并在
多个 target cost 下评估 reward 和 realized cost。

结果显示，该修改在机制层面比较稳定：在 BallRun 和 CarRun 上都显著降低
false negative rate，并在 CarRun 两个 seed 上提高 matched-budget
cost-critic separation。但完整 policy 结果是混合的：BallRun 上部分 seed 和
budget 得到更高 reward 或更好的严格预算安全性，CarRun 上则 reward 提高但
cost 没有改善，两个方法都严重超预算。因此，本实验结论是：focal + soft
OOD weighting 是一个有用的诊断性和小规模改进，但不能声称已经得到稳定的
新 offline safe RL 方法。

## 1. 背景与问题

离线安全强化学习需要在固定数据集上学习既追求 reward 又满足安全约束的
policy。CCAC 通过把 cost threshold 作为条件输入，学习 constraint-conditioned
actor 和 critic，从而支持在不同安全预算下进行 zero-shot evaluation。原始
CCAC 中，生成的 OOD state-action 样本会经过 OOD classifier 判断，再用于
cost-critic 相关更新。

本实验关注一个很小但安全相关的问题：

> 如果 OOD classifier 漏掉 unsafe/OOD 样本，cost critic 和 policy 可能无法
> 获得足够的保守安全信号。能否通过更偏安全侧的 OOD classifier loss 和更平滑
> 的 OOD weighting 改善这一机制？

因此本文比较 original CCAC 和一个选定变体：

- Baseline: original CCAC, `classifier_loss=bce`, `ood_mode=hard`
- Ours: `classifier_loss=focal`, `ood_mode=soft`,
  `focal_alpha=0.75`, `focal_gamma=2.0`

## 2. 方法

### 2.1 Focal OOD Classifier

原始实现使用 BCE 训练 OOD classifier。本文将 classifier loss 替换为 focal
loss：

```text
FL(p_t) = alpha_t * (1 - p_t)^gamma * BCE(p, y)
```

其中 `alpha=0.75`，`gamma=2.0`。直观上，focal loss 会更关注难分类样本。
在本实验中，我们希望它降低 unsafe/OOD 样本被错判为 in-distribution 的比例，
即降低 false negative rate (FNR)。

### 2.2 Soft OOD Weighting

原始 hard OOD mode 使用阈值选择生成样本：

```text
weight = 1(score >= threshold)
```

本文的 soft mode 使用 classifier 输出概率作为权重：

```text
weight = clamp(score, 0, 1)
```

这样做的动机是避免完全依赖单一阈值，让可疑但未超过阈值的样本仍然对 OOD
cost signal 有贡献。

### 2.3 实验范围控制

本实验只做课程项目规模的验证，不尝试复现完整 paper benchmark。范围限制
如下：

- 不加入新的大型 baseline。
- 不跑完整 DSRL benchmark。
- 完整 policy 训练只使用 50k update。
- 主任务 BallRun 跑 3 个 seed，第二任务 CarRun 只跑必要机制实验和 seed 0
  policy 对照。

## 3. 实验设置

运行环境：

- PyTorch `2.4.1+cu121`
- GPU: NVIDIA H100 80GB HBM3
- 任务数据来自 DSRL HDF5 数据集

任务与实验：

| Experiment | Task | Seeds | Steps | Purpose |
| --- | --- | --- | ---: | --- |
| Test A | BallRun, CarRun | BallRun 0; CarRun 0/1 | 5000 | OOD classifier diagnostics |
| Test B | BallRun, CarRun | BallRun 0; CarRun 0/1 | 10000 | matched-budget cost-critic separation |
| Test C | BallRun | 0/1/2 | 50000 | full policy comparison |
| Test C | CarRun | 0 | 50000 | second-task policy check |

Policy evaluation 使用 target costs `[1, 2, 5, 10]`，每个 target cost 评估
20 episodes。安全目标用 `real_cost <= target_cost` 判断。

CarRun 数据集最初因公开服务器 HTTP 502 无法下载，后来手动放置
`SafetyCarRun-v0-40-651.hdf5` 到 `datasets/dsrl/` 后完成实验。BallCircle
fallback 数据集仍然下载失败，因此未纳入最终实验。

## 4. 结果

### 4.1 Test A: OOD Classifier

Test A 关注 OOD classifier 是否减少 unsafe/OOD 样本漏检。表中 FNR 是主要
安全指标，越低表示越少 OOD 样本被误判为 IND。

| Task | Seed | Method | AUROC | AUPRC | FNR | FPR | ECE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| BallRun | 0 | original | 0.94270 | 0.93778 | 0.10380 | 0.17205 | 0.02227 |
| BallRun | 0 | focal_soft | 0.93943 | 0.93326 | 0.03565 | 0.28166 | 0.10002 |
| CarRun | 0 | original | 0.92558 | 0.88536 | 0.13206 | 0.17275 | 0.03152 |
| CarRun | 0 | focal_soft | 0.92649 | 0.88579 | 0.05261 | 0.28111 | 0.13474 |
| CarRun | 1 | original | 0.92868 | 0.89186 | 0.18108 | 0.12899 | 0.01671 |
| CarRun | 1 | focal_soft | 0.92992 | 0.89303 | 0.06185 | 0.25883 | 0.12755 |

结果说明：

- BallRun seed 0 上，FNR 从 `0.10380` 降到 `0.03565`。
- CarRun seed 0 上，FNR 从 `0.13206` 降到 `0.05261`。
- CarRun seed 1 上，FNR 从 `0.18108` 降到 `0.06185`。
- `focal_soft` 的代价是 FPR 明显上升，ECE 变差。

因此 Test A 支持第一个机制假设：focal loss 能让 OOD classifier 更保守，减少
unsafe/OOD 漏检。但这不是无代价提升，而是 safety-oriented tradeoff。

### 4.2 Test B: Matched-Budget Cost-Critic Separation

Test B 检查生成 OOD 样本在相同 query budget 下是否能得到更高 cost critic
估计。主要指标如下：

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

结果说明：

- CarRun seed 0 上，`qc_matched_gap` 从 `2.641` 提升到 `3.632`。
- CarRun seed 1 上，`qc_matched_gap` 从 `3.267` 提升到 `5.031`。
- selected matched gap 也在 CarRun 两个 seed 上提高。
- `focal_soft` 选择更多生成样本作为 OOD，这与 Test A 中更保守的分类行为一致。
- BallRun 上 original 的 `qc_matched_gap` 为负，说明其 cost critic 尺度不稳定；
  `focal_soft` 给出正向 matched gap。

因此 Test B 支持第二个机制假设：soft OOD weighting 可以增强 matched-budget
下 OOD cost signal 的分离。

### 4.3 Test C: Full Policy Evaluation

完整 policy 结果更复杂。下表对每个 task、seed、method 统计 4 个 target cost
上的平均 reward、平均 real cost，以及满足预算的 target 数量。

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

#### BallRun 分析

BallRun 上 `focal_soft` 有正面结果，但不是稳定支配 baseline。

| Seed | Observation |
| ---: | --- |
| 0 | `focal_soft` 在所有 target cost 上提高 reward，同时保持 zero cost。 |
| 1 | `focal_soft` 修复 strict-budget failure：original 在 target `1` 下 cost 为 `74`，`focal_soft` 为 `0`。但在 target `5/10` 下 reward 低于 original。 |
| 2 | `focal_soft` 在 target `2/5` 下 reward 更高且安全，但在 target `1` 下 cost 为 `33.6`，是明确失败。 |

BallRun 说明该方法在部分 setting 下可以提高 reward 或改善 strict-budget
safety，但 budget-conditioned policy stability 仍然不足。

#### CarRun 分析

CarRun policy 是明确负结果。

| Method | Target 1 cost | Target 5 cost | Avg reward | Avg real cost | Safe targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 181.00 | 181.25 | 859.456 | 181.200 | 0/4 |
| focal_soft | 183.05 | 183.55 | 913.301 | 183.213 | 0/4 |

`focal_soft` 在 CarRun 上提高了 reward，但 cost 没有降低。两个方法在所有
target cost 下都严重超预算。因此 CarRun 说明机制层指标改善不能直接保证
policy 层安全。

## 5. 讨论

本实验的主要发现可以分成两层。

第一，机制层结果比较稳定。focal loss 在 BallRun 和 CarRun 上都显著降低
FNR，说明它确实让 classifier 更少漏掉 unsafe/OOD 样本。Test B 中
`focal_soft` 在 CarRun 两个 seed 上都提高 matched-budget `Q_c` separation，
说明 soft OOD weighting 能增强 OOD cost signal。

第二，policy 层结果是 mixed。BallRun 上有若干正面结果，例如 seed 0 全部
target cost reward 提升且保持 zero cost，seed 1 修复 strict-budget failure。
但 seed 2 target cost `1` 出现明显失败。CarRun 上 policy 完全不安全，说明
50k 训练预算下这个轻量修改不足以解决更难任务的 budget-conditioned control。

这两个层面的差异很重要。它说明 Test A/B 是有价值的诊断实验，但不能替代
完整 policy 评估。最终报告不能声称提出了稳定新方法，只能声称提出了一个
机制上合理、部分 policy setting 有收益的轻量扩展。

## 6. 局限性

本实验有以下局限：

- 训练预算较小，只使用 50k update，不能代表完整 benchmark 表现。
- policy-level task 数量有限，完整 50k policy 只覆盖 BallRun 三个 seed 和
  CarRun 一个 seed。
- focal loss 改善 FNR 的同时提高 FPR 和 ECE，说明分类器校准变差。
- soft OOD weighting 增强了 OOD cost signal，但没有解决 CarRun policy 安全
  失败。
- 没有与其他 offline safe RL baseline 做系统比较。
- BallCircle fallback 数据集由于公开服务器 HTTP 502 未能加入实验。

## 7. 结论

本文完成了一个基于 CCAC 的课程项目级扩展实验。我们将原始 OOD classifier
的 BCE loss 替换为 focal loss，并使用 classifier 输出作为 soft OOD weight。
实验表明，该方法在机制层面有效：它能降低 unsafe/OOD 漏检率，并增强
matched-budget cost-critic separation。完整 policy 结果则更加谨慎：BallRun
上存在部分 reward 和 strict-budget safety 改善，但也有失败 case；CarRun 上
两种方法都严重超预算。

因此，本文结论是：focal + soft OOD weighting 是一个有用的诊断性和小规模
改进，可以作为未来更系统方法的基础，但当前实验不足以支持其作为稳定的新
offline safe RL 算法。

## 8. Reproducibility Notes

主要实验脚本：

- `validation_experiments/scripts/run_test_a_classifier.sh`
- `validation_experiments/scripts/run_test_b_cost_critic.sh`
- `validation_experiments/scripts/train_original_ccac_50k.sh`
- `OSRL/examples/train/train_ccac.py`
- `OSRL/examples/eval/eval_ccac.py`

结果文件：

- 完整日志与原始数值：`validation_experiments/results.md`
- 报告用结果表：`validation_experiments/final_report_results.md`
- 实验计划：`validation_experiments/final_project_plan.md`

核心命令示例：

```bash
TASK=OfflineCarRun-v0 SEED=1 VARIANTS="original ours_focal_soft" \
  OUTPUT_DIR=validation_experiments/outputs/test_a_carrun_seed1 \
  bash validation_experiments/scripts/run_test_a_classifier.sh
```

```bash
TASK=OfflineCarRun-v0 SEED=1 VARIANTS="original focal_soft" \
  OUTPUT_DIR=validation_experiments/outputs/test_b_carrun_seed1 \
  bash validation_experiments/scripts/run_test_b_cost_critic.sh
```

```bash
cd OSRL/examples/eval
/mnt/afs/L202500188/RL-mid-pre/CCAC/.venv/bin/python eval_ccac.py \
  --path <ABS_RUN_DIR> \
  --target_costs "[1,2,5,10]" \
  --eval_episodes 20 \
  --device cuda:0
```
