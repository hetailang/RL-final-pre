# RL Mid-term Q&A: Soft and Calibrated OOD Regularization

> 使用建议：这版 Q&A 对应你们现在的方案：不强调“复现某篇论文”，而是说我们以一个 constraint-conditioned actor-critic baseline 为基础，重点改进 OOD classifier 和 cost critic regularization。英文回答控制在 20-35 秒左右。

## Q1. What is the main idea of your project?

**问题中文翻译：你们项目的主要思路是什么？**

**中文回答：**  
我们的项目关注 offline safe RL 中的 dynamic safety budget 问题。核心思路是让 policy 接收当前剩余 safety budget 作为输入，从而根据预算大小调整行为。在此基础上，我们关注 OOD classifier 的不确定性问题，尝试用 soft and calibrated OOD regularization 替代 hard threshold，使 cost critic 的安全正则更加平滑和稳定。

**English answer:**  
Our project focuses on offline safe RL under dynamic safety budgets. The core idea is to condition the policy on the remaining safety budget, so that it can adapt its behavior according to the available budget. On top of that, we focus on the uncertainty of the OOD classifier. We propose to replace hard-threshold OOD filtering with soft and calibrated OOD regularization, so that the cost critic regularization becomes smoother and more stable.

## Q2. What problem do you observe in the original hard OOD threshold?

**问题中文翻译：你们观察到原来的 hard OOD threshold 有什么问题？**

**中文回答：**  
原来的 hard threshold 会把 classifier score 大于 0.5 的样本当成 OOD，小于 0.5 的当成 IND。但 0.49 和 0.51 其实非常接近，却会被完全不同地处理。如果 classifier 没有校准好，这种二值判断可能导致 cost critic 的 overestimation 不稳定，尤其可能漏掉真正危险的 OOD 样本。

**English answer:**  
The hard threshold treats samples with classifier score above 0.5 as OOD and below 0.5 as in-distribution. However, scores such as 0.49 and 0.51 are very close, but they are treated completely differently. If the classifier is not well calibrated, this binary decision may make cost critic overestimation unstable, and may miss truly unsafe OOD samples.

## Q3. What is your proposed modification?

**问题中文翻译：你们提出的修改是什么？**

**中文回答：**  
我们的修改是 Soft and Calibrated OOD Regularization。具体来说，我们不再只用 hard OOD label，而是使用 classifier 的 OOD score 作为连续权重。score 越高，说明样本越可能是 OOD 或 unsafe，那么 cost critic 的 conservative regularization 就越强。这样可以避免 hard threshold 带来的跳变。

**English answer:**  
Our modification is Soft and Calibrated OOD Regularization. Instead of using only a hard OOD label, we use the classifier's OOD score as a continuous weight. A higher score means the sample is more likely to be OOD or unsafe, so the cost critic receives stronger conservative regularization. This avoids the discontinuity caused by a hard threshold.

## Q4. Why do you use weighted BCE or focal loss for the classifier?

**问题中文翻译：为什么你们要用 weighted BCE 或 focal loss 来训练 classifier？**

**中文回答：**  
因为 OOD 和 IND 样本可能不平衡，普通 BCE 可能会偏向多数类，导致 classifier 漏掉真正危险的 OOD 样本。weighted BCE 可以提高少数类权重，focal loss 可以让模型更关注难分类样本。我们会通过 AUROC、AUPRC、false negative rate 和 calibration error 来判断它们是否真的更适合 safe RL。

**English answer:**  
OOD and in-distribution samples may be imbalanced, so standard BCE may be biased toward the majority class and miss truly unsafe OOD samples. Weighted BCE increases the weight of the minority class, while focal loss encourages the model to focus on hard examples. We will evaluate whether they are better for safe RL using AUROC, AUPRC, false negative rate, and calibration error.

## Q5. Why is false negative rate important here?

**问题中文翻译：为什么这里 false negative rate 很重要？**

**中文回答：**  
在 safe RL 中，false negative 指的是把真正 OOD 或 unsafe 的样本误判成 IND。这比 false positive 更危险，因为这些样本可能不会受到 cost overestimation，actor 之后可能会选择它们，导致 constraint violation。因此我们会特别关注 OOD false negative rate。

**English answer:**  
In safe RL, a false negative means that a truly OOD or unsafe sample is incorrectly classified as in-distribution. This is more dangerous than a false positive, because the sample may not receive cost overestimation, and the actor may later choose it, causing constraint violation. Therefore, we pay special attention to the OOD false negative rate.

## Q6. Does focal loss guarantee better calibration?

**问题中文翻译：focal loss 一定能带来更好的 calibration 吗？**

**中文回答：**  
不一定。focal loss 主要解决类别不平衡和难样本学习，但它不一定保证 probability calibration 更好。所以我们不会只看 loss，而会额外看 calibration error 和 OOD score histogram。如果 focal loss 提升了 detection 但 calibration 变差，我们也可以加入 temperature scaling 或者只把它作为一个 ablation。

**English answer:**  
Not necessarily. Focal loss mainly addresses class imbalance and hard examples, but it does not guarantee better probability calibration. Therefore, we will not rely only on the loss value. We will also evaluate calibration error and OOD score histograms. If focal loss improves detection but hurts calibration, we can add temperature scaling or treat it as an ablation.

## Q7. How exactly will the soft OOD score be used?

**问题中文翻译：soft OOD score 具体会怎么使用？**

**中文回答：**  
在 hard version 中，只有被判为 OOD 的样本才会触发更强的 cost overestimation。我们的 soft version 会把 OOD score 作为权重。例如 score 越接近 1，cost critic 对该样本的 conservative penalty 越强；score 越接近 0，penalty 越弱。这样边界样本的处理会更平滑。

**English answer:**  
In the hard version, only samples classified as OOD receive stronger cost overestimation. In our soft version, we use the OOD score as a weight. For example, when the score is close to 1, the cost critic receives a stronger conservative penalty for that sample. When the score is close to 0, the penalty becomes weaker. This makes the treatment of boundary samples smoother.

## Q8. What prior tests will you do before training the full RL policy?

**问题中文翻译：在训练完整 RL policy 之前，你们会做哪些先验测试？**

**中文回答：**  
我们会做三层先验测试。第一，只训练 OOD classifier，比较 BCE、weighted BCE、focal loss，以及 hard threshold 和 soft OOD score。第二，不完整训练 actor，只观察 cost critic 是否能把 OOD 样本的 Q_c 稳定抬高，同时保持 IND 样本在合理范围。第三，在 1 到 2 个小规模 DSRL 任务上跑完整策略，观察 reward、cost、violation rate 和 budget alignment。

**English answer:**  
We plan to conduct three levels of prior tests. First, we train only the OOD classifier and compare BCE, weighted BCE, focal loss, hard threshold, and soft OOD scores. Second, without fully training the actor, we inspect whether the cost critic can consistently assign higher Q_c values to OOD samples while keeping IND samples in a reasonable range. Third, we run the full policy on one or two small DSRL tasks and evaluate reward, cost, violation rate, and budget alignment.

## Q9. Why do you first test only the OOD classifier?

**问题中文翻译：为什么你们要先只测试 OOD classifier？**

**中文回答：**  
因为完整 offline RL 训练比较耗时，而且不稳定因素很多。如果我们先验证 classifier 是否能更准确识别 OOD 或 unsafe 样本，就能提前判断这个改进是否有希望。这个测试成本低，也能为后续完整训练提供先验证据。

**English answer:**  
Full offline RL training can be time-consuming and has many sources of instability. By first testing the OOD classifier, we can check whether our modification can better identify OOD or unsafe samples before running the full pipeline. This test is cheaper and provides prior evidence for whether the full method is promising.

## Q10. What do you expect to see in the cost critic test?

**问题中文翻译：在 cost critic 测试中，你们预期看到什么现象？**

**中文回答：**  
我们期望 OOD samples 的 Q_c 明显高于 IND samples，说明 cost critic 能识别高风险区域。同时，soft OOD version 的 Q_c 分布应该更平滑、更稳定，而不是因为 hard threshold 产生突变。如果没有 classifier，OOD 和 IND 的 Q_c 分离应该不明显。

**English answer:**  
We expect the Q_c values of OOD samples to be higher than those of IND samples, which means the cost critic can identify high-risk regions. At the same time, the soft OOD version should produce a smoother and more stable Q_c distribution, instead of abrupt changes caused by a hard threshold. Without the classifier, the separation between OOD and IND Q_c values should be less clear.

## Q11. What environments will you use?

**问题中文翻译：你们会使用哪些实验环境？**

**中文回答：**  
我们不会一开始跑所有 DSRL 任务，而是先选择 1 到 2 个小规模环境，比如 Ball-Run、Car-Run 或 Ball-Circle。这样可以控制计算成本，也更适合课程项目。等核心实验跑通之后，再考虑增加更多环境。

**English answer:**  
We will not start by running all DSRL tasks. Instead, we will first select one or two smaller environments, such as Ball-Run, Car-Run, or Ball-Circle. This keeps the computational cost manageable and is more suitable for a course project. After the core experiments work, we can consider adding more environments.

## Q12. What metrics will you report?

**问题中文翻译：你们会报告哪些评价指标？**

**中文回答：**  
classifier 层面，我们会报告 AUROC、AUPRC、false negative rate、false positive rate、calibration error 和 OOD score histogram。完整策略层面，我们会报告 normalized reward、normalized cost、constraint violation rate、cost-budget alignment，以及 small budget case 下的安全性。

**English answer:**  
At the classifier level, we will report AUROC, AUPRC, false negative rate, false positive rate, calibration error, and OOD score histograms. At the full policy level, we will report normalized reward, normalized cost, constraint violation rate, cost-budget alignment, and safety performance under small-budget cases.

## Q13. What is cost-budget alignment?

**问题中文翻译：什么是 cost-budget alignment？**

**中文回答：**  
cost-budget alignment 指的是实际产生的 cumulative cost 是否和给定 safety budget 匹配。理想情况下，如果 budget 较大，策略可以更积极；如果 budget 较小，策略应该更保守，并让实际 cost 更接近或低于预算。这个指标可以反映 policy 是否真的学会根据 budget 调整行为。

**English answer:**  
Cost-budget alignment measures whether the actual cumulative cost matches the given safety budget. Ideally, when the budget is large, the policy can be more aggressive; when the budget is small, the policy should become more conservative and keep the actual cost close to or below the budget. This metric shows whether the policy truly adapts to the budget.

## Q14. What if your soft OOD method reduces reward?

**问题中文翻译：如果你们的 soft OOD 方法导致 reward 下降怎么办？**

**中文回答：**  
这是有可能的，因为更强的 safety regularization 可能会让 policy 更保守。我们的目标不是单纯提高 reward，而是在 cost 更稳定、更少 violation 的情况下保持 reward 不明显下降。如果 reward 下降很多，我们会调节 soft weight 的强度，或者把它作为 safety-reward trade-off 的分析结果。

**English answer:**  
This is possible, because stronger safety regularization may make the policy more conservative. Our goal is not simply to maximize reward, but to keep cost more stable and reduce violations without a large reward drop. If the reward decreases significantly, we will tune the strength of the soft weight or analyze it as part of the safety-reward trade-off.

## Q15. What if the classifier is still inaccurate?

**问题中文翻译：如果 classifier 仍然不准确怎么办？**

**中文回答：**  
如果 classifier 仍然不准，我们会先通过 calibration error 和 false negative rate 定位问题。如果是类别不平衡，我们会调整 weighted BCE 或 focal loss；如果是概率不校准，我们会尝试 temperature scaling；如果整体检测能力不足，我们会把完整策略实验限制在 classifier 表现较好的环境上。

**English answer:**  
If the classifier is still inaccurate, we will first diagnose the issue using calibration error and false negative rate. If the problem is class imbalance, we will adjust weighted BCE or focal loss. If the probabilities are poorly calibrated, we can try temperature scaling. If detection is generally weak, we will limit the full policy experiments to environments where the classifier performs reasonably well.

## Q16. Is your method mainly an algorithmic change or an evaluation change?

**问题中文翻译：你们的方法主要是算法改动，还是 evaluation 改动？**

**中文回答：**  
它包含两部分。算法上，我们把 hard OOD filtering 改成 soft OOD regularization，并尝试改进 classifier training。实验上，我们增加 classifier-level 和 critic-level 的先验测试，用来解释为什么这个修改可能提升 safe RL 表现。所以它不仅是 evaluation，也有明确的 training objective 和 regularization change。

**English answer:**  
It has two parts. Algorithmically, we replace hard OOD filtering with soft OOD regularization and improve classifier training. Experimentally, we add classifier-level and critic-level prior tests to explain why this modification may improve safe RL performance. So it is not only an evaluation change; it also includes a clear change in the training objective and regularization.

## Q17. How is this different from just changing the classifier threshold?

**问题中文翻译：这和只是修改 classifier threshold 有什么区别？**

**中文回答：**  
只改 threshold 仍然是 hard decision，只是把边界从 0.5 移到其他位置。我们的 soft version 不再把样本二值化，而是保留 classifier 的连续不确定性信息。这样 0.49 和 0.51 不会被完全不同地处理，cost critic 的更新会更平滑。

**English answer:**  
Changing the threshold is still a hard decision; it only moves the boundary from 0.5 to another value. Our soft version does not binarize the samples. Instead, it keeps the continuous uncertainty information from the classifier. Therefore, scores like 0.49 and 0.51 will not be treated completely differently, and the cost critic update becomes smoother.

## Q18. What is your backup plan if full policy training is too expensive?

**问题中文翻译：如果完整 policy 训练成本太高，你们的 backup plan 是什么？**

**中文回答：**  
我们的 backup plan 是保留前两层实验：只训练 OOD classifier，以及只分析 cost critic 对 OOD 和 IND 样本的分离效果。这两部分不需要完整跑完 actor 训练，也能证明我们的修改是否能改善 OOD detection 和 safety regularization。然后在 final presentation 中把完整策略实验作为 limited-scale result 或 future work。

**English answer:**  
Our backup plan is to keep the first two levels of experiments: training only the OOD classifier and analyzing the cost critic separation between OOD and IND samples. These two parts do not require full actor training, but they can still show whether our modification improves OOD detection and safety regularization. Then in the final presentation, the full policy experiment can be presented as a limited-scale result or future work.

## Short Answers for 1-minute Q&A

### If asked: What is the key novelty?

**问题中文翻译：你们的关键创新点是什么？**

**English:**  
The key novelty is replacing hard OOD filtering with soft and calibrated OOD regularization. Instead of making a binary decision at threshold 0.5, we use the classifier confidence as a continuous weight for cost critic regularization. This is designed to make safety regularization smoother and more robust to classifier uncertainty.

### If asked: Why is this useful for safe RL?

**问题中文翻译：为什么这个修改对 safe RL 有用？**

**English:**  
In safe RL, missing unsafe OOD samples can directly lead to constraint violations. A soft and better-calibrated OOD score can reduce false negatives and make the cost critic more reliable. This should help the actor avoid risky actions, especially under small safety budgets.

### If asked: How will you prove the modification works?

**问题中文翻译：你们怎么证明这个修改有效？**

**English:**  
We will test it at three levels. First, classifier metrics such as AUROC, AUPRC, false negative rate, and calibration error. Second, cost critic separation between OOD and IND samples. Third, full policy metrics such as normalized reward, normalized cost, violation rate, and budget alignment on one or two DSRL tasks.

### If asked: Is this feasible?

**问题中文翻译：这个项目可行吗？**

**English:**  
Yes. We designed the experiments progressively. The classifier-only test is cheap, the cost-critic analysis is more lightweight than full RL, and the final policy evaluation can be limited to one or two small DSRL environments. This makes the project feasible within the course timeline.
