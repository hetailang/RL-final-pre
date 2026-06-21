---
theme: seriph
title: Robust Offline Safe RL with Soft OOD Cost Regularization
info: RL Final Presentation
class: text-slate-950
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Robust Offline Safe RL with Soft OOD Cost Regularization

## A Small-Scale Extension of CCAC

RL Final Project Presentation  
2026.06

<style>
.slidev-layout.cover {
  background: #ffffff !important;
  color: #0f172a !important;
}

.slidev-layout.cover h1,
.slidev-layout.cover h2,
.slidev-layout.cover p {
  color: #0f172a !important;
  text-shadow: none !important;
}

table {
  font-size: 0.78rem;
  line-height: 1.2;
}

th,
td {
  padding: 0.32rem 0.45rem;
}
</style>

---
layout: center
---

# Background: Offline Safe RL

Offline safe reinforcement learning learns a policy from a fixed dataset while
controlling cumulative safety cost.

$$
\max_\pi \ \mathbb{E}_{\pi}[R(\tau)]
\quad
\text{s.t.}
\quad
\mathbb{E}_{\pi}[C(\tau)] \le \epsilon
$$

The setting is difficult because the policy must handle:

1. distribution shift from offline data
2. safety constraints during evaluation
3. different or changing safety budgets

---

# CCAC as the Backbone

CCAC learns a constraint-conditioned actor-critic policy.

$$
\pi_\theta(a_t \mid s_t, \kappa_t),
\qquad
\kappa_{t+1} = \kappa_t - c_t
$$

The remaining budget $\kappa_t$ is part of the policy input, so one trained
policy can be evaluated under multiple target costs.

| Component | Role |
| --- | --- |
| CVAE | generates budget-conditioned state-action samples |
| OOD classifier | identifies generated OOD / unsafe samples |
| cost critic | overestimates cost for risky generated samples |
| actor | maximizes reward while avoiding high predicted cost |

---

# Motivation: Brittle OOD Decisions

The original CCAC implementation uses a hard OOD decision:

$$
\mathbb{I}_{\text{OOD}}(s,a,\kappa)
=
\mathbb{I}\left[h_\psi(s,a\mid\kappa)>0.5\right]
$$

This creates two safety-relevant risks:

1. near-threshold samples such as $0.49$ and $0.51$ are treated very differently
2. false negatives are dangerous because unsafe OOD samples may not be penalized

Our project asks whether a more conservative classifier objective and soft OOD
usage can improve CCAC's safety signal.

---

# Our Modification

We compare original CCAC with one selected variant.

| Method | Classifier loss | OOD usage |
| --- | --- | --- |
| Original CCAC | BCE | hard mask |
| Focal + soft | focal loss | soft OOD weight |

Focal loss:

$$
FL(p_t) = \alpha_t(1-p_t)^\gamma BCE(p, y)
$$

Soft OOD weighting:

$$
w_{\text{OOD}}(s,a,\kappa) = clamp(h_\psi(s,a\mid\kappa), 0, 1)
$$

Final variant: `classifier_loss=focal`, `ood_mode=soft`,
`focal_alpha=0.75`, `focal_gamma=2.0`.

---

# Experiment Design

We use a three-level validation pipeline instead of jumping directly to a full
paper benchmark.

| Test | Question | Cost |
| --- | --- | --- |
| Test A: classifier-only | Does focal loss reduce unsafe OOD false negatives? | cheap |
| Test B: cost-critic separation | Does soft weighting strengthen OOD cost signal? | moderate |
| Test C: full policy | Does the mechanism transfer to policy behavior? | expensive |

Scope:

- `OfflineBallRun-v0`: full policy comparison over seeds `0/1/2`
- `OfflineCarRun-v0`: mechanism repeat over seeds `0/1`, policy seed `0`
- all full policy runs use `50k` update steps

---

# Test A: OOD Classifier

False negative rate is the key safety metric: it measures unsafe/OOD samples
incorrectly treated as in-distribution.

| Task | Seed | Method | AUROC | AUPRC | FNR | FPR | ECE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| BallRun | 0 | original | 0.94270 | 0.93778 | 0.10380 | 0.17205 | 0.02227 |
| BallRun | 0 | focal_soft | 0.93943 | 0.93326 | 0.03565 | 0.28166 | 0.10002 |
| CarRun | 0 | original | 0.92558 | 0.88536 | 0.13206 | 0.17275 | 0.03152 |
| CarRun | 0 | focal_soft | 0.92649 | 0.88579 | 0.05261 | 0.28111 | 0.13474 |
| CarRun | 1 | original | 0.92868 | 0.89186 | 0.18108 | 0.12899 | 0.01671 |
| CarRun | 1 | focal_soft | 0.92992 | 0.89303 | 0.06185 | 0.25883 | 0.12755 |

Take-away: focal loss consistently lowers FNR, but increases FPR and worsens
calibration.

---

# Test A Interpretation

The classifier result supports a safety-oriented trade-off.

<div class="mt-8 grid grid-cols-2 gap-6">
  <div class="border border-slate-200 rounded p-5">
    <div class="text-xl font-semibold">Positive signal</div>
    <div class="mt-3 text-sm leading-relaxed">
      Focal loss misses fewer unsafe/OOD samples. This is useful because false
      negatives can prevent the cost critic from receiving conservative updates.
    </div>
  </div>
  <div class="border border-slate-200 rounded p-5">
    <div class="text-xl font-semibold">Cost of the change</div>
    <div class="mt-3 text-sm leading-relaxed">
      The classifier becomes more conservative: FPR rises, ECE worsens, and the
      score should not be described as better calibrated.
    </div>
  </div>
</div>

---

# Test B: Cost-Critic Separation

Test B compares generated OOD samples with matched in-distribution samples under
the same query budget.

| Task | Seed | Method | qc_matched_gap | qc_selected_matched_gap | OOD selected rate |
| --- | ---: | --- | ---: | ---: | ---: |
| BallRun | 0 | original | -26.829 | 88.324 | 0.4186 |
| BallRun | 0 | focal_soft | 8.711 | 6.131 | 0.7555 |
| CarRun | 0 | original | 2.641 | 4.929 | 0.3106 |
| CarRun | 0 | focal_soft | 3.632 | 6.644 | 0.4868 |
| CarRun | 1 | original | 3.267 | 6.069 | 0.2267 |
| CarRun | 1 | focal_soft | 5.031 | 8.027 | 0.4255 |

Take-away: `focal_soft` strengthens matched-budget OOD-vs-IND cost separation,
especially on CarRun across two seeds.

---

# Test B Interpretation

The mechanism-level evidence is stronger than the policy-level evidence.

| Observation | Meaning |
| --- | --- |
| More generated samples are selected or weighted as OOD | classifier becomes more conservative |
| Positive matched-budget gaps | OOD samples receive higher cost estimates than comparable IND samples |
| CarRun seed `0/1` both improve | mechanism result is not purely seed-specific |
| BallRun original has unstable scale | hard-mask behavior can produce noisy cost-critic separation |

This supports the diagnostic claim: focal + soft OOD weighting improves the cost
signal used by CCAC.

---

# Test C: Full Policy Summary

Policy results are useful but mixed. `Safe targets` counts how many target costs
in `[1, 2, 5, 10]` satisfy `real_cost <= target_cost`.

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

---

# BallRun: Useful but Not Uniform

| Seed | Main observation |
| ---: | --- |
| 0 | `focal_soft` improves reward across all target costs while keeping zero cost. |
| 1 | `focal_soft` fixes the strict-budget failure: original has cost `74` at target `1`, while `focal_soft` has cost `0`. |
| 2 | `focal_soft` improves reward at target `2/5` and remains safe there, but fails at target `1` with cost `33.6`. |

Conclusion for BallRun:

- the variant can improve reward or strict-budget safety in some settings
- it is not uniformly safer across seeds and budgets
- budget-conditioned policy stability remains incomplete

---

# CarRun: Negative Policy Result

CarRun is the clearest policy-level failure case.

| Method | Target 1 cost | Target 5 cost | Avg reward | Avg real cost | Safe targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 181.00 | 181.25 | 859.456 | 181.200 | 0/4 |
| focal_soft | 183.05 | 183.55 | 913.301 | 183.213 | 0/4 |

Interpretation:

- `focal_soft` increases reward on CarRun
- it does not reduce realized cost
- both methods remain far above all requested safety budgets after 50k updates

This shows that better OOD diagnostics do not automatically solve policy-level
safe control.

---

# Overall Findings

| Level | Result | Interpretation |
| --- | --- | --- |
| Classifier | FNR decreases on BallRun and CarRun | fewer unsafe/OOD samples are missed |
| Cost critic | matched-budget separation improves | stronger diagnostic OOD cost signal |
| BallRun policy | several useful improvements, one strict-budget failure | promising but unstable |
| CarRun policy | reward improves, cost remains unsafe | negative result |

Final position:

`focal + soft` is a useful small-scale extension and diagnostic tool, not a
robust new offline safe RL algorithm.

---

# Limitations

1. Full policy training uses only `50k` updates.
2. Policy-level evaluation covers BallRun seeds `0/1/2` and CarRun seed `0`.
3. Focal loss lowers FNR but worsens FPR and ECE.
4. Soft OOD weighting improves the cost signal but does not solve CarRun safety.
5. We do not compare against the full set of OSRL baselines.
6. The public DSRL server blocked the BallCircle fallback dataset during the
   final expansion.

These limits are important: the result is best read as a compact course-project
extension with honest positive and negative evidence.

---

# Conclusion

We extended CCAC with a small OOD regularization change:

- focal loss for the OOD classifier
- soft OOD score weighting in cost-critic updates

The extension improves mechanism-level diagnostics and some BallRun policy
settings. However, full policy behavior is mixed, and CarRun remains unsafe.

<div class="mt-8 border border-slate-200 rounded p-5 text-xl leading-relaxed">
The main contribution is a reproducible and interpretable analysis pipeline for
OOD-based safety regularization in CCAC, with clear evidence about where the
idea helps and where it still fails.
</div>

---
layout: center
---

# Q&A

Thank you!

---

# References

- Guo, Z., Zhou, W., Wang, S., & Li, W. Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning. ICLR 2025.
- Liu, Z. et al. Datasets and Benchmarks for Offline Safe Reinforcement Learning. 2024.
- Liu, Z. et al. Constrained Decision Transformer for Offline Safe Reinforcement Learning. ICML 2023.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS 2020.
