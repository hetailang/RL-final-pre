---
theme: seriph
title: Constraint-Conditioned Actor-Critic for Offline Safe RL
info: RL Mid-term Presentation
class: text-slate-950
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning

## Offline Safe RL under Dynamic Safety Budgets

RL Mid-term Presentation  
2026.06.03

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
</style>

---
layout: center
---

# Background: Why Offline RL?

Traditional reinforcement learning usually requires repeated interaction between an agent and the environment.

In many real-world domains, online exploration is expensive, risky, or unacceptable.

<div class="mt-10 grid grid-cols-3 gap-4 text-center">
  <div class="rounded border border-slate-200 p-4">
    <div class="text-xl font-semibold">Robotics</div>
    <div class="mt-2 text-sm text-slate-700">Avoid collisions, falls, and unsafe motions</div>
  </div>
  <div class="rounded border border-slate-200 p-4">
    <div class="text-xl font-semibold">Autonomous Driving</div>
    <div class="mt-2 text-sm text-slate-700">Unsafe exploration cannot be used to collect data</div>
  </div>
  <div class="rounded border border-slate-200 p-4">
    <div class="text-xl font-semibold">Healthcare</div>
    <div class="mt-2 text-sm text-slate-700">Policies must satisfy strict risk budgets</div>
  </div>
</div>

---

# From Offline RL to Offline Safe RL

Offline RL aims to learn a policy only from a fixed dataset, without additional environment interaction.

In safety-critical settings, the policy must also satisfy a cumulative cost constraint:

$$
\max_\pi \ \mathbb{E}_{\pi}\left[\sum_t r_t\right]
\quad
\text{s.t.}
\quad
\mathbb{E}_{\pi}\left[\sum_t c_t\right] \le \epsilon
$$

where:

- $r_t$: task reward, such as speed, distance, or completion
- $c_t$: safety cost, such as collision, boundary violation, or overspeeding
- $\epsilon$: a fixed safety threshold for the total acceptable risk

---
layout: two-cols
---

# New Challenges in Offline Safe RL

## 1. Offline distribution shift

The dataset only covers historical behavior. At deployment, the learned policy may visit rare or unsupported state-action pairs, making value estimation unreliable.

::right::

# &nbsp;

## 2. Fixed safety constraints

Many Safe RL methods train for a fixed threshold $\epsilon$.  
In deployment, the available safety budget may vary across scenarios, tasks, or time steps.

<div class="mt-8 rounded border border-slate-200 p-4 text-sm">
The core tension: maximize return while avoiding unsafe extrapolation and budget violation.
</div>

---
layout: center
---

# Our Research Question

Can we learn a policy from offline data that, at deployment time, can:

- Achieve high reward
- Satisfy safety constraints
- Adapt to different safety budgets
- Reduce the risk from OOD states and actions

$$
\pi_\theta(a_t \mid s_t, \kappa_t)
$$

<div class="mt-8 text-xl font-semibold">
In short: the policy should adapt its behavior according to the current remaining safety budget.
</div>

---

# Core Idea: Condition on the Safety Budget

Instead of training a separate policy for each fixed threshold $\epsilon$, we condition the policy on the current remaining budget:

$$
\pi_\theta(a_t \mid s_t, \kappa_t)
$$

Here, $\kappa_t$ denotes the remaining safety budget at time step $t$. Given an initial budget $\kappa_1$, it is updated after each step:

$$
\kappa_{t+1} = \kappa_t - c_t
$$

$$
\underbrace{s_t}_{\text{state}}
\ ,\
\underbrace{\kappa_t}_{\text{remaining safety budget}}
\quad \longrightarrow \quad
\underbrace{\pi_\theta(a_t \mid s_t,\kappa_t)}_{\text{constraint-conditioned actor}}
\quad \longrightarrow \quad
\underbrace{a_t}_{\text{action}}
$$

$$
a_t
\quad \Longrightarrow \quad
\begin{cases}
R = \sum_{t=0}^{T} r_t, & \text{maximize reward return} \\
C = \sum_{t=0}^{T} c_t \le \epsilon, & \text{satisfy total cost threshold}
\end{cases}
$$

Changing the initial budget $\kappa_1$ allows the same policy to trade off reward and cost adaptively.

---

# Method: Constraint-Conditioned Actor-Critic

The training pipeline can be summarized in five modules:

| Module | Role | Notation |
|---|---|---|
| Offline data | Build budget-annotated transitions | $D=\{(s_t,a_t,r_t,c_t,s_{t+1},\kappa_t)\}$ |
| Constraint-conditioned CVAE | Generate state-action pairs under a budget | $(\hat{s},\hat{a})\sim p_\phi(s,a\mid z,\kappa)$ |
| OOD classifier | Detect whether $(s,a,\kappa)$ is OOD / unsafe | $h_\psi(s,a\mid\kappa)\in[0,1]$ |
| Cost critic | Overestimate cost values for OOD samples | $Q_c(s,a\mid\kappa)$ |
| Reward critic + Actor | Maximize reward under safety constraints | $Q_r(s,a\mid\kappa),\ \pi_\theta(a\mid s,\kappa)$ |

$$
D
\rightarrow
p_\phi(s,a\mid z,\kappa)
\rightarrow
h_\psi(s,a\mid\kappa)
\rightarrow
Q_c(s,a\mid\kappa)
\rightarrow
\pi_\theta(a\mid s,\kappa)
$$

Take-away: OOD / unsafe samples are assigned high cost, so the actor learns to avoid high-risk actions.

---

# Feasibility and Experimental Plan

We plan to evaluate the framework on the DSRL benchmark.

| Dimension | Plan |
|---|---|
| Environments | Run / Circle / Velocity-style safety control tasks |
| Metrics | Normalized reward, normalized cost, constraint violation |
| Baselines | BC-safe, CQL-based Safe RL, Lagrangian Offline RL, sequence modeling methods |
| Ablations | Remove CVAE, remove OOD classifier, fixed vs. varying safety budgets |

<div class="mt-6 rounded border border-slate-200 p-4">
Feasibility: the datasets and environments are available; Actor-Critic, CVAE, and classifier modules are standard; by the final presentation, we can first reproduce the main reward-cost trade-off and then add one extension such as dynamic budget switching or sparse safe trajectories.
</div>

---

# Related Work

<table class="text-[0.82rem] leading-tight">
  <thead>
    <tr>
      <th>Direction</th>
      <th>Methods</th>
      <th>What they address</th>
      <th>Limitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Offline RL</td>
      <td>CQL, IQL, BCQ, BEAR</td>
      <td>Distribution shift</td>
      <td>No explicit safety constraints</td>
    </tr>
    <tr>
      <td>Safe RL</td>
      <td>Lagrangian, CPO, PID-Lagrangian</td>
      <td>Online constrained optimization</td>
      <td>Needs interaction; often fixed thresholds</td>
    </tr>
    <tr>
      <td>Offline Safe RL</td>
      <td>CPQ, COptiDICE, VOCE, FISOR</td>
      <td>Reward-cost trade-off from offline data</td>
      <td>Hard to adapt to changing budgets</td>
    </tr>
    <tr>
      <td>Sequence / trajectory</td>
      <td>CDT, TREBI</td>
      <td>Condition on reward / cost</td>
      <td>Limited generalization to unseen budgets</td>
    </tr>
  </tbody>
</table>

<div class="mt-4 text-xl font-semibold">
Key gap: adapting behavior under dynamic safety budgets.
</div>

---
layout: center
---

# Q&A

Thank you!

---

# References

- Levine, S., Kumar, A., Tucker, G., & Fu, J. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. 2020.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS 2020.
- Liu, Z. et al. Datasets and Benchmarks for Offline Safe Reinforcement Learning. 2023.
- Liu, Z. et al. Constrained Decision Transformer for Offline Safe Reinforcement Learning. ICML 2023.
