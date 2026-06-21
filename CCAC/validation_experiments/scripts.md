# Validation Experiment Scripts

This file records the implemented and planned validation experiment scripts.

## Test A: Classifier-Only Script

Implemented path:

```text
validation_experiments/scripts/train_ood_classifier.py
```

Purpose:

- Train only the CVAE and OOD classifier.
- Compare BCE, weighted BCE, focal loss, and focal loss with soft OOD score.
- Report classifier metrics without running full RL.

Inputs:

- `--task`
- `--device`
- `--seed`
- `--steps`
- `--batch_size`
- `--classifier_loss {bce,weighted_bce,focal}`
- `--ood_usage {hard,soft}`
- `--output_dir`

Launcher:

```text
validation_experiments/scripts/run_test_a_classifier.sh
```

Outputs:

- `metrics.json`
- `ood_score_histogram.png`
- `history.json`
- `model.pt`
- optional raw prediction table for later analysis

Metrics:

- AUROC
- AUPRC
- False negative rate
- False positive rate
- Expected calibration error

Implementation notes:

- Reuse `set_cost_thresholds(data)`.
- Reuse `CVAE` and `Classifier` from `OSRL/osrl/common/net.py`.
- Construct labels consistently with the existing `CCAC.vae_loss` logic first,
  then refine if needed.
- Keep the script independent from actor/critic training.

## Test B: Cost-Critic Separation Script

Implemented path:

```text
validation_experiments/scripts/train_cost_critic_separation.py
```

Purpose:

- Train CVAE/classifier/cost critic for a limited number of steps.
- Compare IND and OOD `Q_c` values under matched query budgets.
- Test hard OOD mask, soft OOD score weighting, and no-classifier variants.

Inputs:

- `--task`
- `--device`
- `--seed`
- `--steps`
- `--batch_size`
- `--ood_mode {hard,soft,none}`
- `--output_dir`

Launcher:

```text
validation_experiments/scripts/run_test_b_cost_critic.sh
```

Outputs:

- `metrics.json`
- `qc_gap_curve.png`
- `qc_histogram.png`
- `history.json`
- `model.pt`

Metrics:

- IND `Q_c` mean/std
- OOD `Q_c` mean/std
- OOD-minus-IND `Q_c` gap
- matched-budget OOD-minus-IND `Q_c` gap

Implementation notes:

- Start with analysis-only code instead of modifying full CCAC training.
- Generate candidate OOD `(s, a)` with CVAE.
- For `soft`, weight generated samples by classifier probability instead of
  selecting only scores above a threshold.
- The current script pushes generated OOD samples toward `matched IND Q_c +
  margin`, so `qc_matched_gap` is the primary separation metric.

## Test C: Small Full-Policy Variant

Implemented through CCAC training flags:

- `classifier_loss`
- `ood_mode`
- `focal_alpha`
- `focal_gamma`

The first selected variant is `classifier_loss=focal` with `ood_mode=soft`.
This was chosen from Test A/B and has completed 50k runs on
`OfflineBallRun-v0`, seeds `0/1/2`, plus `OfflineCarRun-v0`, seed `0`.

Initial tasks:

- `OfflineBallRun-v0`
- `OfflineCarRun-v0`, completed after local `SafetyCarRun-v0-40-651.hdf5`
  restore
- fallback `OfflineBallCircle-v0`, still blocked by public dataset server HTTP
  502 and no longer needed for the final-project scope

Initial training budget:

- `update_steps=50000`
- three completed `OfflineBallRun-v0` seeds
- one completed `OfflineCarRun-v0` seed

Evaluation target costs:

- `1`
- `2`
- `5`
- `10`

Example focal-soft command:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC/OSRL/examples/train
export WANDB_MODE=offline
export PYTHONPATH="/mnt/afs/L202500188/RL-mid-pre/CCAC/OSRL:/mnt/afs/L202500188/RL-mid-pre/CCAC/DSRL:${PYTHONPATH:-}"
export DSRL_DATASET_DIR="/mnt/afs/L202500188/RL-mid-pre/CCAC/datasets/dsrl"
PYTHON=/mnt/afs/L202500188/RL-mid-pre/CCAC/.venv/bin/python

"$PYTHON" train_ccac.py \
  --task OfflineBallRun-v0 \
  --device cuda:0 \
  --seed 1 \
  --update_steps 50000 \
  --eval_every 10000 \
  --eval_episodes 10 \
  --batch_size 512 \
  --num_workers 4 \
  --classifier_loss focal \
  --ood_mode soft \
  --suffix test_c_focal_soft_50k_seed1
```
