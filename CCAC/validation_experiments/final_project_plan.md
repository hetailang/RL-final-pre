# Final Project Expansion Plan

This plan turns the current CCAC validation work into a compact final-project
experiment. The goal is to provide a complete and defensible course assignment,
not to claim a fully general new offline safe RL method.

## Core Question

Can a small modification to CCAC's OOD classifier and OOD cost-critic usage
improve safety-relevant OOD handling while preserving or improving policy
reward on a small offline safe RL benchmark?

## Method Variant

The final project should compare original CCAC against one selected variant:

- `classifier_loss=focal`
- `ood_mode=soft`
- `focal_alpha=0.75`
- `focal_gamma=2.0`

This variant is selected from the existing validation results:

- Test A: focal loss reduces unsafe/OOD false negatives.
- Test B: focal + soft OOD weighting gives cleaner matched-budget `Q_c`
  separation than the original hard-mask setting.
- Test C seed 0 on `OfflineBallRun-v0`: focal + soft keeps zero realized cost
  across tested budgets and improves reward over the 50k original CCAC baseline.

## Experiment Matrix

Keep the experiment short enough to finish in one or two days.

| Task | Seed | Original CCAC | Focal + soft | Status |
| --- | ---: | --- | --- | --- |
| `OfflineBallRun-v0` | 0 | done | done | completed before expansion |
| `OfflineBallRun-v0` | 1 | done | done | completed in expansion |
| `OfflineBallRun-v0` | 2 | done | done | fallback for blocked second task |
| `OfflineCarRun-v0` | 0 | done | done | completed after local dataset restore |
| `OfflineBallCircle-v0` | 0 | blocked | blocked | fallback dataset still HTTP 502 |

Original fallback design: if `OfflineCarRun-v0` data or environment setup fails,
use `OfflineBallCircle-v0` as the second task.

Execution note on 2026-06-21: the public DSRL server initially returned HTTP
502 for both `OfflineCarRun-v0` and fallback `OfflineBallCircle-v0`. The user
later supplied `SafetyCarRun-v0-40-651.hdf5` locally, so the CarRun Test A,
Test B, and seed-0 policy comparison were completed. `OfflineBallCircle-v0`
remains a blocked fallback and is no longer needed for the final-project scope.

## Required Experiments

### 1. Environment and dataset check

Confirm the current environment still reports:

- PyTorch with CUDA support.
- One available H100 GPU.
- DSRL dataset can be opened by `h5py`.

Download or place the second-task dataset before launching training. In the
completed run, the local file was:

```text
datasets/dsrl/SafetyCarRun-v0-40-651.hdf5
```

The command below is kept for reproducibility if the public dataset server
recovers:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC
export DSRL_DATASET_DIR="$PWD/datasets/dsrl"
.venv/bin/python validation_experiments/scripts/download_dsrl_dataset.py --task OfflineCarRun-v0
```

### 2. Extend Test A to the second task

Run the classifier-only comparison on the second task:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC
TASK=OfflineCarRun-v0 bash validation_experiments/scripts/run_test_a_classifier.sh
```

Report:

- AUROC
- AUPRC
- false negative rate
- false positive rate
- expected calibration error

The main evidence is whether focal loss still lowers false negative rate.

### 3. Extend Test B to the second task

Run matched-budget cost-critic separation on the second task:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC
TASK=OfflineCarRun-v0 bash validation_experiments/scripts/run_test_b_cost_critic.sh
```

Report:

- `qc_matched_gap`
- `qc_selected_matched_gap`
- OOD selected rate

The main evidence is whether `focal_soft` keeps a positive matched-budget OOD
vs IND `Q_c` gap.

### 4. Complete Test C full-policy comparison

Run one additional seed on `OfflineBallRun-v0`.

Original CCAC:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC
SEED=1 SUFFIX=sanity50k_seed1 bash validation_experiments/scripts/train_original_ccac_50k.sh
```

Focal + soft:

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

Run the same original-vs-variant pair on `OfflineCarRun-v0` seed 0:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC
TASK=OfflineCarRun-v0 SEED=0 SUFFIX=sanity50k_car_seed0 bash validation_experiments/scripts/train_original_ccac_50k.sh
```

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC/OSRL/examples/train
export WANDB_MODE=offline
export PYTHONPATH="/mnt/afs/L202500188/RL-mid-pre/CCAC/OSRL:/mnt/afs/L202500188/RL-mid-pre/CCAC/DSRL:${PYTHONPATH:-}"
export DSRL_DATASET_DIR="/mnt/afs/L202500188/RL-mid-pre/CCAC/datasets/dsrl"
PYTHON=/mnt/afs/L202500188/RL-mid-pre/CCAC/.venv/bin/python

"$PYTHON" train_ccac.py \
  --task OfflineCarRun-v0 \
  --device cuda:0 \
  --seed 0 \
  --update_steps 50000 \
  --eval_every 10000 \
  --eval_episodes 10 \
  --batch_size 512 \
  --num_workers 4 \
  --classifier_loss focal \
  --ood_mode soft \
  --suffix test_c_focal_soft_50k_car_seed0
```

After each run, evaluate target costs `[1,2,5,10]`:

```bash
cd /mnt/afs/L202500188/RL-mid-pre/CCAC/OSRL/examples/eval
export PYTHONPATH="/mnt/afs/L202500188/RL-mid-pre/CCAC/OSRL:/mnt/afs/L202500188/RL-mid-pre/CCAC/DSRL:${PYTHONPATH:-}"
export DSRL_DATASET_DIR="/mnt/afs/L202500188/RL-mid-pre/CCAC/datasets/dsrl"
PYTHON=/mnt/afs/L202500188/RL-mid-pre/CCAC/.venv/bin/python

"$PYTHON" eval_ccac.py \
  --path <ABS_RUN_DIR> \
  --target_costs "[1,2,5,10]" \
  --eval_episodes 20 \
  --device cuda:0
```

## Report Tables

Prepare three compact tables:

1. Classifier table: task, variant, AUROC, AUPRC, FNR, FPR, ECE.
2. Cost-critic table: task, variant, `qc_matched_gap`,
   `qc_selected_matched_gap`, OOD selected rate.
3. Policy table: task, seed, method, target cost, reward, normalized reward,
   cost, normalized cost.

## Success Criteria

The final project is strong enough if:

- focal loss reduces false negative rate in at least one task and does not
  completely reverse on the second task, if a second task becomes available;
- `focal_soft` gives positive matched-budget OOD separation in Test B;
- `focal_soft` shows useful strict-budget or reward improvements in the small
  full-policy runs, while any negative seed/budget cases are reported;
- reward is improved or at least competitive on part of the main
  `OfflineBallRun-v0` setting.
- `OfflineCarRun-v0` is reported as a policy-level negative result: Test A/B
  mechanisms improve, but both original CCAC and `focal_soft` remain far above
  the requested cost budgets after 50k updates.

## Scope Limits

Do not run the full paper benchmark. Do not add all OSRL baselines, all DSRL
tasks, more than the completed three `OfflineBallRun-v0` seeds, or
dynamic-budget experiments. Those are beyond the course-project target and would
not change the core story enough to justify the runtime.
