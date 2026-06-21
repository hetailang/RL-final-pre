# CCAC Validation Experiments

This folder records the small validation experiments for improving the CCAC OOD
classifier and OOD usage. The goal is to validate the idea with cheap,
controlled experiments before running a full paper-style benchmark.

## Folder Contents

- `commands.md`: runnable commands for sanity checks, training, and evaluation.
- `final_project_plan.md`: compact plan for extending the validation results
  into a final-project experiment.
- `final_report.md`: complete final report draft.
- `final_report_results.md`: report-ready result tables and conclusions.
- `results.md`: place to record observed outputs and conclusions.
- `scripts.md`: planned experiment scripts and implementation notes.
- `scripts/`: runnable experiment scripts.

## Experiment Order

1. Sanity baseline: short original CCAC run on `OfflineBallRun-v0`.
2. Test A: classifier-only comparison.
3. Test B: cost-critic IND/OOD separation.
4. Test C: small full-policy comparison.

Do not start with the full paper benchmark. First confirm that the original code
runs, then validate the proposed classifier and OOD-use changes.

## Current Status

- Original CCAC sanity and 50k baseline scripts are implemented.
- Test A classifier-only scripts are implemented and initial results are recorded.
- Test B cost-critic separation scripts are implemented; the current version uses
  matched-budget IND/OOD comparison.
- Test C full-policy variant is implemented through CCAC training flags
  (`classifier_loss`, `ood_mode`, `focal_alpha`, `focal_gamma`).
- The final-project expansion has completed `OfflineBallRun-v0` seeds `0/1/2`
  for original CCAC and `focal_soft`.
- The planned second-task extension was initially blocked by HTTP 502 dataset
  downloads, but `SafetyCarRun-v0-40-651.hdf5` was later supplied locally and
  `OfflineCarRun-v0` Test A/B plus seed-0 50k policy comparison are completed.
- `OfflineBallCircle-v0` remains blocked by the public DSRL dataset server and
  is no longer needed for the final-project scope.
- Current conclusions and the fallback policy comparison are recorded in
  `results.md`; the execution plan and remaining blocker are recorded in
  `final_project_plan.md`.
