# RL Course Project: CCAC Extension

This repository contains an RL course project built on top of the original
CCAC implementation. The project studies whether a small change to CCAC's OOD
classifier and OOD cost-critic regularization can improve safety-relevant OOD
handling in offline safe reinforcement learning.

## Repository Layout

```text
.
├── CCAC/                         # CCAC codebase, experiment scripts, and final experiment docs
├── final_report/                 # Final presentation Slidev project and exported PDF
├── midterm_report/               # Midterm presentation Slidev project and exported PDF
├── Guo ... CCAC ... .pdf         # Original CCAC paper reference
└── README.md                     # This repository overview
```

## Main Project Code And Experiments

The main code and experiment records are under:

```text
CCAC/
```

Important entry points:

- `CCAC/README.md`: original CCAC project overview plus our validation entry points.
- `CCAC/EXPERIMENT_PLAN.md`: early small-scale validation plan.
- `CCAC/validation_experiments/README.md`: overview of our added validation experiments.
- `CCAC/validation_experiments/commands.md`: reproducible commands for environment checks, training, and evaluation.
- `CCAC/validation_experiments/results.md`: raw experiment records, run directories, and conclusions.
- `CCAC/validation_experiments/final_project_plan.md`: final-project expansion plan.
- `CCAC/validation_experiments/final_report.md`: final written report draft.
- `CCAC/validation_experiments/final_report_results.md`: report-ready result tables.
- `CCAC/validation_experiments/scripts/`: runnable helper scripts for Test A/B/C.

The final experiment compares original CCAC against:

```text
classifier_loss=focal
ood_mode=soft
focal_alpha=0.75
focal_gamma=2.0
```

The core conclusion is intentionally cautious: the focal + soft variant improves
OOD classifier false negative rate and matched-budget cost-critic separation,
but full policy results are mixed. It helps in several BallRun settings and
fails to solve policy-level safety on CarRun.

## Final Presentation

The final presentation is in:

```text
final_report/
```

Files:

- `final_report/slides.md`: Slidev source for the final presentation.
- `final_report/slides-export.pdf`: exported final presentation PDF.
- `final_report/README.md`: local preview/export instructions.
- `final_report/package.json`: Slidev commands and dependencies.

Preview or export:

```bash
cd final_report
npm install
npm run dev
npm run export
```

## Midterm Presentation

The midterm materials are separated into:

```text
midterm_report/
```

Files:

- `midterm_report/slides.md`: Slidev source for the midterm presentation.
- `midterm_report/slides-export.pdf`: exported midterm presentation PDF.
- `midterm_report/rl_midterm_soft_ood_qa_bilingual.md`: bilingual Q&A notes.
- `midterm_report/README.md`: local preview/export instructions.
- `midterm_report/package.json`: Slidev commands and dependencies.

Preview or export:

```bash
cd midterm_report
npm install
npm run dev
npm run export
```

## Reference Paper

The original paper is kept at the repository root:

```text
Guo 等 - 2025 - CONSTRAINT-CONDITIONED ACTOR-CRITIC FOR OFFLINE SAFE REINFORCEMENT LEARNING.pdf
```

It is the main method reference for CCAC and the baseline that this project
extends.

## Suggested Reading Order

For understanding the project quickly:

1. Read `final_report/slides.md` or open `final_report/slides-export.pdf`.
2. Read `CCAC/validation_experiments/final_report_results.md` for the compact result tables.
3. Read `CCAC/validation_experiments/results.md` for raw records and run directories.
4. Read `CCAC/validation_experiments/commands.md` if you need to reproduce the runs.
5. Read `midterm_report/slides.md` to see how the project evolved from the midterm plan.

## Notes

- The Slidev projects are intentionally separated so the midterm and final
  presentations can be edited independently.
- `final_report/` uses a Node-18-compatible Slidev version because the current
  environment reports Node `v18.19.1`.
- Large training outputs and local environments are not intended to be the main
  reading path; the Markdown documents under `CCAC/validation_experiments/`
  are the source of truth for experiment interpretation.
