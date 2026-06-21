# RL Final Report LaTeX

This folder contains the LaTeX version of the RL final report.

## Files

- `main.tex`: main report source.
- `references.bib`: BibTeX references.

## Build

The report is written in English with the standard `article` class. XeLaTeX was
used for verification:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Or, if `latexmk` is available:

```bash
latexmk -xelatex main.tex
```

## Notes

The report is based on:

- `CCAC/validation_experiments/final_report.md`
- `CCAC/validation_experiments/final_report_results.md`
- `CCAC/validation_experiments/commands.md`
- `final_report/slides.md`

Fill in the name and student ID fields in `main.tex` before submission.
