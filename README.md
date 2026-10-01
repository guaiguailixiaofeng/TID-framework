# TID-framework

This repository contains reproducible materials for the T-I-D framework analysis
of bladder cancer stage-associated network and survival patterns.

## Repository contents

- `data/`: derived tables used to reproduce the revised figures.
- `figures/`: final reproducible figure files generated for the revision.
- `src/`: scripts for regenerating the revision figures from the derived tables
  and the required external public dataset.
- `requirements.txt`: minimal Python dependencies for figure reproduction.

The manuscript text, marked revision, cover letter, response letter, reference
library, and journal-submission files are intentionally not included.

## External public dataset

The analysis uses the public cBioPortal study:

`Bladder Urothelial Carcinoma (TCGA, PanCancer Atlas), blca_tcga_pan_can_atlas_2018`

The raw cBioPortal study folder is not bundled here. Download the study from
cBioPortal and place it at the path configured in `src/reproduce_revision_figures.py`,
or edit the script to point to your local data folder before rerunning.

## Reproducibility notes

The included derived tables support reproduction of the revised module-activity,
ACTN1 survival, and robustness figures. The T-I-D workflow is exploratory:
stage-ordered patterns should not be interpreted as longitudinal prediction, and
candidate-prioritization scores should not be interpreted as direct therapeutic
actionability.

## Quick start

```bash
python -m pip install -r requirements.txt
python src/reproduce_revision_figures.py
```

By default, regenerated figures are written to `figures/`. Use
`--data-dir` and `--figures-dir` to provide custom input and output folders.
