# Code completion tracker

> Update **after each phase** with actual changed files, completed checks, blocking issues, Git commit SHA and Kaggle run evidence. A placeholder is not a completed or validated experiment.

| Phase | Status | Deliverables | Validation evidence | Git SHA |
| --- | --- | --- | --- | --- |
| P0 repo scaffold | In progress | README, docs, config placeholders, Python package, notebooks, CI skeleton | GitHub commits only; no Kaggle execution yet | See latest repo commit |
| P1 real data | Not started | provenance + duplicate groups + frozen splits | Not run | — |
| P2 generation pilot | Not started | checkpoint eligibility + morphology prompt pilot | Not run | — |
| P3 synthetic build | Not started | versioned generation + QA + manifests | Not run | — |
| P4 classifier training | Not started | baseline and controlled augmentation | Not run | — |
| P5 evaluation/stats | Not started | predictions, paired tests, tables | Not run | — |
| P6 extensions | Not started | low-data and optional studies | Not run | — |

## Execution journal format
- Date / phase:
- Branch / commit SHA:
- Code and config changes:
- Local checks (exact command and result):
- Kaggle notebook URL / version / accelerator / wall time:
- Datasets and artifacts IDs:
- Verified outputs and metric files:
- Blockers and next action:

## Next task
Finish P0 with a real Kaggle environment-check notebook, minimal config schema validation, installable package, smoke tests, and CI. **Do not** mark downstream phases complete without actual execution.
