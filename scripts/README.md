# Runnable research CLI wrappers

Run with `PYTHONPATH=src python scripts/<name>.py --help`. Every command uses explicitly provided files and creates no synthetic results.

| Command | Function |
| --- | --- |
| `build_experiment_manifests.py` | audited frozen real/synthetic training combinations E0–E4 |
| `infer_timm_checkpoint.py` | four supported uploaded timm checkpoint types |
| `infer_panderm_checkpoint.py` | official PanDerm + trained linear probe |
| `evaluate_predictions.py` | one real/synthetic score report and optional paired real delta |
| `report_experiments.py` | multi-arm E0–E5 tables with source/class and paired bootstrap |
| `report_calibration.py` | ECE, NLL, multiclass Brier on saved probabilities |
| `audit_generation_quality.py` | reviewer-generated per-class quality counts |

The label gate is also a module CLI: `PYTHONPATH=src python -m dermsynreal.label_gate --help`.

Inputs are actual Kaggle artifacts; no command here trains classifiers or verifies the medical correctness of labels. See `notebooks/Evaluation/README.md`.

| `prepare_public_run_report.py` | Copy only a whitelisted set of aggregate Kaggle results into a versioned `reports/experiments/<run-tag>/` directory for manual privacy review and explicit GitHub publication. It never auto-pushes images or checkpoints. |


## Additional active commands

`combine_reviewed_synthetic_manifests.py`:
```bash
PYTHONPATH=src python scripts/combine_reviewed_synthetic_manifests.py \
  --manifests /kaggle/input/acne-reviewed/review.csv /kaggle/input/normal-reviewed/review.csv \
  --image-roots /kaggle/input/acne-run /kaggle/input/normal-run \
  --out /kaggle/working/combined_reviewed_synthetic_pool.csv
```
These paths are **examples**, not files known to exist. The reviewed CSVs need `reviewer_id`, `reviewer_decision=approved` and `accepted=True` and have to match their mounted image SHA-256 values. A single class cannot silently mix different generator model/adapter revisions.

`report_statistics.py`: `PYTHONPATH=src python scripts/report_statistics.py --results-dir /kaggle/input/EDIT_EVAL_04 --out /kaggle/working/statistical_summary`

`prepare_public_run_report.py`: `PYTHONPATH=src python scripts/prepare_public_run_report.py --from-kaggle-output /kaggle/input/EDIT_EVAL_04 --run-tag pilot_001`. The output is a *candidate* reviewed aggregate GitHub report; never automatic background publishing.
