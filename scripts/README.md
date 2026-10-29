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
