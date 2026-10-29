# dermsynreal research modules

Shared reproducible building blocks, called by the Kaggle notebook orchestration under `notebooks/`. They are Python modules, not pretrained medical models.

| File | Role |
| --- | --- |
| `label_gate.py` | explicit review decision + metadata flags before real-data splitting |
| `experiment_manifest.py` | verified real IDs and matched E0–E5 synthetic training manifest construction |
| `evaluation.py` | six-class metric definitions, source separation and paired visual-group bootstrap |
| `evaluation_pipeline.py` | run registry, E0–E5 reporting, per-source/class metrics and paired contrasts |
| `inference.py` | load real saved weights for ResNet50, EfficientNet-B3, ConvNeXt Tiny and DINOv2 ViT-S/14; export probability predictions |
| `panderm_inference.py` | genuine PanDerm author encoder + saved frozen linear-probe evaluation |
| `calibration.py` | real/synthetic-separated ECE, NLL and multiclass Brier score |
| `quality_audit.py` | manually reviewed synthetic morphology/novelty decision summary |

Use `PYTHONPATH=src` for the current source tree. The GPU modules import third-party deep-learning libraries only at execution time; offline metadata/evaluation tests require only standard Python. No module substitutes missing clinical data, model checkpoints or patient identity.

The original scaffold's future `data/`, `generation/`, `selection/`, and `training/` package subdirectories are not claimed implemented; the actual existing Kaggle generation, grouping and training implementations live under `notebooks/`. Do not add redundant replacement code without a tested integration need.

- `real_collection.py` — Kaggle v2-only image inventory, exact SHA duplicate report, human-review staging and stable frozen-split manifest bridge.
- `kaggle_runs.py` — split-safe generic/morphology prompt loading, per-image generator provenance, approved-only train LoRA staging, CSV/JSONL run logs and model/checkpoint artifact hashes.


- `combine_synthetic.py` — class/generator-compatible reviewed pool combination with remounted image roots, SHA-256 verification and safe prompt-family retention.
- `statistical_analysis.py` — per-arm training-seed means/SD/median/ranges, per-class variability and descriptive matched contrasts, with CSV/JSON evidence. Paired group bootstrap confidence intervals live in `evaluation.py`.
- `kaggle_runs.py` also provides `start_notebook_logging()` to preserve a private console log and the six-LoRA class split audit CSVs without resplitting globally protected groups.
