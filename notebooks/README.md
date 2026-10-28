# Kaggle notebooks — phase by phase

Notebooks are thin runners built on `src/dermsynreal/` and should be independently restartable on Kaggle T4. Each notebook will print exact input/output mounts, package revisions, config hash, compute info, seed, Git SHA and artifacts.

| Notebook | Responsibility | Status |
| --- | --- | --- |
| 00_environment_check.ipynb | Check Kaggle GPU, PyTorch, package versions, disk and mounts | Next |
| 01_real_data_audit.ipynb | Dataset provenance, images, deduplication and splitting | Planned |
| 02_generation_pilot.ipynb | Test generators, prompt families, morphology samples | Planned |
| 03_synthetic_generation.ipynb | Run reproducible batches and QA | Planned |
| 04_train_classifier.ipynb | Fine-tune classifiers for experiment configs | Planned |
| 05_evaluation.ipynb | Fixed real test metrics and per-case predictions | Planned |
| 06_ablation_studies.ipynb | Low-data and optional analyses | Planned |

Do not claim a notebook is executable until it exists and has been tested on a matching Kaggle runtime.
