# DermSynReal — Preprocessing notebooks

## Current real-image grouped split

The current Kaggle 069 run is implemented in [Real_Data/DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb](Real_Data/DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb). See [Real_Data/README.md](Real_Data/README.md) and [WORKFLOW_OVERVIEW.md](Real_Data/WORKFLOW_OVERVIEW.md) for the full procedure, outputs, thresholds, cautions and statistics.

**Source dataset:** [Facial Skin Conditions — Real Image Collection](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/data), `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_datset` (source spelling preserved).

**Kaggle 069 observed outcome:** 9,791 original images → 0 automated quality exclusions → 6 conflicting-label quarantined → **9,785 eligible** in **7,428** groups. Frozen partitions: **6,908 train / 1,420 validation / 1,457 test**, seed 42 and selected embedding cosine threshold 0.90.

The run checks found no identified duplicate group, exact content hash or file path crossing partitions. These are **image-level leakage checks**, not verified patient independence or medical diagnosis. Large groups and label errors still require review.

The older `Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb` and `Real_Data/pipeline.py` remain as a **legacy review-first path**. They do not represent the Kaggle 069 execution. Do not use both paths to create conflicting splits.

### Downstream usage

- Load the fixed partitions directly for real-only classifier training, model evaluation and the E0–E5 comparisons; **never create a new independent 70/15/15 split**.
- Fit diffusion LoRA adapters using real **train** images only; validation and test remain held out.
- Repeating the expensive SHA/pHash/ResNet/Union-Find grouping inside every training notebook is unnecessary. Confirm manifest/dataset identity and rerun audit if input images change.
- Original image rights, provenance, clinical review, potential patient overlap and suspicious large clusters remain open requirements.

## Synthetic-data preprocessing

Use [Synthetic_Data/README.md](Synthetic_Data/README.md) and the [synthetic prompt-group splitting notebook](Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb) only after generation and quality review. Preserve prompt-family isolation and screen for copies of held-out real photographs.

**Repository hygiene:** Keep face photos, raw image manifests with sensitive identifiers, heavy ZIP outputs and model checkpoints in private Kaggle storage rather than public GitHub.
