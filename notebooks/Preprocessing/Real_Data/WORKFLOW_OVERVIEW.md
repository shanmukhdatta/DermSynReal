# Real facial dermatology preprocessing — Kaggle 069

## Purpose

This workflow constructs a **frozen, image-similarity-grouped real-data split** for six-class facial dermatology research. The current standalone notebook is [DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb](DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb). It replaces the older standalone real preprocessing notebook.

**Raw input:** [Facial Skin Conditions — Real Image Collection](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/data), mounted as `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_datset`. The unusual `datset` spelling matches the uploaded directory.

**Processed dataset:** [Facial Dermatology Image Dataset](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-dermatology-image-dataset/data), a downstream result rather than an input for preprocessing.

## Processing procedure

1. Scan six category folders and inventory image files.
2. Validate readability, dimensions and basic image quality; calculate SHA-256 and pixel hashes.
3. Detect exact duplicates and four-orientation 64-bit pHash matches (original, H-flip, V-flip and 180° rotation; Hamming distance ≤5).
4. Compute original/mirrored ImageNet-pretrained ResNet-50 embeddings and compare cosine similarity.
5. Combine similarity links with Union-Find, log image-group assignments and inspect suspicious large clusters.
6. Quarantine groups that contain conflicting class labels.
7. Allocate whole groups using a seeded class-balance-aware search, targeting 70/15/15.
8. Assert that no detected image group or identical content hash crosses train, validation and test; export reports and a class-folder ZIP.

## Observed Kaggle 069 run (30 October 2026)

| Measure | Value |
| --- | ---: |
| Raw image files | 9,791 |
| Quality exclusions | 0 |
| Quarantined conflicting-label images | 6 |
| Eligible images | 9,785 |
| Original groups | 7,430 |
| Final groups | 7,428 |
| Selected embedding cosine threshold | 0.90 |
| Random seed | 42 |

| Split | Images | Share |
| --- | ---: | ---: |
| Train | 6,908 | 70.598% |
| Validation | 1,420 | 14.512% |
| Test | 1,457 | 14.890% |

The supplied audit found no cross-split reuse of paths, exact hashes or identified groups. This is **image-level duplicate-leakage control**, not patient-independent splitting, diagnosis review or evidence of complete source licensing compliance.

## Artifacts

The notebook writes individual CSV/JSON reports for source inventory, image quality and SHA-256, exact duplicate pairs, perceptual hashes and similarity pairs, ResNet-50 candidate pairs, threshold history, group summaries and assignments, quarantine decisions, frozen split manifest, per-class counts and final processing summary. It also produces a review HTML gallery.

The final ZIP, `dermalens_real_70_15_15.zip`, contains `train/`, `validation/` and `test/` directories, each with six original class folders, plus reports. Original image bytes are retained; model-specific RGB resizing, normalization and training-only augmentation belong in the classifier training notebooks.

## Reuse and precautions

- **Do not randomly split again** in downstream classifier notebooks. Freeze and reuse the same manifest for every model.
- LoRA diffusion training uses **only train images**. Keep validation/test images away from generator training.
- No need to repeat the expensive pHash/ResNet similarity or MD5 scan on every training run. Verify that the referenced frozen manifest and dataset version match; rerun audit if the data change.
- Manually review unusually large embedding-connected groups, especially the recorded 161-image Acne group, and check unresolved clinical label concerns in Skin Cancer and Normal Skin.
- Never claim patient identity independence. Original third-party dataset licenses still govern images and redistribution.
- Keep raw face photos, private image-level metadata and heavy ZIPs out of public GitHub; publish only approved aggregate reports.

**Legacy path:** `00_Facial_Real_V2_Kaggle_Start_Here.ipynb` and `pipeline.py` remain in the repository for the older review-first workflow; they are not the notebook used for the recorded Kaggle 069 run. Do not mix the two sets of outputs.
