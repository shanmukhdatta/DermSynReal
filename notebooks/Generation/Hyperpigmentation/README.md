# Hyperpigmentation — SD2.1 generation and LoRA

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


- `01_SD21_Hyperpigmentation_Compel_Inference.ipynb`: [300 authored prompts](../../../prompts/Hyperpigmentation/04_Hyperpigmentation/json/hyperpigmentation_all_300.json) cover melasma-like, post-inflammatory and solar-lentigines-like patterns; 210/45/45 group-disjoint prompt partitioning.
- `02_SD21_Train_Facial_Hyperpigmentation_LoRA.ipynb`: uses only reviewed and frozen **real train** images from Kaggle v2. Published textual-inversion concepts are not equivalent to custom LoRA weights.
- Saved: generation `metadata.jsonl`, review-required `reports/generation_manifest.csv` and `synthetic_pool_review_draft.csv`, `run_log.jsonl`, ZIP; optional LoRA weight, trainer console log and provenance.
