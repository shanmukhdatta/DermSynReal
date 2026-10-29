# Synthetic prompt-family splitting — Kaggle

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


Run only **after** the six `Generation/<class>/01_*.ipynb` generators have produced and you have **manually reviewed** their per-image `reports/generation_manifest.csv` records. Combine the reviewed CSVs with `class_name,image_path,prompt_id,prompt_family_id,prompt_partition,accepted` and other provenance columns into one CSV. Images start `accepted=False`; set `accepted=True` only for qualified reviewed images. Set input paths in `DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb`.

The default preserves the original 210 train/45 val/45 test prompt assignments, protecting all family members from crossing partitions. With only five **train** pilot prompts, you do not yet have synthetic val/test; this is expected. Run approved family-disjoint validation/test generation separately if diagnostic synthetic test metrics are needed, or use only reviewed train generations for E1–E4 augmentation. Exact SHA and RGB pixel hashes deduplicate; this notebook does not perform a clinical check or full pHash/embedding near-duplicate search.

Output: `reports/split_manifest.csv`, duplicate/conflict/invalid CSVs, `reports/dataset_summary.json`, `reports/leakage_audit.json`, `run_log.jsonl`, and `synthetic_facial_split.zip`. Store images privately as versioned Kaggle datasets. Do not commit raw or synthetic face image sets to GitHub.
