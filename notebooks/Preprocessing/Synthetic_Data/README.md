# Synthetic prompt-family splitting — Kaggle

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


Run only **after** the six `Generation/<class>/01_*.ipynb` generators have produced and you have **manually reviewed** their per-image `reports/generation_manifest.csv` records. Combine the reviewed CSVs with `class_name,image_path,prompt_id,prompt_family_id,prompt_partition,accepted` and other provenance columns into one CSV. Images start `accepted=False`; set `accepted=True` only for qualified reviewed images. Set input paths in `DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb`.

The default preserves the original 210 train/45 val/45 test prompt assignments, protecting all family members from crossing partitions. With only five **train** pilot prompts, you do not yet have synthetic val/test; this is expected. Run approved family-disjoint validation/test generation separately if diagnostic synthetic test metrics are needed, or use only reviewed train generations for E1–E4 augmentation. Exact SHA and RGB pixel hashes deduplicate; this notebook does not perform a clinical check or full pHash/embedding near-duplicate search.

Output: `reports/split_manifest.csv`, duplicate/conflict/invalid CSVs, `reports/dataset_summary.json`, `reports/leakage_audit.json`, `run_log.jsonl`, and `synthetic_facial_split.zip`. Store images privately as versioned Kaggle datasets. Do not commit raw or synthetic face image sets to GitHub.


## Base + LoRA combination notebook — run first

`00_Combine_Reviewed_Base_And_LoRA_Generations.ipynb` (or `scripts/combine_reviewed_synthetic_manifests.py`) reads the **reviewed copies** of generation CSVs and their separately mounted image output folders. It re-resolves old Kaggle image paths, verifies the original SHA-256 of each accepted image, requires a reviewer and original prompt-family metadata, and emits `combined_reviewed_synthetic_pool.csv`, a summary JSON, and private logs. Supply separate `image_root` values for each run to avoid missing downloaded paths. It fails on mixed base/LoRA generator revisions within a class unless you filter or run comparisons separately. **No images are automatically approved.**

Next, `DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb` reads the combined CSV, preserves authored prompt-family partitions, performs exact-image dedup/conflict quarantine and writes its split ZIP/CSV reports. A five-prompt train-only pilot does not contain enough authored val/test families for the complete synthetic diagnostic split; preserve it for augmentation or generate diagnostic held-out families separately.


## Final combined base/LoRA dataset handoff

Use `ALLOW_MIXED_GENERATORS=True` in `00_Combine_Reviewed_Base_And_LoRA_Generations.ipynb` only to assemble a single **preprocessing** pool while preserving base/LoRA identity per image. The pool is *not* a single experimental generator arm. Then run `DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb` with `PRESERVE_PROMPT_PARTITIONS=True` and `ONLY_HUMAN_ACCEPTED=True`. It now writes `reports/synthetic_train_pool_for_experiments.csv` containing train-only, reviewed, deduplicated images with complete prompt/generator/adapter provenance and paths **relative** to its output image root. Save the split images/ZIP as Kaggle Input and run the experiment builder with `--synthetic-pool <...>/synthetic_train_pool_for_experiments.csv --synthetic-root <mounted synthetic split root> --generator-mode base` (or the LoRA mode you actually used). The builder then makes matched E0–E5 manifests and uses absolute paths for the selected synthetic images. Never train on synthetic held-out prompt families.
