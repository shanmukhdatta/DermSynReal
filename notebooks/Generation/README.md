# DermSynReal — six-class synthetic generation and LoRA training

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


The class folders are correctly spelled: `Acne/`, `Fungal_Infection/`, `Hyperpigmentation/`, `Normal_Skin/`, `Skin_Cancer/`, `Vitiligo/`.

Every class contains `01_...` inference and `02_...` LoRA training notebooks, plus its class README and audit notes. **01** reads the existing `prompts/` 300-prompt JSON (210 train/45 val/45 test across 60 disjoint families); five train prompts are run by default as a GPU pilot. Set `PROMPT_LIMIT=None` for all train prompts, or `PROMPT_PARTITION='val'/'test'` to create separate diagnostics. Never merge held-out prompt-family generations into training.

**02** requires `real_all_manifest.csv`, approved split images, `label_audit_summary.json` with `passed=true`; subtype LoRAs need independent subtype review. No fallback to the original unsplit Kaggle folders. The generator output includes `metadata.jsonl`, `reports/generation_manifest.csv`, `reports/synthetic_pool_review_draft.csv`, `reports/generation_summary.json`, `run_log.jsonl` and images. The LoRA output includes `trainer_console.log`, checkpoint weights, provenance and `run_log.jsonl`. Save Kaggle outputs as versioned datasets. Do not commit clinical photographs, generated full-image pools, large checkpoints, or private logs to GitHub.

**After generation:** manually review disease morphology, novelty, exact/near-duplicates and source licensing. Approve only reviewed images in the synthetic pool, run `Preprocessing/Synthetic_Data/`, then use `scripts/build_experiment_manifests.py` to build matched E0–E4 training inputs. Test is real-only primary.
