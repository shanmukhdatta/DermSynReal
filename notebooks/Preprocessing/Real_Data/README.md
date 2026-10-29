# Real dataset v2 — approved dedup and frozen split

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


**Start here:** `00_Facial_Real_V2_Kaggle_Start_Here.ipynb`. It maps these exact original folders without renaming the Kaggle upload:

| Canonical class | Raw Kaggle folder |
|---|---|
| Acne | `Acne` |
| Fungal_Infection | `Fungal_infection` |
| Hyperpigmentation | `Hyper_Pigmentation` |
| Normal | `Normal_Skin` |
| Skin_Cancer | `Skin_cancer` |
| Vitiligo | `Vitiligo` |

Step 1 only inventories exact SHA duplicates and flags filenames. Download `review_draft.csv`, record original diagnosis/provenance, actual reviewer decision and reviewer ID for **every** image. No automatic medical labels. Step 2 requires a passing review audit, then stages approved-only links to original images. Step 3 calls `pipeline.py` to discover near-duplicate and cross-label groups and allocate approximately 70/15/15 by protected group; check `review_gallery.html`, `candidate_pairs.csv`, `review_candidates_cross_split.csv`, `quarantined_conflicts.csv`. Step 4 produces stable, reviewer-joined `real_all_manifest.csv`, `real_{train,val,test}_manifest.csv` and a split ZIP with original image bytes.

**Blocked until labels reviewed:** previous source reports flagged potential SJS/TEN in Skin Cancer and dermatitis/rosacea in Normal. The new Kaggle upload's folder names and screenshot cannot establish those cases were corrected. Only a reviewed manifest and image inspection can. No patient ID: inferred visual grouping does not prove patient independence.
