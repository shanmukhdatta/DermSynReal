# DermSynReal — preprocessing

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


**Run in order:**
1. `Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb` — inspect Kaggle v2 originals, collect filename/hash QA, prepare a **manual diagnosis review sheet**, refuse unapproved labels, and run the existing group-aware dedup/split on the approved subset.
2. `Real_Data/DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb` — existing standalone pipeline (SHA, pixel hash, pHash, dHash, ResNet embeddings, optional ORB and Union-Find). The new start-here wrapper connects the exact case-sensitive Kaggle v2 folder names.
3. `Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb` — only after creating and manually reviewing six-class synthetic image pools. Uses authored `prompt_partition` to preserve prompt family isolation by default, with exact hash-based duplicate rejection.

Outputs include inventory and review CSVs, candidate-pair galleries, rejected/quarantined images, frozen real split and synthetic split CSVs, checksums, audits and ZIPs. Final training manifest `real_all_manifest.csv` is output by the start-here notebook and is not available until the reviewed pipeline has actually run.
