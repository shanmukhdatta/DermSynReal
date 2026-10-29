# Skin cancer / precancerous appearance generation

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


- `01_SD21_Skin_Cancer_cgDDI_Compel_Inference.ipynb`: [300 morphology prompts](../../../prompts/Skin_Cancer/05_Skin_Cancer/json/skin_cancer_all_300.json), 75 each of BCC-like, SCC-like, melanoma-suspicious and actinic-keratosis-like patterns.
- **AK is precancerous, not invasive skin cancer.** A generated clinical-looking lesion is not a diagnostic sample.
- `02_SD21_Train_Facial_Skin_Cancer_LoRA.ipynb`: subtype-specific adaptation requires approved Kaggle v2 real train images and a reviewer-confirmed `image_id,subtype,reviewer_decision` CSV. `TRAIN_TARGET` must match reviewed subtype; shared `all_labeled` training is locked until validated per-image labels are integrated.
- Previous suspicious SJS/TEN source photos must not be treated as skin cancer. Save generation/LoRA provenance, reports and logs, and obtain medical review before any publication claim.
