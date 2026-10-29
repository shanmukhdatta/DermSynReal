# Normal facial skin — SDXL generation and LoRA

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


- `01_SDXL_Normal_Skin_Compel_Inference.ipynb`: [300 authored prompts](../../../prompts/Normal_Skin/06_Normal_Skin/json/normal_skin_all_300.json) for balanced-, oily- and mildly dry-looking nonlesional facial texture. These are appearances, not formal diagnoses. SDXL differs from the SD2.1 disease generators; account for generator-style shortcuts in evaluation.
- `02_SDXL_Train_Normal_Facial_Skin_LoRA.ipynb`: only approved frozen **real Normal train** images, never the raw Normal_Skin folder, and uses SDXL-compatible weights.
- Save generated metadata, image hashes, human review CSV and log; optional LoRA weight and console log. Check photographs for dermatologic disorders before approval; a clean filename does not establish healthy skin.
