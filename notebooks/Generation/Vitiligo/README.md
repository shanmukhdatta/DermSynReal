# Vitiligo — SD2.1 generation and optional LoRA

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


- `01_SD21_Vitiligo_Inference.ipynb`: consumes [300 authored prompts](../../../prompts/Vitiligo/03_Vitiligo/json/vitiligo_all_300.json), preserving disjoint train/val/test prompt families.
- `02_SD21_Train_Facial_Vitiligo_LoRA.ipynb`: approved, deduplicated Kaggle v2 **real train** images only, with no provisional re-split.
- The generated appearances may resemble nonsegmental, segmental or focal patterns; do not infer true subtype without clinical review. Keep image, checkpoint and source/hash evidence.
