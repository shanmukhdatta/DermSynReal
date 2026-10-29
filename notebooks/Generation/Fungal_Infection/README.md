# Facial fungal-appearance generation and optional LoRA

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


- `01_SD21_Fungal_cgDDI_TI_Compel_Inference.ipynb`: loads [300 authored prompts](../../../prompts/Fungal_Infection/02_Fungal/json/fungal_all_300.json); controls `tinea_faciei`, `pityriasis_versicolor` and a **clinically unconfirmed candidiasis-like** appearance (base-only until reviewed).
- Published `tinea_pedis` textual inversion concerns a different anatomical site; it is an experimental transfer, **not proof of a facial fungal checkpoint**. Other modes remain base unless a genuine compatible LoRA is supplied.
- `02_SD21_Train_Facial_Fungal_LoRA.ipynb`: requires Kaggle v2 approved train split, passed audit and independent per-image `image_id,subtype,reviewer_decision` CSV. It stages only the reviewed requested subtype; no raw folder fallback.
- Output: generated image ZIP, per-image CSV, unreviewed QA draft, model/source/hash provenance and trainer logs. Human review is essential.
