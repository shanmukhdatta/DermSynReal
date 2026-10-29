# Acne — SD2.1 generation and optional LoRA

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


- `01_SD21_Acne_Inference.ipynb`: uses the 300-prompt [Acne registry](../../../prompts/Acne/01_Acne/json/acne_all_300.json), train-only by default. Uses the published acne textual-inversion concept only where explicitly requested; that TI is **not** a LoRA.
- `02_SD21_Train_Facial_Acne_LoRA.ipynb`: frozen, reviewed real **Acne train** subset only, official Diffusers LoRA procedure; requires approved split and audit, never provisional re-splitting.
- Pilot: first use five train prompts, five images per prompt and inspect saved reports. Increase prompt count and images per prompt only after checkpoint/token/morphology validation.
- Outputs: PNG/metadata, `reports/generation_manifest.csv`, unreviewed pool CSV, `run_log.jsonl`, generation ZIP; optional LoRA `pytorch_lora_weights.safetensors` plus `trainer_console.log`.
- Never use unreviewed synthetic images as classifier ground truth. All disease descriptions and ages are **requested appearances**, not diagnoses.
