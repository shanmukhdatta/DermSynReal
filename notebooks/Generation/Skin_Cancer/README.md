# Skin cancer — SD2.1 + cgDDI TI + custom facial LoRA (Kaggle)

## Contents
1. `01_SD21_Skin_Cancer_cgDDI_Compel_Inference.ipynb` — 4-mode comparison using melanoma / BCC / SCC / optional AK TI concepts, Compel (150 content tokens, 2 CLIP chunks), 5 images per prompt/mode, image previews, metadata, ZIP export.
2. `02_SD21_Train_Facial_Skin_Cancer_LoRA.ipynb` — train a subtype-specific LoRA or one shared label-conditioned LoRA from real images only, using official Diffusers v0.35.1 training script.
3. `AUDIT_REPORT.md`, `OFFLINE_TEST_RESULTS.txt`, and `SHA256SUMS.txt` — tests and limitations.

## Critical distinction
All public `hcarrion/{melanoma,basal_cell_carcinoma,squamous_cell_carcinoma,actinic_keratosis}` checkpoints are **published textual-inversion disease concepts**, not your custom LoRA. LoRA training in Notebook 02 uses the SD2.1 base alone. At inference they can be combined by loading one compatible TI token with your custom LoRA. This does not guarantee that either method generates clinically accurate full-face lesions.

AK (actinic keratosis) is premalignant and not grouped with invasive cancer labels. Melanoma, BCC and SCC require verified real annotations. Do not train a shared subtype-conditioned LoRA from unlabeled, mixed cancer images.

## Kaggle instructions
- Set T4 GPU and Internet ON; the first cell installs pinned packages but preserves Kaggle CUDA PyTorch. If existing imports conflict, restart kernel.
- **Inference first:** Set `MODES=['base','ti']`, edit `PROMPTS`; default five images per prompt/mode. Check whether each TI adapter produces appropriate full-face photos before assuming improved morphology.
- **Training:** Add your dataset. Set `IMAGE_ROOT` to a folder with **train/** inside, or supply existing `SPLIT_CSV`. Choose `TRAIN_TARGET='bcc'` etc. For a shared model use `TRAIN_TARGET='all_labeled'` and a full `LABELS_CSV` with `filename,subtype`. Put verified captions in `CAPTIONS_CSV` (`filename,caption`). Keep 20 steps for first smoke test, inspect results, then increase.
- Export the `.safetensors` plus provenance file. Attach both as Kaggle input and set `LORA_SOURCES` paths in inference. Add `lora` and `combined` to `MODES`. Use exactly the same `PROMPTS`, seeds and generation settings for comparisons.
- Download the generated `.zip` from `/kaggle/working/skin_cancer_generated.zip`.

## Technical constraints
- Near-black generated images raise a visible error rather than silently entering the dataset.
- 150 actual tokenizer content tokens at inference, two windows of 75 content tokens. Compel 2.3.1 with `truncate_long_prompts=False`, encoded positive/negative prompts and padded conditioning vectors.
- Train captions remain ≤75 content tokens because official LoRA training script does not use Compel chunking.
- Best evidence is downstream performance on held-out real faces and qualified morphology review, not visual attractiveness or prompt adherence alone.
- `sd2-community/stable-diffusion-2-1-base` is a **community mirror**, not official upstream; verify provenance and licenses. Kaggle live execution must still be tested.
