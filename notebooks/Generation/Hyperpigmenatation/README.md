# Hyperpigmentation — two Kaggle notebooks (SD2.1)

## Files
- **01_SD21_Hyperpigmentation_Compel_Inference.ipynb**: SD2.1 + selectable cgDDI textual inversions + optional custom LoRA; five outputs per prompt, Compel 150 tokens/two chunks, PNG previews, JSONL metadata, ZIP.
- **02_SD21_Train_Facial_Hyperpigmentation_LoRA.ipynb**: official Diffusers v0.35.1 LoRA trainer on your real hyperpigmentation **train split**.
- **AUDIT_REPORT.md**, **OFFLINE_TEST_RESULTS.txt**, **SHA256SUMS.txt**, **verify_offline.py**.

## Model identification
- `hcarrion/hyperpigmentation`: **textual inversion**, broad pigment concept; not a melasma/PIH LoRA.
- `hcarrion/solar_lentigo`: **textual inversion**, solar lentigo concept.
- `sd2-community/stable-diffusion-2-1-base`: community mirror of SD2.1 512-base. Check weights/provenance/licensing before publication.
- Your output from Notebook 02: **custom LoRA**; NOT present as pretrained weights in this ZIP. The two cgDDI textual inversions are **not loaded during LoRA training**. They are optionally combined with the trained LoRA at inference.

## Usage
1. Kaggle > New Notebook > Settings: **GPU T4**, **Internet ON**. Upload the `.ipynb` files or use their contents.
2. **Quick pilot**: In Notebook 01 leave `LORA_SOURCE=''`. Edit `PROMPTS`. Each entry has `adapter_mode`: `none`, `broad`, `solar`. The corresponding textual-inversion special token is detected automatically after loading.
3. To train your own LoRA: attach only permitted real training images, edit `IMAGE_ROOT` in Notebook 02, run **20 steps first**, then increase `STEPS` after successful smoke testing. Use an existing patient/group-wise split when available.
4. Download `hyperpigmentation_lora_weights.zip` from Notebook 02; it contains `pytorch_lora_weights.safetensors` and `training_provenance.json`. Extract both together into an attached Kaggle dataset, and set `LORA_SOURCE` to the extracted LoRA file or directory. The inference notebook checks matching SD2.1 base revision when provenance is present.
5. Rerun Notebook 01 using **identical prompts/seeds**, compare public TI only vs custom LoRA only vs combination. Evaluate full-face quality, morphology, artifacts, near-copies and diversity before scaling.

## Compel rules
- Max **150 content tokens** for both positive and negative prompt, using actual SD2.1 tokenizer.
- Automatically reject over-budget prompts; Compel `truncate_long_prompts=False`.
- Max **two chunks** of 77 positions, 75 content tokens each.
- Compel encodes positive and negative prompts, pads embedding lengths to equality, passes them to the pipeline.
- For textual inversion, count tokens **after multi-vector expansion** and use `DiffusersTextualInversionManager`.
- LoRA **training captions** (in Notebook 02) remain limited to 75 content tokens (one standard CLIP window).

## Scientific precautions
- Prompted labels are *requested* morphology, not real medical diagnoses.
- No verified melasma- or PIH-specific textual inversion weights were found in the three supplied reports.
- Solar lentigo is a specific phenotype; do not equate it with all hyperpigmentation.
- Synthetic photos require a medical morphology review, copy/memorization screening and a frozen real test split.
- Actual Kaggle CUDA training/inference has **not** been executed in this offline environment.

## Version pins
Diffusers 0.35.1, Transformers 4.56.2, Accelerate 1.10.1, PEFT 0.17.1, Compel 2.3.1; do not use native BF16 on T4.


## Improvements from additional independent audit
- Training input photographs are EXIF-corrected and aspect-ratio-padded to 512 × 512, protecting forehead/cheek/chin framing. Transparency is composited onto neutral grey rather than becoming black pixels.
- LoRA training writes `training_provenance.json` with base ID and exact revision. Inference checks it when provided.
- This ZIP is an **offline-checked code package**, not a trained checkpoint or a guarantee of Kaggle GPU execution.
