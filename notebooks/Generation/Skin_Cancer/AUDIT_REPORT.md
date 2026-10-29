# Offline audit — facial skin cancer SD2.1 Kaggle notebooks

## Scope and published checkpoints

Compared with the user's deep-research reports and the cgDDI paper repository (MICCAI 2026): `hcarrion/melanoma`, `hcarrion/basal_cell_carcinoma`, `hcarrion/squamous_cell_carcinoma`, and `hcarrion/actinic_keratosis` are published **textual-inversion disease-concept adaptations** of the SD2.1 line. They are not clinical full-face image quality guarantees or pre-trained facial skin-cancer LoRAs. AK is **premalignant** and has a separate category. The notebook trains a **new** custom UNet LoRA on the user's real data and only optionally combines it with the public TI at inference.

## Verified locally (CPU-safe)

- Both notebooks conform to notebook JSON schema, and every Python code cell compiled.
- Training command includes the options documented by pinned official Diffusers v0.35.1 (`train_text_to_image_lora.py`); **static check only**, not actual execution.
- Simulated SD2.1 tokenizer counted 150 content tokens and rejected 151; also checked expansion of multi-vector TI token.
- Simulated Compel uses `LAST_HIDDEN_STATES_NORMALIZED`, `truncate_long_prompts=False`, pads positive/negative to equal lengths (up to two 77-position chunks), and passes embedding tensors to a mock image pipeline.
- Rejects near-black generated PNGs instead of silently accepting them.
- Simulated all 4 model modes (`base`, `ti`, `lora`, `combined`) generated five PNGs per mode and recorded correct mode-specific metadata, matching seeds.
- Actual preprocessing code exercised on toy PNGs: held-out duplicate group was excluded, output square images had correct RGB mode and no black alpha padding, and ImageFolder `metadata.csv` image paths resolved.
- Shared-LoRA training refuses the setup without a verified subtype-label CSV; optional LoRA checkpoint provenance is checked against the requested target and SD2.1 base revision.
- ZIP contents and SHA256 digests are checked after packaging.

## Not verified; must be tested on Kaggle

- Installing pinned external packages in the current Kaggle environment; this offline container lacks internet and model libraries.
- Download/access and exact special tokens of all four cgDDI repositories; inference discovers tokens at runtime and raises if ambiguous.
- Real SD2.1 + Compel + TI embedding forward pass and generated image quality on a CUDA GPU.
- Full LoRA training, 8-bit optimizer initialization, Hugging Face ImageFolder loading and checkpoint production.
- Clinical morphological correctness, memorization risk, fairness, or synthetic-to-real classifier improvement.

## Preconditions for a valid experiment

- Real dataset split/grouping is fixed before LoRA training. `train/` or `SPLIT_CSV` must be provided; `AUTO_SPLIT=False` by default.
- Single-target LoRAs must be trained on confirmed disease images only. Shared LoRAs require per-image verified subtype labels and ideally reliable morphology captions.
- Use `MODES=['base','ti']` until a LoRA is genuinely trained. Then use all four modes with consistent seeds and a documented base-model revision.
- Do not treat dermoscopy, isolated lesion images, and full-face clinical images as interchangeable domains. Limit clinical validity claims without expert review.
