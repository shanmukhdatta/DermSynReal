# Facial-acne SD2.1 Kaggle notebooks — corrected audit edition (29 Oct 2026)

This package includes two Kaggle Python notebooks, a CPU-only regression test, and `AUDIT_REPORT.md`.

## Notebook 1: `01_SD21_Acne_Inference.ipynb`

Generates **five images per prompt** using the `sd2-community/stable-diffusion-2-1-base` base checkpoint and `hcarrion/acne-cystic` textual-inversion (TI) embedding. Your custom SD2.1 LoRA can be loaded later via `LORA_SOURCE`. Change prompts in the clearly labelled `PROMPTS = [...]` cell; for noncystic prompts set `use_cystic_token: False` and remember that the cystic adapter does not affect those prompts unless its special token is included. Output images, metadata, and downloadable `/kaggle/working/acne_generated.zip` are saved by the notebook.

**Runtime setup:** Start a fresh Kaggle notebook, select **T4 GPU** and **Internet ON**, run the pinned dependency-install cell **before importing Diffusers**, then execute top to bottom. First model downloads can require several GB of disk space and network. Use one GPU, FP16. Set `IMAGES_PER_PROMPT=5` (already default).

## Notebook 2: `02_SD21_Train_Facial_Acne_LoRA.ipynb`

- Attach your Kaggle facial-acne dataset via *Add Input* and change `IMAGE_ROOT`.
- Prefer your **already frozen real-train partition**. If available, set `SPLIT_CSV` or point `IMAGE_ROOT` to its train folder. Otherwise the notebook makes **provisional** pHash-based group splits; those are NOT patient-independent and are not publication-quality without auditing the full multisource dataset. The notebook now removes near-duplicate representatives from the *train* subset.
- General-acne images need no subtype labels for basic LoRA training, but cannot provide verified subtype-controlled guidance. Leave `TRAIN_WITH_CYSTIC_TI=False` with unlabeled acne data.
- The default uses **PyTorch AdamW** (`USE_8BIT_ADAM=False`), avoiding an unnecessary `bitsandbytes` dependency. Optional 8-bit Adam explicitly installs `bitsandbytes` when selected.
- The installation cell includes `tensorboard`, required by the official trainer's `--report_to tensorboard` setting.
- Both the Accelerate launcher **and** the official training script receive `--mixed_precision fp16` to prevent incorrectly leaving the LoRA trainable weights in FP16. The script receives explicit image/caption column names.
- The trainer is the **official Diffusers v0.35.1 `train_text_to_image_lora.py`** fetched from the fixed GitHub release tag. This is intentional reproducibility pinning, not the latest Diffusers version. Any download/script shape mismatch stops with an explicit error instead of silently patching unfamiliar code.
- The produced LoRA is stored at `/kaggle/working/acne_lora_output/pytorch_lora_weights.safetensors`, plus an export ZIP. Attach its output directory/file to Notebook 1, set `LORA_SOURCE`, and rerun inference.

## Scientific experiment plan

1. **A:** SD2.1 + published acne textual inversion, with the learned token only for cystic prompts.
2. **B:** SD2.1 + custom general facial-acne LoRA (without using the cystic token).
3. **C:** SD2.1 + custom LoRA + published cystic textual inversion (on cystic prompts).

Compare A/B/C with matched seeds and morphology prompts. Loading both trained components together is supported but improvement is not guaranteed. Human/clinical review and similarity screening are required before treating generated images as valid research training samples.

## What was actually tested

- Notebook JSON/schema validation and compilation of every code cell.
- CPU synthetic-image preprocessing: image reading, duplicate grouping, provisional group splits, imagefolder/`metadata.jsonl` preparation.
- Static checks: prompt batches, explicit trainer FP16 argument, correct dataset columns, conditional optimizer dependencies, TI mode safeguards.
- Official source documentation confirmed the `--mixed_precision`, `--image_column`, `--caption_column`, `--report_to`, `--use_8bit_adam` and LoRA saving arguments.

**Not tested:** pip installation in the actual 2026 Kaggle image; GPU memory availability; live model downloads; actual trained LoRA integration; CUDA image synthesis; quality of generated images. Thus no scientifically honest guarantee of zero Kaggle runtime errors is possible.

## External resources

- [cgDDI acne-cystic textual inversion](https://huggingface.co/hcarrion/acne-cystic)
- [Stable Diffusion 2.1 community model mirror](https://huggingface.co/sd2-community/stable-diffusion-2-1-base)
- [Official pinned Diffusers trainer](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/text_to_image/train_text_to_image_lora.py)
- [Textual-inversion loader documentation v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/api/loaders/textual_inversion)

> **Licensing/provenance:** The SD2.1 model is in an unofficial community mirror. The TI resource is cystic-acne-specific. Before publication verify the model's identity and licenses. For healthcare research, preserve all dataset consent and access constraints.

## New inference prompt strategy — Compel 150-content-token mode (29 Oct 2026)

**Notebook 1 now uses `compel==2.3.1`** to match the pinned `transformers==4.56.2` stack. Do not use Compel 2.4+/2.5+ with these pins without a separate compatibility test; those versions target Transformers 5.x.

- Edit the `PROMPTS = [...]` list. Each item still generates **five images**; `use_cystic_token` appends the TI token only when selected.
- Default **maximum content-token budget: 150**, applied separately to positive and negative prompts. SD2.1 tokenizer has a 77-position context with 2 special positions, giving 75 content tokens/chunk; 150 corresponds to **at most two chunks** (154 conditioning positions). Shorter prompts may use a single chunk.
- Prompt validation is performed **after the real SD2.1 tokenizer and textual inversion load**, with `add_special_tokens=False`, no truncation and a second check after multi-vector TI expansion.
- `Compel(..., truncate_long_prompts=False, split_long_text_mode=SplitLongTextMode.WORDS, textual_inversion_manager=DiffusersTextualInversionManager(pipe))` encodes both prompts. The positive and negative tensors are padded to identical shapes and passed to `StableDiffusionPipeline(prompt_embeds=..., negative_prompt_embeds=...)`.
- Conditioning is precomputed once per prompt, reused across five seeds, and recorded in JSONL metadata (`content_tokens`, `negative_content_tokens`, `conditioning_chunks`, `compel_version`).
- Overlong text or a third embedding chunk stops generation explicitly; **nothing is silently shortened**.

**Offline check:** `python test_compel_offline.py && python offline_validation.py`. The Compel-specific test uses a CPU toy tokenizer and mock encoder to verify the notebook's logic and error handling; it **does not import Compel or perform actual diffusion**. The Kaggle GPU and live checkpoint load remain necessary for true end-to-end confirmation.

**Version sources:** https://pypi.org/project/compel/2.3.1/ ; https://github.com/damian0815/compel/blob/v2.3.1/src/compel/compel.py ; https://github.com/damian0815/compel/blob/v2.3.1/src/compel/diffusers_textual_inversion_manager.py
