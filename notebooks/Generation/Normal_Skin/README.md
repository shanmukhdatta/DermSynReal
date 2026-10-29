# Normal Facial Skin — SDXL Kaggle Notebooks (Final Re-Audit)

## Why SDXL instead of SD2.1
The project research report specifically recommends **SDXL Base 1.0** for normal full-face photographs. Acne uses SD2.1 and its acne-cystic **textual-inversion** embedding. For normal faces, there is no analogous clinically validated normal-skin SD2.1 disease adapter. Do **not** load SD2.1 textual inversion into SDXL.

**This package matches the acne experimentation workflow, not its underlying model:** base-only → published optional adapter → custom LoRA → combination.

| Mode | Model combination | Intended use |
|---|---|---|
| `base` | `stabilityai/stable-diffusion-xl-base-1.0` | First baseline; recommended starting point |
| `texture` | SDXL + `profpeng/skinenhancer` | Optional experimental skin-texture LoRA; not medically validated, prompt trigger mentions acne |
| `custom` | SDXL + your trained normal-face LoRA | Main customized generator |
| `combined` | SDXL + texture LoRA + custom LoRA | Experimental test; may not be better than `custom` |

### Notebook 1 — Inference

1. Enable **T4 GPU** and **Internet** in Kaggle settings.
2. Run the install cell in a fresh session. The notebook installs pinned Diffusers/Transformers/Compel; keep Kaggle's GPU-enabled torch. Restart the kernel if the environment already imported conflicting libraries.
3. Edit the **configuration cell**. Set `MODES=['base']` for a first smoke test, or `['base','texture']` for the published optional adapter comparison. When you have trained a custom SDXL LoRA, set `CUSTOM_LORA_SOURCE` to its Kaggle file/folder and try `['base','custom']`, then optionally all four modes.
4. Paste your prompt dictionaries into **`PROMPTS`**, one ID and text per prompt. `IMAGES_PER_PROMPT=5`. The same seeds are used across mode comparisons.
5. Compel 2.3.1's `CompelForSDXL` wrapper is **non-truncating by default**. Both actual SDXL tokenizers are checked for a **150-content-token maximum**; prompts above this limit raise `ValueError`. Two 77-position CLIP chunks are supported per encoder. Positive and negative embeddings, including SDXL pooled embeddings, are checked for matching shape before model inference.
6. Results save to `/kaggle/working/normal_skin_generated/` and `/kaggle/working/normal_skin_images.zip`. Choose a fresh `OUT_DIR` for subsequent runs.

The optional texture adapter is not proof of normal-skin morphology: its Hugging Face model card describes acne/blemishes. Generated images must be visually screened for acne, pigment changes, AI smoothing, facial artifacts, and similarity to any source images.

### Notebook 2 — LoRA training

1. Attach the real normal-face dataset in Kaggle, then edit `IMAGE_ROOT`.
2. Provide an existing `train/` folder or `SPLIT_CSV` (`filename,split`) made from your **global six-class grouped split**. `CAPTIONS_CSV` (`filename,caption`) is optional. If you opt into provisional splitting, it is not patient independent.
3. The notebook checks for near-duplicates across the local training/held-out images (pHash <=5) and stops on conflicts. This is a safety gate; review false-positive pairs before modifying your actual dataset.
4. Confirm preprocessed RGB training images and captions. Preprocessing preserves facial framing with neutral-grey padding and prevents transparent pixels turning black.
5. A **Diffusers v0.35.1** official SDXL LoRA trainer is downloaded and its CLI preflight is run on Kaggle. Default `STEPS=20` is only a GPU smoke test; increase steps after it succeeds, using a fresh output directory. The trainer uses FP16, 8-bit Adam and gradient checkpointing at 512×512, batch size 1/rank 8 by default.
6. Export the `.safetensors` file and the `training_provenance.json` via the notebook ZIP cell. Attach it in Notebook 1 and set `CUSTOM_LORA_SOURCE`.

### Verification status

This package's two notebooks passed **Python compilation and `nbformat` validation**, plus CPU mock tests of token limits, 4 inference modes, five outputs per mode, matching seeds, metadata and preprocessing. A real model/Compel GPU execution and actual LoRA training **have not run in the local sandbox**. SDXL training could exceed Kaggle T4 memory. `verify_final.py` contains the offline tests.

### External model resources

- Official SDXL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- Optional texture LoRA: https://huggingface.co/profpeng/skinenhancer
- Official SDXL LoRA training script: https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora_sdxl.py
- Compel project and API: https://github.com/damian0815/compel
