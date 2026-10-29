# Final cross-check — Normal facial skin versus acne workflow

## Report-matched pretrained resource

- Base: `stabilityai/stable-diffusion-xl-base-1.0` (SDXL, **not** SD2.1)
- Optional pretrained adapter: `profpeng/skinenhancer` (SDXL LoRA; **not clinically validated**, card trigger refers to acne/blemishes)
- Independent custom normal-face LoRA: produced by Notebook 02 using approved real normal-skin training images
- The pretrained texture LoRA is **not part of Notebook 02's training procedure**. Notebook 02 trains a new LoRA on SDXL base. Notebook 01 can load both at inference.

## Notebook 01 — confirmed comparison modes

- `base`: SDXL only.
- `texture`: SDXL + pretrained `profpeng/skinenhancer`.
- `custom`: SDXL + user's own trained normal-face LoRA.
- `combined`: SDXL + both LoRAs.

The inference notebook implements five images per prompt per selected mode with matched seeds; CompelForSDXL nontruncating conditioning, both SDXL tokenizers capped at 150 content tokens, positive/negative pooled embeddings and conditioning shape checks; previews, metadata and zip output.

## Notebook 02 — confirmed workflow

- Real training images only, using `train/` or an explicit `SPLIT_CSV` (provisional split requires opt-in).
- RGB preprocessing with aspect ratio preserved; per-image captions and ImageFolder metadata.
- Pinned Diffusers v0.35.1 official SDXL text-to-image LoRA training example.
- A 20-step GPU smoke test by default, FP16, 8-bit Adam, gradient checkpointing; trained weights plus provenance exported.
- It does NOT initialize from or continue training `profpeng/skinenhancer`.

## Offline validation

The package's `verify_final.py` passes Python compilation, mock tokenizer boundaries, four-mode inference with five mocked images per mode, train-only image processing and duplicate/held-out conflict checks. ZIP integrity and included files were verified separately.

## Not verified

There has been **no real Kaggle GPU generation or training** using SDXL and both actual LoRA files in this environment. A successful smoke test on Kaggle remains necessary. Existing adapter quality for normal clinical faces is unproven.
