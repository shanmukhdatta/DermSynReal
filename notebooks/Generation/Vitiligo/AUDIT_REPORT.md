# Audit report — Vitiligo SD2.1 Kaggle package

## Previous package problems found and repaired

1. **Broken Python code:** the inference notebook had an unterminated string literal in the metadata writer. Fixed and compiled every notebook code cell.
2. **Wrong Compel mode for SD2.1:** replaced SDXL-style penultimate non-normalized embeddings and pooled list settings with the documented SD2.1 `LAST_HIDDEN_STATES_NORMALIZED` and `requires_pooled=False`.
3. **Incorrect Hugging Face ImageFolder paths:** `metadata.csv` now uses `images/000000.png` relative paths, pointing to actual images. Dataset loading is explicitly validated inside Kaggle before training.
4. **Provisional split mislabeled:** implemented two-stage group split producing train/val/test, with split manifest; near-duplicate groups cannot cross a partition. Held-out group leakage in externally supplied splits triggers exclusion from training.
5. **Trainer safeguards:** added pinned official script, FP16 in both launcher and trainer, 8-bit Adam dependency, gradient checkpointing, disabled expensive validation generation, 20-step smoke-test default, caption validation, and weight existence checks.
6. **Token logic:** added strict positive/negative checks with the loaded tokenizer, a 150-token maximum, no truncation, two-chunk embedding bounds, and padded embeddings.
7. **Source reproducibility:** records base-model revision and saves generation metadata.

## Tests completed in the local CPU sandbox

Run: `python verify_offline.py`

- PASS: notebook JSON validation and Python compile for all cells
- PASS: synthetic sample image scanning and duplicate/flip grouping
- PASS: provisional group-isolated train/val/test and manifest generation
- PASS: ImageFolder metadata paths exist, images decode, captions nonempty
- PASS (mock tokenizer): 150 tokens accepted and 151 rejected
- PASS (mock Compel): two 77-position conditioning chunks padded to `[1,154,1024]`
- PASS (mock pipeline): five images per prompt across two prompts, 10 PNGs plus 10 valid metadata JSONL lines
- PASS (static): training launcher flags and required FP16/gradient checkpoint options

## Explicitly NOT verified

- Installing pinned packages on the actual Kaggle image
- Importing real Compel 2.3.1 / Diffusers / PEFT in the offline container (not installed, outbound package requests blocked)
- Loading the real model weights or confirming T4 memory consumption
- Running the actual official training script, completing steps, or writing trained LoRA weights
- Medical quality, subtype correctness, or data confidentiality of any synthetic image

**Verdict:** corrected and CPU-mock validated, **not GPU/end-to-end verified**. Only a real Kaggle T4 smoke run can establish that the whole workflow works in that environment.

## Verified external documentation used

- Compel 2.3.1: https://pypi.org/project/compel/2.3.1/
- Compel v2.3.1 source: https://raw.githubusercontent.com/damian0815/compel/v2.3.1/src/compel/compel.py
- Diffusers v0.35.1 LoRA trainer: https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/text_to_image/train_text_to_image_lora.py
- ImageFolder metadata: https://huggingface.co/docs/datasets/v2.20.0/en/image_dataset
- SD2.1 mirror: https://huggingface.co/sd2-community/stable-diffusion-2-1-base
