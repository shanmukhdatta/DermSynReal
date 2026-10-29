# Normal Skin SDXL — Final Code Audit

## Confirmed offline (real code cells executed with CPU mock objects)

- Both notebook files parse and all code cells compile under the local Python interpreter.
- The actual dual-tokenizer prompt-validation cell accepts 150 content tokens and rejects 151; it rejects a prompt that exceeds 150 in the second tokenizer independently.
- The actual inference loop was executed with a fake SDXL pipeline for **four modes**, **five seeds per mode** and **20 saved images** with metadata; seeds match between modes.
- Embedding computation was moved *after* adapter selection, preventing stale text-encoder conditioning across modes.
- The actual training-data selection and preprocessing cells were executed with transparent and rectangular CPU test images; produced RGB nonblack images and valid `images/...` metadata paths.
- A duplicate across the training/held-out split now causes a stop; a separate safe test dataset passes.
- The reference to the official Diffusers v0.35.1 SDXL LoRA script and expected CLI flags were checked statically; the script's upstream location and arguments were compared with its published source.
- Hugging Face Compel v2.3.1 documents non-truncating defaults for CompelForSDXL and pooled positive/negative embeddings.

## Not verified in the sandbox

- Installing pinned dependencies into a fresh Kaggle runtime; this sandbox lacks Diffusers, Compel, GPU and the full weights.
- Loading actual SDXL Base 1.0, `profpeng/skinenhancer`, custom LoRA and VAE together.
- Actual long-prompt embeddings from a real CLIP tokenizer or full GPU generation.
- Any LoRA training step with CUDA, 8-bit Adam or its memory consumption.
- Generated image realism, absence of pathology or identity copying.

**No 100% runtime guarantee.** First perform one-prompt inference and a 20-step LoRA training smoke test on Kaggle. The optional `profpeng/skinenhancer` is acne/blemish-oriented and may hurt normal-skin suitability.
