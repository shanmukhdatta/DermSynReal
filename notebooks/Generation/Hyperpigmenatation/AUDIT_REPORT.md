# Hyperpigmentation Kaggle Notebook Audit

## Result

**Eight offline checks passed after the second audit.** Both notebook files are valid Jupyter JSON and each Python code cell compiles. The inference data-saving cell executed with a fake CPU diffusion pipeline and produced 10 PNG images plus 10 metadata JSONL rows for two prompts (five seeds per prompt). Training preprocessing/splitting cells executed against actual temporary PIL images and verified that a duplicate spanning training and held-out subsets was excluded from training. Tests also exercised 150/151 token boundaries, expanded textual-inversion placeholders and mocked prompt embedding shape padding.

**This does not mean error-free Kaggle GPU execution has been established.** No Hugging Face model download, actual `compel`, `diffusers`, `transformers`, GPU diffusion inference, or LoRA training was run in this offline container. Those packages and pretrained weights are not available locally; version compatibility was reviewed against the listed upstream APIs and documentation.

## Main implementation checks

| Feature | Check | Status |
|---|---|---|
| Jupyter notebook structure | `nbformat.validate` | Passed |
| Python syntax | `compile` every code cell | Passed |
| Hugging Face base + adapter identifiers | Official model cards checked externally | Verified names/type, not model executed |
| Existing adapters are textual inversion, not LoRA | Official model cards | Verified |
| TI special tokens | Detect registered tokenizer additions at runtime | Implemented, Kaggle result unverified |
| Multi-vector TI + Compel | `DiffusersTextualInversionManager` and `maybe_convert_prompt` | API verified, runtime unverified |
| Two-CLIP-chunk target | 75 content tokens × 2; 150 max | Passed mock acceptance/rejection |
| Positive/negative embedding padding | Equal shape | Passed simulated tensor test |
| Five PNGs per prompt + metadata | Execute actual inference code cell with fake pipeline | Passed |
| Real image preparation | Decode, pHash-group, select training split, metadata.csv paths | Passed with synthetic CPU image fixtures |
| Facial framing and transparency | Square aspect-ratio padding, alpha composition on neutral background | Passed with rectangular RGBA fixture |
| Custom LoRA provenance | Save training backbone/revision; verify on inference | Passed static guards; real LoRA loading untested |
| LoRA trainer script | Pinned Diffusers v0.35.1 raw URL and CLI switches | Matched reviewed official source, script not executed |
| FP16 and 8-bit optimizer setup | Static command inspection | Passed static check; CUDA not tested |
| Kaggle CUDA full pipeline | Need model download + T4 | Not tested |

## Model warnings

1. Both `hcarrion/hyperpigmentation` and `hcarrion/solar_lentigo` publish **textual-inversion** weights for SD2.1; neither constitutes an existing generalized facial LoRA.
2. The former is not a melasma/PIH-specific proven model, and the latter is explicitly solar-lentigo-specific. Output morphology must be evaluated.
3. A custom trained SD2.1 LoRA can be loaded along with the published textual-inversion concept **during inference**. Training notebook trains only the custom LoRA on SD2.1; it does not initialize training from existing TI weights.
4. The SD2.1 base uses a community mirror; verify origin, exact commit, license and weight hashes before publication.
5. Real training captions are checked against 75 CLIP content tokens. Only inference prompts support 150 through Compel.
6. pHash groups are provisional and do not prove distinct patients. Use verified patient/group splits where available.

## Kaggle smoke test

1. Enable **GPU T4** and **Internet ON**; run Notebook 01 with `LORA_SOURCE=''`, one short prompt and `adapter_mode='broad'`. Confirm first 5 images and ZIP.
2. Run Notebook 02 with a real **training-only** dataset and `STEPS=20`. Verify output `pytorch_lora_weights.safetensors`.
3. Attach that LoRA and test Notebook 01 again with `adapter_mode='none'` (LoRA only) and `adapter_mode='broad'` (combined). Review outputs and failures, do not assume quality from successful execution.

## Verification sources

- https://huggingface.co/hcarrion/hyperpigmentation
- https://huggingface.co/hcarrion/solar_lentigo
- https://huggingface.co/sd2-community/stable-diffusion-2-1-base
- https://pypi.org/project/compel/2.3.1/
- https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/text_to_image/train_text_to_image_lora.py


## Second audit — code changes
- Added alpha-safe, aspect-preserving square preprocessing to avoid black backgrounds and unwanted face crops.
- Added a `training_provenance.json` file to the LoRA export ZIP and a base ID/revision check when loading it for inference.
- Confirmed from published model cards: `hcarrion/hyperpigmentation` and `hcarrion/solar_lentigo` are textual-inversion adapters, **not** pretrained facial LoRAs. The LoRA trainer **does not train with these adapters loaded**; inference can combine them later.
- Diffusers training CLI and output `pytorch_lora_weights.safetensors` checked against the pinned v0.35.1 official source.
- Eight local tests passed, including 5 PNGs/prompt via mocked CPU diffusion; real HF/Compel imports, packages, GPUs and clinically meaningful results remain unverified.
