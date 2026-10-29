# Code audit — 29 October 2026

## Verdict

Offline syntax/schema, CPU image processing and static consistency tests **PASS** after fixes. Kaggle GPU **NOT RUN**, so the notebooks are **offline-validated, not end-to-end-verified**.

## Issues identified and repaired

| Finding | Original risk | Correction |
|---|---|---|
| Missing bitsandbytes with `USE_8BIT_ADAM=True` | `ImportError` in official trainer | Changed default to AdamW; optional 8-bit path checks and installs `bitsandbytes` |
| `--mixed_precision=fp16` only in Accelerate launch | Official v0.35.1 trainer checks its **own** `args.mixed_precision` before upcasting LoRA weights | Added `--mixed_precision fp16` to trainer args as well |
| `--report_to tensorboard` without explicit dependency | Optional tracker import can fail | Explicitly install `tensorboard==2.20.0` |
| Dataset columns inferred positionally | Incorrect image/text columns if schema changes | Pass `--image_column image --caption_column text` |
| Model/runtime mismatch hard to diagnose | Hard-to-debug import failures | Added version checks, GPU inspection and early `torchvision` import check |
| Training duplicate groups retained | Unnecessary repeated samples | Keep one image per identified near-duplicate group in training only; preserve manifest |
| Optional TI training misleading for unlabeled images | General-acne data incorrectly treated as cystic | Keep disabled by default and require explicit cystic annotations |

## Compatibility decisions

- Pinned **Diffusers v0.35.1**, **Transformers 4.56.2**, **Accelerate 1.10.1**, **PEFT 0.17.1**, **Datasets 3.6.0**; uses Kaggle's GPU-enabled PyTorch.
- This older stable pin corresponds to the official tagged trainer script; upgrading only one of the packages could introduce API changes.
- `hcarrion/acne-cystic` is a **textual-inversion resource**, not an all-subtypes LoRA.
- `sd2-community/stable-diffusion-2-1-base` is a third-party SD2.1-compatible mirror; exact upstream equivalence is not established.

## Tests actually executed in CPU sandbox

`python offline_validation.py`:

- Notebook `nbformat` schema and `ast.parse`/`compile` on all code cells.
- 27 fabricated image files (including one copy), 26 provisional duplicate groups, group split and training metadata generation.
- Prompt batch count and configuration.
- Static training configuration checks for FP16, optimizer dependency, explicit dataset columns and TI safety.

## Tests not executed

- Kaggle package installation and base checkpoint download.
- Hugging Face inference with TI or TI+LoRA.
- End-to-end official LoRA GPU training.
- Image morphology/clinical validity and actual runtime/VRAM benchmarks.

The user should run a **4-image inference smoke test** and a **20-step LoRA training smoke test** in Kaggle before scaling to 1,000 images or a large batch.

## Addendum — Compel 150-content-token mode (29 Oct 2026)

Updated **Notebook 01 only**; Notebook 02 remains the previously audited training notebook. Additions: pinned `compel==2.3.1`; actual SD2.1 tokenizer-based content-token validation, with TI-expanded counts; two-CLIP-chunk maximum; `truncate_long_prompts=False`; positive/negative Compel embedding generation; `pad_conditioning_tensors_to_same_length`; direct pipeline conditioning; metadata and pre-generation failsafe checks.

**Verified locally:** every notebook cell compiles and conforms to notebook JSON/schema; the existing image-preprocessing regression passes; Compel wrapper simulation passes 75/76/150/151 token boundaries, simulated multi-vector TI expansion rejection, embedding padding to `[1,154,1024]`, more-than-two-chunk rejection, and code-level check that the pipeline receives `prompt_embeds` and `negative_prompt_embeds`. These were mock tests and were not run against the real `compel` Python package.

**Still unverified:** Compel 2.3.1 + pinned packages install on current Kaggle, actual SD2.1 tokenization, actual Compel text encoding with published TI, downloading weights, LoRA compatibility, CUDA inference, morphology quality. The local container is offline and has no diffusers/compel packages or GPU. Never claim a 100%-error-free GPU run based on the offline checks.
