# Revised fungal adapter comparison — offline audit

Date: 2026-10-29

## Verified source strategy
- The cgDDI repository lists `hcarrion/tinea_pedis` as a disease-conditioned generative resource.
- **Tinea pedis is an infection of feet**, not a validated facial infection image generator. This notebook only uses its textual-inversion weights in an **experimental domain-transfer comparator** for facial tinea, never for pityriasis versicolor.
- Training notebook fine-tunes one custom LoRA from verified real facial subtype training data. The published TI embedding is **not** used to initialize the LoRA and is only combined during downstream inference.

## CPU validations performed
1. Both notebook JSON documents validated and every Python code cell compiled.
2. Exact checkpoint identifier and Diffusers textual-inversion API use checked statically.
3. Added textual inversion token discovery and multi-vector length accounting.
4. Positive and negative Compel conditioning, 150-token boundary and 2-chunk limit checked statically, and boundary tested with a tokenizer mock.
5. Fake CPU pipeline ran 4 requested comparisons for facial tinea and 2 for versicolor, 5 images/mode, 30 total files with metadata and matched seeds.
6. Existing training notebook unchanged; its original fixture-based tests passed (7 tests including image preparation, split isolation, rectangular/transparent handling and training command flags).
7. README and package contents validated.

## NOT verified — important
- `hcarrion/tinea_pedis` Hugging Face repository exact file contents / full Diffusers loading are not independently executable in this offline environment; the cgDDI README lists it, but the HF resource did not open in the current web retrieval.
- No real imports of `diffusers`, `compel`, `transformers`, and no Hugging Face downloads in sandbox (dependencies not installed).
- No Kaggle CUDA execution, real images, GPU LoRA fine-tuning, clinical validity validation, or guaranteed absence of runtime errors.
- No claim that combining the published adapter with your trained LoRA improves facial tinea. It may worsen morphology.

## Runtime launch order
1. Run inference with `base` / `ti_only` first. If public TI load fails, inspect the exact Hub repo files on Kaggle and correct the loader; do not guess model compatibility.
2. Train the tinea-faciei LoRA on **real train only** and save `.safetensors` + provenance.
3. Attach LoRA, set path and enable `lora_only` and `ti_plus_lora` experiments.
4. Evaluate morphology with expert reviewers; do not add out-of-domain synthetic images to real medical training data without validation.
