# Acne — Stable Diffusion 2.1 generation and LoRA

## Working LoRA trainer

- **Training notebook:** [02_SD21_Train_Facial_Acne_LoRA.ipynb](02_SD21_Train_Facial_Acne_LoRA.ipynb) — Kaggle T4, official Diffusers v0.35.1 LoRA example, class-specific train-only input, logging, checkpoints, and standalone inference adapter export.
- **Inference notebook:** [01_SD21_Acne_Inference.ipynb](01_SD21_Acne_Inference.ipynb) — existing generation experiment. Its loading/configuration must be checked for compatibility with the newly exported LoRA bundle before using the adapter.

## Dataset and environment

Attach the finalized [Facial Dermatology Image Dataset](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-dermatology-image-dataset/data). The trainer discovers `train/Acne` under `/kaggle/input` and **does not train on validation or test images**. Do not replace it with the older raw dataset. Keep the fixed image splits unchanged.

Use an NVIDIA T4 accelerator and enable Kaggle Internet. The notebook pins the Diffusers training script and uses the community-hosted SD2.1 Base repository `sd2-community/stable-diffusion-2-1-base`. The manifest records model provenance; verify its provenance for a final research release. If access requires authentication, provide `HF_TOKEN` through Kaggle Secrets; never commit tokens.

## Running the trainer

1. Start with `RUN_MODE = 'smoke'` (10 optimizer steps).
2. Execute all cells, checking dataset preflight, model access, training exit status, adapter reload, generated sample grid, and inference ZIP.
3. Inspect the saved run logs and sample images.
4. For a distinct 500-step experiment, switch to `RUN_MODE = 'full'` and rerun in a fresh session, keeping outputs separate.
5. Select a trained checkpoint using qualitative and quantitative validation before producing your research synthetic-image pool.

## Verified execution record (2026-10-30)

User-provided Kaggle execution logs for the 10-step smoke test show:

- Final `output/pytorch_lora_weights.safetensors` written.
- `checkpoint-5` and `checkpoint-10` saved.
- Adapter reloaded and a sample generation grid saved.
- Standalone `*_INFERENCE_BUNDLE.zip` created (approximately 12.4 MB).
- Complete smoke-run archive created (approximately 4.8 GB).

These results validate **the smoke path only**; they do not validate 500-step training, clinical image fidelity, or downstream classification performance. The log's `torch_dtype` deprecation, absent CLIP text-encoder LoRA keys, and nbconvert warnings were nonfatal.

## Reusing the adapter

The inference bundle contains `pytorch_lora_weights.safetensors`, `adapter_manifest.json`, and an example loader. Attach the extracted bundle to a separate Kaggle inference notebook; load it with the same base model recorded in the manifest, e.g.:

```python
pipe.load_lora_weights(adapter_dir, weight_name="pytorch_lora_weights.safetensors")
```

Keep optimizer-state checkpoints for *training resumption*, but use the smaller inference bundle for synthetic-image generation. Save Kaggle outputs explicitly. Do not commit private real photographs, secrets, large generated-image pools, or large checkpoint archives to GitHub.

## Research cautions

The current trainer's generic acne caption is a technical baseline. Before final controlled generation experiments, review image-specific lesion captions, model license/provenance, generated morphology, diversity, memorization/near-duplicate checks, and the held-out prompt-family protocol. Generated medical images are not diagnostic ground truth without expert review.
