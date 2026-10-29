# Facial Vitiligo — SD2.1 Kaggle Notebooks (Verified Offline)

## Contents

- `01_SD21_Vitiligo_Inference.ipynb`: 5 generated images per prompt; uses Stable Diffusion 2.1 plus an optional **custom facial-vitiligo LoRA** and Compel long-prompt conditioning.
- `02_SD21_Train_Facial_Vitiligo_LoRA.ipynb`: train your own SD2.1 LoRA using real vitiligo face photos and captions; follows the **version-pinned official Diffusers v0.35.1** training script.
- `verify_offline.py`: reproducible CPU-only notebook syntax, dataset, and mock-generation checks.
- `AUDIT_REPORT.md`: identified issues, fixes, and what remains untested.

**No pretrained vitiligo LoRA is included.** No publicly verified, clinically validated full-face vitiligo SD2.1 adapter was established in the source reports. This package makes a LoRA using your actual training data; it does not include already trained weights.

## Start on Kaggle

1. Upload `02_SD21_Train_Facial_Vitiligo_LoRA.ipynb` to Kaggle. Set **GPU T4** and enable **Internet**.
2. Add your vitiligo dataset via **Add Input**. Change `IMAGE_ROOT` in configuration to the actual `/kaggle/input/...` directory. Best: supply your frozen `train/` folder or `SPLIT_CSV`; the automated group split is provisional only.
3. Optionally provide `CAPTIONS_CSV` with `filename,caption` columns for accurate visible morphology; default captions use only the general vitiligo label. No automatic medical subtype inference is performed.
4. Run with `STEPS=20` initially. Confirm the trainer produces `/kaggle/working/vitiligo_lora_output/pytorch_lora_weights.safetensors`. Only then increase steps, e.g., to 1000, and evaluate the results.
5. Save the custom `.safetensors` file as a Kaggle dataset or otherwise make it accessible in the **inference notebook**.
6. Open `01_SD21_Vitiligo_Inference.ipynb`, enable GPU T4 and Internet. Set `LORA_SOURCE` to the LoRA file or directory.
7. Edit the `PROMPTS` list in **Section 3**. There will be **five images for each prompt**. The output ZIP is `/kaggle/working/vitiligo_generated.zip`.

## Prompt strategy

- `Compel==2.3.1`, `truncate_long_prompts=False`
- SD2.1 text tokenizer: 77 positions per chunk, 75 content tokens per chunk
- Maximum 150 content tokens, **two chunks** per positive or negative prompt
- The exact tokenizer from the loaded pipeline counts content tokens; prompts over 150 are rejected before generation
- Positive and negative text is encoded with Compel and padded to identical embedding length
- Padded embeddings are passed as `prompt_embeds` and `negative_prompt_embeds` directly to Diffusers
- **Training is not long-prompt trained:** the official LoRA trainer tokenizes captions to a single CLIP window; the 150-token capability applies to inference.

## Compatible dependencies

Notebook installation pins `diffusers==0.35.1`, `transformers==4.56.2`, `accelerate==1.10.1`, `peft==0.17.1`, and Hugging Face Hub 0.34.4; inference also installs `compel==2.3.1`, training additionally installs `datasets==3.6.0`, `imagehash==4.3.2`, `bitsandbytes==0.47.0`. The notebook retains Kaggle's GPU-capable PyTorch instead of replacing it.

The model is loaded from `sd2-community/stable-diffusion-2-1-base`, a **community-maintained mirror**. Check provenance, terms, and SHA/revision before publishing research artifacts.

## Scientific limitations

- General vitiligo captions do **not** provide reliable supervision for segmental versus nonsegmental patterns.
- Face framing and depigmentation fidelity require manual evaluation. Generated images are not clinically validated diagnoses.
- The automatic group split uses flip-aware perceptual hashing and cannot establish patient identity; freeze the real-data splits across all six disease classes before serious experiments.
- LoRA training must not see real validation or test photos.
- Generated images must undergo privacy/near-copy checks before classifier training.

## Offline verification vs Kaggle

The local CPU test (`python verify_offline.py`) validates full notebook syntax, dataset splitting, metadata paths, mocked Compel padding/token limits, and mock image generation including JSONL output. **No real model weights, GPU inference, Kaggle package installation, or GPU LoRA training were executed locally.** Hence an end-to-end zero-error guarantee is not possible yet.
