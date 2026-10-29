# Facial fungal SD2.1 / cgDDI adapter pilot + custom LoRA

Two Kaggle notebooks:

1. `01_SD21_Fungal_cgDDI_TI_Compel_Inference.ipynb`: compares the same morphology prompts across four settings: **base**, **published TI only**, **custom LoRA only**, **published TI + custom LoRA**. Five images per prompt **per selected setting**, with identical matched seeds. Published resource: `hcarrion/tinea_pedis` (cgDDI; an athlete's-foot concept), NOT a validated facial-fungal adapter. Compel 2.3.1 is used with `truncate_long_prompts=False`, a **150 content-token** ceiling, maximum **two 77-position SD2.1 CLIP chunks**, positive and negative prompt embedding padding, and metadata/ZIP output.
2. `02_SD21_Train_Facial_Fungal_LoRA.ipynb`: train your **own SD2.1 LoRA** using a verified `tinea_faciei` OR `pityriasis_versicolor` training partition. Run it separately for each subtype. It does **not** initialize LoRA from the published tinea-pedis textual inversion. The LoRA is combined with the public textual inversion *after training*, during inference (the user's requested approach).

## How to run

- Kaggle notebook settings: Internet ON and T4 GPU. Keep Kaggle's CUDA PyTorch installation.
- Initially open inference notebook and use default `MODES_BY_SUBTYPE`: tinea faciei `['base','ti_only']` and versicolor `['base']`. Set your own `PROMPTS` and run five images per prompt per active mode. The published adapter must be downloadable and must load via Diffusers textual inversion; if it does not, the notebook raises an explicit actionable error rather than faking the adapter.
- Review these test outputs, checking that **full-face morphology** rather than skin patches of an unrelated body site is produced. Do not use unverified images to train the medical classifier.
- Train the LoRA using Notebook 02 on **real training split only**. Choose `SUBTYPE='tinea_faciei'` or `'pityriasis_versicolor'`, set `IMAGE_ROOT` and optionally `SPLIT_CSV`/`CAPTIONS_CSV`. Start with the short 20-step smoke test. Complete a full run only after checking real GPU training behavior.
- Save the `.safetensors` and `training_provenance.json` files and attach them in Kaggle. In inference set `LORA_PATHS['tinea_faciei']` or `LORA_PATHS['pityriasis_versicolor']` to the saved checkpoint. Change `MODES_BY_SUBTYPE` to enable `lora_only` and `ti_plus_lora` for tinea-faciei comparisons. Only `base` and `lora_only` are permitted for versicolor, unless a genuinely matching public adapter is independently verified.
- The output ZIP and metadata file live in `/kaggle/working/`; each folder is nested by prompt ID and inference mode.

## Why this is an exploratory baseline, not a recommended medical generator

`hcarrion/tinea_pedis` is **tinea pedis (foot/athlete's foot)**, not facial tinea. The cgDDI paper lists this resource but the supplied reports did not identify verified T2I adapters for *tinea faciei*, *facial pityriasis versicolor* or *facial candidiasis*. Consequently, the published adapter can be assessed for **out-of-domain transfer only**. It must not be represented as a prevalidated facial generator, and its inclusion is not evidence that the result is medically correct or will improve quality. Do not use it for versicolor images.

The community SD2.1 base mirror `sd2-community/stable-diffusion-2-1-base` is not the original Stability AI repo and should be checked for licensing/provenance before research publication.

## Verification boundaries

The included CPU audit tests notebook compilation, prompt budget including expanded multi-vector textual-inversion tokens, comparison-mode routing, mocked positive/negative embedding compatibility, five-image saving and metadata. The training notebook was previously CPU tested on mock/fixture images, and retained without changing its original training logic. **Actual Hugging Face downloads, Compel imports against model weights, Kaggle CUDA inference, full LoRA training, and clinical image validity have not been verified here.** A zero-runtime-error guarantee is not possible from offline testing alone.

## References

- [cgDDI model list and methodology](https://github.com/hectorcarrion/ControllableGenDDI)
- [Tinea-pedis concept resource](https://huggingface.co/hcarrion/tinea_pedis)
- [Compel 2.3.1 and textual-inversion manager](https://pypi.org/project/compel/2.3.1/)
- [Diffusers LoRA](https://huggingface.co/docs/diffusers/training/lora)
