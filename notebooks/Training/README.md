# DermSynReal — real-image classifier training (Kaggle)

**Current notebook set: six SMOKE / FULL progressive fine-tuning classifiers, six facial categories.**
These notebooks replace the older, architecture-mismatched training examples. They are research prototypes, not clinical diagnostic systems.

## Data: use the finalized split dataset, not the raw collection

- [Facial Dermatology Image Dataset](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-dermatology-image-dataset/data)
- Kaggle may mount it at `/kaggle/input/facial-dermatology-image-dataset` or `/kaggle/input/datasets/shanmukhadattaboda/facial-dermatology-image-dataset`. ResNet-50 and ConvNeXt V2 now detect both; the other notebooks also search the nested layout.
- A nested folder containing exactly `train/`, `validation/` and `test/` is supported.
- All splits must contain these **exact** folder names:
  `Acne`, `Fungal_infection`, `Hyper_Pigmentation`, `Normal_Skin`, `Skin_cancer`, `Vitiligo`.
- **Do not attach** the older unsplit `facial-skin-conditions-real-image-collection` as a substitute.
- Dataset labels, provenance, licensing and leakage audits require independent human verification. A passed loader/smoke test cannot establish medical correctness.

## Current notebooks

| Notebook | Backbone / initialization | Staged fine-tuning |
|---|---|---|
| [01_ResNet50_SMOKE_FULL.ipynb](01_ResNet50_SMOKE_FULL.ipynb) | torchvision ResNet-50, ImageNet V2 | Head → layer4 → layers3–4 |
| [02_ConvNeXtV2_Tiny_SMOKE_FULL.ipynb](02_ConvNeXtV2_Tiny_SMOKE_FULL.ipynb) | timm ConvNeXt **V2** Tiny, ImageNet | Head → final stage → final two stages |
| [03_DINOv2_ViTB14_SMOKE_FULL.ipynb](03_DINOv2_ViTB14_SMOKE_FULL.ipynb) | timm DINOv2 ViT-**B/14** | Head → last 2 transformer blocks → last 4 |
| [04_VMamba_Tiny_SMOKE_FULL.ipynb](04_VMamba_Tiny_SMOKE_FULL.ipynb) | official VMamba Tiny state-space model | Head → final state-space stage → final two |
| [05_PanDerm_Base_SMOKE_FULL.ipynb](05_PanDerm_Base_SMOKE_FULL.ipynb) | **genuine** official PanDerm Base weights | Head → final 2 transformer blocks → final 4 |
| [06_EfficientNetB0_SMOKE_FULL.ipynb](06_EfficientNetB0_SMOKE_FULL.ipynb) | torchvision **EfficientNet-B0** (ImageNet V1) | Head → final MBConv stage → final two MBConv stages; lightweight efficiency baseline |

**Do not relabel** previous EfficientNet-B3, ConvNeXt Tiny (non-V2), or DINOv2 ViT-S/14 checkpoints as the architectures above. The earlier PanDerm linear probe is also a different experiment.

## Run one notebook on Kaggle

1. Open the model notebook in Kaggle, attach the finalized dataset above, enable a GPU (T4) and Internet if official weights are not cached.
2. Keep `RUN_MODE = "SMOKE"`; use **Save Version → Save & Run All**. Smoke uses a *small balanced sample* and one epoch per enabled stage, and validates the load/train/evaluate/export pipeline.
3. Check for `SMOKE PASSED` and `run_status.json` recording `smoke_passed`. Inspect the **actual logs** for official backbone/checkpoint success, especially VMamba's CUDA selective-scan and PanDerm's official weights.
4. Change **only** `RUN_MODE = "FULL"`, save a **new** Kaggle version and Run All. FULL uses the complete fixed splits and the configured progressive training schedule. Optional full-backbone stage is **off** by default.
5. Confirm final `run_status.json` records `full_completed`; download/retain the model-specific ZIP and Kaggle output version. A smoke pass does **not** guarantee a complete full run.

**Kaggle caution:** Do not reuse smoke checkpoints as research results. SMOKE and FULL write to different output directories. No FULL run is certified by this Git commit.

## Shared methods

- **Real-only training:** ImageNet or officially released pretrained weights; six-class classification head; sequential block-wise unfreezing; discriminative learning rates.
- Mild **train-only** augmentation: preserve aspect ratio, conservative horizontal flips and small rotations; do not augment real validation/test images. Strong color distortions are avoided.
- Model selection: **validation macro-F1** (validation-loss tie-break where implemented), early stopping and best checkpoint reload.
- Final evaluation: one held-out real test after selecting the configuration with validation data. Do **not** use test metrics to tune hyperparameters.
- Log model name, true pretrained checkpoint, library versions, random seed, class order, dataset/split identity, preprocessing, augmentation, and measured results. For publication-quality comparisons, add multiple seeds and confidence intervals.
- **SMOKE and FULL both evaluate a test subset/full test respectively** by notebook design; the smoke test is a *software validation*, not a valid model-comparison result. For strictly one-time test inference during experimental selection, avoid examining test metrics until the protocol is frozen.

## Expected Kaggle output / downstream evaluation

Outputs are written under `/kaggle/working/<model_specific_name>_SMOKE/` or `_FULL/`, with a downloadable artifact ZIP. Look for:

```text
checkpoints/best_model.pt
checkpoints/                 # stage checkpoints (model-specific)
reports/history.csv
reports/final_metrics.json
reports/                     # predictions / classification reports (model-specific)
plots/
class_to_idx.json
preprocessing.json
config.json
training.log
run_status.json
inference_example.py         # available for most models; check the artifact manifest
```

**Important:** Each checkpoint must be loaded by its **own architecture** and exact preprocessing. The common folder names do **not** imply the serialized state dictionaries are interchangeable. PanDerm uses its own official architecture/source and may not export exactly the same per-image prediction filenames as the other models.

For later real-vs-synthetic evaluation, reuse a real-trained `best_model.pt` *without retraining* on independently held-out real and synthetic images. Synthetic-only and real+synthetic conditions instead require **new training runs**. Maintain the same class mapping, frozen test split, and prompt/identity separation. The pre-existing notebooks in `../Evaluation/` were written for earlier checkpoint formats; their checkpoint loaders **must be adapted and smoke-tested** before using these artifacts. Do not assume drop-in compatibility.

## Model-specific dependency risks

- **DINOv2:** This version includes the corrected `timm.list_models(..., pretrained=True)` check and does not intentionally throw a restart error after detecting a missing registration.
- **VMamba:** Its official implementation depends on a selective-scan CUDA extension and may require compatible compilation tools. Do not silently substitute another model if the kernel fails.
- **EfficientNet-B0:** Uses `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`. Local workflow tests used random initialization; genuine weights and Kaggle T4 must still be tested.
- **PanDerm:** Requires an authentic released PanDerm Base checkpoint. Automatic Google Drive download may be restricted; attach the original file as a second Kaggle input if needed.
- **ResNet-50 / ConvNeXt V2:** Dataset-path detection supports the flat and nested Kaggle mounts; attach the correct split dataset first.

## Provenance and status

These files are the current **real-only** Kaggle training notebooks. Earlier notebook implementations were removed from this folder but remain recoverable through Git history. The repository commit records code, **not measured model accuracy**. The sixth EfficientNet-B0 notebook uses the official torchvision architecture without requiring `timm`; its SMOKE and reduced FULL paths were locally tested with that architecture and synthetic fixture images, not pretrained Kaggle weights. Local syntax and reduced/substitute-model workflow tests were performed during notebook development; authentic pretrained-weight loading and complete FULL runs must be documented with actual Kaggle logs and artifact versions.

Keep raw clinical images, private per-image metadata, API keys, model weights, and large Kaggle artifacts **out of Git**. Share reviewed aggregate outcomes and versioned run references only.
