# DermaLens AI — ResNet-50 Kaggle training notebook

**File:** `DermaLens_ResNet50_Kaggle_Training.ipynb` (fully self-contained; no `pip install` cell, no external Python module needed).

## 1. Purpose

Fine-tune a 6-class ImageNet-V2-pretrained ResNet-50 using already frozen train/validation/test folders. Do **not** split images again inside the training notebook.

Six class folder names (case-sensitive):

`Acne`, `Fungal_Infection`, `Vitiligo`, `Hyperpigmentation`, `Skin_Cancer`, `Normal`.

The notebook supports 3 runs:

| `RUN_MODES` entry | Train | Model selection | Evaluations after training |
|---|---|---|---|
| `real` | real/train | real/val | real/test; synthetic/test if attached |
| `synthetic` | synthetic/train | synthetic/val | real/test and synthetic/test if attached |
| `combined` | real/train + synthetic/train | real/val | real/test and synthetic/test |

To avoid excessive training time, **default `RUN_MODES=["real"]`**. Once it works, attach both sources and choose `RUN_MODES=["real", "synthetic", "combined"]`. Each run constructs a *new* pretrained model, not one continued from the preceding experiment.

## 2. Kaggle steps

1. Upload the `.ipynb` notebook in Kaggle.
2. Set the accelerator to **GPU (T4 is adequate for the pilot)**; one GPU is used. The first run may need Internet to download official ImageNet V2 weights. If Internet is disabled, attach the official `resnet50-11ad3fa6.pth` weights and set `PRETRAINED_WEIGHTS_PATH`.
3. Add the real and/or synthetic split from our prior notebooks as Kaggle Inputs. Each source may be a `.zip` containing `train/`, `val/`, `test/` or an extracted folder with those directories.
4. Edit **configuration cell 1**:

```python
REAL_SOURCE = "/kaggle/input/YOUR-REAL-SPLIT/facial_real_split.zip"
SYN_SOURCE = ""  # Supply this for synthetic-only/combined training or synthetic evaluation.
RUN_MODES = ["real"]
PRETRAINED_WEIGHTS_PATH = ""  # Set to attached official weights file when Internet is disabled.
```

5. Keep `ALLOW_RANDOM_WEIGHTS_FOR_SMOKE_TEST = False` for every research experiment. Do not change it to True because pretrained weights fail to download.
6. Click **Run All**, view the preprocessing preview, inspect `invalid_images.csv`, and download results from Kaggle **Output**.

For combined experiments, input ZIPs can require substantial extraction storage; attach extracted folders when disk space is tight.

## 3. Training and preprocessing

- Preflight inventory checks full image decoding, 64px minimum dimension, extreme near-blank/black/white images, and exact SHA-256/pixel-hash cross-partition leakage. Corrupt/blank training images are excluded and listed in `invalid_images.csv`. Invalid validation/test images **stop** the notebook by default; repair the split, do not silently change the test set.
- Correct EXIF orientation, safe RGB conversion (alpha composited on white, not black), **preserve full face** with a 224×224 center letterbox padded with neutral gray (not a central crop). ImageNet mean/std normalization is applied. This is a deliberate clinical full-face alternative to the official V2 resize + center-crop transform; ensure the preview accurately preserves facial features and lesion positions.
- Training augmentation: mild horizontal reflection and ±7° rotation; no aggressive color distortions, elastic deformation or lesion-removing crops. Val/test uses deterministic preprocessing only.
- Stage 1: freeze pretrained backbone and BatchNorm, train six-class head (`EPOCHS_HEAD=6`). A **separate head-only baseline checkpoint** is saved.
- Stage 2: reload best validation checkpoint, unfreeze ResNet `layer4` + head, use lower differential learning rates (`EPOCHS_PARTIAL=15`). BatchNorm stays frozen.
- Stage 3: optional full fine-tuning (`EPOCHS_FULL=0` by default). It can be enabled after seeing validation evidence.
- Optimizer AdamW; cosine LR decay, gradient clipping, CUDA mixed precision via **modern** `torch.amp` API, early stopping and checkpoint selection by validation macro-F1 (validation loss is tie-breaker).
- Combined training uses balanced **domain × class sampling** on the training split, not on validation or tests. This is an initial 50:50 real-to-synthetic mixture strategy; log different mixtures as separate experiments.
- A saved checkpoint embeds architecture, class mapping, preprocessing, weight origin, training seed, and best validation metrics. Load it later for evaluation **without retraining**.

## 4. Outputs

With `OUTPUT_ROOT="/kaggle/working/dermalens_resnet50"`, results include:

```text
/kaggle/working/
  resnet50_training_results.zip          # optional, contains checkpoints and reports
  dermalens_resnet50/
    preprocessing_preview.png
    invalid_images.csv
    validated_image_inventory.csv
    all_test_metrics.csv
    resnet50_real/
      resnet50_real_best.pth
      resnet50_real_head_only_best.pth
      training_history.csv
      test_metrics.csv
      test_predictions_real.csv
      test_predictions_synthetic.csv   # when synthetic input attached
      classification_report_real.json
      confusion_matrix_real.csv
      metadata.json
    resnet50_synthetic/ ...             # if mode selected
    resnet50_combined/ ...              # if mode selected
```

If all three modes run with `SAVE_HEAD_ONLY_BASELINE=True`, the optional ZIP can be **hundreds of megabytes**. This is normal for six full ResNet50 state-dict checkpoints. To reduce file size, use one mode per Kaggle run or set `PACKAGE_RESULTS_AS_ZIP=False` and save individual `.pth` files.

## 5. Verification and limitations

**Verified in a local CPU sandbox (2026-10-29):** Python 3.13, torch 2.10.0+cpu, torchvision 0.25.0+cpu, Pillow 12.3.0, scikit-learn 1.8.0. Ran the complete code in three shortened training modes against six-class sample data. Confirmed handling of a transparent PNG, blank-black/corrupt training images, image shape (3, 224, 224), six main/head-only checkpoints, checkpoint reload, metrics, and separate real/synthetic test predictions. **Local smoke testing used random initial weights only to exercise code; its accuracy/F1 scores are not real medical classification results.** Official pretrained-weight download, actual Kaggle GPU, and training on your real images have **not** yet been run or verified.

This pipeline cannot prove absence of patient overlap or visual near-duplicates that were missed by the upstream splitter; review the upstream grouping audit. It also cannot detect synthetic generator/LoRA training contamination by a held-out real test image. These must be handled separately. This is a research benchmark, not a clinical diagnosis system.

## 6. Reference for API

Official PyTorch torchvision weights: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

Official PyTorch modern AMP: https://docs.pytorch.org/docs/stable/amp.html
