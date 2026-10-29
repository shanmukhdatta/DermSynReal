# DermSynReal — six-class classifier training

The **only real-image dataset** is Kaggle v2 `shanmukhadattaboda/facial-skin-conditions-real-image-collection`: `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset`. The raw class folder spellings are preserved as uploaded. **Never train on the raw collection**: first run `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb`, complete real-label review, and freeze the protected 70/15/15 split. Labels and original source licenses still need independent verification. No Kaggle GPU executions are claimed yet.


**Only train from saved Kaggle approved real-split artifacts**:
```python
MANIFEST_CSV = "/kaggle/input/EDIT_APPROVED_REAL_SPLIT/real_all_manifest.csv"
IMAGE_ROOT = "/kaggle/input/EDIT_APPROVED_REAL_IMAGES/facial_real_split_v2"
EXPERIMENT = "real_only"  # E0 initial pilot
RUN_FINAL_TEST = False     # keep frozen until method choices complete
```

The five current notebooks are `01_ResNet50_6Class_Train.ipynb`, `02_EfficientNet_B3_6Class_Train.ipynb`, `03_ConvNeXt_Tiny_6Class_Train.ipynb`, `04_DINOv2_ViTS14_6Class_Train.ipynb`, and `05_PanDerm_Base_6Class_Linear_Probe.ipynb`. Exact checkpoint architectures are ResNet-50, **EfficientNet-B3**, **ConvNeXt Tiny (not V2)**, **DINOv2 ViT-S/14 (not B/14)** and **PanDerm Base frozen-feature linear probe (not full fine-tuning)**. The older `DermaLens_ResNet50_Kaggle_Training.ipynb` is a separate optional trainer, not interchangeable with the new experiment-runner manifests.

Start with E0, real-only and a small GPU smoke test. Choose model stages on **real validation only**. The **four timm** notebooks now include head warmup → partial final-block unfreezing → optional full fine-tuning (default off), with a matched real-train-based number of optimizer updates per epoch across E0–E5. This is offline validated only; the exact architecture parameter prefix and GPU optimizer behavior still require Kaggle validation. PanDerm remains a frozen-feature linear probe, not progressive fine-tuning. The manuscript must describe what was *actually* run.

For E1–E4 use reviewed synthetic data in train **only** and the same frozen real val/test. The `scripts/build_experiment_manifests.py` builder outputs compatible E0–E4 CSVs after reviewed real and synthetic pools are supplied. Do not use synthetic test data to train or choose checkpoints.

Each notebook saves `best_weights.pth` or the PanDerm probe, `history.csv`/`val_tuning.csv`, run config, `artifact_inventory.csv`, `run_log.jsonl`, and (when explicitly unlocked) real-test predictions with stable IDs, six class probabilities and grouped/real source metadata. PanDerm needs genuine pretrained weights and the official encoder source. Kaggle execution and clinical validity are unverified until actually run.
