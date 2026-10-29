# Kaggle notebooks — current repository inventory

Notebook code has been uploaded, but **no GPU run or clinical experiment is verified simply by its presence**.

- `Generation/`: two disease-specific diffusion / optional LoRA notebooks for each of six target classes, with existing README and offline validation notes.
- `Preprocessing/Real_Data/`: real deduplication and group-aware split.
- `Preprocessing/Synthtic_data/`: synthetic prompt-family split (folder spelling preserved for compatibility).
- `Training/`: five explicit backbone notebooks (ResNet-50, EfficientNet-B3, ConvNeXt Tiny, DINOv2 ViT-S/14, PanDerm Base linear probe) plus a separate self-contained ResNet50 real/synthetic/combined training notebook.
- **`Evaluation/`: seven Kaggle-ready notebooks**, from real-label preflight to model inference, generation QA, cross-condition paired reporting, calibration, and LaTeX result export. Start with [Evaluation README](Evaluation/README.md).

Train notebooks 01–05 export locked real-test prediction CSVs. For full class probabilities or a separate synthetic-test set, use the dedicated evaluation inference notebooks. All source image inputs and final run flags are intentionally configured by the researcher.

See [execution guide](../docs/EXECUTION_GUIDE.md) and [evaluation protocol](../docs/EVALUATION_PROTOCOL.md).

**Important:** The present EfficientNet-B3, ConvNeXt Tiny and DINOv2 ViT-S/14 notebooks are not the separately planned EfficientNet-B0, ConvNeXt V2 Tiny and DINOv2 ViT-B/14 variants. PanDerm is a linear probe. Do not misname experiments.
