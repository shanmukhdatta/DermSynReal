# DermSynReal notebooks

Research notebooks grouped by task:

- [Generation](Generation/README.md): disease-specific image synthesis and LoRA experiments.
- [Preprocessing](Preprocessing/README.md): real-image auditing/grouped splits and synthetic prompt-family splitting.
- [Training](Training/README.md): **current** six-class real-image classifier pipeline with model-specific SMOKE/FULL Kaggle notebooks for ResNet-50, ConvNeXt V2 Tiny, DINOv2 ViT-B/14, VMamba Tiny, and PanDerm Base, and **EfficientNet-B0**. All six current classifier notebooks are present.
- [Evaluation](Evaluation/README.md): evaluation and reporting notebooks. **Check checkpoint format compatibility first**: the evaluation scripts predate the new SMOKE/FULL model artifacts and are not automatically interchangeable.

## Fixed real dataset

Use the finalized [Facial Dermatology Image Dataset](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-dermatology-image-dataset/data) containing **train / validation / test** splits. It is *not* the older raw [facial-skin-conditions-real-image-collection](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection).

See the [Training README](Training/README.md) for class names, Kaggle mount paths, exact model variants, instructions, expected outputs, compatibility caveats, and reproducibility requirements. Notebook presence and SMOKE tests do not establish clinically meaningful or verified FULL-run results. Historical training notebooks remain available in Git history.
