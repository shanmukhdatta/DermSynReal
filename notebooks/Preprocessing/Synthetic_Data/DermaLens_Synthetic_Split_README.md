# DermaLens synthetic prompt-family splitter — Kaggle

**Contents:** `DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb`, `synthetic_metadata_example.csv`, this README.

1. Upload the `.ipynb` notebook to Kaggle (Code > New Notebook > File > Import Notebook).
2. Attach your synthetic images and a generator-produced metadata CSV via Add Input.
3. Edit notebook **configuration cell**: `INPUT_MODE = "manifest"`, `MANIFEST_CSV`, `IMAGE_ROOT` (absolute `image_path` is also supported).
4. **Run All**. Download `/kaggle/working/synthetic_facial_split.zip` from Kaggle Notebook Output.
5. If you lack a manifest, `INPUT_MODE = "filenames"` uses each of six `CLASS_DIRS` plus `FILENAME_PROMPT_REGEX`. E.g., `ACNE_P001_seed101.png` becomes prompt ID `ACNE_P001`. Don't use filename mode if filenames do not faithfully encode prompts.

### Manifest columns
Required: `image_path,class_name,prompt_id`. Strongly recommended: `prompt_family_id`. Optional: `seed,model_id`. Relative image paths are resolved under `IMAGE_ROOT` (or manifest parent if root blank). Names must match six configured classes exactly.

A single prompt ID must map to a single prompt family ID. Minor variants/paraphrases of a prompt should share a family ID, even if seeds differ. If family ID is empty, the prompt ID becomes its family ID; this is fine only when prompts are genuinely independent. Save your prompt metadata during generation.

### Dataset split and dedup decisions
- Prompt-family holdout; target percentages 70 train, 15 validation, 15 test, with indivisible groups and 3+ independent groups per class.
- SHA-256 and canonical decoded RGB + EXIF orientation hash. Exact duplicate files/pixels are removed; across-class conflicts quarantined. Families connected by exact copies are merged into one split.
- No pHash/dHash/ResNet; these can incorrectly flag distinct but morphologically similar synthetic images.
- Preserves original image bytes, no crop, resize, augmentation or normalization. Run model-specific preprocessing later.
- Post-split checksum and prompt/exact-hash separation audit, checks ZIP, reports all exclusions. A clean audit does not prove zero resemblance-based or upstream generator leakage.

**Runtime:** Requires standard Kaggle Python 3 and Pillow (normally preinstalled); no GPU, Internet, Torch or pip install needed. Disk usage: materialized split + its ZIP both reside under `/kaggle/working`, so have roughly twice the final dataset size of free space. If ZIP is too large, use Kaggle notebook Output or change available disk/storage accordingly. ZIP uses ZIP64.
