# DermaLens — Skin Cancer + AK 300 Morphology Prompts

**Download all 300 complete prompts, not templates.** 75 BCC-appearance, 75 SCC-appearance, 75 melanoma-suspicious appearance, 75 actinic keratosis-like appearances.

AK is **precancerous**, not an invasive skin cancer. No prompt implies histological diagnosis; synthetic images are for research only and cannot be used as real diagnostic ground truth. Skin tones and sexes are intentionally balanced for controlled sampling, not epidemiological prevalence. Some ambiguous appearances (especially melanoma-like and SCC-like) require dermatology review.

## GitHub + Kaggle use
Upload the unzipped folder to your public GitHub repository. Obtain the **Raw** URL for `05_Skin_Cancer/json/skin_cancer_all_300.json` and use:

```python
from load_skin_cancer_prompts import load_prompts
rows = load_prompts("https://raw.githubusercontent.com/USER/REPO/BRANCH/05_Skin_Cancer/json/skin_cancer_all_300.json")
for row in rows:
    positive_prompt = row['positive_prompt']
    negative_prompt = row['negative_prompt']
    seed = int(row['seed_base'])
    split = row['split']
    # Validate raw positive text using your actual SD2.1 tokenizer first.
    # Encode it through Compel (truncate_long_prompts=False) as prompt_embeds.
    # For each row, vary seed for 5 candidates but retain row['split'].
```

The helper `load_skin_cancer_prompts.py` can be loaded as a file from the same GitHub repo, or use requests.get(...).json() with the Raw JSON URL.

### Files
- CSV: `05_Skin_Cancer/csv/skin_cancer_001_100.csv` and two subsequent batches
- JSON: `05_Skin_Cancer/json/skin_cancer_all_300.json` (canonical complete prompts), plus 3 batch files and JSONL
- TXT: 3 batch text files and 300 full individual text files
- Preview `FIRST_FIVE_PREVIEW.md`, QA report `QA_REPORT.json`, `prompt_manifest.json`
- `validate_skin_cancer_tokens.py`: run with the exact inference `pipe.tokenizer` **before generating any images**; fail on overlength

### Prompt length and training evaluation
Our design target is 110–145 SD 2.1 content tokens, **maximum 150** (Compel at most 2 chunks); these prompts have NOT been verified with the official SD 2.1 tokenizer in this offline environment. Word counts are NOT token counts. A tokenizer fail must be fixed before inference. Compel does not guarantee medical accuracy. Generate a small pilot and assess morphology, location, diversity and image artifacts before bulk inference.

The 210 train / 45 validation / 45 test *synthetic prompt-group* split is fixed in the files. It does not replace a real-image holdout, and generated images must not be mixed across splits by prompt ID, seed, or near-duplicate content. Note the V1-V5 tags denote **variation emphasis**, not a strict single-factor intervention.

Sources:
- https://dermnetnz.org/topics/basal-cell-carcinoma
- https://dermnetnz.org/topics/cutaneous-squamous-cell-carcinoma
- https://dermnetnz.org/topics/melanoma
- https://dermnetnz.org/topics/actinic-keratosis
