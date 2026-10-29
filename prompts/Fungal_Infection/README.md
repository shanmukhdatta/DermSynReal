# DermaLens — 300 full fungal facial appearance prompts (v3)

**Direct inference entrypoint:** `02_Fungal/json/fungal_all_300.json` contains 300 FULL `positive_prompt` strings and 300 `negative_prompt` values; these are NOT templates and do not need construction from metadata. It is intended for upload to GitHub and direct download in Kaggle.

## Structure
- `02_Fungal/csv/fungal_001_100.csv`, `_101_200.csv`, `_201_300.csv`: 100 full records each.
- `02_Fungal/json/fungal_all_300.json`: complete 300-row JSON list; 3 matching 100-row JSONs and JSONL.
- `02_Fungal/txt/fungal_all_300.txt`, 3 100-prompt TXT batches and **300 separate individual TXT prompt files**.
- `load_fungal_prompts.py`: local/raw-GitHub JSON loader; does NOT reconstruct prompts. Returns all 300 records by default; use `exclude_uncertain=True` to omit 30 medically uncertain angular-cheilitis records.
- `validate_fungal_prompts_for_kaggle.py`: checks real SD 2.1 text tokenizer and fails rather than silently truncating.
- `QA_REPORT.json`, `prompt_manifest.json`, `FIRST_FIVE_PREVIEW.md`.

## Example Kaggle usage
```python
from load_fungal_prompts import load_fungal_prompts
url = 'https://raw.githubusercontent.com/USER/REPO/main/02_Fungal/json/fungal_all_300.json'
rows = load_fungal_prompts(url)  # All 300 records with clinical caution flags.
reviewed_subset = load_fungal_prompts(url, exclude_uncertain=True)  # 270 without angular-cheilitis candidates.
for item in rows:
    positive = item['positive_prompt']
    negative = item['negative_prompt']
    # Use loaded SD 2.1 (+ LoRA) Compel long-prompt embeddings after tokenizer validation.
```
**No generated images or LoRA training are included.** `images_per_prompt=5` refers to planned candidates and is not an output claim.

## Scientific cautions
- Tinea faciei: annular or oval scaly plaques, often asymmetric, sometimes central clearing; can resemble dermatitis. Source: https://dermnetnz.org/topics/tinea-faciei
- Facial pityriasis versicolor: typically flat, pale/brown/pink macules with fine surface scale; may occur on face but is more typical on the trunk. Source: https://dermnetnz.org/topics/pityriasis-versicolor
- Angular cheilitis: fissuring at mouth corners with many possible etiologies; **cannot label Candida infection using appearance alone**. Thirty prompts deliberately retained as `HOLD_FOR_DIAGNOSTIC_REVIEW`, not automatic confirmed fungal training images. Source: https://dermnetnz.org/topics/angular-cheilitis
- Assigned Fitzpatrick references are sampling labels, not a valid substitute for measuring actual phototype.
- V1–V5 are sampling variations around an anchor, not a fully randomized causal study.
- **Token limit:** SD 2.1 standard 77 positions, typical chunk contains up to 75 content tokens. This release targets 150 content tokens maximum and Compel 2 chunks. Exact token counts have NOT been checked here; run the included validation with `pipe.tokenizer` before image generation.
- Any candidate images need clinical assessment, defect checks, class review, deduplication and grouping; synthetic test images do not replace the held-out real-image clinical test set.
