# DermaLens — Facial Hyperpigmentation Morphology Prompts (300)

All 300 **complete positive and negative prompts** are stored directly in CSV, JSON and TXT. **No prompt reconstruction or prompt assembly is necessary** in the inference notebook.

## What is included
- 100 *melasma-like* appearance prompts (centrofacial, malar and mandibular distributions).
- 100 *post-inflammatory hyperpigmentation-like* prompts, including residual macules associated with previously resolved acne-like, dermatitis-like or irritant-like areas. Clinical history is **not** confirmed by an image.
- 100 *solar lentigo-like* benign-appearing pigmented macule prompts, mostly older adults and sun-exposed face sites. A generated spot **must not be labeled clinically confirmed benign**.
- 60 distinct morphological archetypes, with five variation focuses each: **V1 age, V2 extent, V3 skin tone, V4 facial location, V5 combined**.
- 150 male and 150 female descriptors; 50 prompts each of six intended skin-tone reference categories across the full set. These are **synthetic sampling targets**, not population incidence or clinical prevalence.
- 70/15/15 prompt-group split = 210 train / 45 val / 45 test. Keep all five seeds and all group variants in the same split.

## Main file for GitHub/Kaggle
`04_Hyperpigmentation/json/hyperpigmentation_all_300.json`

Example:

```python
import requests
url = 'https://raw.githubusercontent.com/OWNER/REPO/main/04_Hyperpigmentation/json/hyperpigmentation_all_300.json'
resp = requests.get(url, timeout=60)
resp.raise_for_status()
rows = resp.json()
for item in rows:
    prompt = item['positive_prompt']
    negative = item['negative_prompt']
    seed = item['seed_base']
    split = item['split']
    # Use Compel to encode complete prompt as embeddings; do not rebuild from metadata.
```

The included `load_hyperpigmentation_prompts.py` handles local files and raw GitHub JSON URLs with schema checks.

## Compel token limits
Designed for **at most 150 SD 2.1 content tokens**; this is NOT guaranteed by word counts. `sd21_content_tokens` is null pending real tokenizer validation. Run `validate_hyperpigmentation_tokens.py` on Kaggle with the same tokenizer loaded in your inference pipeline; it refuses prompts over 150 content tokens and writes a token report. Use `truncate_long_prompts=False`, `prompt_embeds` and matching negative conditioning embeddings. For two chunks expect 154 text-conditioning positions; one-chunk prompts produce 77. Do not require every prompt to fill 150 tokens.

## Generation discipline
Start with small batches and inspect morphology versus the text. Generate the planned 5 seeds per prompt in small GPU batches; save results/metadata continuously and resume. **1,500 candidates does not mean 1,500 clinically usable images.** Avoid distributing prompt variants across different classifier splits, even if image seeds are different. Also deduplicate generated samples against all real datasets and evaluate the classifier on independent real-image groups.

## Clinical caution
- Melasma: usually bilaterally distributed flat brownish facial patches with irregular borders; not all facial brown patches are melasma.
- PIH: leftover flat pigment at healed sites; etiology is usually established by medical history, not a photograph.
- Solar lentigo: commonly discrete, largely uniform flat macules in older adults; an unusual pigmented lesion can resemble cancer and needs clinician review.
- These are descriptions designed to *ask* a model for a visual appearance, not clinician-verified diagnoses.

## File layout
- `04_Hyperpigmentation/csv/hyperpigmentation_001_100.csv`, `_101_200.csv`, `_201_300.csv`
- `04_Hyperpigmentation/json/hyperpigmentation_all_300.json`, batch JSONs, JSONL
- `04_Hyperpigmentation/txt/hyperpigmentation_all_300.txt`, three batch TXT files, 300 per-prompt TXT files
- `load_hyperpigmentation_prompts.py`, `validate_hyperpigmentation_tokens.py`, `prompt_manifest.json`, `QA_REPORT.json`, `FIRST_FIVE_PREVIEW.md`

## References
- https://dermnetnz.org/topics/melasma
- https://dermnetnz.org/topics/postinflammatory-hyperpigmentation
- https://dermnetnz.org/topics/solar-lentigo
- https://www.aad.org/public/diseases/acne/derm-treat/scars/symptoms/
