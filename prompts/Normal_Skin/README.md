# DermaLens — Normal Facial Skin Prompts (300)

This release contains **300 complete, ready-to-use positive and negative prompts** in CSV, JSON, and TXT formats. Never reconstruct prompt text from metadata; load `positive_prompt` and `negative_prompt` verbatim.

## Target phenotype appearances
- **100 balanced-looking, non-lesional facial skin appearances:** visible pores and normal microtexture; no exaggerated beauty retouching.
- **100 oily-looking, non-lesional appearances:** natural localized sebum sheen and pores, without acne lesions.
- **100 mildly dry-looking, non-lesional appearances:** matte microtexture and fine skin lines, without inflammatory scaling, fissures or dermatitis-like plaques.

These three appearance categories are **not medical diagnoses** and should not be treated as hard ground truth from a photograph. Do not conflate the class named Normal with perfectly smooth or cosmetically ideal skin. There is normal variation in pores, shine, texture and tone. Five different focus variations per morphology family (age, surface expression, skin tone, camera/region, combined) are designed for diversity. V1–V5 denote intended axes of variation, not a claim of statistically isolated experiments.

## Main GitHub/Kaggle inference file
`06_Normal_Skin/json/normal_skin_all_300.json`

```python
import requests
url = 'https://raw.githubusercontent.com/OWNER/REPO/main/06_Normal_Skin/json/normal_skin_all_300.json'
r = requests.get(url, timeout=60)
r.raise_for_status()
rows = r.json()
assert len(rows) == 300
for rec in rows:
    positive_prompt = rec['positive_prompt']
    negative_prompt = rec['negative_prompt']
    seed = rec['seed_base']
    split = rec['split']
    # Encode full text with Compel using the same SD2.1 pipeline tokenizer/text encoder.
```

The included `load_normal_skin_prompts.py` validates schema from either local JSON or a public raw GitHub URL; its script entrypoint can print a sample.

## Dataset grouping, splits and output
60 distinct morphology groups × 5 controlled prompt variants. 70/15/15 split at the **prompt-group level** (210 train, 45 validation, 45 test). Keep the entire five-seed image set and every variant of one morphology group in its assigned split. To avoid synthetic leakage in classifier experiments, **do not treat synthetic validation or test as a replacement for independent held-out real patient/image-group evaluation**.

## SD 2.1 tokenizer / Compel
The design target is 110–145 content tokens where useful, with a **hard maximum of 150 CLIP content tokens** (two 75-token chunks), not 150 words. **Exact token lengths have NOT been measured in this offline environment.** Run `validate_normal_skin_tokens.py` on Kaggle after you load the exact SD 2.1 model/tokenizer (or with `--tokenizer` set to your pipeline's model). The validator reports actual counts and rejects any overlength prompt before image generation. Never silently truncate. Use `truncate_long_prompts=False` with Compel and supply positive/negative `prompt_embeds`, padding to the same sequence length. Two chunks have 154 cross-attention positions; one chunk has 77. The adapter weights and tokenizer must be compatible with your base model.

For 300 prompts × 5 seed images, plan for 1,500 **candidate** images. Generate in small batches with continuous save/resume, then visually inspect for realistic non-lesional skin, lack of disease contamination, unwanted prompt features, near-duplicates and demographic balance. Generated-image validation has NOT been done.

## Files
- `06_Normal_Skin/csv/normal_skin_001_100.csv`, `_101_200.csv`, `_201_300.csv`
- `06_Normal_Skin/json/normal_skin_all_300.json`, three JSON batches and JSONL
- `06_Normal_Skin/txt/normal_skin_all_300.txt`, three TXT batches and 300 individual TXT files
- `load_normal_skin_prompts.py`, `validate_normal_skin_tokens.py`, `prompt_manifest.json`, `QA_REPORT.json`, `FIRST_FIVE_PREVIEW.md`, `BATCH_QA.json`

## Medical and reference notes
- https://health.clevelandclinic.org/understanding-skin-types
- https://www.aad.org/public/everyday-care/skin-care-basics/dry/oily-skin
- https://www.aad.org/public/everyday-care/skin-care-basics/care/skin-care-for-men

Fitzpatrick phototype is defined partly by sun response, not a skin-color label. Here `fitzpatrick_reference` is a **planned synthetic-sampling category**, not a clinically measured phototype or a claim of ethnicity.
