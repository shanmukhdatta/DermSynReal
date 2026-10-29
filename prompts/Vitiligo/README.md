# DermaLens — Facial Vitiligo Morphology Prompt Collection (300)

This is an **inference-ready prompt collection** (fully written text, not prompt templates). For Standard Diffusion 2.1 + compatible LoRA, use `03_Vitiligo/json/vitiligo_all_300.json`. Each record already includes `positive_prompt` and `negative_prompt`; never construct a prompt from metadata.

## Dataset
- 300 positive prompts and matched negative prompts; 60 distinct clinical morphology archetypes with five controlled appearances each.
- 120 bilateral non-segmental-appearing, 90 unilateral segmental-appearing, 90 localized focal/undetermined-appearing facial patterns. These labels are **appearance intents**, not clinical diagnoses or verified subtypes.
- Variations: V1 age, V2 affected-area extent, V3 observed skin tone, V4 facial site, V5 combined. Multiple subject characteristics are balanced across the dataset; these are not strict controlled causal intervention experiments.
- 150 male and 150 female descriptors; 50 prompts for each intended Fitzpatrick I–VI reference category (use descriptive visible tones in the actual prompts).
- Group split: 210 training prompts, 45 validation, 45 test. **All 5 seeds from the same prompt and all 5 variations of one archetype stay within one split.** This prevents same-prompt leakage, not similarity of new images from other classes or datasets.
- Default 5 images / prompt (1,500 generated candidate images). Generate sequentially or in small batches, checkpoint outputs, and filter before classifier training.

## Morphology rules
- Vitiligo typically produces smooth flat clearly defined white depigmented macules/patches, without superficial scale or raised plaques.
- Segmental appearing prompts remain unilateral; non-segmental appearing prompts are bilateral/roughly symmetrical; focal appearing patterns are single-site, with unspecified longitudinal clinical course.
- No synthetic image is verified as a true patient diagnosis. Do not use prompts or generated images as diagnostic ground truth without clinician review.
- Exclude visually inaccurate lesions and prompt-to-image contradictions. Evaluate synthetic classifier effect on **held-out real-image groups**.

## Prompt length and Compel
- Positive prompts are written concisely for up to 150 SD 2.1 content tokens (two 75-content-token chunks).
- **Actual CLIP tokenizer counts have not been computed here.** Run the included `validate_vitiligo_tokens.py` in Kaggle after downloading your exact SD 2.1 tokenizer. It fails on overlength records; do not silently truncate. The count of words is not a token count.
- Add Compel to the inference notebook with `truncate_long_prompts=False`, pad positive and negative conditioning tensors to equal length, then use `prompt_embeds`/`negative_prompt_embeds`. Do not train the LoRA for chunking.

## GitHub and Kaggle
1. Upload extracted `03_Vitiligo/` alongside `prompt_manifest.json` to a public or authenticated GitHub repository.
2. Supply the RAW GitHub URL for `03_Vitiligo/json/vitiligo_all_300.json` to the loader.
3. Validate real SD2.1 tokenizer token counts before bulk inference.
4. For each record, use `positive_prompt`, `negative_prompt`, `seed_base`, and `images_per_prompt`, and record the `prompt_group_id` and `split`. Do not perform a new random split on generated images.

## Package structure
- `03_Vitiligo/csv/vitiligo_001_100.csv`, `vitiligo_101_200.csv`, `vitiligo_201_300.csv`
- `03_Vitiligo/json/vitiligo_all_300.json` (recommended inference source), JSON batches and JSONL
- `03_Vitiligo/txt/vitiligo_all_300.txt`, batch TXT, and 300 individual TXT prompts
- `load_vitiligo_prompts.py`, `validate_vitiligo_tokens.py`, `prompt_manifest.json`, `QA_REPORT.json`, `FIRST_FIVE_PREVIEW.md`

## Clinical references
- https://dermnetnz.org/topics/vitiligo
- https://www.aad.org/public/diseases/a-z/vitiligo-symptoms
- https://www.ncbi.nlm.nih.gov/books/NBK559149/
- https://onlinelibrary.wiley.com/doi/10.1111/jdv.19451

Note: clinical descriptions derive from general clinical resources, not from proof of any generated output. Review by a qualified dermatologist is needed.
