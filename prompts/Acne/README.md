# DermaLens — Acne Morphology Prompts for SD 2.1 + LoRA

**Status:** 300 prompts authored; **tokenizer verification pending execution with the actual SD 2.1 tokenizer**. No clinical images were generated or medically verified in this package.

## Main deliverables

- `01_Acne/acne_001_100.csv` — prompts ACNE_001…ACNE_100
- `01_Acne/acne_101_200.csv` — prompts ACNE_101…ACNE_200
- `01_Acne/acne_201_300.csv` — prompts ACNE_201…ACNE_300
- `FIRST_5_PREVIEW.md` — representative first batch spanning five distinct lesion families
- `ACNE_QC_REPORT.json` — programmatic checks, balance summaries, five-prompt review batch results
- `validate_acne_prompts_for_kaggle.py` — fail-closed exact-token checking after loading your SD 2.1 pipeline

## Clinical foundation

Six **appearance families**, not mutually exclusive diagnostic subtypes:
1. Closed comedones (noninflamed whiteheads)
2. Open comedones (blackheads)
3. Inflammatory papules
4. Papules with superficial pustules
5. Deep inflammatory nodules
6. Mixed comedonal and inflammatory lesions

Each family has 50 prompt rows, containing diverse visual lesion characteristics and facial distributions. Medical morphology follows DermNet, AAD and EADV descriptions:

- https://dermnetnz.org/topics/acne-vulgaris
- https://dermnetnz.org/topics/comedonal-acne
- https://www.aad.org/public/diseases/acne/really-acne/symptoms
- https://www.aad.org/public/diseases/acne/diy/types-breakouts/
- https://eadv.org/wp-content/uploads/2024/04/ACNE-production-evolution-and-diagnosis-EADV.pdf

**Caution:** Severity is a *descriptive generation label*, not a diagnosis determined from a photograph alone. `fitzpatrick_reference` is intended as a skin-appearance sampling proxy; actual Fitzpatrick phototype cannot be determined solely from appearance. Manually review the desired skin-tone and disease-feature distributions.

## Five variation groups

- `V1` — age-focused (other major morphological factors held approximately constant within a group)
- `V2` — lesion extent/severity-focused
- `V3` — visible skin-tone-focused
- `V4` — facial region/distribution-focused
- `V5` — combined controlled variations across age, extent, tone, location, camera

Each `prompt_group_id` comprises 5 closely related prompts. The group is assigned to exactly one split to prevent leakage between train, validation and test, even when it crosses a CSV boundary. The 60 consecutive **review batches** in `ACNE_QC_REPORT.json` are *presentation batches*, intentionally diversified across phenotype families; they are not the same as the leak-safe grouping unit.

## Counts

| Property | Count |
|---|---:|
| Total prompts | 300 |
| Distinct prompt groups | 60 (5 related prompts each) |
| `train` | 210 prompts |
| `val` | 45 prompts |
| `test` | 45 prompts |
| Lesion families | 6 × 50 prompts |
| Variation groups | 5 × 60 prompts |
| Male descriptor | 150 prompts |
| Female descriptor | 150 prompts |
| Visual skin-tone references | 6 × 50 prompts |
| Planned image candidates | 300 prompts × 5 independent seeds = 1,500 |

## Prompt length and Compel

**Design maximum:** 150 **content** CLIP tokens, intended to fit two 75-content-token chunks (two 77-position sequences including CLIP markers) with `truncate_long_prompts=False`.

**Do not mistake the CSV `word_count` for CLIP token count.** The CSV `sd21_content_tokens` cell is intentionally blank and `token_validation_status` is `PENDING_MODEL_TOKENIZER`, because the exact SD 2.1 tokenizer was not available in the offline authoring environment. This avoids fabricating token counts. The prompts target compact clinical descriptions but **may require manual trimming** after the true tokenizer check. The checker rejects any overlength prompt rather than silently truncating it.

From the Kaggle notebook, after loading the same `pipe` (including adaptations) that generates images:

```python
from pathlib import Path
import sys
root = Path('/kaggle/input/YOUR_PROMPT_DATASET/DermaLens_SD21_Prompts')
sys.path.insert(0, str(root))
from validate_acne_prompts_for_kaggle import validate_and_write
report = validate_and_write(
    source_dir=root/'01_Acne',
    tokenizer=pipe.tokenizer,
    output_dir=Path('/kaggle/working/acne_prompts_validated'),
    max_content_tokens=150,
)
print(report)
```

Then use the **validated CSV copies** in `/kaggle/working/acne_prompts_validated`. In the inference step, use Compel with `truncate_long_prompts=False`, pad positive/negative conditioning and pass the embeddings via `prompt_embeds` / `negative_prompt_embeds`. Do not pass the long text as raw `prompt=` through a standard truncated path.

## Generating and splitting images

`seed_base` is a reproducible per-prompt integer; obtain five independent seeds by using `seed_base + image_index` with image_index 0…4. Save model/LoRA revision, exact prompt, negative prompt, seed, sampler, guidance, steps and image filename. Process sequentially or in small batches to fit Kaggle T4, saving every completed image and allowing resume.

The split assignment of each generated image must be inherited from its `prompt_group_id` / original CSV row, never assigned by random image splitting. Do not let these synthetic prompts compromise the **separate, grouped real-image evaluation** for the skin classifier. For model selection and final reporting, prefer a real held-out test set, because synthetic test examples share the generator's biases.

## QC completed and QC not yet done

Checked: unique IDs, unique full positive prompts, 300 exact rows, diversity across every consecutive batch of five, six equal phenotype-family counts, five equal variation counts, sex and appearance-label balance, 210/45/45 group-leak-safe split, separate seed bases, cross-group TF-IDF near-copy check.

**Not checked here:** real tokenizer token counts, actual generated-image diversity, diffusion checkpoint/LoRA compatibility during running inference, anatomical plausibility in rendered images, dermatologist review, image-vs-prompt feature accuracy. Those require model execution and image auditing; prompt text alone cannot guarantee them.

---

## NEW: Direct prompt exports for GitHub / Kaggle inference

**The `positive_prompt` and `negative_prompt` strings in the JSON and TXT exports are copied verbatim from the existing CSVs. No second prompt-generation or morphology guesswork was used.** All metadata (IDs, lesion phenotype, prompt-group splits, seeds) is retained in JSON. Token checks still must be run on the actual SD 2.1 pipeline tokenizer.

### File layout

```text
DermaLens_SD21_Prompts/
  prompt_manifest.json
  01_Acne/
    acne_001_100.csv, acne_101_200.csv, acne_201_300.csv
    acne_all_300.jsonl
    json/
      acne_all_300.json                 <- easiest for GitHub URL loading
      acne_001_100.json
      acne_101_200.json
      acne_201_300.json
    txt/
      acne_all_300.txt                  <- readable, labeled positive + negative
      batches/acne_001_100.txt         <- readable batches (100 each)
      batches/acne_101_200.txt
      batches/acne_201_300.txt
      individual/ACNE_001.txt          <- EXACT positive prompt only
      .../ACNE_300.txt
  load_acne_prompts.py                  <- loader / jobs from local path or GitHub URL
  validate_acne_prompts_for_kaggle.py   <- check 150 content-token maximum
  FIRST_5_PREVIEW.md
  ACNE_QC_REPORT.json
  SHA256SUMS.txt
```

### Single URL to load all 300 prompts

Push **the entire `DermaLens_SD21_Prompts` directory** to a GitHub repository. Then point Kaggle to the raw master JSON, for example:

```
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/DermaLens_SD21_Prompts/01_Acne/json/acne_all_300.json
```

The repo and branch are placeholders, not actual links. Enable Internet in Kaggle if downloading directly.

Use this **exact Python cell** after defining the raw URL:

```python
import json
from urllib.request import Request, urlopen
url = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/DermaLens_SD21_Prompts/01_Acne/json/acne_all_300.json'
with urlopen(Request(url, headers={'User-Agent':'DermaLens-PromptLoader/1.0'}), timeout=60) as response:
    prompts = json.loads(response.read().decode('utf-8'))
assert len(prompts) == 300
assert len({r['prompt_id'] for r in prompts}) == 300
for r in prompts:
    prompt = r['positive_prompt']      # Already a complete morphology-first prompt
    negative = r['negative_prompt']    # Do NOT regenerate / prepend / concat wording
    # after exact-token validation, encode with Compel and run your loaded SD2.1+LoRA pipeline
```

For a GitHub repository URL rather than a raw JSON URL, the included `load_acne_prompts.py` supports public repo archives (best if prompt files live in a small standalone repo):

```python
from load_acne_prompts import load_prompts, iter_generation_jobs
prompts = load_prompts('https://github.com/YOUR_USERNAME/YOUR_REPO')
for task in iter_generation_jobs(prompts, split='train'):
    print(task['prompt_id'], task['positive_prompt'], task['seed'])
```

The loader also accepts a local folder, `acne_all_300.json`, or a downloaded package `.zip`. For split-safe generation, each of the five outputs for a prompt inherits its `split` and `prompt_group_id`; never randomly redistribute images. The generator should checkpoint metadata after each image to allow resume.

### Correct SD 2.1 + LoRA + Compel integration

Use the already loaded `pipe` and LoRA in your inference notebook. Validate with the **actual `pipe.tokenizer`** using `validate_acne_prompts_for_kaggle.py` and reject any prompt over 150 content tokens; do not silently truncate. Then configure Compel with `truncate_long_prompts=False`, encode the provided `positive_prompt` and `negative_prompt` fields, pad both embeddings to equal sequence length, and supply `prompt_embeds` and `negative_prompt_embeds` to `pipe`. Use `seed_base + image_index` for image_index 0..4. **The TXT files are not templates; the full literal text is the prompt.** Do not duplicate the morphology by independently rebuilding it from columns.

### Honesty about validation

The JSON and TXT output has been checked for byte-identical prompt text compared with CSV and 300 unique prompt IDs. The original prompt descriptions are research-oriented but not clinically certified; the token field remains null / pending until actual SD2.1 tokenizer validation, and visual fidelity cannot be validated without image generation and manual clinical review.
