# DermSynReal

**Morphology-controlled synthetic facial dermatology images and real-world classifier evaluation**

> Research project with uploaded notebooks, prompt libraries, offline analysis helpers and a LaTeX manuscript. **GPU experiments and clinical validity have not been established.** This repository is not a diagnostic tool.

## Research question

Does clinically described, morphology-controlled synthetic augmentation improve six-category facial dermatology classification on **unseen real photographs** over real-only and disease-name-prompt baselines? Which selection protocols, training-data regimes and disease categories benefit or fail?

## Intended categories (taxonomy under audit)

1. Acne
2. Vitiligo
3. Hyperpigmentation
4. Facial fungal infections
5. Skin cancer / precancerous lesions — *label definition to be finalized; actinic keratosis is premalignant*
6. Normal facial skin

Subtypes guide generation and metadata; they do not automatically become separate classifier labels. Dataset provenance, patient consent/licensing, label quality, and meaningful clinical boundaries must be reviewed before training.

## Execution sequence

| Phase | Aim | First executable notebook | Status |
|---|---|---|---|
| P0 | Initialize repo, reproducibility contract, config schemas | `notebooks/00_environment_check.ipynb` | Scaffold |
| P1 | Real-image ingest, metadata, deduplication and frozen group-aware splits | `notebooks/01_real_data_audit.ipynb` | Planned |
| P2 | Generator eligibility check and disease-specific 15–20 image/prompt pilots | `notebooks/02_generation_pilot.ipynb` | Planned |
| P3 | Versioned synthetic generation, QA, near-copy and coverage controls | `notebooks/03_synthetic_generation.ipynb` | Planned |
| P4 | Train real-only baseline, then equal-budget synthetic variants | `notebooks/04_train_classifier.ipynb` | Planned |
| P5 | Evaluate fixed real test sets, intervals and main experiments | `notebooks/05_evaluation.ipynb` | Planned |
| P6 | Low-data, prompt, quality/diversity selection and optional analyses | `notebooks/06_ablation_studies.ipynb` | Planned |

Each notebook must execute one phase from config and use shared tested Python modules under `src/`. Kaggle datasets/artifacts are mounted separately; do not put clinical image collections, model weights, secrets or Kaggle outputs into Git.

## Planned primary experiment conditions

- **E0**: real training images only.
- **E1**: real + equal-budget disease-name-prompt synthetic images.
- **E2**: real + equal-budget morphology-controlled synthetic images.
- **E3**: real + matched-count morphology-controlled, quality-screened synthetic images.
- **E4**: real + matched-count morphology-controlled, diversity/coverage-selected synthetic images (extension).
- **E5**: synthetic-only training, evaluated exclusively on real images (diagnostic extension).

The same fixed real-test partition, controlled train budgets, seed schedules and documented selection constraints apply. Selection must never use test metrics. Low-data claims must separately track which real images were accessible to generator adaptation.

**Primary outcome:** six-class real-test macro-F1. Secondary outcomes include balanced accuracy, per-class recall, confidence intervals, artifact audits and useful failure analyses.

## Project directories

- `configs/`: validated dataset, generation, training and experiment definitions.
- `data/`: documentation, metadata schemas, split-manifest templates; large/private imagery stays outside Git.
- `docs/`: scientific assumptions, protocols, decisions, reporting, and phase trackers.
- `notebooks/`: one Kaggle-ready orchestrating notebook per phase (to be implemented sequentially).
- `src/dermsynreal/`: reusable Python modules, independently testable.
- `scripts/`: command-line entry points.
- `tests/`: unit/integration tests and synthetic fixtures.
- `reports/`: report templates and publication-ready analyses; no personal data.
- `.github/workflows/`: lightweight lint, validation and unit-test CI.

## Critical safety/reproducibility rules

- Keep images and patient metadata outside Git and respect dataset licenses.
- Deduplicate **before** freezing splits; no guessed patient identity claims.
- Record original source, label provenance, SHA-256, grouping method, hash, split version and augmentation lineage.
- Keep validation and test photographs away from generator adaptation and selection.
- Define fixed prompt families, model identifiers, revisions, generated image IDs, seeds and acceptance reasons.
- Match counts, epochs/optimizer steps and real/synthetic sampling policies for comparisons.
- Treat clinical realism, fairness, novelty, compute times and statistically significant results as hypotheses until verified.
- Never present synthetic images as genuine patient photographs.

## Paper and reproducibility additions (29 October 2026)

- [Complete LaTeX methodology draft with blank results tables](paper/dermsynreal_manuscript.tex).
- [Step-by-step Kaggle execution and critical data-label gate](docs/EXECUTION_GUIDE.md).
- [Reviewed-label metadata gate](src/dermsynreal/label_gate.py) and [source-separated real-test evaluation](src/dermsynreal/evaluation.py).
- [Research unit tests](tests/test_research_gates.py) and [GitHub Actions offline gate](.github/workflows/research-gates.yml).
- Existing generation, preprocessing, training and prompt assets are preserved; see `notebooks/` and `prompts/`.
- The five current classifier notebooks now export immutable test-image/group-linked CSV files after the final-test flag is enabled.

**Important:** current notebooks use EfficientNet-B3, ConvNeXt Tiny and DINOv2 ViT-S/14; these are not the separately planned EfficientNet-B0, ConvNeXt V2 Tiny and DINOv2 ViT-B/14. PanDerm Base is currently implemented as a linear probe. Do not report results under mismatched model names.

## New Kaggle evaluation workflow

The [seven evaluation notebooks](notebooks/Evaluation/README.md) now cover reviewed real-label preflight, real/synthetic held-out checkpoint inference, official PanDerm linear probe, morphology QA, paired E0–E5 comparisons, calibration and LaTeX results export. New [empty CSV templates](data/manifests/) and [source/class reporting protocol](docs/EVALUATION_PROTOCOL.md) are available. These notebooks **have not been executed against real images on Kaggle**; results remain blank. The first blocking task is to correct and clinically review the real Skin Cancer and Normal label issues.

## Starting point

Read [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md), [`docs/CODE_COMPLETION_TRACKER.md`](docs/CODE_COMPLETION_TRACKER.md) and [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md). Develop and test one phase at a time, then add its notebook and run record. The repository contains existing notebook implementations and offline checks, but no verified end-to-end study results on Kaggle. Work must begin by correcting the suspect Cancer and Normal class labels.

## October 29 Kaggle v2 integration

**Only real-image source:** `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection` (Kaggle dataset v2). The screenshot displays six folders totaling **9,791 uploaded files, not 9,791 verified independent images**. The collection is not yet clinically reviewed, deduplicated or trained against. Original permissions must be checked; the uploader's CC0 choice does not prove third-party image rights.

**Start:** [real v2 inventory → review → group-disjoint split → train manifest](notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb). See [Preprocessing README](notebooks/Preprocessing/README.md), [Generation README](notebooks/Generation/README.md), [Training README](notebooks/Training/README.md), and [Evaluation README](notebooks/Evaluation/README.md).

Each generator now consumes the authored **300-prompt** bank (`PROMPT_VARIANT='generic'` for E1 or `morphology` for E2) and writes unreviewed per-image manifests, SHA metadata and run logs. LoRA notebooks train only from the reviewed frozen **real training** split, never directly from the raw dataset; subtype LoRA adaptation requires reviewed subtype mapping. The four timm classifiers have head/partial/optional full stages with matched optimizer update budget; PanDerm remains a frozen-feature linear probe. E5 synthetic-only is optional. The held-out real test stays locked until experiment selection is complete.

**Future GitHub results:** Kaggle keeps images and heavy model files; the new `scripts/prepare_public_run_report.py` prepares whitelisted aggregate CSV/JSON reports under `reports/experiments/<run-tag>/` for explicit, reviewed GitHub commits. [Handoff steps](docs/RUN_RESULTS_HANDOFF.md). No code here performs unattended future Kaggle training or publishing.


## Verified orchestration code, not verified Kaggle experiments

The [real v2 start-here notebook](notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb) prepares one globally reviewed group-safe real split, **not independent resplits per disease**. All six LoRA notebooks filter this frozen manifest to their selected class, report its train/val/test counts, and train on **train only**. `class_frozen_split_counts.csv` and private assignment records are saved alongside the LoRA weights; `notebook_console.log`, `trainer_console.log` and `run_log.jsonl` preserve runtime evidence.

The [reviewed synthetic combiner](notebooks/Preprocessing/Synthetic_Data/00_Combine_Reviewed_Base_And_LoRA_Generations.ipynb) accepts separately reviewed base or LoRA generation manifests after remounting Kaggle outputs, verifies image SHA-256 and generator/prompt provenance, and produces one CSV for [synthetic group preprocessing](notebooks/Preprocessing/Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb). It does **not** silently merge different generator variants into a single E1/E2 research arm.

The five classifiers (plus the older separate ResNet50 notebook) now save private full console logs and structured artifact provenance. [Evaluation 04](notebooks/Evaluation/04_Experiment_Suite_Real_Test_Reporting.ipynb) generates paired real-test group-bootstrap confidence intervals and seed-summary CSVs; [Evaluation 07](notebooks/Evaluation/07_Statistical_Analysis_Across_Seeds.ipynb) inspects seed variability and per-class uncertainty descriptively. Every evaluation notebook logs its successful outputs.

For public GitHub research evidence use **only** the whitelisted, privacy-reviewed aggregate results exported using `scripts/prepare_public_run_report.py` and the [run evidence ledger](reports/templates/run_evidence_ledger_template.csv). Keep per-image manifests, patient IDs, image paths, photographs, raw GPU logs, private credentials, and heavy weights in private Kaggle storage; link to versioned Kaggle run evidence instead. **No experiments or performance statistics have been run or measured by these repository commits.**

**Final generation-source bridge:** after the separate reviewed base and/or LoRA image pools are attached to Kaggle, run [synthetic combination](notebooks/Preprocessing/Synthetic_Data/00_Combine_Reviewed_Base_And_LoRA_Generations.ipynb), [synthetic preprocessing](notebooks/Preprocessing/Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb), then `scripts/build_experiment_manifests.py --synthetic-root <saved-synthetic-root> --generator-mode <matched-mode>`. The builder's E0–E5 manifests then feed the existing training notebooks. Mixed-generator preprocessing is allowed explicitly; mixing generator configurations in matched experimental arms is not. See [full execution guide](docs/EXECUTION_GUIDE.md).


## Research paper — complete pre-results draft

The new research manuscript is maintained separately at [paper/manuscript/main.tex](paper/manuscript/main.tex), with nine complete sections and [verified research references](paper/manuscript/references.bib). Its contextual and methodological content is written; measured results, empirical figures, result-dependent discussion and final numerical conclusions remain **deliberately blank** until the audited Kaggle experiments run. For the exact update locations, compilation instructions and CSV-to-table mapping, read [paper/manuscript/README.md](paper/manuscript/README.md). The older paper/dermsynreal_manuscript.tex protocol draft remains unchanged for comparison.
