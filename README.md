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

## Starting point

Read [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md), [`docs/CODE_COMPLETION_TRACKER.md`](docs/CODE_COMPLETION_TRACKER.md) and [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md). Develop and test one phase at a time, then add its notebook and run record. The repository contains existing notebook implementations and offline checks, but no verified end-to-end study results on Kaggle. Work must begin by correcting the suspect Cancer and Normal class labels.
