# Research project plan

## Scientific objective
Investigate whether morphology-controlled **synthetic facial photographs** improve six-category classification on unseen real facial photographs. Do not claim clinical validation without an appropriate clinical review or proven dataset labels.

## Stage gates
1. **P0 — Infrastructure:** repository tree, Kaggle environment and dependencies, seed control, config validation, baseline smoke tests.
2. **P1 — Real data:** provenance, licensing, image modality, label taxonomy, hash/near-duplicate groups, immutable group-aware split manifests. Hold out complete patient groups when true IDs exist; otherwise mark groups as visually inferred.
3. **P2 — Generative pilot:** check official checkpoint IDs, weights, licenses, T4 memory and clinical-photo compatibility. For each disease/subtype, generate 15–20 image pilots per prompt family and inspect morphology rather than simply aesthetics.
4. **P3 — Synthetic pool:** fixed generation manifests (checkpoint revision, LoRA, prompt ID, prompt text, seed, scheduler, settings, source). Record accepted/rejected images and flags for suspicious similarity to train/validation/test data. Never train a generator with held-out real images.
5. **P4 — Classification:** real-only (E0); disease-only prompt augmentation (E1); morphology prompt augmentation (E2); quality-selected morphology augmentation (E3); optional diversity/coverage selection (E4). Hold augmentation counts, model, schedule and evaluation set fixed as appropriate.
6. **P5 — Evaluation:** fixed real-only test set; macro-F1 primary; balanced accuracy/per-class sensitivity secondary. Compare paired predictions and bootstrap uncertainty at valid unit of analysis. Different seeds reflect training instability, not more independent patients.
7. **P6 — Ablations and extensions:** low-data 25%/100% first, alternate architectures, prompt attributes, synthetic quantity, calibration, cross-generator, external test, clinical check if available.

## Main comparison contract
| ID | Training data | Question |
| --- | --- | --- |
| E0 | Real only | Baseline |
| E1 | Real + generic disease-prompt synth | Does generic generation help? |
| E2 | Real + morphology-prompt synth | Do morphological prompts help vs E1 and E0? |
| E3 | Real + quality-screened morphology synth | Does controlled filtering help at matched N? |
| E4 | Real + diversity-screened morphology synth | Optional, same pool and N as E3/random |
| E5 | Synthetic only | Optional diagnostic for domain gap |

**Do not conflate prompt ablation with selection ablation.** For E1/E2 generate matched pools with identical base model/seed grids where feasible and select by a predeclared common policy. For selection ablations, all selected subsets must come from the **same underlying pool**.

## Decision log
- Class definitions and annotation quality to be finalized from actual data.
- Skin cancer versus premalignant actinic keratosis must be explicitly defined, not silently combined.
- Fixed accepted synthetic budget around 1,500/class is a **target, not a requirement** or a claim of achieved quality.
- Primary two classifier families provisionally ResNet-50 and EfficientNet-B3; three seeds for selected runs, limited Kaggle T4.
- True low-data studies must ensure disease-specific generator adaptation did not access excluded real training images; otherwise label explicitly as classifier-only scarcity.
- Literature novelty remains unproven until independent paper audit.

## Gate before any reported result
Dataset manifest version, code SHA, exact model/weight revisions, seeds, configuration hash, train-data references, verified test partition, unit of statistical analysis, output logs and hardware recorded.

See `docs/RESEARCH_PROTOCOL.md` and `docs/CODE_COMPLETION_TRACKER.md`.
