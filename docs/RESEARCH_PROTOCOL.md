# Reproducible research protocol — initial draft

## Inputs and partitions
Represent each real image with stable `image_id`, source dataset, relative storage URI, raw SHA-256, label, subtype (nullable), consent/licensing status and group ID (nullable). Keep image files off GitHub. If group IDs are inferred from image similarity, mark their provenance as `inferred_visual`, not verified patient identity. The six-class label mapping must be explicitly signed off before freezing partitions.

Detect exact duplicates with byte/pixel SHA-256, orientation-aware perceptual hashes, and candidate embedding matches plus manual review where needed. Verify no cross-split matches. Freeze train/val/test group assignments into versioned manifests and never change them to improve a test score.

## Generation
First identify checkpoint and adapters with legitimate download, compatible license and full-face clinical-photo ability. A 15–20 image/prompt pilot is **screening**, not statistical proof of medical correctness. Store image ID, generator checkpoint/revision, prompt family, exact prompt, negative prompt, seed, scheduler, steps, guidance, resolution, accepted/rejected decision, rejection reason, model license and hash. Morphology quality needs auditing; reject near-reproductions of source images. Generic FID/CLIP similarity alone does not prove clinical correctness.

## Training
Define configs with model, input size, optimizer, number of steps or epochs, train-only augmentation, train-real/synthetic mixture, class sampling policy, checkpointing, deterministic seed and data-manifest hash. Train/val/test must be independent of generator tuning and selection. Use validation for model selection, test once per finalized setting.

## Evaluation
Primary endpoint: macro-F1 across the six target labels, on *real test images only*. Secondary: balanced accuracy, per-class recall, confusion matrix and, if useful, probabilities/calibration. Across-seed summary and test-sampling uncertainty must be reported separately. Prefer paired bootstrap for metric differences (group-resampled when valid groups exist); specify multiplicity family and Holm adjustment when conducting multiple primary comparisons. Disclose incomplete labels, weak class evidence and uncertainty honestly.

## Low-data experiment
At first test real train fractions 25% and 100% with fixed independent real validation/test groups. If a disease-specific generator has been trained with the full pool, do **not** call the downstream classifier condition strictly 25%-real-data. Instead label as a fixed-generator classifier training-subset study. For strict low-data, constrain all dataset-specific generator adaptation to the fraction's allowed real subset.

## Main statistical comparisons
- E2 vs E0: morphology-controlled synthetic utility.
- E2 vs E1: morphology-specific prompting over generic prompting.
- E3 vs E2: quality selection utility, but only if E2 is a matched-count random selection from the *same morphology pool*.
- E4 vs E2: optional diversity selection under the same restriction.

Avoid unsupported p-values, medical claims, fairness conclusions, and cross-source generalization claims.
