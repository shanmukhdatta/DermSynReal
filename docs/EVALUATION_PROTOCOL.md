# Evaluation protocol — DermSynReal

**Research protocol, not completed medical validation.** The only headline endpoint is six-class macro-F1 on an **unchanged, label-reviewed, genuinely real facial image test set**.

## Run organization

- Frozen real test: same `image_id`, label, and visual/patient grouping across every model and condition. One frozen split SHA across one evaluation registry.
- E0 real-only. E1 real+generic-prompt synthetic. E2 real+morphology-prompt synthetic sampled randomly. E3 same E2 pool quality-selected, matched count. E4 same E2 pool diversity-selected, matched count. E5 synthetic-only, tested on real images. The five current classifiers now accept an optional E5 synthetic-only training mode, retaining frozen real validation and test. The standalone older ResNet50 trainer remains separate. All GPU execution remains unverified.
- Report training-mode differences only when the starting weights, optimizer, class budgets, seed and effective training budgets are controlled. Matched-arm manifest generation is in `scripts/build_experiment_manifests.py`.
- Evaluation `notebooks/Evaluation/01` and `02` load `best_weights.pth` or `linear_probe.joblib` and **do not train** or choose hyperparameters on test data. They fail if test is not explicitly unlocked.
- Real-test and synthetic-test metrics are separate. Synthetic-test split must be independent by prompt family from synthetic train. A synthetic-only model must still have real-test predictions.
- Retain dataset acquisition source as `dataset_source`, distinct from `source` which labels the real/synthetic domain. Never treat an image source containing too few classes as a complete six-class macro-F1 comparison.
- For each class report support, precision, recall and F1. Save confusion matrices. If `p_*` probabilities exist, calculate calibration separately for each domain, and do not tune temperature using real-test labels.
- Paired comparison is restricted to exact matched `backbone + seed + frozen real-test IDs` with validated group IDs. The code returns a paired bootstrap interval for the observed macro-F1 difference, **not a statistical significance p-value**. Three seeds describe training variability, not three times as many patients. If no patient IDs exist, group IDs are inferred visual components and independence remains uncertain.
- Source-stratified analyses are descriptive unless there is adequate class coverage and source support. Expose failures, missing data, weak classes and test-count limitations.
- The run-registry and audit files must be provenance-preserving; `label_audit_summary.json` must contain `passed: true`. This flags documented structural checks, not guaranteed clinical accuracy or complete patient leakage prevention.

## Reproducibility ledger

At each run keep code SHA; versioned approval and source manifests; real split SHA and raw image checksums; model ID/version; weight and probe SHA-256; generator base+revision/adapter; prompt ID/family/seed; input transforms; class ordering; hardware; software versions; split/label QA evidence; actual prediction files; wall time and failure logs. Do not commit identifying clinical images, source license violations, secrets or raw patient metadata to GitHub.

## Publication and results

The LaTeX paper is `paper/dermsynreal_manuscript.tex`, with blank tables. Evaluation 04 exports `run_summary.csv`, `per_class_real_test.csv`, `confusion_real_test.csv`, `per_source_real_test.csv`, `real_test_errors.csv`, `paired_real_test.csv` and a reproducibility manifest. Evaluation 06 creates a separate tabular fragment, but does **not automatically overwrite the paper**. A final results claim requires matching checkpoint, true test set, verified labels, reviewed source artifacts, and appropriately qualified statistical uncertainty.
