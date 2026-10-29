# After Kaggle execution — research output handoff

Code is versioned in the [DermSynReal GitHub repo](https://github.com/shanmukhdatta/DermSynReal). **Neither this guide nor GitHub Actions performs future Kaggle tasks or pushes future results automatically.**

1. Save each Kaggle notebook Output as a private versioned Kaggle Dataset. Keep full generated images, original medical images, large LoRA/model checkpoints, patient information and private logs **out of public GitHub**; the notebook output panel holds them.
2. Preserve run metadata and aggregate counts as CSV/JSON, including source v2, code SHA, real split/label audit SHA, generator/prompt family/seed/adapter revision, full train configuration, validation results, checkpoint SHA, hardware and software environment.
3. On completing the frozen real-test evaluations, run `notebooks/Evaluation/04_Experiment_Suite_Real_Test_Reporting.ipynb`; don't synthesize results or update the LaTeX paper before checking labels, grouped splits and source permissions.
4. Share **only aggregate report files** (e.g., `run_summary.csv`, `per_class_real_test.csv`, `paired_real_test.csv`, `generation_quality_counts.csv`, error counts **without image paths/IDs**, and redacted run config) to commit them to `reports/experiments/<run-tag>/`. Do not commit `real_test_errors.csv`, `misclassified_image_ids.csv`, per-image synthetic manifests, private trace logs, or screenshots of patient images to public GitHub by default.
5. When the actual Kaggle run URLs and approved aggregate output are available, update `docs/CODE_COMPLETION_TRACKER.md` with verified run links/SHAs, failures, fixes and evidence. This preserves reproducibility; nothing is claimed complete until an actual run succeeds.
