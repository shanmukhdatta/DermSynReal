# Research reports and templates

- `templates/runs_registry_template.csv` contains **headers only**; fill one row per verified run.
- Evaluation 04 produces `run_summary.csv`, `per_class_real_test.csv`, `confusion_real_test.csv`, `per_source_real_test.csv`, `real_test_errors.csv`, `paired_real_test.csv`, and `report_manifest.json`.
- Evaluation 05 produces probability calibration JSON and optional reliability plots.
- Evaluation 06 can export an empirical LaTeX fragment only after verified run outputs exist.
- The manuscript `paper/dermsynreal_manuscript.tex` retains blank tables until actual Kaggle execution and review.

Do not commit raw clinical photographs, restricted datasets, sensitive patient identifiers, credentials or fabricated performance claims. Store private run artifacts in an authorized, versioned location and record their hashes/identifiers in the tracker.
