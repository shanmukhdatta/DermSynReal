# Research reports and templates

- `templates/runs_registry_template.csv` contains **headers only**; fill one row per verified run.
- Evaluation 04 produces `run_summary.csv`, `per_class_real_test.csv`, `confusion_real_test.csv`, `per_source_real_test.csv`, `real_test_errors.csv`, `paired_real_test.csv`, and `report_manifest.json`.
- Evaluation 05 produces probability calibration JSON and optional reliability plots.
- Evaluation 06 can export an empirical LaTeX fragment only after verified run outputs exist.
- The manuscript `paper/dermsynreal_manuscript.tex` retains blank tables until actual Kaggle execution and review.

Do not commit raw clinical photographs, restricted datasets, sensitive patient identifiers, credentials or fabricated performance claims. Store private run artifacts in an authorized, versioned location and record their hashes/identifiers in the tracker.


## Actual research proof workflow

Save a fresh versioned Kaggle Output for each experiment, keep private face/identifier-level evidence in Kaggle, and add the run's notebook URL, frozen reviewed-split SHA, checkpoint SHA and aggregate SHA to `templates/run_evidence_ledger_template.csv`. After final real-test evaluation, `scripts/prepare_public_run_report.py` accepts only an allowlisted set of **aggregate** measured CSV/JSON reports, including `seed_summary_real_test.csv`, `seed_summary_per_class.csv`, `seed_summary_paired_contrasts.csv`, and `statistical_analysis_summary.json`. It does not send a GitHub commit automatically; you must review and explicitly commit the exported aggregate report folder.

Raw per-image outputs, paths, original diagnoses, private patient or synthetic-image review rows, GPU logs and checkpoints are **not** appropriate in a public research Git repo.
