# DermSynReal — Kaggle Evaluation Notebooks

**Seven independent Kaggle notebooks are implemented.** These are code-ready, not experimentally validated. The model checkpoints, six-class reviewed images, and prediction CSVs must still be supplied when running.

| Order | Notebook | Produces | Required data |
| --- | --- | --- | --- |
| 00 | `00_Real_Label_And_Split_Preflight.ipynb` | reviewed label CSV, split check and SHA | human-reviewed source manifest, frozen real split |
| 01 | `01_Timm_Frozen_Checkpoint_Inference.ipynb` | final test predictions + provenance | one of the four current timm model checkpoints, run config, frozen real test |
| 02 | `02_PanDerm_Official_Probe_Inference.ipynb` | PanDerm probe predictions + provenance | genuine PanDerm Base encoder/weights and saved linear probe |
| 03 | `03_Generation_Morphology_QA.ipynb` | per-class/generator reviewed acceptance | human morphology review CSV |
| 04 | `04_Experiment_Suite_Real_Test_Reporting.ipynb` | E0-E5 real/synthetic-source tables, paired CIs, confusion | scored per-run CSVs, passed label audit, run registry |
| 05 | `05_Probabilities_Calibration_and_Errors.ipynb` | Brier, ECE, NLL, reliability plots per domain | probability-bearing predictions from 01/02 |
| 06 | `06_Results_Export_For_LaTeX.ipynb` | observed results LaTeX fragment | verified run_summary.csv from 04 |

## Kaggle setup

1. Add the existing DermSynReal code repository as a Kaggle Dataset, or enable **Internet** and leave `CLONE_REPO_IF_MISSING=True` (notebook clones `https://github.com/shanmukhdatta/DermSynReal.git` into `/kaggle/working`). Use a fixed Git SHA for official results, not mutable `main`.
2. Set the notebook **repository path and data inputs**. Each notebook deliberately ships with `EDIT_...` paths and a disabled execution flag. These are intentional safety gates, not fabricated run results.
3. For inference, select **GPU T4**. The standalone inference routines need `torch`, `torchvision`, `timm`, `Pillow`, and optionally PanDerm official dependencies. Match your training notebook package versions; use its original environment rather than blindly upgrading dependencies. Analysis notebooks use ordinary Python/pandas/matplotlib where available.
4. Run 00 only after manual label review. If the Skin Cancer or Normal class includes known wrongly labeled photos, stop and fix the data before training.
5. For each final model checkpoint, run 01 (ResNet50, EfficientNet-B3, ConvNeXt Tiny, DINOv2 ViT-S/14) or 02 (PanDerm official frozen encoder + linear probe). One evaluation produces a six-class `final_test_predictions.csv` and a JSON provenance sidecar. **Do not** import weights from another model because the names look similar.
6. Optional synthetic test is only a diagnostic. Provide a synthetic test manifest with prompt-family IDs and the corresponding training-family manifest to rule out prompt-family overlap. Real test is the single primary evaluation set.
7. Create the run registry from the template at `reports/templates/runs_registry_template.csv`. Each row represents an actual trained checkpoint, condition E0–E5, exact model, training seed, config/checkpoint/split hashes and prediction CSV.
8. Use 04 to compute evidence-backed real-test metrics and paired E2-E0, E2-E1, E3-E2, E4-E2 contrasts whenever both matched runs exist. Calibration (05), manual synthetic quality audit (03), and LaTeX export (06) are independent optional downstream jobs.

## Manifest requirements

`real_test_manifest.csv` — **image_id,image_path,label,split,group_id**; optional `dataset_source`. The images must all have `split=test` and the same stable IDs/groups across model arms. Canonical training label names are `acne, fungal_infection, vitiligo, hyperpigmentation, skin_cancer, normal`.

`synthetic_test_manifest.csv` — same columns **plus prompt_family_id**; prompt families must not overlap those used in synthetic training. Its diagnostics are reported separately and never merged with the real-test primary endpoint.

`final_test_predictions.csv` — includes `image_id,true_label,pred_label,source,split,group_id,dataset_source,prompt_family_id` and, for 01/02, the six `p_*` probability fields. No patient photographs need to enter a public report.

`runs_registry.csv` — `run_id,condition,backbone,seed,predictions_csv,checkpoint_sha,config_sha,real_split_sha`. Distinct runs for each arm/seed; all final comparisons must share the same frozen `real_split_sha`. Pairwise comparison additionally checks actual image IDs, labels and group IDs.

## Missing experimental prerequisites (expected)

Actual clinical-label review, properly licensed facial photos, data-split frozen manifests, compatible checkpoints, generated synthetic pools, full model training, and final real/synthetic predictions **are not provided by these evaluation notebooks**. They are not evidence of clinical validity. Use `docs/CODE_COMPLETION_TRACKER.md` to record runs and blockers.

## CLI equivalents

```bash
PYTHONPATH=src python scripts/infer_timm_checkpoint.py --help
PYTHONPATH=src python scripts/infer_panderm_checkpoint.py --help
PYTHONPATH=src python scripts/audit_generation_quality.py --help
PYTHONPATH=src python scripts/report_experiments.py --help
PYTHONPATH=src python scripts/report_calibration.py --help
```

Read `docs/EXECUTION_GUIDE.md` for the full upstream acquisition, generation, splitting and training sequence.

**Version 2 integration:** Final real test comes from `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb` and its saved, reviewer-approved deduplicated split. Classifier outputs now include stable IDs and six prediction probability columns. E1/E2 generic-vs-morphology prompts retain matched authored families, and the optional E5 synthetic-only arm is supported by the current training manifests. Missing GPU checkpoints and unreviewed labels remain blockers to final metrics.


## Notebook 07 and paper statistics

- `07_Statistical_Analysis_Across_Seeds.ipynb`: reads **actual** Evaluation 04 results, reports training-seed means, sample SD (blank for one seed), medians/ranges, per-class scores and matched E0–E5 contrasts, and produces a plot. These are *not* repeated independent patients. Evaluation 04 already automatically generates the three `seed_summary_*.csv` files, plus paired group-bootstrap 95% intervals.
- Every 00–07 evaluation notebook now preserves `notebook_console.log`, `run_log.jsonl`, output checksums and CSV/JSON reports **when run**. No notebook fabricates class accuracies.
- `scripts/prepare_public_run_report.py` handles only reviewed aggregate CSV/JSON files; never upload private per-image `real_test_errors.csv`, `misclassified_image_ids.csv` or raw console output to public GitHub.


The paper's statistics are computationally prepared but unpopulated: paired group-bootstrap 95% intervals in Evaluation 04, train-seed mean/sample SD and matched arm differences in Evaluation 07, source-specific class metrics, confusion matrices and calibration. Check genuine patient/visual group independence and multiple comparisons before inferential claims; do not invent p-values or present train seeds as independent real test cohorts.
