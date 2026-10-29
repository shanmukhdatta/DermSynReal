# DermSynReal — complete research manuscript source

This is the **new, stand-alone LaTeX research paper**, separate from the earlier protocol draft at \`paper/dermsynreal_manuscript.tex\`. The main source is \`main.tex\`. The prose sections are original academic writing based on the repository's actual six-class research design, its current training/evaluation modules, and verified research references. **It is a full theoretical and methodological draft, not a claim of completed experiments.**

## Compile

From this folder on Overleaf or a local TeX installation:

\`\`\`bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
\`\`\`

The CI workflow also builds \`paper/manuscript/main.tex\` and archives the compiled PDF artifact. Keep \`references.bib\` and \`sections/\` beside \`main.tex\` in Overleaf. Use a conventional natbib-compatible LaTeX installation.

## What is already written

| File | Contents | Status |
| --- | --- | --- |
| \`main.tex\` | Full abstract, study metadata, manuscript structure | Complete factual pre-results draft |
| \`sections/01_introduction.tex\` | Motivation, problem, RQ1–RQ3, contribution | Complete |
| \`sections/02_related_work.tex\` | Dataset work, diffusion synthesis, pretrained backbones, gap | Complete |
| \`sections/03_data.tex\` | Six-class taxonomy, real-image provenance, label review, hashes and visual-group splitting | Complete theory |
| \`sections/04_generation.tex\` | Prompt-bank design, generic/morphology variants, SD2.1/SDXL, LoRA, QA, workflow diagram | Complete theory |
| \`sections/05_classification.tex\` | Exact five current backbones, progressive fine-tuning, E0–E5 experimental arms | Complete theory |
| \`sections/06_evaluation.tex\` | Held-out real macro-F1, paired differences, bootstrap, seed summaries, subgroup and calibration reporting | Complete theory |
| \`sections/07_results.tex\` | Four empty tables, actual-figure slots, results synthesis | **Fill after measured runs** |
| \`sections/08_discussion.tex\` | Interpretation framework, anticipated limitations, separate final result-dependent discussion paragraph | Theory complete; final paragraph pending |
| \`sections/09_reproducibility.tex\` | Ethics, data governance, reproducibility, AI-assistance disclosure, conclusion | Theory complete; final measured sentence pending |
| \`references.bib\` | Published research and dataset references | Included; final citation check required before submission |

## How to complete the paper after Kaggle runs

1. **Complete the real label audit.** Use \`notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb\` and verify the actual real-image licensing and approved class mapping. Only then fill Table 1 of the new Results section. The uploaded folder image count is **not** an accepted diagnostic dataset count.
2. **Generate and review the synthetic images.** The six \`notebooks/Generation/\` class folders produce metadata and image files. Review candidate clinical morphology and near-copies. Use the combined synthetic manifest and \`notebooks/Preprocessing/Synthetic_Data/\` to obtain the partitioned, reviewed training pool. Fill Table 2 from accepted/rejected summaries; do not quote the planned 1,500 candidates as a measured accepted count.
3. **Train E0–E5.** \`notebooks/Training/\` contains the current ResNet-50, EfficientNet-B3, ConvNeXt Tiny, DINOv2 ViT-S/14 and PanDerm Base frozen linear-probe training files. Record actual stages and software/weight revisions. The newer PanDerm experiment is a linear probe, not a PanDerm full fine-tune.
4. **Run checkpoint inference and evaluation.** \`notebooks/Evaluation/04_Experiment_Suite_Real_Test_Reporting.ipynb\` produces \`run_summary.csv\`, \`per_class_real_test.csv\`, \`paired_real_test.csv\`, confusion and source reports, plus \`seed_summary_*.csv\`. \`07_Statistical_Analysis_Across_Seeds.ipynb\` provides a secondary analysis of variability and plots. Use only verified real-test results, with the same real test and group IDs.
5. **Replace the dash cells and pending lines** in \`sections/07_results.tex\`, then update the final result-dependent paragraph in \`sections/08_discussion.tex\` and final measured conclusion sentence in \`sections/09_reproducibility.tex\`. When measurements exist, revise the *abstract* in \`main.tex\` so it states the observed direction, effect magnitude and limitations rather than simply the protocol.
6. **Insert figures** in the Results section only after generating them and checking clinical privacy/permissions. A confusion matrix may be inserted with \`\\includegraphics[width=0.86\\linewidth]{figures/confusion_real_test.pdf}\` (place your own generated PDF in a new \`figures/\` directory); use caption text stating dataset domain, backbone, seed and sample count. Do not invent figure filenames in the manuscript before those files exist.
7. Confirm author/co-author contributions, institution, venue formatting, acknowledgment permissions, AI-assistance disclosure and citation accuracy. **No ethical approval or clinical validation is claimed unless actually documented.**

## Manuscript-to-result correspondence

| Paper item | Evidence expected |
| --- | --- |
| Real reviewed class counts | \`real_all_manifest.csv\`, real split/audit summary, reviewed labels |
| Synthetic accepted count and morphology QA | Reviewed \`generation_manifest.csv\`, quality summary |
| E0–E5 real-test metric table | \`run_summary.csv\` (exact backbone/condition/seed) |
| Paired delta and 95% CI | \`paired_real_test.csv\` from identical real-test groups |
| Per-class error and domain breakdown | \`per_class_real_test.csv\`, \`confusion_real_test.csv\`, \`per_source_real_test.csv\` |
| Seed mean/SD and differences | \`seed_summary_real_test.csv\`, \`seed_summary_paired_contrasts.csv\` |
| Checkpoint provenance | \`artifact_inventory.csv\`, \`run_log.jsonl\`, run evidence ledger |
| Figure provenance | Actual exported plot artifact and matching model/split SHA |

## Scientific boundaries

- Never describe images from unlabeled or incorrectly labeled source folders as diagnosed cases.
- The broad skin-cancer label needs a prespecified choice about the inclusion/exclusion of premalignant actinic keratosis.
- A visually inferred duplicate group does not prove a patient-independent holdout.
- An improvement in image realism is not evidence of real-image classifier improvement.
- Do not pool synthetic-test scores with the primary real-test score.
- Public GitHub should contain **only reviewed aggregate metrics, code, manuscript and safe provenance**. Keep clinical photos, patient-identifying metadata, per-image failure lists, original image paths, checkpoint weights and sensitive logs in appropriately controlled storage.
- Because this study has not yet been executed in Kaggle, its results and empirical conclusion remain empty. Complete measured results first, then perform an independent scientific and language edit before journal or conference submission.

## Previous draft

The original \`paper/dermsynreal_manuscript.tex\` is preserved for comparison. This folder is the recommended full-length manuscript to continue editing after experiments.
