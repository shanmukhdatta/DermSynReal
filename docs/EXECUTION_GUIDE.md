# DermSynReal — execution guide (29 October 2026)

This is the **only** target repository: https://github.com/shanmukhdatta/DermSynReal.

The paper source is [paper/dermsynreal_manuscript.tex](../paper/dermsynreal_manuscript.tex). Its results are intentionally blank. This guide is a planned run order; it does **not** claim completed Kaggle experiments.

## 0. Confirm existing code and versions

Existing notebooks should be reused instead of recreated:

| Function | Current repository path | Execution status |
| --- | --- | --- |
| Real-data dedup + group split | `notebooks/Preprocessing/Real_Data/DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb` | GPU and image-based audit still required |
| Synthetic prompt-family split | `notebooks/Preprocessing/Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb` | Needs actual generated images |
| Acne, fungal, vitiligo, hyperpigmentation, skin cancer, normal generation | `notebooks/Generation/<class>/01_*.ipynb` | Offline checks in class audit reports; no general claim of Kaggle GPU verification |
| Optional custom LoRA adaptation | `notebooks/Generation/<class>/02_*.ipynb` | Train only on approved real training partitions |
| ResNet-50, EfficientNet-B3, ConvNeXt Tiny, DINOv2 ViT-S/14, PanDerm Base probe | `notebooks/Training/01_*.ipynb` through `05_*.ipynb` | Final test locked until experiment settings are frozen |

**Variant discrepancy to resolve before writing results:** the previously discussed EfficientNet-B0, ConvNeXt **V2** Tiny and DINOv2 ViT-**B**/14 are not the same models as the uploaded EfficientNet-B3, ConvNeXt Tiny and DINOv2 ViT-**S**/14 notebooks. PanDerm's uploaded fifth notebook is a **linear probe**, not full PanDerm fine-tuning. Name models by the exact checkpoint, not the intended future architecture.

Use Kaggle T4 for GPU pipelines. Pin model/revision, package versions, notebook version and random seed. Do not disable safety assertions just to finish a run.

## 1. FIRST: repair the real-label dataset

The earlier audit flagged an apparent Skin Cancer directory containing SJS/TEN (not skin cancer) and a Normal folder with multiple diseases. Replacing the Kaggle dataset name does not change labels. **Stop experiments until images and original diagnoses are reviewed and corrected**; filename-based suspicion alone is not a clinical diagnosis.

Prepare one source-approved CSV, with required fields:

```csv
image_id,source,class_name,original_diagnosis,reviewer_decision,reviewer_id,image_path
IMG0001,licensed_source,Acne,acne vulgaris,approved,reviewer_A,Acne/img0001.jpg
IMG0002,licensed_source,Skin_Cancer,SJS-TEN,rejected,reviewer_A,Skin_Cancer/img0002.jpg
IMG0003,licensed_source,Normal,healthy facial skin,approved,reviewer_A,Normal/img0003.jpg
```

The rows above illustrate **format only**, not verified patient data. The CSV must include all six categories and real review decisions before it can pass. Use `approved`, `rejected` or `needs_review`; missing or suspicious approved records are queued for manual review. This tool checks metadata only; a clinical reviewer must verify uncertain diagnoses.

```bash
PYTHONPATH=src python -m dermsynreal.label_gate \
  --manifest /path/to/reviewed_real_manifest.csv \
  --out /path/to/label_review_output
```

It writes `approved_for_split.csv`, `needs_manual_review.csv`, `label_audit_summary.json`, and exits nonzero if blocked. Only approved and legitimately licensed real images should proceed to the split notebook. **The output is not a clinical certificate**. Record source-specific counts, class disagreements, review criteria and exclusions.

## 2. Frozen real split

Run `notebooks/Preprocessing/Real_Data/DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb` on the corrected six-class dataset; follow its own README for input mounts and outputs. Review exact hashes, pHash/embedding candidate pairs, cross-class conflicts, group assignments and any review gallery. Confirm no visual/patient group spans train, validation and test.

Store the exact split output and audit artifacts in versioned private/Kaggle datasets. Do not shuffle/split again in a training notebook. A visually inferred group is **not** proof of patient identity. The real-test partition is immutable after final approval.

## 3. Checkpoints and controlled synthetic generation

Use each disease-specific `Generation` README and audit report. For a pilot, inspect **15–20 examples per prompt family** at fixed seeds; do not start bulk production until quality and facial morphology are acceptable. Check that each base, textual-inversion adapter, optional LoRA, tokenizer and license are actually compatible. The Normal Skin pathway uses SDXL; do not pretend it is the same generator as the SD2.1 disease pathways.

Run custom LoRA training only with the approved **real-train** subset. Do not use validation/test images even for captions, prompt search, checkpoint choice or memorization checks that feed back into generation. Record prompt ID/family, source class/subtype, morphology, negative prompt, exact model revision, LoRA/TI checkpoint, random seed, scheduler, steps, acceptance/rejection, quality reviewer, and image hashes.

Use the existing `notebooks/Preprocessing/Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb` when an independent synthetic diagnostic set is needed. It requires a prompt-family manifest; read its README. Do not confuse this synthetic-test subset with the primary real-test endpoint.

## 4. Experiment design and frozen config

Use `configs/experiments/main.yaml` as a draft, then version exact configurations for each run. Main comparisons:

- **E0** real-only baseline.
- **E1** real + generic disease-name-prompt synthetic.
- **E2** real + morphology-prompt synthetic, common matched selection.
- **E3** real + quality-selected subset from the **same E2 morphology pool** at matched N.
- **E4** optional diversity-selected subset from that same pool at matched N.
- **E5** optional synthetic-only diagnostic; evaluate real test independently.

E1/E2 compare **prompt design**; E2/E3/E4 compare **selection**. Match class budgets, optimization steps, real data and model settings, and retain seed schedules. If models or generator bases differ, present them as different factors instead of attributing all change to prompt design.

Training notebooks `01`–`05` currently use a manifest with:

```csv
image_path,label,split,group_id,is_synthetic
```

Labels are lowercase `acne,fungal_infection,vitiligo,hyperpigmentation,skin_cancer,normal`. Split is `train`, `val` or `test`. The `is_synthetic` field is boolean; synthetic data may be used for *training only* in these five notebooks. Construct this manifest from **validated split outputs**, not by randomly rescanning unsplit class folders. Maintain a corresponding provenance join to original image ID, source ID, and label-audit approval. An upstream scanner must not silently generate false patient IDs.

### Matched E0–E4 manifest builder (new reusable code)

After auditing all real labels and freezing the real group split, prepare two provenance CSVs:

- `real_split.csv` with `image_id,image_path,label,split,group_id`. **Preserve the original approved image IDs** even if split copies have different paths; join the split report back to the original source manifest. Never assign guessed patient identities.
- `synthetic_pool.csv` with `image_id,image_path,label,prompt_family_id,prompt_type,model_id,accepted`; optionally `quality_score,diversity_cluster_id,split`. Here `prompt_type` is `generic` or `morphology`; each class must have sufficient accepted samples from both types. `model_id` must match between generic and morphology candidates *within each class* to isolate prompt design.

Use the actual `approved_for_split.csv` from the label gate as the approval join:

```bash
PYTHONPATH=src python scripts/build_experiment_manifests.py \
  --real-split /path/real_split.csv \
  --approved-real /path/approved_for_split.csv \
  --synthetic-pool /path/synthetic_pool.csv \
  --n-per-class 100 --seed 42 \
  --quality --diversity \
  --out /path/experiment_manifests
```

The builder writes `E0_training_manifest.csv` through `E4_training_manifest.csv` as requested, and `manifest_build_summary.json` with the selected synthetic image IDs. Omit `--quality` or `--diversity` until all the corresponding metadata is available. It verifies approved real IDs, mapped labels, group isolation, six-class split coverage, and matched per-class synthetic counts. It does **not** substitute for near-duplicate image analysis, generator quality review or clinical verification. E5 (synthetic-only) requires a separate classifier runner because the five uploaded classifier notebooks currently require real train/val/test partitions.

## 5. Train and export real-test predictions

Start with the actual real-only baseline and keep `RUN_FINAL_TEST=False` while selecting hyperparameters. Apply frozen splits, train-only augmentation, pretrained checkpoint and correct model-specific preprocessing, best-checkpoint reload and validation macro-F1. Do not call results clinically validated.

After **all** arms/seeds/configs are frozen, enable `RUN_FINAL_TEST=True` once for each final run. The five uploaded model notebooks now have an additional cell that checks the saved NPZ predictions against test-manifest order and writes `FINAL_real_test_predictions.csv` with:

```csv
image_id,true_label,pred_label,source,split,group_id
```

Those image IDs are stable SHA-256 hashes of the existing `image_path` strings. All comparable arms must use the **same exact manifest paths** and groups. The exporter asserts row count, label order and six-class probability shape. It **does not** recover identity if the input manifests changed between arms. This export cell runs only when the final test flag is enabled. Re-run the model notebook after obtaining valid data; code alone does not produce predictions.

The existing self-contained `DermaLens_ResNet50_Kaggle_Training.ipynb` offers separate real/synthetic test reporting; if using it, export the same column schema and preserve prompt-family separation before comparing its outputs.

## 6. Evaluate, then fill manuscript tables

Using the exact two CSVs from matched final runs:

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --predictions /path/E0/FINAL_real_test_predictions.csv \
  --treatment /path/E2/FINAL_real_test_predictions.csv \
  --bootstrap 2000 --seed 42 \
  --out /path/metrics/E2_vs_E0.json
```

The script verifies identical real-test IDs, labels and resampling-group IDs and reports six-class macro-F1, balanced accuracy, confusion matrix, per-class metrics, and paired bootstrap delta/95% percentile interval. It returns **no invented p-value**. If synthetic-test rows are provided in another properly labeled CSV, it reports them under a **different source key**; no pooled real/synthetic metric.

Resampling group IDs must be characterized as actual patient IDs or visually inferred protective groups. Do not interpret image-level bootstrap as patient-independent certainty when identities may overlap. Keep trained seeds and patient/image bootstrap uncertainty separate. For source-stratified metrics, retain source metadata through the real-data join; the basic CSV's `source` field distinguishes real/synthetic only, not originating atlas. Multiple-comparison adjustment and calibrated interval estimation require a separately documented statistics plan if used.

Then fill **only observed data** in `paper/dermsynreal_manuscript.tex`: source counts, audited synthetic acceptance, real-test results, per-class and real-vs-synthetic diagnostics, paired comparisons and limitations.

## 7. Offline code checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_research_gates.py -v
python -m compileall -q src/dermsynreal/evaluation.py src/dermsynreal/label_gate.py scripts/evaluate_predictions.py
```

GitHub Actions contains a lightweight `research-gates.yml`. It does not run GPU generation, clinical label verification or accuracy benchmarks. Save Kaggle outputs and update `docs/CODE_COMPLETION_TRACKER.md` with actual run URLs, exact SHA, configs and blockers for every phase.

**Do not rerun expensive training while label quality is unresolved.**

## Sole real collection v2 — exact Kaggle execution paths

`/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_dataset` contains the original six folders `Acne`, `Fungal_infection`, `Hyper_Pigmentation`, `Normal_Skin`, `Skin_cancer`, `Vitiligo`. Use `notebooks/Preprocessing/Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb` for discovery, SHA duplicate inventory, diagnosis/source review, and the frozen global protected split. These real photos are **not** accessed by generators directly; LoRA training requires the approved frozen *train* subset and audit. All six 300-prompt banks are now loaded automatically with generic E1 or morphology E2 modes, 210/45/45 disjoint prompt partitions and unapproved generated-image CSV output.

Recommended run order: (1) inventory + human labels + licensing; (2) dedup/group split and review; (3) real-only E0 head/partial training smoke test; (4) E1 generic and E2 morphology 5-prompt pilot with morphology review; (5) optional real TRAIN-only LoRA; (6) all 210 training-family prompts and reviewed synthetic pools; (7) E1–E5 run manifests, fixed-budget classifier training across seeds; (8) real held-out evaluation and publication figures. Results stay blank until observed. For approved aggregate GitHub reports see `docs/RUN_RESULTS_HANDOFF.md`.


## Final verified-input synthetic and E0–E5 handoff (code, not executed)

1. Generate generic or morphology images with the authored prompt-family labels; review images and complete `accepted=True,reviewer_decision=approved,reviewer_id` CSVs. For a full synthetic diagnostic split, separately generate the originally allocated train/val/test prompt families.
2. Attach saved generator output folders; run `notebooks/Preprocessing/Synthetic_Data/00_Combine_Reviewed_Base_And_LoRA_Generations.ipynb`, using `ALLOW_MIXED_GENERATORS=True` if you need one **preprocessing** pool containing both base and LoRA, preserving their adapter metadata.
3. Run `notebooks/Preprocessing/Synthetic_Data/DermaLens_Synthetic_Prompt_Split_Kaggle.ipynb` with the combined reviewed CSV, exact hashes and original prompt partitions. It now emits `reports/synthetic_train_pool_for_experiments.csv`, a ZIP, leakage audit and class split counts. Attach both its CSV and saved image root for the next step.
4. Run `PYTHONPATH=src python scripts/build_experiment_manifests.py --real-split <reviewed-real-all.csv> --approved-real <reviewed-approved.csv> --synthetic-pool <synthetic_train_pool_for_experiments.csv> --synthetic-root <mounted-synthetic-split-directory> --generator-mode <matched-mode> --n-per-class <budget> --out <output> --quality --diversity --synthetic-only`. The flags `--quality` and `--diversity` need measured quality scores/clusters; omit until actually reviewed/computed. Every E1/E2 comparison must have matching base model/adapter revisions and synthetic budgets.
5. Train each of the five classifier notebooks using the produced real/synthetic manifest and original approved real image root. Save checkpoint/validation/log artifacts in Kaggle. Run Evaluation 01/02 on the saved checkpoint, Evaluation 04 to generate paired real-test bootstrap CIs and seed summaries, Evaluation 07 for statistical plots. Publish only reviewed aggregate results via the safe export script and explicit GitHub commit.

All steps require real Kaggle execution; a successful static CI run proves only code syntax/unit checks, not GPU/clinical behavior.


**1,500 image allocation:** 300 authored prompts/class × 5 images = 1,500 only when **train (210)/val (45)/test (45)** prompt partitions are each generated separately (1,050 / 225 / 225 before acceptance). `PROMPT_LIMIT=None` on `train` alone generates 1,050, not 1,500. Output paths now include variant, partition and offset; preserve the prompt-family metadata when merging reviewed base/LoRA outputs. All generated samples require review.
