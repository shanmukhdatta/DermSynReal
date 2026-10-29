# Code completion tracker

> This records **repository code availability** separately from completed **scientific experiments**. A notebook present on GitHub or a successful offline test does not prove a Kaggle run, clinical correctness or an accurate classification result. Updated 29 October 2026.

| Phase | Code status | Scientific execution | Evidence / next action |
| --- | --- | --- | --- |
| P0 infrastructure | Existing scaffold, real generation/training notebooks uploaded, offline CI added | **Kaggle runtime not verified** | Run a real Kaggle environment/integration check; freeze requirements and seed manifest |
| P1 real-label + dedup | Real group split notebook present; metadata label gate added | **BLOCKED** until incorrect Skin Cancer and Normal labels are replaced/reviewed | Review diagnosis/source and rerun real split; record group, source and class counts |
| P2 generation pilot | Six class-pathway inference notebooks and offline audit notes exist | **Actual GPU morphology pilot not verified** | Run 15–20 images per prompt family, assess compatibility and clinical morphology |
| P3 synthetic build | Prompt libraries, custom LoRA and synthetic family split notebooks exist | **Synthetic accepted pools not verified** | Generate image IDs, revisions, prompt IDs, QA verdicts and near-copy audit |
| P4 classifier training | ResNet-50, EfficientNet-B3, ConvNeXt Tiny, DINOv2 ViT-S/14, PanDerm Base probe notebooks exist; final-test CSV export cell added to each | **No verified final Kaggle training/test results** | Run fixed E0 baseline, then matched E1–E5 and seeds; verify checkpoint reload |
| P5 evaluation | Seven new Kaggle evaluation notebooks; checkpoint inference for four timm backbones and official PanDerm probe; source/class, paired, calibration and morphology QA modules; CSV schemas; expanded CI unit tests | **Offline code checks only; no GPU evaluation** | Supply true frozen predictions, approved audit and source metadata; populate paper without inventing scores |
| P6 research extensions | Low-data and ablation protocol written | Not executed | Evaluate only after primary comparisons are validated |
| Paper | Complete LaTeX draft with methods, limitations, blank tables, verified reference list | No experimental results; LaTeX successfully compiled in GitHub Actions (run 40839443094) | Review paper after real runs; add figure panels and statistical results |

## Changes committed to this repository on 29 October 2026

- `paper/dermsynreal_manuscript.tex`: draft full research paper; all scores blank.
- `docs/EXECUTION_GUIDE.md`: ordered Kaggle execution, datasets, safety and reporting requirements.
- `src/dermsynreal/label_gate.py`: metadata-driven, explicit-review gate; not automatic clinical diagnosis.
- `src/dermsynreal/evaluation.py` and `scripts/evaluate_predictions.py`: source-separated six-class metrics and paired real-test confidence interval.
- `src/dermsynreal/experiment_manifest.py` and `scripts/build_experiment_manifests.py`: create controlled matched E0--E4 train manifests after approved-ID and real-split checks.
- `tests/test_research_gates.py` and `.github/workflows/research-gates.yml`: no-GPU validation.
- Five `notebooks/Training/0[1-5]_*.ipynb`: only append final-test CSV export, preserve existing training procedures.
- Seven `notebooks/Evaluation/0[0-6]_*.ipynb`: Kaggle-ready label/split preflight, frozen inference, generation QA, real/synthetic evaluation, calibration and paper export.
- `src/dermsynreal/{inference,panderm_inference,evaluation_pipeline,calibration,quality_audit}.py`: reusable evaluation components; corresponding `scripts/` CLI wrappers and `tests/test_evaluation_workflow.py`.
- Empty evaluation CSV templates in `data/manifests/` and `reports/templates/`; protocol in `docs/EVALUATION_PROTOCOL.md`.
- `README.md`: links to current artifacts and correct model variants.

**Verified LaTeX compilation:** https://github.com/shanmukhdatta/DermSynReal/actions/runs/40839443094 (PDF artifact available on that run page).\n\n**Verified GitHub Actions:** https://github.com/shanmukhdatta/DermSynReal/actions/runs/40839301076 — status SUCCESS on commit `5f1c9830bb163d9190ece7d7c60110fe67770d2c`. This workflow checks pure Python unit tests and JSON validity; it does not run models on real clinical images.

**Model-name discrepancy:** current uploaded notebooks are EfficientNet-B3, ConvNeXt Tiny, and DINOv2 ViT-S/14 (not the newer proposed EfficientNet-B0 / ConvNeXt V2 Tiny / DINOv2 ViT-B/14); PanDerm is a linear probe, not full fine-tuning. Correct this through separate evaluated notebook implementations if those variants become the experimental protocol.

## Journal entry template

- Date / phase / code commit SHA:
- Kaggle notebook URL/version and runtime:
- Dataset source licenses and approved manifest hash:
- Generator checkpoint + adapter versions:
- Exact config hash, seeds, class counts:
- Checkpoint hash and output prediction file:
- Unit tests / GPU checks / clinical QA performed:
- Blockers and next phase:

## Immediate next action

**P1:** review and replace mislabeled Skin Cancer and Normal data; run metadata gate on the reviewed source manifest, then rerun approved real-image grouping and immutable group-aware splitting. **Do not train with the known suspect real dataset.**

## Evaluation checkpoint (29 October 2026)

Seven Evaluation notebooks and their supporting modules/configs/report schemas are committed. Unit tests and notebook AST validation are configured in GitHub Actions. Kaggle dependencies, real checkpoint loading, source reviews, image QA, model output distributions and research metrics **have not been executed**. The next data-dependent gate remains clinical-label review and P1 real splitting. Check the latest GitHub Actions run for pass/fail evidence; do not infer a pass merely because a workflow was created.

## Kaggle v2 code-integration checkpoint (29 October 2026)

- **Sole real dataset configured:** `shanmukhadattaboda/facial-skin-conditions-real-image-collection` v2, six uploaded folders (1,457 Acne; 576 Fungal; 1,077 Hyperpigmentation; 2,792 Normal; 1,597 Skin Cancer; 2,292 Vitiligo). **These are Kaggle listing counts only; duplicates, diagnostic labels and licenses are not verified.**
- **Preprocessing:** `Real_Data/00_Facial_Real_V2_Kaggle_Start_Here.ipynb` inventories SHA duplicates and source/label flags, requires reviewed-label approval before SHA/pHash/dHash/ResNet/ORB grouping and frozen train/val/test. The join writes canonical `real_all_manifest.csv` for classifier/LoRA use. Synthetic_Data notebook preserves original prompt-family splits and accepts only reviewed synthetic manifests.
- **Generation:** all six `01_` inference notebooks import their 300-prompt bank via `kaggle_runs.py`; `PROMPT_VARIANT` selects E1 generic or E2 morphology with matched families and original negative prompts. CSV+JSON+JSONL and SHA report outputs follow each run. Pilot defaults to five train prompts.
- **LoRA:** all six `02_` training notebooks accept only passed-audit real TRAIN partitions from the sole Kaggle v2 dataset. Fungal and skin-cancer subtype LoRAs additionally require reviewer-verified subtype CSV. Console output saved; frozen model weights and runs logged. Original Kaggle raw folders are *not* directly used for training.
- **Training:** exact current five models; the four timm classifiers now offer head → partial last-stage → optional full fine-tune and fixed optimizer-update budgets for E0–E5. The PanDerm baseline is a frozen linear probe; it is not full PanDerm fine-tuning. E5 is an optional synthetic-only arm with unchanged real validation/test. No GPU benchmark has yet passed.
- **Results:** aggregation and paired evaluation remain test-locked and data dependent. `scripts/prepare_public_run_report.py` safely stages whitelisted aggregate reports for deliberate GitHub updates. Private photos, original image IDs, raw logs, checkpoints and bulk synthetic images should stay in private Kaggle storage.
- **Next action:** run the image/label review and inventory, inspect suspect classes and original image rights, then first GPU pretrained ResNet/group-split Kaggle smoke test. Do **not** claim validated research experiments from static CI tests.


## Evidence and statistical analysis update (29 October 2026)

- `src/dermsynreal/kaggle_runs.py` now emits each LoRA class's **global frozen** train/val/test counts and private assignments, stages *train only*, and enables private full notebook console logging.
- All six `Generation/**/02_*.ipynb` LoRA notebooks save these split counts and trainer + notebook run logs; all six diffusion generation notebooks save notebook console logs; five primary training notebooks and the older ResNet50 trainer save their own console/provenance logs; evaluation 00–07 save console and structured artifact metadata when executed.
- `notebooks/Preprocessing/Synthetic_Data/00_Combine_Reviewed_Base_And_LoRA_Generations.ipynb` and `src/dermsynreal/combine_synthetic.py` fill the missing approved generation-output combination step without silently mixing incompatible generators or skipping manual review.
- Evaluation 04 now automatically writes across-seed tables; Evaluation 07 graphs/reports seed variability. Group bootstrap paired 95% confidence intervals remain in Evaluation 04. No unsupported p-values are invented.
- `scripts/prepare_public_run_report.py` and `reports/templates/run_evidence_ledger_template.csv` enable *explicit* public GitHub reporting of reviewed aggregate metrics; per-image data and raw case logs remain in private Kaggle storage.
- **Unverified:** no real Kaggle GPU generation/training/inference, dataset label audit or medical evidence has been completed in this chat. Static CI is not a substitute for clinical review or completed experiments.
