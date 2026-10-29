# Code completion tracker

> This records **repository code availability** separately from completed **scientific experiments**. A notebook present on GitHub or a successful offline test does not prove a Kaggle run, clinical correctness or an accurate classification result. Updated 29 October 2026.

| Phase | Code status | Scientific execution | Evidence / next action |
| --- | --- | --- | --- |
| P0 infrastructure | Existing scaffold, real generation/training notebooks uploaded, offline CI added | **Kaggle runtime not verified** | Run a real Kaggle environment/integration check; freeze requirements and seed manifest |
| P1 real-label + dedup | Real group split notebook present; metadata label gate added | **BLOCKED** until incorrect Skin Cancer and Normal labels are replaced/reviewed | Review diagnosis/source and rerun real split; record group, source and class counts |
| P2 generation pilot | Six class-pathway inference notebooks and offline audit notes exist | **Actual GPU morphology pilot not verified** | Run 15–20 images per prompt family, assess compatibility and clinical morphology |
| P3 synthetic build | Prompt libraries, custom LoRA and synthetic family split notebooks exist | **Synthetic accepted pools not verified** | Generate image IDs, revisions, prompt IDs, QA verdicts and near-copy audit |
| P4 classifier training | ResNet-50, EfficientNet-B3, ConvNeXt Tiny, DINOv2 ViT-S/14, PanDerm Base probe notebooks exist; final-test CSV export cell added to each | **No verified final Kaggle training/test results** | Run fixed E0 baseline, then matched E1–E5 and seeds; verify checkpoint reload |
| P5 evaluation | Strict per-source metrics, real-test paired bootstrap CLI and unit tests committed | **Offline code checks only** | Supply real predictions from frozen runs; populate paper tables without inventing scores |
| P6 research extensions | Low-data and ablation protocol written | Not executed | Evaluate only after primary comparisons are validated |
| Paper | Complete LaTeX draft with methods, limitations, blank tables, verified reference list | No experimental results; TeX compilation still needs Overleaf/CI check | Review paper after real runs; add figure panels and statistical results |

## Changes committed to this repository on 29 October 2026

- `paper/dermsynreal_manuscript.tex`: draft full research paper; all scores blank.
- `docs/EXECUTION_GUIDE.md`: ordered Kaggle execution, datasets, safety and reporting requirements.
- `src/dermsynreal/label_gate.py`: metadata-driven, explicit-review gate; not automatic clinical diagnosis.
- `src/dermsynreal/evaluation.py` and `scripts/evaluate_predictions.py`: source-separated six-class metrics and paired real-test confidence interval.
- `tests/test_research_gates.py` and `.github/workflows/research-gates.yml`: no-GPU validation.
- Five `notebooks/Training/0[1-5]_*.ipynb`: only append final-test CSV export, preserve existing training procedures.
- `README.md`: links to current artifacts and correct model variants.

**Verified GitHub Actions:** https://github.com/shanmukhdatta/DermSynReal/actions/runs/40839301076 — status SUCCESS on commit `5f1c9830bb163d9190ece7d7c60110fe67770d2c`. This workflow checks pure Python unit tests and JSON validity; it does not run models on real clinical images.

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
