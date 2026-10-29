# DermaLens — Kaggle real-image deduplication and group split

**Open [`DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb`](DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb) in Kaggle.**

1. Click **Add Input** to attach your real-image dataset(s); six class folders are required. The notebook's configuration cell accepts independent folder paths per class. Use `/kaggle/input/...`, not public dataset URLs. Subdirectories inside each class are supported.
2. Choose GPU (optional but helpful). Enable Kaggle **Internet** (if available) for the official ImageNet ResNet-50 V2 pretrained weights; or attach `resnet50-11ad3fa6.pth` with **Add Input**, and set `RESNET_WEIGHTS_PATH` to the mounted file. This code never substitutes untrained embeddings during research runs.
3. Edit only the config cell with six folder paths. Click **Run All**, then use Kaggle notebook **Output** to download `/kaggle/working/facial_real_split.zip`.
4. Final ZIP contains original-byte copies under `train/`, `val/`, `test/` (each with all six classes), and `reports/` with inventory, removed pixel copies, protected group assignments, quarantine, candidate pairs, review gallery, split statistics, source distribution, and leakage audit.

## Default matching rules (tunable, not clinically validated)

- SHA-256: byte-identical. Pixel hash: identical EXIF-normalized RGB pixels.
- 64-bit pHash, four orientations: Hamming <=5 generates a *protective split link*; strong similarity does not prove patient identity.
- Horizontal and vertical dHash, each <=4 **plus** ResNet cosine >=0.95 generates a protective split link.
- ResNet-50 V2, 2048-dimensional normalized embeddings: top-15 cosine neighbors >=0.90 are review candidates; >=0.985 creates a precautionary protective link.
- ORB geometric verification is attempted on up to 1,000 higher-cosine uncertain candidates; matched geometry creates a protective link.
- Union-Find ensures each protected component belongs to exactly one split. Exact pixel duplicates collapse to a single representative. Near-duplicates *remain present*, together in one split. Cross-class protected groups are quarantined because a single-label classifier cannot safely assign them without review.
- A fixed seed 42, group-aware approximate 70/15/15 split, six-class presence checks, and SHA-256 verification of exported bytes.

## Cautions

- **Cannot guarantee zero leakage**: patient IDs may be absent, visual detection is heuristic, and ResNet retrieval is capped at top-k. Check `review_candidates_cross_split.csv`, especially against disease/source artifacts. Strict patient-disjoint datasets require usable patient IDs or manual annotations.
- **No hidden overwrite:** if output exists, change `OUTPUT_DIR` or remove the prior output intentionally before rerunning.
- If a large visual cluster is formed or too many conflicting labels are quarantined, the notebook **stops** with `reports/STOP_REASON.json`, so you can inspect and adjust rather than unknowingly train on a broken split. An accepted ZIP is not produced in that case.
- Highly similar images from different *people* can trigger group links; review giant groups, class conflicts and candidates before publication.
- `resnet_weights_path` must be ImageNet-1K V2 ResNet-50 state_dict; pass no local path and enable Internet to use Torchvision's verified automatic weight loading.
- Train-time random crop/flips/color jitter must be applied **only after splitting** in the later training notebook; no augmentation is done here.

## Files

- `DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb`: fully self-contained Kaggle notebook; only this file is required.
- `pipeline.py`: optional identical module for local testing/inspection.
