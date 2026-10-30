# Real Image Preprocessing — Current Kaggle Workflow

**Current notebook:** [DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb](DermaLens_Real_Dedup_GroupSplit_Kaggle.ipynb) — replaced with the 26-cell real-data grouping and split notebook executed on Kaggle account 069.

See [WORKFLOW_OVERVIEW.md](WORKFLOW_OVERVIEW.md) for the processing algorithm, expected outputs, instructions, limitations and recorded summary.

## Dataset sources

- **Raw:** [Facial Skin Conditions — Real Image Collection](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/data) (9,791 image files).
- **Processed:** [Facial Dermatology Image Dataset](https://www.kaggle.com/datasets/shanmukhadattaboda/facial-dermatology-image-dataset/data).
- **Kaggle input root:** `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_datset`

Original class directories are `Acne`, `Fungal_infection`, `Hyper_Pigmentation`, `Normal_Skin`, `Skin_cancer` and `Vitiligo`. Preserve these spellings for direct use with the exported image folders.

## Recorded output from Kaggle 069

| Stage | Count |
|---|---:|
| Source images | 9,791 |
| Automated quality exclusions | 0 |
| Quarantined conflicting-label images | 6 |
| Eligible images | 9,785 |
| Final similarity groups | 7,428 |
| Train | 6,908 |
| Validation | 1,420 |
| Test | 1,457 |

The run selected cosine threshold 0.90 and used seed 42. The split passed the recorded path/hash/group isolation checks. Large same-class similarity clusters, clinical labels and source licenses still require review.

## Which notebook should I run?

Run **only the current standalone notebook** above for reproduction of the Kaggle 069 preprocessing procedure. The older `00_Facial_Real_V2_Kaggle_Start_Here.ipynb` and `pipeline.py` are preserved as an alternate review-first route; they are not the recorded 069 run and should not generate competing splits for this study.

Use the frozen 70/15/15 output directly in classifier training notebooks; **do not rerun grouping or independently split again**. Train LoRA adapters on real train images only, never validation/test. Keep derived images and image-level private manifests outside GitHub.

**Important:** Duplicate-aware image grouping is not clinical ground truth or patient-level independence.
