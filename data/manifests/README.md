# Image manifests

Required schema (initial): `image_id,source_id,relative_path,sha256,label,subtype,group_id,group_id_kind,split,license_status`.

`group_id_kind` must distinguish `verified_patient`, `inferred_visual` and `unknown`.

Use pseudonymous IDs and permitted relative paths; do not store real names, patient IDs or addresses in the repository. CSV/Parquet schema and validators will follow in P1.

## Ready-to-fill evaluation schemas

- `real_split_manifest_template.csv`: stable real `image_id,image_path,label,split,group_id`.
- `real_test_manifest_template.csv`: held-out real images plus `dataset_source` for real-source reporting.
- `synthetic_test_manifest_template.csv`: independent synthetic test plus `prompt_family_id`.
- `generation_quality_review_template.csv`: actual manual morphology rating and near-copy review.

All templates have **only headers**, not fabricated subjects or results. Labels must match the exact six training labels, and every final test prediction must have explicit `group_id`. Use pseudonymous IDs and no identifying clinical photos. Metadata checks alone cannot guarantee diagnostic accuracy or patient-independent samples.
