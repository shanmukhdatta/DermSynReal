# Image manifests

Required schema (initial): `image_id,source_id,relative_path,sha256,label,subtype,group_id,group_id_kind,split,license_status`.

`group_id_kind` must distinguish `verified_patient`, `inferred_visual` and `unknown`.

Use pseudonymous IDs and permitted relative paths; do not store real names, patient IDs or addresses in the repository. CSV/Parquet schema and validators will follow in P1.
