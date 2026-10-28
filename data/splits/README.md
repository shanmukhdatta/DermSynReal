# Frozen splits

P1 will create group-aware, stratified train/val/test manifests *after* deduplication. Preserve their hashes and Git SHA. Never tune split assignments based on the held-out test score.
