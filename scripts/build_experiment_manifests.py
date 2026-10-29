"""CLI for matched DermSynReal E0-E5 experimental manifest creation.

Example:
PYTHONPATH=src python scripts/build_experiment_manifests.py \
 --real-split audit/real_split.csv \
 --approved-real audit/approved_for_split.csv \
 --synthetic-pool audit/synthetic_pool.csv \
 --n-per-class 100 --out outputs/manifests --quality --diversity --synthetic-only
"""
from dermsynreal.experiment_manifest import main

if __name__ == "__main__":
    main()
