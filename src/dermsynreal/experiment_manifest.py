"""Prepare matched, leakage-controlled E0-E4 CSV manifests for existing notebooks.

Nothing is trained or evaluated here. Never select with real validation/test scores.
Requires the audited real image ID mapping and a generator-provenance table.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from dermsynreal.evaluation import CLASSES, _label


def _read(path, required):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path} missing fields: {set(required) - set(reader.fieldnames or [])}")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _unique(rows, source):
    ids, paths = set(), set()
    for row in rows:
        if not row["image_id"] or row["image_id"] in ids:
            raise ValueError(f"Missing or repeated {source} image_id: {row['image_id']!r}")
        if not row["image_path"] or row["image_path"] in paths:
            raise ValueError(f"Missing or repeated {source} image_path: {row['image_path']!r}")
        ids.add(row["image_id"])
        paths.add(row["image_path"])


def _training_row(r, synthetic):
    return {
        "image_path": r["image_path"],
        "label": _label(r["label"]).lower(),
        "split": "train" if synthetic else r["split"],
        "group_id": f"synthetic_{r['prompt_family_id']}" if synthetic else r["group_id"],
        "is_synthetic": "True" if synthetic else "False",
    }


def prepare(
    real_split_csv, approved_csv, synthetic_pool_csv, out_dir, *,
    budget_per_class: int, seed: int = 42,
    include_quality: bool = False, include_diversity: bool = False,
    include_synthetic_only: bool = False,
):
    if budget_per_class < 1:
        raise ValueError("Synthetic budget per class must be positive")
    real = _read(real_split_csv, {"image_id", "image_path", "label", "split", "group_id"})
    approved = _read(approved_csv, {"image_id", "class_name", "reviewer_decision"})
    synth = _read(synthetic_pool_csv, {
        "image_id", "image_path", "label", "prompt_family_id",
        "prompt_type", "model_id", "accepted",
    })
    _unique(real, "real")
    _unique(synth, "synthetic")
    if {r["image_id"] for r in real} & {s["image_id"] for s in synth}:
        raise ValueError("A synthetic image_id collides with a real image_id")
    if {r["image_path"] for r in real} & {s["image_path"] for s in synth}:
        raise ValueError("A synthetic image_path collides with a real image_path")
    by_id = {}
    for record in approved:
        key = record["image_id"]
        if not key or key in by_id:
            raise ValueError(f"Duplicate approved ID {key!r}")
        by_id[key] = record
    groups = defaultdict(set)
    for row in real:
        record = by_id.get(row["image_id"])
        if record is None or record["reviewer_decision"].lower() != "approved":
            raise ValueError(f"Real ID lacks approved original label: {row['image_id']}")
        if _label(record["class_name"]) != _label(row["label"]):
            raise ValueError(f"Real diagnosis mapping differs for {row['image_id']}")
        if row["split"] not in {"train", "val", "test"} or not row["group_id"]:
            raise ValueError(f"Missing split or protective group: {row['image_id']}")
        groups[row["group_id"]].add(row["split"])
    if any(len(s) != 1 for s in groups.values()):
        raise ValueError("A real protective group crosses the train/val/test boundary")
    for split in ("train", "val", "test"):
        seen = {_label(r["label"]) for r in real if r["split"] == split}
        if set(CLASSES) != seen:
            raise ValueError(f"Real {split} split missing classes: {set(CLASSES) - seen}")
    real_out = [_training_row(r, False) for r in real]
    rng = random.Random(seed)
    pool = defaultdict(lambda: defaultdict(list))
    model_check = defaultdict(lambda: defaultdict(set))
    for row in synth:
        if row["accepted"].lower() not in {"yes", "true", "1"}:
            continue
        if not row["prompt_family_id"] or not row["model_id"]:
            raise ValueError(f"Accepted synthetic ID missing provenance: {row['image_id']}")
        group = row["prompt_type"].lower()
        if group not in {"generic", "morphology"}:
            raise ValueError(f"Unknown prompt_type: {group}")
        if row.get("prompt_partition", "train") not in {"", "train"}:
            continue
        if row.get("split", "train") not in {"", "train"}:
            # Optional diagnostic synthetic test data belongs elsewhere.
            continue
        cls = _label(row["label"])
        pool[cls][group].append(row)
        # To isolate prompt wording (E1 vs E2), compare the same generator
        # backbone AND adapter mode/revision. Mixing base, TI and LoRA runs
        # would confound the experiment even if the backbone name is identical.
        model_check[cls][group].add((
            row["model_id"],
            row.get("generator_mode", "base") or "base",
            row.get("generator_revision", "") or row.get("base_revision", "")
        ))
    sets = {"E0": []}
    selected = {"E1": [], "E2": [], "E3": [], "E4": []}
    for cls in CLASSES:
        generic = sorted(pool[cls]["generic"], key=lambda x: x["image_id"])
        morph = sorted(pool[cls]["morphology"], key=lambda x: x["image_id"])
        if min(len(generic), len(morph)) < budget_per_class:
            raise ValueError(f"Insufficient accepted generic/morphology pool for {cls}")
        if len(model_check[cls]["generic"]) != 1 or model_check[cls]["generic"] != model_check[cls]["morphology"]:
            raise ValueError(f"E1/E2 generator backbone/mode/revision mismatch for {cls}; filter to one matched generator configuration")
        selected["E1"].extend(rng.sample(generic, budget_per_class))
        selected["E2"].extend(rng.sample(morph, budget_per_class))
        if include_quality:
            if any(not r.get("quality_score") for r in morph):
                raise ValueError(f"Quality selection requires scores for all {cls} morphology candidates")
            selected["E3"].extend(sorted(morph, key=lambda r: (-float(r["quality_score"]), r["image_id"]))[:budget_per_class])
        if include_diversity:
            if any(not r.get("diversity_cluster_id") for r in morph):
                raise ValueError(f"Diversity selection requires cluster IDs for all {cls} morphology candidates")
            clusters = defaultdict(list)
            for row in morph:
                clusters[row["diversity_cluster_id"]].append(row)
            for rows in clusters.values():
                rng.shuffle(rows)
            chosen = []
            # Seeded round-robin coverage, then fill with remaining distinct items.
            while len(chosen) < budget_per_class:
                made_progress = False
                for cluster in sorted(clusters):
                    if len(chosen) >= budget_per_class:
                        break
                    if clusters[cluster]:
                        chosen.append(clusters[cluster].pop())
                        made_progress = True
                if not made_progress:
                    raise ValueError("Insufficient distinct diversity candidates")
            selected["E4"].extend(chosen)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced = {"E0": real_out}
    for arm in ("E1", "E2", "E3", "E4"):
        if arm in {"E3", "E4"} and not (include_quality if arm == "E3" else include_diversity):
            continue
        if len(selected[arm]) != len(CLASSES) * budget_per_class:
            raise AssertionError(f"Unexpected sample count for {arm}")
        produced[arm] = real_out + [_training_row(r, True) for r in selected[arm]]
    if include_synthetic_only:
        # The CSV retains the frozen real partitions for downstream validation and
        # test evaluation; the training notebook's synthetic_only setting uses ONLY
        # the synthetic subset for optimizer steps.
        produced["E5"] = real_out + [_training_row(r, True) for r in selected["E2"]]
        selected["E5"] = list(selected["E2"])
    fields = ["image_path", "label", "split", "group_id", "is_synthetic"]
    for arm, records in produced.items():
        with (out / f"{arm}_training_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "generated_arms": sorted(produced),
        "budget_per_class": budget_per_class,
        "seed": seed,
        "real_rows_each_arm": len(real_out),
        "real_group_count": len(groups),
        "selected_synthetic_ids": {arm: [r["image_id"] for r in selected[arm]]
                                   for arm in produced if arm != "E0"},
        "warning": "Manifest construction cannot certify image morphology, clinical labels, or near-copy absence.",
    }
    (out / "manifest_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description="Build matched E0-E5 training manifests")
    p.add_argument("--real-split", required=True)
    p.add_argument("--approved-real", required=True)
    p.add_argument("--synthetic-pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-class", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality", action="store_true")
    p.add_argument("--diversity", action="store_true")
    p.add_argument("--synthetic-only", action="store_true")
    a = p.parse_args()
    print(json.dumps(prepare(
        a.real_split, a.approved_real, a.synthetic_pool, a.out,
        budget_per_class=a.n_per_class, seed=a.seed,
        include_quality=a.quality, include_diversity=a.diversity,
        include_synthetic_only=a.synthetic_only,
    ), indent=2))


if __name__ == "__main__":
    main()
