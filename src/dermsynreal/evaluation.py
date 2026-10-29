"""Strict six-class, source-separated evaluation for DermSynReal.

Requires exported prediction rows; this module NEVER runs a model or creates scores.
No external dependencies. Statistical results are valid only to the extent that the
prediction provenance, labels, and resampling groups are valid.
"""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("Acne", "Fungal_Infection", "Vitiligo", "Hyperpigmentation", "Skin_Cancer", "Normal")
_ALIASES = {
    "acne": "Acne",
    "fungal": "Fungal_Infection",
    "fungal_infection": "Fungal_Infection",
    "vitiligo": "Vitiligo",
    "hyperpigmentation": "Hyperpigmentation",
    "skin_cancer": "Skin_Cancer",
    "skin_cancer_or_precancer": "Skin_Cancer",
    "normal": "Normal",
}


def _label(value: str) -> str:
    result = _ALIASES.get(value.strip().lower().replace(" ", "_").replace("-", "_"))
    if result is None:
        raise ValueError(f"Unrecognized class label: {value!r}")
    return result


def read_predictions(path: str | Path) -> list[dict]:
    """Expect image_id,true_label,pred_label,source,split; optional group_id.

    source must be explicit 'real' or 'synthetic'; split must be 'test'.
    An image_id must appear once per model run, across both source domains.
    """
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "true_label", "pred_label", "source", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        ids = set()
        for line, raw in enumerate(reader, 2):
            identifier = (raw["image_id"] or "").strip()
            if not identifier or identifier in ids:
                raise ValueError(f"Missing or duplicate image_id on line {line}: {identifier!r}")
            ids.add(identifier)
            source = (raw["source"] or "").strip().lower()
            if source not in {"real", "synthetic"}:
                raise ValueError(f"Explicit source must be real/synthetic on line {line}")
            if (raw["split"] or "").strip().lower() != "test":
                raise ValueError(f"Only frozen test partitions may be evaluated: line {line}")
            rows.append({
                "image_id": identifier,
                "true_label": _label(raw["true_label"] or ""),
                "pred_label": _label(raw["pred_label"] or ""),
                "source": source,
                "split": "test",
                "group_id": (raw.get("group_id") or "").strip() or identifier,
            })
    if not rows:
        raise ValueError("Prediction CSV is empty")
    return rows


def metrics(rows: list[dict], *, require_all_classes: bool = False) -> dict:
    if not rows:
        raise ValueError("No evaluation rows")
    supports = Counter(row["true_label"] for row in rows)
    if require_all_classes and any(supports[c] == 0 for c in CLASSES):
        missing = [c for c in CLASSES if supports[c] == 0]
        raise ValueError(f"No true test images for classes: {missing}")
    confusion = {c: {d: 0 for d in CLASSES} for c in CLASSES}
    for row in rows:
        confusion[row["true_label"]][row["pred_label"]] += 1
    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[d][c] for d in CLASSES if d != c)
        fn = sum(confusion[c][d] for d in CLASSES if d != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"support": supports[c], "precision": precision, "recall": recall, "f1": f1}
    return {
        "n": len(rows),
        "accuracy": sum(confusion[c][c] for c in CLASSES) / len(rows),
        "macro_f1": sum(per_class[c]["f1"] for c in CLASSES) / len(CLASSES),
        "balanced_accuracy": sum(per_class[c]["recall"] for c in CLASSES) / len(CLASSES),
        "per_class": per_class,
        "confusion": confusion,
        "missing_true_classes": [c for c in CLASSES if supports[c] == 0],
    }


def source_metrics(rows: list[dict]) -> dict:
    """Always return separate, never pooled, real/synthetic test scores."""
    return {
        source: metrics([r for r in rows if r["source"] == source])
        for source in ("real", "synthetic")
        if any(r["source"] == source for r in rows)
    }


def _quantile(sorted_numbers: list[float], probability: float) -> float:
    position = (len(sorted_numbers) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return sorted_numbers[lower] * (upper - position) + sorted_numbers[upper] * (position - lower)


def compare_on_real_test(
    baseline_rows: list[dict], treatment_rows: list[dict], *,
    iterations: int = 2000, seed: int = 42,
) -> dict:
    """Paired group-resampled difference, treatment minus baseline.

    User must confirm whether group_id is a genuine patient ID or an
    inferred visual group; the code does not infer patient independence.
    This produces percentile confidence intervals, not p-values.
    """
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations required")
    left = {r["image_id"]: r for r in baseline_rows if r["source"] == "real"}
    right = {r["image_id"]: r for r in treatment_rows if r["source"] == "real"}
    if not left or set(left) != set(right):
        raise ValueError("Both runs must predict identical nonempty real-test image IDs")
    for identifier in left:
        a, b = left[identifier], right[identifier]
        if a["true_label"] != b["true_label"] or a["group_id"] != b["group_id"]:
            raise ValueError(f"Label or group mismatch at test image_id {identifier}")
    ordered = sorted(left)
    base = [left[i] for i in ordered]
    treatment = [right[i] for i in ordered]
    base_score = metrics(base, require_all_classes=True)["macro_f1"]
    treatment_score = metrics(treatment, require_all_classes=True)["macro_f1"]
    grouped = defaultdict(list)
    for identifier in ordered:
        grouped[left[identifier]["group_id"]].append(identifier)
    keys = sorted(grouped)
    if len(keys) < 2:
        raise ValueError("At least two independent resampling groups needed")
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        sampled_groups = rng.choices(keys, k=len(keys))
        sampled_ids = [identifier for g in sampled_groups for identifier in grouped[g]]
        diff = (
            metrics([right[i] for i in sampled_ids])["macro_f1"]
            - metrics([left[i] for i in sampled_ids])["macro_f1"]
        )
        diffs.append(diff)
    diffs.sort()
    return {
        "baseline_macro_f1": base_score,
        "treatment_macro_f1": treatment_score,
        "delta_macro_f1": treatment_score - base_score,
        "ci_95_percentile": [_quantile(diffs, 0.025), _quantile(diffs, 0.975)],
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "real_test_images": len(ordered),
        "resampling_groups": len(keys),
        "group_id_provenance": "must be documented separately (patient or inferred_visual)",
        "p_value": None,
    }
