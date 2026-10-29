"""Metadata-only label approval gate for real facial images.

This flags suspicious labels, but never performs or claims clinical diagnosis.
Only an approved export may be used to create frozen research splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from dermsynreal.evaluation import CLASSES, _label

CANCER_MISMATCH = re.compile(
    r"stevens[\s-]*johnson|(?:^|[^a-z])sjs(?:[^a-z]|$)|toxic\s+epidermal|(?:^|[^a-z])ten(?:[^a-z]|$)",
    re.IGNORECASE,
)
NORMAL_MISMATCH = re.compile(
    r"dermatitis|rosacea|eczema|psoriasis|acne|vitiligo|fungal|tinea|melasma|"
    r"carcinoma|melanoma|lentigo|seborrh|perioral",
    re.IGNORECASE,
)
APPROVALS = {"approved", "rejected", "needs_review"}


def inspect_manifest(path: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Returns approved records, review queue and summary.

    Mandatory columns: image_id,source,class_name,original_diagnosis,
    reviewer_decision,reviewer_id. Each included row must have all fields.
    """
    approved: list[dict] = []
    queue: list[dict] = []
    counts = Counter()
    ids = set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        needed = {"image_id", "source", "class_name", "original_diagnosis",
                  "reviewer_decision", "reviewer_id"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"Missing manifest columns: {sorted(needed - set(reader.fieldnames or []))}")
        for number, raw in enumerate(reader, 2):
            row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
            image_id = row["image_id"]
            if not image_id or image_id in ids:
                raise ValueError(f"Missing/duplicate image ID at line {number}: {image_id!r}")
            ids.add(image_id)
            cls = _label(row["class_name"])
            row["class_name"] = cls
            decision = row["reviewer_decision"].lower()
            if decision not in APPROVALS:
                raise ValueError(f"Unknown reviewer decision on line {number}: {decision!r}")
            original = row["original_diagnosis"]
            combined = original + " " + row.get("image_path", "")
            mismatch = (
                (cls == "Skin_Cancer" and bool(CANCER_MISMATCH.search(combined)))
                or (cls == "Normal" and bool(NORMAL_MISMATCH.search(combined)))
            )
            needs_details = not all(
                [row["source"], original, row["reviewer_id"]]
            )
            counts[f"decision_{decision}"] += 1
            if mismatch:
                counts["incompatible_label_flags"] += 1
            if decision == "approved" and not mismatch and not needs_details:
                approved.append(row)
                counts[f"accepted_{cls}"] += 1
            elif decision != "rejected":
                row["gate_issue"] = (
                    "mismatched_original_diagnosis_or_filename" if mismatch
                    else "missing_required_provenance" if needs_details
                    else "manual_review_unfinished"
                )
                queue.append(row)
            # Rejected rows remain excluded and are counted; never included.
    summary = {
        "total_records": len(ids),
        "approved_records": len(approved),
        "review_queue_records": len(queue),
        "rejected_records": counts["decision_rejected"],
        "incompatible_label_flags": counts["incompatible_label_flags"],
        "approved_by_class": {c: counts[f"accepted_{c}"] for c in CLASSES},
    }
    summary["passed"] = bool(approved) and not queue and all(
        summary["approved_by_class"][c] > 0 for c in CLASSES
    )
    return approved, queue, summary


def write_audit(
    source: str | Path, destination: str | Path, *, fail_if_blocked: bool = True
) -> dict:
    approved, queue, summary = inspect_manifest(source)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    with open(source, newline="", encoding="utf-8-sig") as f:
        columns = csv.DictReader(f).fieldnames
    assert columns
    for filename, rows, extra in [
        ("approved_for_split.csv", approved, []),
        ("needs_manual_review.csv", queue, ["gate_issue"]),
    ]:
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(columns) + extra, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    (out / "label_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if fail_if_blocked and not summary["passed"]:
        raise ValueError(
            f"Label gate BLOCKED: {summary['review_queue_records']} unresolved; "
            f"class counts={summary['approved_by_class']}. Inspect {out}."
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Review metadata, not clinical diagnosis")
    parser.add_argument("--manifest", required=True, help="Reviewed real image manifest CSV")
    parser.add_argument("--out", required=True, help="Directory for QA outputs")
    args = parser.parse_args()
    try:
        print(json.dumps(write_audit(args.manifest, args.out), indent=2))
        return 0
    except ValueError as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
