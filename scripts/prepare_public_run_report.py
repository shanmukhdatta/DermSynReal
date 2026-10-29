"""Prepare aggregate Kaggle research reports that are safe to review for public GitHub.

Does not push or authenticate to GitHub. Full image files, per-image IDs, model
weights, review manifests and private logs remain in versioned Kaggle storage.
Only whitelisted *aggregate* result artifacts are considered.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime,timezone
from pathlib import Path

ALLOWED={
    "run_summary.csv",
    "per_class_real_test.csv",
    "confusion_real_test.csv",
    "per_source_real_test.csv",
    "paired_real_test.csv",
    "generation_quality_counts.csv",
    "domain_class_error_rates.csv",
    "calibration_by_domain.json",
    "generation_quality_summary.json",
}
IMAGE_ID_COLUMNS={"image_id","image_path","filename","path","source_path","original_path",
                  "patient_id","patient_name","file"}
MAX_BYTES=4*1024*1024


def copy_aggregate_reports(source_dir,out_dir,tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}",tag):
        raise ValueError("Tag must be a short, safe ASCII run identifier")
    source=Path(source_dir)
    if not source.is_dir():raise FileNotFoundError(source)
    dest=Path(out_dir)/tag
    if dest.exists():raise FileExistsError("Run report directory already exists; use a versioned tag")
    selected=[]
    for name in sorted(ALLOWED):
        choices=list(source.rglob(name))
        if len(choices)>1:
            raise ValueError(f"Ambiguous {name}: multiple run outputs; package one run suite at a time")
        if not choices:continue
        p=choices[0]
        if not p.is_file() or p.stat().st_size>MAX_BYTES:
            raise ValueError(f"Unexpected report size: {name}")
        raw=p.read_bytes()
        if name.endswith(".csv"):
            with p.open(newline="",encoding="utf-8-sig") as f:
                reader=csv.DictReader(f)
                if not reader.fieldnames or IMAGE_ID_COLUMNS & {x.lower() for x in reader.fieldnames}:
                    raise ValueError(f"Potential per-image sensitive columns in {name}")
                if not next(reader,None):
                    raise ValueError(f"Empty aggregate results CSV: {name}")
        elif name.endswith(".json"):
            obj=json.loads(raw.decode("utf-8"))
            if not isinstance(obj,dict):
                raise ValueError(f"Unexpected summary JSON format in {name}")
        selected.append((name,raw))
    if not selected:
        raise ValueError("No verified aggregate result files found")
    dest.mkdir(parents=True)
    records=[]
    for name,data in selected:
        (dest/name).write_bytes(data)
        records.append({"file":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    (dest/"artifact_manifest.json").write_text(json.dumps({
        "run_tag":tag,"created_utc":datetime.now(timezone.utc).isoformat(),
        "artifact_files":records,
        "note":"Aggregate research outputs only; check source licenses, diagnosis review, "
               "model names and scientific claims manually before pushing to public GitHub."
    },indent=2)+"\n")
    return dest

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--from-kaggle-output",required=True)
    parser.add_argument("--out",default="reports/experiments")
    parser.add_argument("--run-tag",required=True)
    args=parser.parse_args()
    print(copy_aggregate_reports(args.from_kaggle_output,args.out,args.run_tag))
