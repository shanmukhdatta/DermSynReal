"""Kaggle Version 2 facial real-image dataset -> reviewed, leakage-conscious training input.

Only the Kaggle collection uploaded by the project user is permitted. No labels are
inferred or automatically approved from directory names or computer vision.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import re
from collections import Counter,defaultdict
from pathlib import Path

from .evaluation import CLASSES, _label
from .label_gate import CANCER_MISMATCH,NORMAL_MISMATCH

DATASET_SLUG="facial-skin-conditions-real-image-collection"
KAGGLE_OWNER="shanmukhadattaboda"
DATASET_SOURCE=f"kaggle:{KAGGLE_OWNER}/{DATASET_SLUG}:v2"
CLASS_FOLDERS={
    "Acne":"Acne",
    "Fungal_Infection":"Fungal_infection",
    "Vitiligo":"Vitiligo",
    "Hyperpigmentation":"Hyper_Pigmentation",
    "Skin_Cancer":"Skin_cancer",
    "Normal":"Normal_Skin",
}
IMAGE_EXTENSIONS={".png",".jpg",".jpeg",".webp",".bmp",".tiff",".tif"}
ROOT_CANDIDATES=(
    f"/kaggle/input/datasets/{KAGGLE_OWNER}/{DATASET_SLUG}",
    f"/kaggle/input/{DATASET_SLUG}",
)


def discover_root(path=None):
    """Find screenshot's Face_Skin_dataset directory without switching datasets."""
    bases=[Path(path)] if path else [Path(x) for x in ROOT_CANDIDATES]
    for base in bases:
        for candidate in (base,base/"Face_Skin_dataset"):
            if candidate.is_dir() and all((candidate/folder).is_dir() for folder in CLASS_FOLDERS.values()):
                return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate all six original Face_Skin_dataset folders. "
        f"Attach only {KAGGLE_OWNER}/{DATASET_SLUG}, and inspect /kaggle/input. "
        f"Paths attempted: {bases}"
    )


def class_directories(root):
    root=discover_root(root)
    return {name:str(root/folder) for name,folder in CLASS_FOLDERS.items()}


def stable_id(relative_path):
    return "real_"+hashlib.sha256(
        (DATASET_SOURCE+"|"+str(relative_path)).encode("utf-8")
    ).hexdigest()[:24]


def _csv(path,rows,columns):
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path, columns):
    with Path(path).open(newline="",encoding="utf-8-sig") as source:
        reader=csv.DictReader(source)
        if not reader.fieldnames or not set(columns).issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing {sorted(set(columns)-set(reader.fieldnames or []))}")
        return list(reader)


def _sha256(path):
    sha=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            sha.update(chunk)
    return sha.hexdigest()


def _suspicion(label,relative):
    text=str(relative)
    if label=="Skin_Cancer" and CANCER_MISMATCH.search(text):
        return "possible_SJS_TEN_not_skin_cancer"
    if label=="Normal" and NORMAL_MISMATCH.search(text):
        return "possible_disorder_not_normal"
    return ""


def inventory_collection(root,out_dir):
    """Hash all images, collect filename-level flags, create a human review draft.

    Filename flags are screening hints, never ground-truth diagnoses. This detects
    exact-byte copies; semantic/flip-aware near-duplicates use the later pipeline.
    """
    root=discover_root(root)
    out=Path(out_dir)
    if out.exists():
        raise FileExistsError(f"Inventory output exists; use a new versioned path: {out}")
    out.mkdir(parents=True)
    records=[]
    for cls,foldername in CLASS_FOLDERS.items():
        folder=root/foldername
        files=sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"{cls}: {len(files)} candidate image files")
        for path in files:
            rel=path.relative_to(root).as_posix()
            flag=_suspicion(cls,rel)
            records.append({
                "image_id":stable_id(rel),"image_path":rel,"class_name":cls,
                "source":DATASET_SOURCE,"original_diagnosis":"",
                "reviewer_decision":"needs_review","reviewer_id":"",
                "filename_flag":flag,"sha256":_sha256(path),
                "bytes":path.stat().st_size,
            })
    if not records:raise ValueError("No images found")
    hashes=defaultdict(list)
    for row in records:hashes[row["sha256"]].append(row)
    duplicate_groups=[]
    for sha,group in sorted(hashes.items()):
        if len(group)>1:
            labels=sorted(set(row["class_name"] for row in group))
            duplicate_groups.append({
                "sha256":sha,"num_files":len(group),"cross_class":len(labels)>1,
                "classes":";".join(labels),
                "image_paths":";".join(row["image_path"] for row in group)
            })
    counts=Counter(row["class_name"] for row in records)
    flags=Counter(row["filename_flag"] for row in records if row["filename_flag"])
    columns=list(records[0])
    _csv(out/"inventory.csv",records,columns)
    # All rows need human review, including "normal" images and unflagged filenames.
    _csv(out/"review_draft.csv",records,
         ["image_id","source","class_name","original_diagnosis","reviewer_decision",
          "reviewer_id","image_path","filename_flag"])
    _csv(out/"exact_sha_duplicate_groups.csv",duplicate_groups,
         ["sha256","num_files","cross_class","classes","image_paths"])
    summary={
        "source":DATASET_SOURCE,"dataset_root":str(root),
        "files_screened":len(records),"files_by_class":dict(counts),
        "exact_duplicate_groups":len(duplicate_groups),
        "redundant_exact_file_copies":sum(len(g)-1 for g in hashes.values()),
        "cross_class_exact_duplicate_groups":sum(g["cross_class"] for g in duplicate_groups),
        "suspect_filename_flags":dict(flags),
        "clinical_labels_verified":False,"near_duplicates_verified":False,
        "next":"Manually review and edit review_draft.csv; do not auto-approve from directory names",
    }
    (out/"inventory_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def stage_approved(root,approved_csv,audit_json,stage_dir,*,inventory_csv):
    """Symlink ONLY manually approved exact-source photos, no huge staging copy.

    Requires a passed label gate and a matching *original inventory* (sha256).
    This excludes rejected/unreviewed images from the full dedup pipeline.
    """
    root=discover_root(root)
    audit=json.loads(Path(audit_json).read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError("Real-label review is not passed; refusing split and training")
    original=_read_csv(inventory_csv,{"image_id","image_path","class_name","sha256"})
    original_by_id={r["image_id"]:r for r in original}
    if len(original_by_id)!=len(original):raise ValueError("Nonunique inventory image IDs")
    approved=_read_csv(approved_csv,{"image_id","image_path","class_name","reviewer_decision",
                                     "original_diagnosis","reviewer_id","source"})
    if not approved:raise ValueError("Zero approved real images")
    if len(approved)!=int(audit.get("approved_records",-1)):
        raise ValueError("Audit count does not match approved CSV")
    stage=Path(stage_dir)
    if stage.exists():raise FileExistsError(f"Approved staging path exists: {stage}")
    stage.mkdir(parents=True)
    rows=[]
    ids=set()
    for row in approved:
        identifier=row["image_id"]
        if identifier in ids or identifier not in original_by_id:
            raise ValueError(f"Unrecognized/repeated approved image_id {identifier}")
        ids.add(identifier)
        ref=original_by_id[identifier]
        cls=_label(row["class_name"])
        rel=row["image_path"].replace("\\","/")
        if (cls!=ref["class_name"] or rel!=ref["image_path"]
            or row["reviewer_decision"].strip().lower()!="approved"
            or not row["original_diagnosis"].strip() or not row["reviewer_id"].strip()
            or row["source"]!=DATASET_SOURCE):
            raise ValueError(f"Review metadata not verified: {identifier}")
        if _suspicion(cls,rel):
            raise ValueError(f"Suspect filename requires correcting/excluding source: {rel}")
        relpath=Path(rel)
        if not relpath.parts or relpath.parts[0]!=CLASS_FOLDERS[cls] or ".." in relpath.parts:
            raise ValueError(f"Approved path outside expected label folder: {rel}")
        source=(root/relpath).resolve()
        if not source.is_file() or not source.is_relative_to(root):
            raise ValueError(f"Path is missing or outside sole real dataset: {source}")
        if _sha256(source)!=ref["sha256"]:
            raise ValueError(f"Input image changed since inventory: {rel}")
        target=stage/cls/relpath.relative_to(CLASS_FOLDERS[cls])
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(source)
        rows.append({
            "image_id":identifier,
            "stage_path":str(target.absolute()),
            "class_name":cls,
            "source_rel_path":rel,
            "sha256":ref["sha256"],
            "dataset_source":DATASET_SOURCE,
        })
    if set(CLASSES)-set(x["class_name"] for x in rows):
        raise ValueError("Approved images do not cover all six classes")
    _csv(stage/"stage_to_original.csv",rows,list(rows[0]))
    summary={"approved_images":len(rows),"by_class":dict(Counter(r["class_name"] for r in rows)),
             "staging_root":str(stage),"images_copied":0,
             "note":"The symlinks refer only to the user-supplied Kaggle real dataset. "
                    "Original files remain unchanged."}
    (stage/"stage_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def export_training_manifests(split_csv,stage_map_csv,output_dir):
    """Bridge existing dedup pipeline's numeric IDs to stable reviewed IDs.

    The classifier requires image_path,label,split,group_id,is_synthetic.
    This is for REAL images only; synthetic augmentation is a later step.
    """
    rows=_read_csv(split_csv,{"image_id","label","split","group_id","source_path","relative_path","sha256"})
    stage=_read_csv(stage_map_csv,{"image_id","stage_path","class_name","sha256","dataset_source"})
    by_path={Path(r["stage_path"]).resolve():r for r in stage}
    split_root=Path(split_csv).resolve().parent.parent
    if not rows:raise ValueError("No deduplication split records")
    ids=set();paths=set();groups={};output=[]
    for r in rows:
        orig=by_path.get(Path(r["source_path"]).resolve())
        if orig is None:raise ValueError(f"Dedup exported an unapproved source: {r['source_path']}")
        if r["sha256"]!=orig["sha256"]:
            raise ValueError("Mismatch against original approved photo hash")
        if r["relative_path"] in paths:
            raise ValueError("Duplicate split output path")
        paths.add(r["relative_path"])
        identifier=orig["image_id"]
        if identifier in ids:raise ValueError("Duplicate stable reviewed image ID")
        ids.add(identifier)
        split=r["split"].lower()
        if split not in {"train","val","test"}:raise ValueError("Invalid split")
        group="real_"+str(r["group_id"])
        if group in groups and groups[group]!=split:
            raise ValueError("Protected group crosses splits")
        groups[group]=split
        label=_label(r["label"])
        if label!=orig["class_name"]:
            raise ValueError("Approved-label mismatch")
        if not (split_root/r["relative_path"]).is_file():
            raise FileNotFoundError(split_root/r["relative_path"])
        output.append({
            "image_id":identifier,"image_path":r["relative_path"],"label":label.lower(),
            "split":split,"group_id":group,"is_synthetic":"False",
            "dataset_source":orig["dataset_source"],
        })
    counts=Counter((r["label"],r["split"]) for r in output)
    for cl in CLASSES:
        for split in ("train","val","test"):
            if counts[(cl.lower(),split)]==0:
                raise ValueError(f"Missing {cl}/{split} after deduplication")
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    _csv(out/"real_all_manifest.csv",output,list(output[0]))
    for split in ("train","val","test"):
        partition=[r for r in output if r["split"]==split]
        _csv(out/f"real_{split}_manifest.csv",partition,list(output[0]))
    summary={
        "real_images":len(output),"split_root":str(split_root),
        "class_split_counts":{f"{k[0]}/{k[1]}":v for k,v in sorted(counts.items())},
        "protected_groups":len(groups),
        "source_dataset":DATASET_SOURCE,
        "clinical_labels_verified_by_code":False,
        "patient_independence_guaranteed":False,
    }
    (out/"real_manifest_bridge_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary
