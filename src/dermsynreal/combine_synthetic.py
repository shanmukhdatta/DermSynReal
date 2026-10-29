"""Safely combine *reviewer-approved* diffusion/LoRA synthetic image manifests.

Combines metadata only. Keeps all prompt families for the downstream synthetic
group-aware split. Does not copy medical images or assert clinical correctness.
"""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

from .evaluation import CLASSES,_label
from .kaggle_runs import sha256_file

REQUIRED={
    "image_id","image_path","label","class_name","prompt_id","prompt_family_id",
    "prompt_partition","prompt_type","model_id","generator_mode",
    "accepted","reviewer_decision","reviewer_id","image_sha256",
}
PARTITIONS={"train","val","test"}
ACCEPT_TRUE={"true","yes","1"}
ACCEPT_REVIEW={"approved"}


def _read(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        missing=REQUIRED-set(reader.fieldnames or [])
        if missing:raise ValueError(f"{path} is missing reviewed fields {sorted(missing)}")
        return list(reader)


def _resolve_image(row,root):
    root=Path(root).resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    saved=(row["source_file"] if row.get("source_file") else row["image_path"]).replace("\\","/")
    rel=Path(saved)
    if ".." in rel.parts:raise ValueError("Unsafe generated source_file relative path")
    # Existing absolute path may refer to the original Kaggle working mount.
    candidates=[Path(row["image_path"]),root/rel] if Path(row["image_path"]).is_absolute() else [root/rel]
    for candidate in candidates:
        resolved=candidate.resolve()
        if resolved.is_file() and resolved.is_relative_to(root):
            return resolved
    raise FileNotFoundError(f"Image not found inside mounted generation output: {saved!r}; root={root}")


def combine_approved(manifests,image_roots,out_csv,*,generator_mode=None,
                     prompt_partition=None,require_six_classes=False):
    if len(manifests)!=len(image_roots) or not manifests:
        raise ValueError("Provide one image-root mount per input reviewed manifest")
    if prompt_partition is not None and prompt_partition not in PARTITIONS:
        raise ValueError("Invalid prompt partition")
    selected=[]
    skipped=Counter()
    seen_ids=set()
    sha_labels=defaultdict(set)
    sha_groups=defaultdict(set)
    manifest_hashes={}
    for manifest,root in zip(manifests,image_roots):
        path=Path(manifest)
        manifest_hashes[path.name]=sha256_file(path)
        for number,row in enumerate(_read(path),start=2):
            if row["accepted"].strip().lower() not in ACCEPT_TRUE:
                skipped["not_accepted"]+=1;continue
            if row["reviewer_decision"].strip().lower() not in ACCEPT_REVIEW or not row["reviewer_id"].strip():
                raise ValueError(f"{path}:{number} accepted without an accountable reviewer")
            if row.get("near_copy_flag","").strip().lower() in {"true","yes","1"}:
                raise ValueError(f"{path}:{number} accepted near-copy flag must be resolved")
            label=_label(row["label"])
            if _label(row["class_name"])!=label:
                raise ValueError(f"{path}:{number} label/class mismatch")
            if row["prompt_partition"] not in PARTITIONS:
                raise ValueError(f"{path}:{number} missing authored prompt split")
            if row["prompt_type"] not in {"generic","morphology"}:
                raise ValueError(f"{path}:{number} invalid prompt type")
            if any(not row[k].strip() for k in
                   ("prompt_id","prompt_family_id","model_id","generator_mode","image_sha256")):
                raise ValueError(f"{path}:{number} missing generator provenance")
            if generator_mode and row["generator_mode"]!=generator_mode:
                skipped["other_generator_mode"]+=1;continue
            if prompt_partition and row["prompt_partition"]!=prompt_partition:
                skipped["other_prompt_partition"]+=1;continue
            identifier=row["image_id"].strip()
            if not identifier or identifier in seen_ids:
                raise ValueError(f"Missing or duplicated synthetic image_id in {path}: {identifier}")
            seen_ids.add(identifier)
            image=_resolve_image(row,root)
            if sha256_file(image).lower()!=row["image_sha256"].strip().lower():
                raise ValueError(f"{path}:{number} image content changed since generation")
            sha_labels[row["image_sha256"]].add(label)
            sha_groups[row["image_sha256"]].add((label,row["prompt_family_id"]))
            item=dict(row)
            item["image_path"]=str(image)
            item["label"]=label.lower()
            item["class_name"]=label
            item["accepted"]="True"
            item["reviewer_decision"]="approved"
            selected.append(item)
    conflicting=[sha for sha,v in sha_labels.items() if len(v)>1]
    if conflicting:raise ValueError("Same generated-image SHA assigned to different classes: manually quarantine")
    if not selected:raise ValueError("No human-approved synthetic samples")
    if require_six_classes and {_label(r["label"]) for r in selected}!=set(CLASSES):
        raise ValueError("Approved synthetic pool is missing target classes")
    # A random mixture of base-only, textual inversion, and custom LoRA can
    # invalidate E1/E2 comparisons. Require an explicit choice when a class
    # has multiple modes; document separate runs instead of silent blending.
    per_class_modes=defaultdict(set)
    for row in selected:per_class_modes[row["class_name"]].add((
        row["model_id"],row["generator_mode"],
        row.get("generator_revision",""),row.get("adapter_source",""),
        row.get("adapter_revision","")))
    for cls,modes in per_class_modes.items():
        if len(modes)>1:
            raise ValueError(f"{cls} mixes generator checkpoints/modes. Use matched inputs or filter before combining")
    out=Path(out_csv)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    fields=sorted(set().union(*(row.keys() for row in selected)))
    rows=sorted(selected,key=lambda r:(r["class_name"],r["prompt_partition"],
                                       r["prompt_type"],r["prompt_family_id"],r["image_id"]))
    with out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    metadata={
        "approved_images":len(rows),
        "classes":dict(Counter(r["class_name"] for r in rows)),
        "prompt_types":dict(Counter(r["prompt_type"] for r in rows)),
        "prompt_partitions":dict(Counter(r["prompt_partition"] for r in rows)),
        "generator_modes":dict(Counter(r["generator_mode"] for r in rows)),
        "distinct_prompt_families":len({(r["class_name"],r["prompt_family_id"]) for r in rows}),
        "same_class_exact_duplicate_hashes_left_for_preprocessing":sum(
             len(groups)>1 for groups in sha_groups.values()),
        "skipped":dict(skipped),
        "input_manifest_sha256":manifest_hashes,
        "research_use":"Reviewed pool only. Synthetic preprocessing must independently scan and group exact duplicates.",
        "clinical_correctness_guaranteed":False
    }
    out.with_suffix(".summary.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata
