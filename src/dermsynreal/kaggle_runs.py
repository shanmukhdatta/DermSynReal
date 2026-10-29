"""Shared Kaggle orchestration for the existing DermSynReal notebooks.

All functions are stdlib only and never approve a clinical label.
The only real-image source is the user's Kaggle v2 collection, filtered
through previously reviewed and frozen real train partitions.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import _label
from .real_collection import DATASET_SOURCE, discover_root, CLASS_FOLDERS

PROMPT_JSON = {
    "Acne": "Acne/01_Acne/json/acne_all_300.json",
    "Fungal_Infection": "Fungal_Infection/02_Fungal/json/fungal_all_300.json",
    "Vitiligo": "Vitiligo/03_Vitiligo/json/vitiligo_all_300.json",
    "Hyperpigmentation": "Hyperpigmentation/04_Hyperpigmentation/json/hyperpigmentation_all_300.json",
    "Skin_Cancer": "Skin_Cancer/05_Skin_Cancer/json/skin_cancer_all_300.json",
    "Normal": "Normal_Skin/06_Normal_Skin/json/normal_skin_all_300.json",
}
PROMPT_FOLDER_COMPAT={
    "Hyperpigmentation":["Hyperpigmentation","Hyperpigmenatation"],
    "Vitiligo":["Vitiligo","Vitligo"],
}
SUBTYPES={
    "Fungal_Infection":{
        "tinea_faciei":"tinea_faciei",
        "pityriasis_versicolor":"pityriasis_versicolor",
        "angular_cheilitis_candida_unconfirmed": "unconfirmed_candida",
    },
    "Skin_Cancer":{
        "bcc_appearance":"bcc","scc_appearance":"scc",
        "melanoma_appearance":"melanoma","ak_precancerous_appearance":"ak",
    },
}
def git_sha(repo):
    repo=Path(repo)
    if not (repo/".git").is_dir():
        return "source_snapshot_no_git_sha"
    try:
        return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
                                       text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return "unverified"

def _read_csv(path,needed):
    with Path(path).open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing {sorted(needed-set(reader.fieldnames or []))}")
        return list(reader)
def _csv(path,rows,columns):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()

def prompt_collection(repo_root, class_name, *, partition="train", limit=5, offset=0):
    """Load exact authored prompts; preserve prompt-family partition and negative prompts.

    Defaults to a small train-only pilot. For 210 train prompts set limit=None.
    Validation/test prompts are available only for separate diagnostic generation;
    their family IDs must not enter augmentation or generator fine-tuning.
    """
    cls=_label(class_name)
    if partition not in ("train","val","test"):
        raise ValueError("partition must be train, val, test")
    if limit is not None and limit<1 or offset<0:
        raise ValueError("limit must be positive and offset nonnegative")
    candidates=[Path(repo_root)/"prompts"/PROMPT_JSON[cls]]
    for old in PROMPT_FOLDER_COMPAT.get(cls,[]):
        target=PROMPT_JSON[cls].split("/",1)[1]
        candidates.append(Path(repo_root)/"prompts"/old/target)
    source=next((p for p in candidates if p.is_file()),None)
    if source is None:
        raise FileNotFoundError(f"Missing 300-prompt registry: {candidates}")
    rows=json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(rows,list) or len(rows)!=300:
        raise ValueError("Expected a full authored 300-prompt JSON list")
    groups=defaultdict(set)
    unique=set()
    for r in rows:
        required=("prompt_id","prompt_group_id","positive_prompt","negative_prompt","split")
        if not all(str(r.get(k) or "").strip() for k in required):
            raise ValueError(f"Required prompt metadata absent: {r}")
        if r["prompt_id"] in unique:raise ValueError("Duplicate prompt_id")
        unique.add(r["prompt_id"])
        groups[r["prompt_group_id"]].add(r["split"])
    if any(len(s)!=1 for s in groups.values()):
        raise ValueError("Prompt group crosses train/val/test partitions")
    subset=[r for r in rows if r["split"]==partition]
    selected=subset[offset:offset+limit if limit is not None else None]
    if not selected:raise ValueError(f"Empty prompt selection for {cls}/{partition} offset={offset}")
    normalized=[]
    for r in selected:
        phenotype=r.get("phenotype") or "unspecified"
        subtype=SUBTYPES.get(cls,{}).get(phenotype,phenotype)
        normalized.append({
            "id":r["prompt_id"],
            "text":r["positive_prompt"],
            "negative_prompt":r["negative_prompt"],
            "prompt_family_id":r["prompt_group_id"],
            "prompt_partition":r["split"],
            "subtype":subtype,
            "phenotype":phenotype,
            "pattern":phenotype,
            "distribution":r.get("facial_location",""),
            "adapter_mode":"none",
            "category":cls.lower(),
            "use_cystic_token":False,
            "seed_base":r.get("seed_base"),
        })
    return normalized

def resolve_repo_root(candidate=None):
    paths=([Path(candidate)] if candidate else [])+[
        Path("/kaggle/working/DermSynReal"),
        Path("/kaggle/input/dermsynreal/DermSynReal"),
    ]
    for path in paths:
        if (path/"src/dermsynreal/kaggle_runs.py").is_file() and (path/"prompts").is_dir():
            return path
    raise FileNotFoundError("Attach a full DermSynReal repository snapshot or clone it with Kaggle Internet")

def generation_manifest(metadata,out_dir,*,root,class_name,repo_root,
                        prompt_partition,model_id=None):
    """Convert existing metadata.jsonl to robust CSV without changing generator logic.

    Every image starts UNREVIEWED and is not accepted for training until a human
    morphological audit. Explicit source/shas/modes/families are exported.
    """
    cls=_label(class_name)
    root=Path(root)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    registry={p["id"]:p for p in prompt_collection(repo_root,cls,
                    partition=prompt_partition,limit=None)}
    records=[]
    for source_row in metadata:
        prompt_id=str(source_row.get("prompt_id") or "")
        if prompt_id not in registry:
            raise ValueError(f"Missing prompt provenance: {prompt_id}")
        p=registry[prompt_id]
        relative=source_row.get("file","")
        if not relative:raise ValueError("No generated file path recorded")
        given=Path(relative)
        if given.is_absolute() or ".." in given.parts:
            raise ValueError("Invalid generated image relative path")
        image=root/given
        if not image.is_file() or image.stat().st_size==0:
            raise FileNotFoundError(image)
        digest=sha256_file(image)
        record={
            "image_id":"synthetic_"+digest[:16]+"_"+hashlib.sha256((p["id"]+relative).encode()).hexdigest()[:12],
            "image_path":str(image.resolve()),
            "label":cls.lower(),"class_name":cls,
            "prompt_id":prompt_id,"prompt_family_id":p["prompt_family_id"],
            "prompt_partition":p["prompt_partition"],"prompt_type":"morphology",
            "model_id":str(model_id or source_row.get("base_model") or source_row.get("base") or ""),
            "generator_mode":str(source_row.get("mode") or source_row.get("requested_adapter_mode") or "base"),
            "seed":source_row.get("seed",""),
            "image_sha256":digest,
            "accepted":"False", # no automated medical approval
            "reviewer_decision":"needs_review",
            "morphology_score":"",
            "near_copy_flag":"",
            "source_dataset":DATASET_SOURCE,
            "source_file":relative
        }
        records.append(record)
    if not records:raise ValueError("No images generated")
    seen=set()
    for r in records:
        if r["image_id"] in seen:raise ValueError("Duplicate generated image ID")
        seen.add(r["image_id"])
    _csv(out/"synthetic_pool_review_draft.csv",records,list(records[0]))
    _csv(out/"generation_manifest.csv",records,list(records[0]))
    counts=Counter(r["class_name"] for r in records)
    modes=Counter(r["generator_mode"] for r in records)
    manifest={
        "generated_images":len(records),"labels":dict(counts),"modes":dict(modes),
        "prompt_partition":prompt_partition,
        "distinct_prompt_families":len({r["prompt_family_id"] for r in records}),
        "model_id":model_id or "see individual rows",
        "source_repo_sha":git_sha(repo_root),
        "review_state":"ALL_UNREVIEWED",
        "source_real_dataset":DATASET_SOURCE,
        "files_sha256_verified":True,
        "note":"Do not enter synthetic_pool_review_draft into training until human clinical/novelty review."
    }
    (out/"generation_summary.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def append_run_log(output_dir,*,kind,config,artifacts=(),repo_root=None):
    """Append one JSONL record. Never silently overwrite audit/history."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for f in artifacts:
        p=Path(f)
        if p.is_file():hashes[p.name]=sha256_file(p)
    row={
        "utc":datetime.now(timezone.utc).isoformat(),
        "kind":kind,"config":config,"artifacts_sha256":hashes,
        "repository_sha":git_sha(repo_root) if repo_root else "unverified",
        "python":sys.version.split()[0],"platform":platform.platform()
    }
    with (out/"run_log.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False,default=str)+"\n")
    return row

def prepare_lora_train_only(*,real_split_manifest,real_image_root,cls,
                            stage_dir,origin_root=None):
    """Fail closed unless image is explicitly from reviewed REAL train split.

    Symlinks only; avoids copying photos before the original LoRA training
    notebook prepares its imagefolder. NEVER run on unsplit raw Kaggle folders.
    """
    cls=_label(cls)
    dataset_root=discover_root(origin_root)
    allowed=_read_csv(real_split_manifest,
        {"image_id","image_path","label","split","group_id","is_synthetic"})
    candidate=[]
    root=Path(real_image_root).resolve()
    source_dir=(dataset_root/CLASS_FOLDERS[cls]).resolve()
    real_groups=defaultdict(set)
    for r in allowed:
        if str(r["is_synthetic"]).lower() in {"true","1","yes"}:
            continue
        if not r["group_id"]:raise ValueError("Missing real group ID")
        real_groups[r["group_id"]].add(r["split"])
    if any(len(v)>1 for v in real_groups.values()):
        raise ValueError("Real patient/visual group crosses splits")
    for r in allowed:
        if _label(r["label"])!=cls or r["split"]!="train":
            continue
        if str(r["is_synthetic"]).lower() not in {"false","0","no"}:
            raise ValueError("Only reviewed real source images may train the generator")
        rel=Path(r["image_path"])
        if ".." in rel.parts:raise ValueError("Unsafe relative manifest path")
        src=(rel if rel.is_absolute() else root/rel).resolve()
        if not src.is_file() or src.is_relative_to(dataset_root)==False:
            # Final dedup ZIP copy can reside in a saved Kaggle dataset;
            # the review gate is established by the explicit frozen manifest.
            if not src.is_file() or not src.is_relative_to(root):
                raise ValueError(f"File outside approved real split: {src}")
        candidate.append((r,src))
    if not candidate:raise ValueError(f"No approved train images for {cls}")
    out=Path(stage_dir)
    if out.exists():raise FileExistsError(f"LoRA stage already exists: {out}")
    (out/"train").mkdir(parents=True,exist_ok=False)
    index=[]
    for r,src in candidate:
        name=r["image_id"]+"_"+src.name
        dest=out/"train"/name
        if dest.exists():raise ValueError("Stable identifier collision")
        dest.symlink_to(src)
        index.append({"image_id":r["image_id"],"image_path":str(src),
                      "staged_path":str(dest),"class_name":cls,"split":"train",
                      "group_id":r["group_id"],"source_dataset":DATASET_SOURCE})
    _csv(out/"lora_train_source_manifest.csv",index,list(index[0]))
    note={"class":cls,"count":len(index),"source":DATASET_SOURCE,
          "reviewed_train_only":True,"image_root":str(root),
          "raw_input":str(dataset_root),
          "manifest_sha256":sha256_file(real_split_manifest),
          "train_group_count":len({x["group_id"] for x in index}),
          "note":"No validation/test images were staged. Clinical review is assumed from the approved split provenance."}
    (out/"lora_train_source_summary.json").write_text(json.dumps(note,indent=2)+"\n")
    return out,note

def training_run_report(*,model_output,manifest_csv,model_name,experiment,seed,repo_root=None):
    """Create checkpoint/validation artifact inventory without inventing scores."""
    out=Path(model_output);out.mkdir(parents=True,exist_ok=True)
    if not Path(manifest_csv).is_file():
        raise FileNotFoundError(manifest_csv)
    audit=[]
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name not in {"run_log.jsonl","artifact_inventory.csv"}:
            audit.append({"filename":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    _csv(out/"artifact_inventory.csv",audit,["filename","bytes","sha256"])
    record=append_run_log(out,kind="classifier_checkpoint_inventory",
        config={"model_name":model_name,"experiment":experiment,"seed":seed,
                "real_manifest_sha256":sha256_file(manifest_csv),
                "manifest_path":str(manifest_csv),
                "checkpoint_count":len([r for r in audit if r["filename"].endswith((".pth",".joblib"))])},
        artifacts=[out/"artifact_inventory.csv"],repo_root=repo_root)
    return record
