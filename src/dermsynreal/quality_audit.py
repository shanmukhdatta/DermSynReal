"""Review-based audit of generated images, not an automatic clinical validator."""
from __future__ import annotations
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path


def summarize_quality(review_csv, out_dir):
    required={"image_id","class_name","prompt_id","prompt_family_id","generator_id",
              "reviewer_id","decision","morphology_score","near_copy_flag"}
    with Path(review_csv).open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing review columns: {sorted(required-set(reader.fieldnames or []))}")
        data=list(reader)
    if not data: raise ValueError("Empty generation review")
    ids=set(); grouped=defaultdict(lambda:Counter())
    for row in data:
        image_id=row["image_id"].strip()
        if not image_id or image_id in ids:
            raise ValueError(f"Missing/repeated image ID: {image_id!r}")
        ids.add(image_id)
        if not all(row[field].strip() for field in required):
            raise ValueError(f"Missing review metadata for {image_id}")
        if row["decision"] not in {"accepted","rejected","needs_review"}:
            raise ValueError(f"Unexpected review decision for {image_id}")
        try:
            score=float(row["morphology_score"])
        except ValueError as ex:
            raise ValueError(f"Invalid morphology score for {image_id}") from ex
        if not 0 <= score <= 4: raise ValueError(f"Outside [0,4] morphology scale: {image_id}")
        if row["near_copy_flag"].lower() not in {"yes","no","true","false","1","0"}:
            raise ValueError(f"Invalid near_copy_flag: {image_id}")
        near=row["near_copy_flag"].lower() in {"yes","true","1"}
        if row["decision"]=="accepted" and near:
            raise ValueError(f"Accepted near-copy flagged image: {image_id}")
        grouped[(row["class_name"],row["generator_id"])][row["decision"]]+=1
        if near: grouped[(row["class_name"],row["generator_id"])]["near_copy_flagged"]+=1
    result=[{
        "class_name":k[0],"generator_id":k[1],
        "screened":sum(c[d] for d in ("accepted","rejected","needs_review")),
        "accepted":c["accepted"],"rejected":c["rejected"],"needs_review":c["needs_review"],
        "near_copy_flagged":c["near_copy_flagged"]
    } for k,c in sorted(grouped.items())]
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"generation_quality_counts.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    meta={"screened":len(data),"accepted":sum(r["accepted"] for r in result),
          "all_reviewed":all(r["needs_review"]==0 for r in result),
          "morphology_scores_are_subjective_not_diagnostic":True}
    (out/"generation_quality_summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta
