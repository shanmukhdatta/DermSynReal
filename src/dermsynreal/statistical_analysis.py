"""Descriptive across-seed analysis of actual real-test experiment reports.

Not independent patient cohorts: all training seeds share the same real test.
Paired group-bootstrap confidence intervals are produced separately.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS=("real_test_macro_f1","real_test_balanced_accuracy","real_test_accuracy")
CLASS_METRICS=("precision","recall","f1")


def _csv_read(path,required):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"Missing {sorted(set(required)-set(reader.fieldnames or []))} in {path}")
        return list(reader)


def _save(path,rows,fields):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def _describe(values):
    values=[float(v) for v in values]
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite or absent measured statistic")
    return {"n_seeds":len(values),
            "mean":statistics.mean(values),
            "sample_sd":statistics.stdev(values) if len(values)>1 else "",
            "median":statistics.median(values),
            "min":min(values),"max":max(values)}


def summarize_across_seeds(results_dir, output_dir=None):
    """Output paper-safe real-test summaries without fabricated p-values."""
    root=Path(results_dir)
    out=Path(output_dir) if output_dir else root
    out.mkdir(parents=True,exist_ok=True)
    runs=_csv_read(root/"run_summary.csv",
                  {"run_id","condition","backbone","seed","real_test_n","real_split_sha",*METRICS})
    classes=_csv_read(root/"per_class_real_test.csv",
                      {"run_id","condition","backbone","seed","class_name","support",*CLASS_METRICS})
    pairs=_csv_read(root/"paired_real_test.csv",
                    {"backbone","seed","contrast","delta_macro_f1"})
    if not runs:raise ValueError("Run summary contains no actual results")
    shas={r["real_split_sha"] for r in runs}
    if len(shas)!=1:raise ValueError("Frozen real split SHA differs across runs")
    test_sizes={int(r["real_test_n"]) for r in runs}
    if len(test_sizes)!=1 or not next(iter(test_sizes)):
        raise ValueError("Frozen test size differs across runs")
    seeds=defaultdict(set)
    grouped=defaultdict(list)
    for r in runs:
        key=(r["backbone"],r["condition"])
        if r["seed"] in seeds[key]:raise ValueError("Repeated training seed for one backbone/arm")
        seeds[key].add(r["seed"])
        grouped[key].append(r)
    by_arm=[]
    for (backbone,arm),rows in sorted(grouped.items()):
        for metric in METRICS:
            by_arm.append({"backbone":backbone,"condition":arm,"metric":metric,
                           **_describe([r[metric] for r in rows]),
                           "real_test_n":next(iter(test_sizes)),
                           "note":"Across training seeds, same real test images"})
    class_groups=defaultdict(list)
    class_seeds=set()
    for r in classes:
        key=(r["backbone"],r["condition"],r["class_name"])
        k=(*key,r["seed"])
        if k in class_seeds:raise ValueError("Duplicated class/seed report")
        class_seeds.add(k)
        if r["seed"] not in seeds[(r["backbone"],r["condition"])]:
            raise ValueError("Per-class seed missing matching run")
        class_groups[key].append(r)
    by_class=[]
    for (backbone,arm,cls),rows in sorted(class_groups.items()):
        if len(rows)!=len(seeds[(backbone,arm)]):
            raise ValueError("Incomplete class scores across seeds")
        supports={int(r["support"]) for r in rows}
        if len(supports)!=1:raise ValueError("Per-class test support differs across seeds")
        for metric in CLASS_METRICS:
            by_class.append({"backbone":backbone,"condition":arm,"class_name":cls,
                             "metric":metric,"real_test_class_n":next(iter(supports)),
                             **_describe([r[metric] for r in rows])})
    paired_groups=defaultdict(list)
    pair_seeds=defaultdict(set)
    for row in pairs:
        key=(row["backbone"],row["contrast"])
        if row["seed"] in pair_seeds[key]:raise ValueError("Duplicate paired contrast seed")
        pair_seeds[key].add(row["seed"]);paired_groups[key].append(row)
    by_paired=[]
    for (backbone,contrast),rows in sorted(paired_groups.items()):
        diffs=[float(r["delta_macro_f1"]) for r in rows]
        by_paired.append({"backbone":backbone,"contrast":contrast,
                          **_describe(diffs),
                          "improved_seeds":sum(x>0 for x in diffs),
                          "tied_seeds":sum(x==0 for x in diffs),
                          "worse_seeds":sum(x<0 for x in diffs),
                          "note":"Matched training seeds, not independent patient cohorts"})
    _save(out/"seed_summary_real_test.csv",by_arm,
          ["backbone","condition","metric","n_seeds","mean","sample_sd","median","min","max",
           "real_test_n","note"])
    _save(out/"seed_summary_per_class.csv",by_class,
          ["backbone","condition","class_name","metric","real_test_class_n",
           "n_seeds","mean","sample_sd","median","min","max"])
    _save(out/"seed_summary_paired_contrasts.csv",by_paired,
          ["backbone","contrast","n_seeds","mean","sample_sd","median","min","max",
           "improved_seeds","tied_seeds","worse_seeds","note"])
    meta={"format_version":1,
          "real_split_sha":next(iter(shas)),"real_test_n":next(iter(test_sizes)),
          "backbone_arm_groups":len(grouped),
          "paired_contrast_groups":len(paired_groups),
          "source_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                           for name in ("run_summary.csv","per_class_real_test.csv",
                                        "paired_real_test.csv")},
          "interpretation":"Descriptive mean, sample SD, median and range across training seeds; "
                           "single-seed SD is blank. These are not independent-patient "
                           "confidence intervals. Use Evaluation 04 grouped bootstrap CIs. "
                           "No p-values or significance claims generated."}
    (out/"statistical_analysis_summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta
