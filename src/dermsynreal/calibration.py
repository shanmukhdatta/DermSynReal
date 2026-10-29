"""Optional calibration, reliability and Brier scores on predicted probabilities.

Pure Python, no plotting dependencies. Returns per-domain results separately.
Evaluate only after selecting hyperparameters on validation (never on final test).
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from .evaluation import _label
from .inference import TRAIN_ORDER


def calibration_by_domain(path, bins=10):
    if bins<2:raise ValueError("bins must be >=2")
    with Path(path).open(newline="",encoding="utf-8-sig") as handle:
        reader=csv.DictReader(handle)
        required={"image_id","true_label","pred_label","source","split"} | {"p_"+c for c in TRAIN_ORDER}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Prediction CSV lacks six probability columns; rerun checkpoint inference")
        rows=list(reader)
    by_domain={}
    for source in ("real","synthetic"):
        subset=[r for r in rows if r["source"]==source]
        if not subset:continue
        cells=[{"count":0,"correct":0.0,"confidence":0.0} for _ in range(bins)]
        nll,brier=0.0,0.0
        for r in subset:
            if r["split"].lower()!="test":raise ValueError("Only test rows allowed")
            p=[float(r["p_"+name]) for name in TRAIN_ORDER]
            if any(not math.isfinite(x) or x<0 or x>1 for x in p) or abs(sum(p)-1)>0.002:
                raise ValueError("Invalid probability distribution")
            truth=_label(r["true_label"]).lower()
            if truth not in TRAIN_ORDER:raise ValueError("Unrecognized true label")
            ti=TRAIN_ORDER.index(truth)
            chosen=max(range(6),key=p.__getitem__)
            if _label(r["pred_label"]).lower()!=TRAIN_ORDER[chosen]:
                raise ValueError("Stored predicted class differs from probability argmax")
            conf=p[chosen]
            bucket=min(int(conf*bins),bins-1)
            cells[bucket]["count"]+=1
            cells[bucket]["correct"]+=float(chosen==ti)
            cells[bucket]["confidence"]+=conf
            nll+=-math.log(max(1e-12,p[ti]))
            brier+=sum((p[j]-float(j==ti))**2 for j in range(6))
        details=[]
        ece=0.0
        for k,c in enumerate(cells):
            count=c["count"]
            acc=c["correct"]/count if count else None
            confidence=c["confidence"]/count if count else None
            if count:ece+=(count/len(subset))*abs(acc-confidence)
            details.append({"bin":k,"n":count,"accuracy":acc,"mean_confidence":confidence})
        by_domain[source]={"n":len(subset),"ece":ece,"multiclass_brier":brier/len(subset),
                           "nll":nll/len(subset),"bins":details}
    if not by_domain:raise ValueError("No evaluation rows")
    return by_domain
