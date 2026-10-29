"""Generate publication-stage reports from explicitly frozen prediction CSVs.

No generated output is a medical validation. Results are never invented.
"""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .evaluation import CLASSES, read_predictions, metrics, source_metrics, compare_on_real_test

CONDITIONS = ("E0","E1","E2","E3","E4","E5")
CONTRASTS = (("E2","E0"), ("E2","E1"), ("E3","E2"), ("E4","E2"))


def _read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        r = csv.DictReader(stream)
        if not r.fieldnames or not set(required).issubset(r.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(set(required)-set(r.fieldnames or []))}")
        return list(r)


def _save_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path):
    """Minimal columns: run_id,condition,backbone,seed,predictions_csv,checkpoint_sha,
    config_sha,real_split_sha. Each run is a *trained* checkpoint, not a seed replicate
    of the test set. Paths are relative to the registry's directory unless absolute.
    """
    path = Path(path).resolve()
    names = ("run_id","condition","backbone","seed","predictions_csv",
             "checkpoint_sha","config_sha","real_split_sha")
    entries = _read_csv(path, names)
    if not entries:
        raise ValueError("Run registry is empty")
    seen = set()
    for r in entries:
        if not all(str(r[k]).strip() for k in names):
            raise ValueError(f"Registry row contains missing metadata: {r}")
        if r["run_id"] in seen:
            raise ValueError(f"Duplicate run_id {r['run_id']}")
        seen.add(r["run_id"])
        if r["condition"] not in CONDITIONS:
            raise ValueError(f"Unsupported arm {r['condition']}")
        try:
            int(r["seed"])
        except ValueError as ex:
            raise ValueError("seed must be an integer") from ex
        p = Path(r["predictions_csv"])
        r["resolved_predictions_csv"] = str(p if p.is_absolute() else path.parent / p)
        if not Path(r["resolved_predictions_csv"]).is_file():
            raise FileNotFoundError(r["resolved_predictions_csv"])
    return entries


def _source_rows(rows, min_support):
    """Source-specific macro-F1 is only displayed for all-six-class strata."""
    breakdown = defaultdict(list)
    for r in rows:
        if r["source"] != "real":
            continue
        source = r.get("dataset_source","").strip()
        if source:
            breakdown[source].append(r)
    output = []
    for source, sub in sorted(breakdown.items()):
        m = metrics(sub)
        all_classes = not m["missing_true_classes"]
        output.append({
            "dataset_source":source,"n":m["n"],"all_six_classes":all_classes,
            "macro_f1":m["macro_f1"] if all_classes and m["n"] >= min_support else "",
            "balanced_accuracy":m["balanced_accuracy"] if all_classes and m["n"] >= min_support else "",
            "note":"" if all_classes and m["n"] >= min_support else "descriptive only: missing classes/low N",
        })
    return output


def run_reporting(registry_csv, output_dir, *, label_audit_json,
                  iterations=2000, bootstrap_seed=42, min_source_support=24):
    """Write reproducible metrics without contacting any API or training a model.

    A passed label audit is mandatory, but cannot independently establish
    clinical correctness. All compared runs must have the same split SHA.
    Paired comparison is within the same backbone and seed, on exact image IDs.
    """
    audit = json.loads(Path(label_audit_json).read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError("The real label audit has not passed: refuse publication metrics")
    registry = load_registry(registry_csv)
    real_split_shas = {r["real_split_sha"] for r in registry}
    if len(real_split_shas) != 1:
        raise ValueError("All runs in one evaluation suite must share the frozen real_split_sha")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    indexed = {}
    summaries, class_rows, confusion_rows, source_rows, problems = [], [], [], [], []
    for run in registry:
        pred_path = Path(run["resolved_predictions_csv"])
        data = read_predictions(pred_path)
        real = [r for r in data if r["source"] == "real"]
        if not real:
            raise ValueError(f"{run['run_id']} has no REAL test predictions")
        report = metrics(real, require_all_classes=True)
        by_source = source_metrics(data)
        key = (run["backbone"], run["seed"], run["condition"])
        if key in indexed:
            raise ValueError(f"Repeated backbone/seed/arm combination: {key}")
        indexed[key] = (run, data)
        summaries.append({
            "run_id":run["run_id"],"condition":run["condition"],"backbone":run["backbone"],
            "seed":run["seed"],"real_test_n":report["n"],
            "real_test_macro_f1":report["macro_f1"],
            "real_test_balanced_accuracy":report["balanced_accuracy"],
            "real_test_accuracy":report["accuracy"],
            "synthetic_test_n":by_source.get("synthetic",{}).get("n",""),
            "synthetic_test_macro_f1":by_source.get("synthetic",{}).get("macro_f1",""),
            "synthetic_test_accuracy":by_source.get("synthetic",{}).get("accuracy",""),
            "real_split_sha":run["real_split_sha"],
            "checkpoint_sha":run["checkpoint_sha"],"config_sha":run["config_sha"]
        })
        for cls in CLASSES:
            class_rows.append({"run_id":run["run_id"],"condition":run["condition"],
                               "backbone":run["backbone"],"seed":run["seed"],
                               "class_name":cls,**report["per_class"][cls]})
            for predicted, count in report["confusion"][cls].items():
                confusion_rows.append({"run_id":run["run_id"],"true_label":cls,
                                       "pred_label":predicted,"count":count})
        for item in _source_rows(data, min_source_support):
            source_rows.append({"run_id":run["run_id"],**item})
        problems += [{"run_id":run["run_id"],"image_id":p["image_id"],
                      "true_label":p["true_label"],"pred_label":p["pred_label"],
                      "dataset_source":p.get("dataset_source","")}
                     for p in real if p["true_label"] != p["pred_label"]]
    paired = []
    for (backbone, seed, condition), (run, data) in sorted(indexed.items()):
        for treatment, baseline in CONTRASTS:
            if condition != treatment:
                continue
            other = indexed.get((backbone, seed, baseline))
            if other is None:
                continue  # Avoid pretending every optional arm exists.
            base_run, base_data = other
            pair = compare_on_real_test(base_data, data, iterations=iterations, seed=bootstrap_seed)
            paired.append({
                "backbone":backbone,"seed":seed,
                "treatment_run":run["run_id"],"baseline_run":base_run["run_id"],
                "contrast":f"{treatment}-{baseline}","delta_macro_f1":pair["delta_macro_f1"],
                "ci95_lower":pair["ci_95_percentile"][0],
                "ci95_upper":pair["ci_95_percentile"][1],
                "paired_n":pair["real_test_images"],"resampling_groups":pair["resampling_groups"],
                "p_value":"", "test_independence_note":"Group IDs may be inferred visual clusters"
            })
    _save_csv(out/"run_summary.csv", summaries, list(summaries[0]))
    _save_csv(out/"per_class_real_test.csv", class_rows, list(class_rows[0]))
    _save_csv(out/"confusion_real_test.csv", confusion_rows, list(confusion_rows[0]))
    _save_csv(out/"per_source_real_test.csv", source_rows,
              ["run_id","dataset_source","n","all_six_classes","macro_f1","balanced_accuracy","note"])
    _save_csv(out/"real_test_errors.csv", problems,
              ["run_id","image_id","true_label","pred_label","dataset_source"])
    _save_csv(out/"paired_real_test.csv", paired,
              ["backbone","seed","treatment_run","baseline_run","contrast","delta_macro_f1",
               "ci95_lower","ci95_upper","paired_n","resampling_groups","p_value","test_independence_note"])
    manifest = {
        "format_version":1,
        "registry":str(Path(registry_csv).resolve()),
        "registry_sha256":hashlib.sha256(Path(registry_csv).read_bytes()).hexdigest(),
        "label_audit":str(Path(label_audit_json).resolve()),
        "label_audit_sha256":hashlib.sha256(Path(label_audit_json).read_bytes()).hexdigest(),
        "real_split_sha":next(iter(real_split_shas)),
        "primary_test_domain":"real",
        "runs":len(summaries),"pairs":len(paired),
        "bootstrap_iterations":iterations,"bootstrap_seed":bootstrap_seed,
        "interpretation":"descriptive outputs only; uncertainty depends on true independence of resampling groups; no p-values calculated",
        "no_clinical_validation":True
    }
    (out/"report_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest
