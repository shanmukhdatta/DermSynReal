"""Evaluate frozen per-image prediction CSVs.

Examples:
  PYTHONPATH=src python scripts/evaluate_predictions.py \
      --predictions results/E0_test.csv --out results/E0_metrics.json
  PYTHONPATH=src python scripts/evaluate_predictions.py \
      --predictions results/E0_test.csv --treatment results/E2_test.csv \
      --out results/E2_vs_E0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dermsynreal.evaluation import compare_on_real_test, read_predictions, source_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked-test prediction analysis")
    parser.add_argument("--predictions", required=True, help="Baseline run CSV")
    parser.add_argument("--treatment", help="Optional matched treatment-run CSV")
    parser.add_argument("--out", required=True, help="Output JSON")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    baseline = read_predictions(args.predictions)
    output = {
        "baseline_file": str(args.predictions),
        "baseline_by_test_source": source_metrics(baseline),
        "primary_test_source": "real",
    }
    if args.treatment:
        treatment = read_predictions(args.treatment)
        output["treatment_file"] = str(args.treatment)
        output["treatment_by_test_source"] = source_metrics(treatment)
        output["paired_real_test_comparison"] = compare_on_real_test(
            baseline, treatment, iterations=args.bootstrap, seed=args.seed
        )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "baseline_real_test": output["baseline_by_test_source"].get("real"),
        "delta_macro_f1": output.get("paired_real_test_comparison", {}).get("delta_macro_f1"),
    }, indent=2))


if __name__ == "__main__":
    main()
