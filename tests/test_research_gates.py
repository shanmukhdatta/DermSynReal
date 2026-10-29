"""Offline unit tests for strict labels, source-separated metrics and paired IDs.

Run: PYTHONPATH=src python -m unittest discover -s tests -p test_research_gates.py
"""
import csv
import tempfile
import unittest
from pathlib import Path

from dermsynreal.evaluation import (
    CLASSES, compare_on_real_test, metrics, read_predictions, source_metrics,
)
from dermsynreal.label_gate import inspect_manifest, write_audit


def prediction_rows(*, source="real", perfect=True):
    rows = []
    for i, label in enumerate(CLASSES):
        rows.append({
            "image_id": f"img-{i}",
            "true_label": label,
            "pred_label": label if perfect or i != 0 else "Normal",
            "source": source,
            "split": "test",
            "group_id": f"g-{i}",
        })
    return rows


def csv_file(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EvaluationTests(unittest.TestCase):
    def test_perfect_six_class_macro_f1(self):
        m = metrics(prediction_rows(), require_all_classes=True)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)

    def test_source_metrics_never_pool_domains(self):
        real = prediction_rows()
        synthetic = prediction_rows(source="synthetic", perfect=False)
        for index, row in enumerate(synthetic):
            row["image_id"] = f"s-{index}"
        result = source_metrics(real + synthetic)
        self.assertEqual(result["real"]["n"], 6)
        self.assertEqual(result["synthetic"]["n"], 6)
        self.assertEqual(result["real"]["macro_f1"], 1.0)
        self.assertLess(result["synthetic"]["macro_f1"], 1.0)

    def test_paired_bootstrap_is_reproducible(self):
        bad = prediction_rows(perfect=False)
        good = prediction_rows()
        first = compare_on_real_test(bad, good, iterations=200, seed=5)
        second = compare_on_real_test(bad, good, iterations=200, seed=5)
        self.assertEqual(first, second)
        self.assertGreater(first["delta_macro_f1"], 0)
        self.assertEqual(first["real_test_images"], 6)

    def test_paired_comparison_refuses_wrong_test_image_ids(self):
        baseline = prediction_rows()
        treatment = prediction_rows()
        treatment[0]["image_id"] = "DIFFERENT"
        with self.assertRaises(ValueError):
            compare_on_real_test(baseline, treatment, iterations=100)

    def test_read_rejects_duplicates_and_val_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            records = prediction_rows()
            records[-1]["image_id"] = records[0]["image_id"]
            csv_file(path, records)
            with self.assertRaises(ValueError):
                read_predictions(path)
            records = prediction_rows()
            records[1]["split"] = "val"
            csv_file(path, records)
            with self.assertRaises(ValueError):
                read_predictions(path)


class LabelGateTests(unittest.TestCase):
    def example_manifest(self):
        return [{
            "image_id": f"real-{i}",
            "source": "licensed_dataset",
            "class_name": label,
            "original_diagnosis": "melanoma" if label == "Skin_Cancer" else "healthy" if label == "Normal" else label,
            "reviewer_decision": "approved",
            "reviewer_id": "reviewer_a",
            "image_path": f"{label}/{i}.jpg",
        } for i, label in enumerate(CLASSES)]

    def test_six_reviewed_classes_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            csv_file(path, self.example_manifest())
            approved, queue, summary = inspect_manifest(path)
            self.assertEqual(len(approved), 6)
            self.assertEqual(queue, [])
            self.assertTrue(summary["passed"])
            self.assertTrue(write_audit(path, Path(tmp) / "out")["passed"])

    def test_reject_sjs_ten_as_skin_cancer(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.example_manifest()
            rows[CLASSES.index("Skin_Cancer")]["original_diagnosis"] = "SJS-TEN"
            path = Path(tmp) / "manifest.csv"
            csv_file(path, rows)
            _, queue, summary = inspect_manifest(path)
            self.assertEqual(len(queue), 1)
            self.assertFalse(summary["passed"])
            with self.assertRaises(ValueError):
                write_audit(path, Path(tmp) / "out")

    def test_reject_dermatitis_as_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.example_manifest()
            rows[CLASSES.index("Normal")]["image_path"] = "normal/perioral_dermatitis_22.png"
            path = Path(tmp) / "manifest.csv"
            csv_file(path, rows)
            _, queue, summary = inspect_manifest(path)
            self.assertEqual(len(queue), 1)
            self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
