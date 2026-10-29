"""Offline evaluation pipeline tests using generated *unit-test fixtures only*.

No clinical images, patient information or measured experiment metrics.
"""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path

from dermsynreal.calibration import calibration_by_domain
from dermsynreal.evaluation import CLASSES, read_predictions
from dermsynreal.evaluation_pipeline import run_reporting
from dermsynreal.inference import _read_manifest, TRAIN_ORDER, MODEL_ALLOWLIST
from dermsynreal.quality_audit import summarize_quality

def write_csv(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_predictions(treatment=False):
    rows=[]
    for i,name in enumerate(CLASSES):
        predicted = name if treatment or i != 0 else "Normal"
        row={"image_id":f"fake-id-{i}","true_label":name,"pred_label":predicted,
             "source":"real","split":"test","group_id":f"g-{i}",
             "dataset_source":"fabricated_test_fixture","prompt_family_id":""}
        probs=[0.02]*6
        index=TRAIN_ORDER.index(predicted.lower())
        probs[index]=0.90
        for j,c in enumerate(TRAIN_ORDER):row["p_"+c]=str(probs[j])
        rows.append(row)
    return rows


class EvaluationPipelineTests(unittest.TestCase):
    def test_report_outputs_and_paired_source_but_no_pvalues(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_csv(root/"e0.csv",fake_predictions())
            write_csv(root/"e2.csv",fake_predictions(treatment=True))
            write_csv(root/"runs.csv",[
                {"run_id":"model_e0","condition":"E0","backbone":"ResNet50","seed":"13",
                 "predictions_csv":"e0.csv","checkpoint_sha":"hash0","config_sha":"cfg0","real_split_sha":"split0"},
                {"run_id":"model_e2","condition":"E2","backbone":"ResNet50","seed":"13",
                 "predictions_csv":"e2.csv","checkpoint_sha":"hash2","config_sha":"cfg2","real_split_sha":"split0"},
            ])
            (root/"labels.json").write_text('{"passed":true}')
            info=run_reporting(root/"runs.csv",root/"reports",
                               label_audit_json=root/"labels.json",iterations=100)
            self.assertEqual(info["runs"],2)
            self.assertEqual(info["pairs"],1)
            self.assertTrue((root/"reports/paired_real_test.csv").is_file())
            with (root/"reports/paired_real_test.csv").open() as f:
                pair=next(csv.DictReader(f))
            self.assertEqual(pair["contrast"],"E2-E0")
            self.assertEqual(pair["p_value"],"")
            with (root/"reports/per_source_real_test.csv").open() as f:
                sources=list(csv.DictReader(f))
            self.assertEqual(len(sources),2)
            self.assertEqual(sources[0]["dataset_source"],"fabricated_test_fixture")
            self.assertEqual(sources[0]["macro_f1"],"")

    def test_paper_report_rejects_failed_audit_or_split_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_csv(root/"e0.csv",fake_predictions())
            write_csv(root/"e2.csv",fake_predictions(treatment=True))
            rows=[
                {"run_id":"a","condition":"E0","backbone":"ResNet50","seed":"13","predictions_csv":"e0.csv",
                 "checkpoint_sha":"h","config_sha":"c","real_split_sha":"A"},
                {"run_id":"b","condition":"E2","backbone":"ResNet50","seed":"13","predictions_csv":"e2.csv",
                 "checkpoint_sha":"h","config_sha":"c","real_split_sha":"B"}
            ]
            write_csv(root/"registry.csv",rows)
            (root/"audit.json").write_text('{"passed":false}')
            with self.assertRaises(ValueError):
                run_reporting(root/"registry.csv",root/"output",label_audit_json=root/"audit.json")
            (root/"audit.json").write_text('{"passed":true}')
            with self.assertRaises(ValueError):
                run_reporting(root/"registry.csv",root/"output",label_audit_json=root/"audit.json")

    def test_source_metadata_survives_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"rows.csv"
            write_csv(file,fake_predictions())
            rows=read_predictions(file)
            self.assertEqual(rows[0]["dataset_source"],"fabricated_test_fixture")

    def test_calibration_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"rows.csv"
            write_csv(file,fake_predictions(treatment=True))
            result=calibration_by_domain(file,bins=5)
            self.assertEqual(result["real"]["n"],6)
            self.assertNotIn("synthetic",result)
            self.assertGreaterEqual(result["real"]["ece"],0.0)

    def test_calibration_refuses_invalid_probability_sum(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"rows.csv"
            rows=fake_predictions()
            rows[0]["p_acne"]="999"
            write_csv(file,rows)
            with self.assertRaises(ValueError):
                calibration_by_domain(file)

    def test_timm_backbone_names_are_actual_repo_backbones(self):
        self.assertEqual(MODEL_ALLOWLIST["vit_small_patch14_dinov2.lvd142m"],518)
        self.assertNotIn("vit_base_patch14_dinov2.lvd142m",MODEL_ALLOWLIST)


class InferenceManifestTests(unittest.TestCase):
    def test_synthetic_test_must_not_share_prompt_family_with_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            test=[{"image_id":"s1","image_path":"s1.png","label":"acne","split":"test",
                   "group_id":"visual_1","prompt_family_id":"ACNE_GROUP_1"}]
            training=[{"image_id":"s0","image_path":"s0.png","label":"acne","split":"train",
                       "group_id":"synthetic_ACNE_GROUP_1","is_synthetic":"True"}]
            write_csv(root/"test.csv",test)
            write_csv(root/"train.csv",training)
            with self.assertRaises(ValueError):
                _read_manifest(root/"test.csv",domain="synthetic",
                               train_synthetic_manifest=root/"train.csv")
            test[0]["prompt_family_id"]="ACNE_GROUP_2"
            write_csv(root/"test.csv",test)
            rows=_read_manifest(root/"test.csv",domain="synthetic",
                                train_synthetic_manifest=root/"train.csv")
            self.assertEqual(len(rows),1)

    def test_real_manifest_rejects_validation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"test.csv"
            write_csv(path,[{"image_id":"a","image_path":"a.png","label":"acne",
                            "split":"val","group_id":"G1"}])
            with self.assertRaises(ValueError):
                _read_manifest(path,domain="real")


class GenerationReviewTests(unittest.TestCase):
    def test_quality_audit_records_review_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{"image_id":"i1","class_name":"Acne","prompt_id":"p1",
                   "prompt_family_id":"family1","generator_id":"mock-generator",
                   "reviewer_id":"mock-reviewer","decision":"accepted",
                   "morphology_score":"3","near_copy_flag":"no"}]
            write_csv(root/"reviews.csv",rows)
            result=summarize_quality(root/"reviews.csv",root/"out")
            self.assertEqual(result["accepted"],1)
            self.assertTrue(result["all_reviewed"])

    def test_accepting_flagged_near_copy_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{"image_id":"i1","class_name":"Acne","prompt_id":"p1",
                   "prompt_family_id":"family1","generator_id":"mock-generator",
                   "reviewer_id":"mock-reviewer","decision":"accepted",
                   "morphology_score":"4","near_copy_flag":"yes"}]
            write_csv(root/"reviews.csv",rows)
            with self.assertRaises(ValueError):
                summarize_quality(root/"reviews.csv",root/"out")


if __name__=="__main__":
    unittest.main()
