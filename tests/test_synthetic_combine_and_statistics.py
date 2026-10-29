"""Offline statistical and reviewed synthetic-combination tests; synthetic bytes only."""
from __future__ import annotations
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from dermsynreal.combine_synthetic import combine_approved
from dermsynreal.statistical_analysis import summarize_across_seeds

def csv_write(path,records):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


class ReviewedCombinerTests(unittest.TestCase):
    def sample(self,image,*,mode="base",flag="True",review="approved",label="Acne",identifier="A1"):
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        return {
            "image_id":identifier,
            "image_path":"/kaggle/working/obsolete/image.png",
            "source_file":"images/example.png",
            "label":label.lower(),"class_name":label,
            "prompt_id":"P1","prompt_family_id":"F1",
            "prompt_partition":"train","prompt_type":"morphology",
            "model_id":"fake_model","generator_mode":mode,
            "accepted":flag,"reviewer_decision":review,"reviewer_id":"fixture_reviewer",
            "image_sha256":digest,"near_copy_flag":"no"
        }

    def test_remounts_reviews_and_preserves_prompt_id(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mount=root/"mounted"
            (mount/"images").mkdir(parents=True)
            image=mount/"images/example.png";image.write_bytes(b"synthetic unit-test bytes")
            manifest=root/"approved.csv";csv_write(manifest,[self.sample(image)])
            out=root/"combined.csv"
            result=combine_approved([manifest],[mount],out)
            self.assertEqual(result["approved_images"],1)
            with out.open(newline="") as f:record=next(csv.DictReader(f))
            self.assertEqual(record["image_path"],str(image))
            self.assertEqual(record["prompt_family_id"],"F1")
            self.assertTrue(out.with_suffix(".summary.json").is_file())

    def test_unreviewed_never_auto_approved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mount=root/"mounted"
            (mount/"images").mkdir(parents=True)
            image=mount/"images/example.png";image.write_bytes(b"fake")
            manifest=root/"raw.csv";csv_write(manifest,[self.sample(image,flag="False",review="needs_review")])
            with self.assertRaises(ValueError):
                combine_approved([manifest],[mount],root/"combined.csv")

    def test_reject_mixed_generator_modes_even_same_class(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mount=root/"mounted"
            (mount/"images").mkdir(parents=True)
            x=mount/"images/example.png";x.write_bytes(b"fake_a")
            y=mount/"images/other.png";y.write_bytes(b"fake_b")
            first=self.sample(x)
            second=self.sample(y,mode="custom_lora",identifier="A2")
            second["source_file"]="images/other.png"
            second["prompt_id"]="P2";second["prompt_family_id"]="F2"
            manifest=root/"raw.csv";csv_write(manifest,[first,second])
            with self.assertRaises(ValueError):
                combine_approved([manifest],[mount],root/"combined.csv")
            # Explicitly permitted for preprocessing; experiment builder must
            # filter the resulting pool to a matched generator mode.
            result=combine_approved([manifest],[mount],root/"mixed.csv",
                                    allow_mixed_generators=True)
            self.assertEqual(result["approved_images"],2)
            self.assertEqual(result["mixed_generator_classes"],["Acne"])

    def test_accepted_bad_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mount=root/"mounted"
            (mount/"images").mkdir(parents=True)
            image=mount/"images/example.png";image.write_bytes(b"factual fixture")
            row=self.sample(image);row["image_sha256"]="wrong"
            manifest=root/"bad.csv";csv_write(manifest,[row])
            with self.assertRaises(ValueError):
                combine_approved([manifest],[mount],root/"combined.csv")


class AcrossSeedSummaryTests(unittest.TestCase):
    def test_two_seed_mean_and_sample_sd(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            csv_write(root/"run_summary.csv",[
                {"run_id":"r1","condition":"E0","backbone":"ResNet50","seed":"13",
                 "real_test_n":"60","real_split_sha":"frozen-a",
                 "real_test_macro_f1":"0.6","real_test_balanced_accuracy":"0.6",
                 "real_test_accuracy":"0.6"},
                {"run_id":"r2","condition":"E0","backbone":"ResNet50","seed":"42",
                 "real_test_n":"60","real_split_sha":"frozen-a",
                 "real_test_macro_f1":"0.8","real_test_balanced_accuracy":"0.8",
                 "real_test_accuracy":"0.8"},
            ])
            csv_write(root/"per_class_real_test.csv",[
                {"run_id":rid,"condition":"E0","backbone":"ResNet50","seed":seed,
                 "class_name":"Acne","support":"10","precision":"0.5",
                 "recall":"0.7","f1":"0.6"} for rid,seed in (("r1","13"),("r2","42"))
            ])
            (root/"paired_real_test.csv").write_text(
                "backbone,seed,contrast,delta_macro_f1\n",encoding="utf-8")
            result=summarize_across_seeds(root)
            self.assertEqual(result["real_test_n"],60)
            with (root/"seed_summary_real_test.csv").open(newline="") as f:
                metrics=list(csv.DictReader(f))
            f1=next(x for x in metrics if x["metric"]=="real_test_macro_f1")
            self.assertAlmostEqual(float(f1["mean"]),0.7)
            self.assertAlmostEqual(float(f1["sample_sd"]),0.1414213562,places=6)
            self.assertTrue((root/"seed_summary_paired_contrasts.csv").is_file())

    def test_reject_different_test_split_fingerprints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            csv_write(root/"run_summary.csv",[
                {"run_id":"r1","condition":"E0","backbone":"ResNet50","seed":"13",
                 "real_test_n":"60","real_split_sha":"a",
                 "real_test_macro_f1":"0.6","real_test_balanced_accuracy":"0.6",
                 "real_test_accuracy":"0.6"},
                {"run_id":"r2","condition":"E0","backbone":"ResNet50","seed":"42",
                 "real_test_n":"60","real_split_sha":"b",
                 "real_test_macro_f1":"0.6","real_test_balanced_accuracy":"0.6",
                 "real_test_accuracy":"0.6"},
            ])
            (root/"per_class_real_test.csv").write_text(
                "run_id,condition,backbone,seed,class_name,support,precision,recall,f1\n")
            (root/"paired_real_test.csv").write_text(
                "backbone,seed,contrast,delta_macro_f1\n")
            with self.assertRaises(ValueError):
                summarize_across_seeds(root)

if __name__=="__main__":
    unittest.main()
