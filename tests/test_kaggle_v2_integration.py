"""Offline no-image, no-GPU integration tests for the Kaggle v2 research contract."""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path

from dermsynreal.kaggle_runs import (
    prompt_collection,generation_manifest,prepare_lora_train_only,
    training_run_report,DATASET_SOURCE,
)
from dermsynreal.real_collection import (
    CLASS_FOLDERS,discover_root,inventory_collection,stable_id,
    export_training_manifests,
)

REPO=Path(__file__).resolve().parents[1]

def write_rows(path,rows):
    with open(path,"w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

class KaggleV2ContractTests(unittest.TestCase):
    def test_authored_prompt_banks_have_family_disjoint_splits(self):
        for cls in ("Acne","Fungal_Infection","Vitiligo","Hyperpigmentation","Skin_Cancer","Normal"):
            a=prompt_collection(REPO,cls,partition="train",limit=None)
            b=prompt_collection(REPO,cls,partition="val",limit=None)
            c=prompt_collection(REPO,cls,partition="test",limit=None)
            self.assertEqual((len(a),len(b),len(c)),(210,45,45))
            groups=[{r["prompt_family_id"] for r in x} for x in (a,b,c)]
            self.assertFalse(groups[0]&groups[1])
            self.assertFalse(groups[0]&groups[2])
            self.assertFalse(groups[1]&groups[2])
            self.assertTrue(all(r["negative_prompt"] for r in a+b+c))
            self.assertTrue(all(r["text"] for r in a+b+c))

    def test_generated_samples_require_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"test.png").write_bytes(b"ONLY A FAKE NO-IMAGE TEST FIXTURE")
            prompt=prompt_collection(REPO,"Acne")[0]
            r=generation_manifest([{"file":"test.png","prompt_id":prompt["id"],
                                     "seed":42,"base_model":"test-no-model"}],p/"reports",
                                  root=p,class_name="Acne",repo_root=REPO,
                                  prompt_partition="train",model_id="fake")
            self.assertEqual(r["generated_images"],1)
            with (p/"reports/generation_manifest.csv").open() as f:
                v=next(csv.DictReader(f))
            self.assertEqual(v["accepted"],"False")
            self.assertEqual(v["prompt_family_id"],prompt["prompt_family_id"])
            self.assertEqual(v["source_dataset"],DATASET_SOURCE)

    def test_raw_folder_names_and_exact_duplicate_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"Face_Skin_dataset"
            for folder in CLASS_FOLDERS.values():(root/folder).mkdir(parents=True)
            (root/"Acne/img1.jpg").write_bytes(b"a fake file")
            (root/"Acne/img2.jpg").write_bytes(b"a fake file")
            (root/"Skin_cancer/sjs-ten-photo.jpg").write_bytes(b"another fake")
            found=discover_root(Path(tmp))
            self.assertEqual(found,root.resolve())
            result=inventory_collection(found,Path(tmp)/"inventory")
            self.assertEqual(result["files_screened"],3)
            self.assertEqual(result["redundant_exact_file_copies"],1)
            self.assertEqual(result["suspect_filename_flags"]["possible_SJS_TEN_not_skin_cancer"],1)
            with (Path(tmp)/"inventory/review_draft.csv").open() as f:
                allrows=list(csv.DictReader(f))
            self.assertTrue(all(r["reviewer_decision"]=="needs_review" for r in allrows))

    def test_lora_train_only_never_sees_test_and_needs_passed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/"Face_Skin_dataset"
            for folder in CLASS_FOLDERS.values():(dataset/folder).mkdir(parents=True)
            exported=root/"saved_approved_splits"
            (exported/"train/Acne").mkdir(parents=True)
            (exported/"test/Acne").mkdir(parents=True)
            (exported/"train/Acne/a.png").write_bytes(b"fake train")
            (exported/"test/Acne/b.png").write_bytes(b"fake test")
            manifest=root/"real.csv"
            write_rows(manifest,[
                {"image_id":"train1","image_path":"train/Acne/a.png","label":"acne",
                 "split":"train","group_id":"G1","is_synthetic":"False"},
                {"image_id":"test1","image_path":"test/Acne/b.png","label":"acne",
                 "split":"test","group_id":"G2","is_synthetic":"False"},
            ])
            audit=root/"audit.json"
            audit.write_text('{"passed":false}')
            with self.assertRaises(ValueError):
                prepare_lora_train_only(real_split_manifest=manifest,
                    real_image_root=exported,cls="Acne",stage_dir=root/"lora",
                    origin_root=dataset,label_audit_json=audit)
            audit.write_text('{"passed":true}')
            staged,summary=prepare_lora_train_only(real_split_manifest=manifest,
                    real_image_root=exported,cls="Acne",stage_dir=root/"lora",
                    origin_root=dataset,label_audit_json=audit)
            self.assertEqual(summary["count"],1)
            self.assertEqual(len(list((staged/"train").glob("*.png"))),1)
            self.assertTrue((staged/"train/train1_a.png").is_symlink())

    def test_training_artifact_inventory_no_fake_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/"m.csv";manifest.write_text("image_path\n")
            model=root/"model";model.mkdir()
            (model/"best_weights.pth").write_bytes(b"fake no weights")
            record=training_run_report(model_output=model,manifest_csv=manifest,
                model_name="fixture_model",experiment="real_only",seed=13)
            self.assertEqual(record["config"]["checkpoint_count"],1)
            self.assertTrue((model/"artifact_inventory.csv").is_file())
            self.assertTrue((model/"run_log.jsonl").is_file())
            self.assertNotIn("macro_f1",record)

if __name__=="__main__":
    unittest.main()
